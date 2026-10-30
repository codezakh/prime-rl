import copy

import torch
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer

from prime_rl.trainer.optim.base import OffloadOptimizer


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor._local_tensor if isinstance(tensor, DTensor) else tensor


class CPUOffloadOptimizer(OffloadOptimizer):
    """Wraps an optimizer to keep states on CPU, moving to GPU only for step().

    Unlike FSDP's CPUOffload which offloads weights too, this keeps weights on GPU.
    With activation checkpointing, activations and optimizer states are never on GPU
    at the same time.

    step() runs the wrapped optimizer over the parameters in chunks of at most
    ``step_chunk_bytes`` of local parameter shard, with only that chunk's states on
    GPU. Bringing every state up at once puts weights, gradients and all of AdamW's
    moments on GPU together: for a 9B full finetune on two ranks that is about
    72 GiB per card. Per-parameter optimizers such as AdamW compute the same update
    either way; gradient clipping has already run on the full set.
    """

    def __init__(self, optimizer: Optimizer, pin_memory: bool = True, step_chunk_bytes: int = 1 << 30):
        self.optimizer = optimizer
        self.pin_memory = pin_memory
        self.step_chunk_bytes = step_chunk_bytes
        self._initialized = False

    def _move_states(self, device: str, params=None):
        """Move optimizer states to CPU or back to GPU, for *params* or for every parameter."""
        for param in self.optimizer.state if params is None else params:
            if param not in self.optimizer.state:
                continue
            state = self.optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, DTensor):
                    local_tensor = value._local_tensor
                    if device == "cpu":
                        non_blocking = not self.pin_memory
                        new_local = local_tensor.to("cpu", non_blocking=non_blocking)
                        if self.pin_memory and not new_local.is_pinned():
                            new_local = new_local.pin_memory()
                    else:
                        new_local = local_tensor.to(device, non_blocking=True)
                    new_dtensor = copy.copy(value)
                    new_dtensor._local_tensor = new_local
                    state[key] = new_dtensor
                elif isinstance(value, torch.Tensor):
                    if device == "cpu":
                        non_blocking = not self.pin_memory
                        cpu_tensor = value.to("cpu", non_blocking=non_blocking)
                        if self.pin_memory and not cpu_tensor.is_pinned():
                            cpu_tensor = cpu_tensor.pin_memory()
                        state[key] = cpu_tensor
                    else:
                        state[key] = value.to(device, non_blocking=True)

    def _step_chunks(self) -> list[list[list[torch.Tensor]]]:
        """Each group's parameters, split into chunks of about ``step_chunk_bytes`` of local shard."""
        chunks, current, size = [], [[] for _ in self.optimizer.param_groups], 0
        for index, group in enumerate(self.optimizer.param_groups):
            for param in group["params"]:
                local = _local(param)
                if size and size + local.numel() * local.element_size() > self.step_chunk_bytes:
                    chunks.append(current)
                    current, size = [[] for _ in self.optimizer.param_groups], 0
                current[index].append(param)
                size += local.numel() * local.element_size()
        chunks.append(current)
        return chunks

    def step(self, closure=None):
        if closure is not None:
            raise NotImplementedError("CPUOffloadOptimizer steps in chunks and cannot re-evaluate a closure")
        groups = self.optimizer.param_groups
        all_params = [group["params"] for group in groups]
        try:
            for chunk in self._step_chunks():
                params = [param for group_params in chunk for param in group_params]
                if not params:
                    continue
                # States not created yet (the first step) are created beside their
                # parameters by the step itself.
                self._move_states(str(_local(params[0]).device), params)
                for group, group_params in zip(groups, chunk):
                    group["params"] = group_params
                self.optimizer.step()
                self._move_states("cpu", params)
        finally:
            for group, group_params in zip(groups, all_params):
                group["params"] = group_params
        self._initialized = True

    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        # Move to GPU temporarily for consistent state dict
        if self._initialized:
            self._move_states("cuda")
            torch.cuda.synchronize()
        state_dict = self.optimizer.state_dict()
        if self._initialized:
            self._move_states("cpu")
        return state_dict

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
        self._move_states("cpu")
        self._initialized = True

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optimizer.param_groups = value

    @property
    def state(self):
        return self.optimizer.state

    @property
    def base_optimizer(self) -> Optimizer:
        return self.optimizer

    def checkpoint_optimizer(self) -> Optimizer:
        return self.optimizer

    def prepare_checkpoint_save(self) -> None:
        if self._initialized:
            self._move_states("cuda")
            torch.cuda.synchronize()

    def finish_checkpoint_save(self) -> None:
        self._move_states("cpu")

    def finish_checkpoint_load(self) -> None:
        self._initialized = True
