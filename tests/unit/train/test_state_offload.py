import pytest
import torch

from prime_rl.trainer.optim.state_offload import CPUOffloadOptimizer


def _model(device: str) -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Linear(16, 4)).to(device)


def _groups(model: torch.nn.Module) -> list[dict]:
    weights = [p for n, p in model.named_parameters() if n.endswith("weight")]
    biases = [p for n, p in model.named_parameters() if n.endswith("bias")]
    return [{"params": weights}, {"params": biases, "weight_decay": 0.0}]


def _train(model: torch.nn.Module, optimizer, steps: int = 3) -> None:
    generator = torch.Generator().manual_seed(1)
    device = next(model.parameters()).device
    for _ in range(steps):
        inputs = torch.randn(5, 8, generator=generator).to(device)
        model(inputs).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad()


def _offloaded_matches_plain_adamw(device: str) -> None:
    plain, chunked = _model(device), _model(device)
    _train(plain, torch.optim.AdamW(_groups(plain), lr=1e-2, weight_decay=0.1))
    # A budget smaller than one parameter: every parameter is stepped on its own.
    offloaded = CPUOffloadOptimizer(
        torch.optim.AdamW(_groups(chunked), lr=1e-2, weight_decay=0.1), pin_memory=False, step_chunk_bytes=1
    )
    _train(chunked, offloaded)
    for expected, actual in zip(plain.parameters(), chunked.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert [len(group["params"]) for group in offloaded.param_groups] == [2, 2]
    for state in offloaded.state.values():
        assert all(value.device.type == "cpu" for value in state.values() if isinstance(value, torch.Tensor))


def test_chunked_offloaded_step_matches_plain_adamw_and_restores_param_groups():
    _offloaded_matches_plain_adamw("cpu")


@pytest.mark.gpu
def test_chunked_offloaded_step_matches_plain_adamw_with_states_back_on_cpu():
    _offloaded_matches_plain_adamw("cuda")
