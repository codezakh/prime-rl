"""Exact finite-vocabulary checks of the production OPD objective."""

from types import SimpleNamespace

import torch

from prime_rl.configs.trainer import FakeDataLoaderConfig
from prime_rl.trainer.rl import data
from prime_rl.trainer.rl.loss import LossInputs, compute_loss, ref_kl_loss_fn


def test_full_length_debug_batch_selects_only_opd(monkeypatch):
    monkeypatch.setattr(data, "get_world", lambda: SimpleNamespace(rank=0, world_size=1))
    loader = data.FakeDataLoader(FakeDataLoaderConfig(batch_size=1, loss_component="ref_kl"), 65536, 1)
    batch = loader.get_batch()[0]
    assert batch["sequence_lengths"] == [65536]
    assert batch["position_ids"][0, -1] == 65535
    assert torch.all(batch["rl_weights"] == 0)
    assert torch.all(batch["ref_kl_weights"] == 1)
    assert torch.all(batch["ref_logprobs"] - batch["inference_logprobs"] == 1)


def test_on_policy_gradient_matches_exact_reverse_kl():
    logits = torch.tensor([0.4, -0.3, 1.2], dtype=torch.float64, requires_grad=True)
    logp = logits.log_softmax(-1)
    logq = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float64).log()
    inputs = LossInputs(
        trainer_logprobs=logp,
        inference_logprobs=logp.detach(),
        ref_logprobs=logq,
        advantages=torch.zeros_like(logp),
        loss_mask=torch.ones(3, dtype=torch.bool),
        loss_weights=logp.detach().exp(),
    )
    actual = torch.autograd.grad(ref_kl_loss_fn(inputs).loss, logits, retain_graph=True)[0]
    expected = torch.autograd.grad((logp.exp() * (logp - logq)).sum(), logits)[0]
    torch.testing.assert_close(actual, expected)


def test_stale_rollout_signal_is_frozen_and_environment_tokens_are_ignored():
    policy = torch.tensor([-1.0, -2.0, float("nan")], requires_grad=True)
    behavior = torch.tensor([-2.0, -1.0, float("nan")], requires_grad=True)
    teacher = torch.tensor([-1.5, -1.5, float("nan")], requires_grad=True)
    inputs = LossInputs(
        trainer_logprobs=policy,
        inference_logprobs=behavior,
        ref_logprobs=teacher,
        advantages=torch.zeros(3),
        loss_mask=torch.tensor([True, True, False]),
    )
    loss = ref_kl_loss_fn(inputs).loss
    loss.backward()
    expected = -(teacher[:2] - behavior[:2]).detach() * (policy[:2] - behavior[:2]).detach().exp()
    torch.testing.assert_close(policy.grad[:2], expected)
    assert policy.grad[2] == 0
    assert behavior.grad is None
    assert teacher.grad is None
    assert torch.isfinite(loss)


def test_production_dispatch_normalizes_opd_and_ignores_masked_nan_tokens():
    policy = torch.tensor([-1.0, -2.0, float("nan"), -3.0], requires_grad=True)
    behavior = torch.tensor([-1.2, -2.2, float("nan"), -2.8], requires_grad=True)
    teacher = torch.tensor([-0.8, -1.5, float("nan"), -3.2], requires_grad=True)
    mask = torch.tensor([True, True, False, True])

    def unused_rl_loss(inputs):
        raise AssertionError("Pure OPD must not call the reward-based loss")

    loss, metrics = compute_loss(
        trainer_logprobs=list(policy.split(2)),
        inference_logprobs=list(behavior.split(2)),
        ref_logprobs=list(teacher.split(2)),
        advantages=list(torch.zeros(4).split(2)),
        loss_mask=list(mask.split(2)),
        rl_weights=list(torch.zeros(4).split(2)),
        ce_weights=None,
        ref_kl_weights=list(mask.float().split(2)),
        rl_loss_fn=unused_rl_loss,
        rl_scale=1,
        ce_scale=1,
        ref_kl_scale=3,
    )
    loss.backward()
    expected = -(teacher[mask] - behavior[mask]).detach() * (policy[mask] - behavior[mask]).detach().exp() / 3
    torch.testing.assert_close(policy.grad[mask], expected)
    torch.testing.assert_close(loss.detach(), expected.sum())
    assert policy.grad[2] == 0
    assert behavior.grad is None and teacher.grad is None
    assert "ref_kl/teacher_kl" in metrics
