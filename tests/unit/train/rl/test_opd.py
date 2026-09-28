"""Exact finite-vocabulary checks of the production OPD objective."""

import torch

from prime_rl.trainer.rl.loss import LossInputs, ref_kl_loss_fn


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
