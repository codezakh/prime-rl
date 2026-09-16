from __future__ import annotations

import math

import verifiers.v1 as vf

from prime_rl.orchestrator.algo.base import Algorithm, iter_trainable_traces
from prime_rl.transports.batch import TrainingSample


class SFTDistillAlgorithm(Algorithm):
    """Hard distillation. Needs a teacher: the frozen model that generates the
    rollouts (``sampling.source``); the policy trains with CE on its tokens.

    Assigns no advantage — the ``ce`` loss ignores credit, and SFT trains on
    every sampled token. A curriculum can reject results using reward or any
    other finalized rollout data."""

    action_loss_type = "ce"


class OnlineSFTAlgorithm(Algorithm):
    """Consume prepared trajectories with scalar reward weights. No rewriting or baseline."""

    action_loss_type = "ce"

    async def score_episode(self, episode: vf.Episode) -> None:
        for _, trace in iter_trainable_traces([episode]):
            weight = self.action_weight(trace)
            for node in trace.nodes:
                streams = dict(node.loss_weights or {})
                streams["ce"] = [weight if target else 0.0 for target in node.mask]
                node.loss_weights = streams

    def action_weight(self, trace: vf.Trace) -> float:
        weight = float(trace.reward)
        if not math.isfinite(weight):
            raise ValueError(f"online_sft requires a finite reward for trace {trace.id!r}")
        return weight

    def prepare_sample(self, trace: vf.Trace, sample: TrainingSample, temperature: float) -> None:
        super().prepare_sample(trace, sample, 1.0)
        # SFT optimizes the model distribution, not the producer's truncated sampler.
        sample.sampling_mask = None
