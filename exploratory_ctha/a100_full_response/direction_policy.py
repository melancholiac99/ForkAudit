"""Explicit NEW protocol: full-response gate, action-span diagnostic."""
import math


def evaluate(means):
    if not all(math.isfinite(value) for value in means.values()):
        raise ValueError('Nonfinite direction diagnostic')
    return dict(means,
        full_response_passed=means['positive_delta_mean_response_logprob'] > 0 > means['negative_delta_mean_response_logprob'],
        action_span_signs_ok=means['positive_delta_mean_action_logprob'] > 0 > means['negative_delta_mean_action_logprob'],
        action_span_blocking=False, protocol='a100-full-response-r3-20261113',
        passed=means['positive_delta_mean_response_logprob'] > 0 > means['negative_delta_mean_response_logprob'])

