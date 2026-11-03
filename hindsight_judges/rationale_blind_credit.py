"""Minimal candidate method, pending the action diagnostic.

A rationale-blind SAME-model teacher measures only information gain of F
relative to M for an action, using the SAME no-Think prefix. This signal
reweights reward-defined (G)RPO advantages and NEVER sets their sign.
Student's complete <think> and action tokens retain the policy gradient.

This is an algorithm module and CPU-tested sketch, NOT completed RL training.
"""
from __future__ import annotations

import argparse
import json
import math


def blind_weight(advantage: float, logp_future: float, logp_memory: float,
                 alpha: float = 0.5, lower: float = 0.5, upper: float = 2.0) -> float:
    """Positive weight; reward advantage sign remains authoritative.

    Same tokenization / same no-Think context for F and M, except privilege.
    If F boosts a good action or penalizes a bad one, magnify reward gradient.
    If F contradicts reward, attenuate, but never invert the update.
    """
    for value in (advantage, logp_future, logp_memory, alpha, lower, upper):
        if not math.isfinite(value):raise ValueError('Nonfinite blind-credit input')
    if not 0<lower<=1<=upper or alpha<0:raise ValueError('Invalid weight bounds')
    if advantage==0:return 1.0
    direction=1.0 if advantage>0 else -1.0
    delta=max(-20.0,min(20.0,logp_future-logp_memory))
    return max(lower,min(upper,math.exp(max(-50.,min(50.,alpha*direction*delta)))))


def weighted_ppo_policy_loss(new_logprobs, old_logprobs, reward_advantage,
                             blind_future_logp, blind_memory_logp,
                             response_mask, alpha=0.5, clip_eps=0.2):
    """One-trajectory PPO-style reward policy loss; NO teacher-imitation KL.

    All response positions, including Think, can carry gradients. `response_mask`
    is one for every visited response token and zero for prompt/padding.
    Advantage computed EXTERNALLY from independent environment rewards.
    """
    import torch
    if not isinstance(reward_advantage,(int,float)):
        raise TypeError('Reward advantage must be a scalar per sampled response')
    if response_mask.shape!=new_logprobs.shape or new_logprobs.shape!=old_logprobs.shape:
        raise ValueError('Token shapes do not match')
    if not 0 <= clip_eps < 1:raise ValueError('clip_eps must lie in [0,1)')
    w=blind_weight(float(reward_advantage),float(blind_future_logp),
                   float(blind_memory_logp),alpha=alpha)
    adv=torch.as_tensor(float(reward_advantage)*w,device=new_logprobs.device,dtype=new_logprobs.dtype)
    ratio=(new_logprobs-old_logprobs).clamp(-20.,20.).exp()
    objective=torch.minimum(ratio*adv,ratio.clamp(1-clip_eps,1+clip_eps)*adv)
    mask=response_mask.to(objective.dtype)
    if not torch.any(mask>0):raise ValueError('No response tokens selected')
    return -(objective*mask).sum()/mask.sum(),w


def selftest(_):
    import torch
    assert blind_weight(+1.,-3.,-4.)>1
    assert blind_weight(-1.,-4.,-3.)>1
    assert blind_weight(+1.,-4.,-3.)<1
    assert blind_weight(-1.,-3.,-4.)<1
    assert blind_weight(0,-5,0)==1
    log=torch.zeros(6,requires_grad=True)
    old=torch.zeros(6)
    # Includes four Think tokens and two action tokens: none is masked.
    loss,w=weighted_ppo_policy_loss(log,old,1.,-1.,-2.,torch.ones(6))
    loss.backward()
    assert (log.grad.abs()>0).all() and w>0
    print('SELFTEST PASS',json.dumps(dict(loss=float(loss.detach()),weight=w)))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['selftest'])
    globals()[p.parse_args().command](None)

if __name__=='__main__':main()
