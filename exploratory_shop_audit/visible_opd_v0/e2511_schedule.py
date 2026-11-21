"""Registered log-log interpolation and independent update indices for Section 25.11."""
from math import exp, log, sqrt

CAL = 'calibration_2511'
MATCH = 'R-matched'
ARMS = ('R', 'P', MATCH)


def interpolate(rate1, movement1, rate2, movement2, target):
    if not (0 < rate1 < rate2 and 0 < movement1 < target < movement2):
        raise ValueError('The target must lie between the two observed positive movements')
    power = log(movement2 / movement1) / log(rate2 / rate1)
    return rate1 * exp(log(target / movement1) / power), power


def classify(matched, target):
    if matched < 0 or target <= 0:
        raise ValueError('Movement must be nonnegative and reference movement positive')
    ratio = sqrt(matched / target)
    return ratio, 'MATCHED' if .8 <= ratio <= 1.25 else 'MOVEMENT_MISMATCH'


def scheduled(round_number, stopped):
    if not 1 <= round_number <= 110:
        raise ValueError(round_number)
    jobs = [('formal', round_number, a) for a in ('R', 'P')
            if round_number <= 100 and a not in stopped]
    if round_number <= 10:
        jobs.append((CAL, round_number, MATCH))
    elif MATCH not in stopped:
        jobs.append(('formal', round_number - 10, MATCH))
    return jobs
