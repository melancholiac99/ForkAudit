"""November 20 amendment: P-pre behavior, evidence coverage and P2 exposure."""
from math import ceil

TASKS_PER_UPDATE = 32
ROLLOUTS_PER_TASK = 2
DECISION_CLASSES = ('search', 'click', 'option')

def behavior_result(estimate, ci95):
    lo, hi = (round(float(x),12) for x in ci95)
    point = round(float(estimate),12)
    if point >= .10 and lo > 0:
        return 'improved'
    if point < .10 and hi < .10:
        return 'no_improvement'
    return 'PREFLIGHT_INCONCLUSIVE'

def scoreable(selected, page, structurally_valid=True):
    return structurally_valid and (selected != 'evidence' or page == 'search_results')

def freeze_p2_minima(cal_counts, cal_tasks):
    """Counts include all eligible response positions from one rollout per distinct T_cal task."""
    if cal_tasks < 512 or any(cal_counts[c]<0 for c in DECISION_CLASSES):
        raise ValueError('incomplete calibration counts')
    return {c: max(1,ceil(.5*10*TASKS_PER_UPDATE*ROLLOUTS_PER_TASK*cal_counts[c]/cal_tasks))
            for c in DECISION_CLASSES}

def p2_coverage(minima, class_counts, distinct_tasks):
    return {c: class_counts[c]>=minima[c] and distinct_tasks[c]>=50 for c in DECISION_CLASSES}
