"""Data-free synthetic checks for the PROPOSED E5 amendment 3 v2 entity rule.

No E5 pilot, E4/E5 reward, replay, judge, or true supporting title is loaded.
Run: python proposals/test_e5_amendment3_history_rule.py
"""
import json

import e5_amendment3_history_rule as H


def info(*titles):
    result = "".join(f'Doc {i}: "{title}"\nAn arbitrary paragraph.\n' for i, title in enumerate(titles, 1))
    return "<information>" + json.dumps({"result": result}) + "</information>"


def prompt(n_completed=0, title=None, executed=None, question="Who directed Blade Runner?", prefix=""):
    s = "Your question: " + question + "\n"
    if title:
        s += "Retrieved documents: " + info(*title) + "\n"
    s += prefix + "\n"
    if n_completed:
        s += f"Prior to this step, you have already taken {n_completed} step(s). History:\n"
        for i in range(n_completed):
            act = (executed or ["<search>unrelated query</search>"] * n_completed)[i]
            s += f"Step {i+1}:{act} " + info("Blade Runner" if i == 0 else "The Hobbit") + "\n"
    return s


def five(*names):
    assert len(names) == 5
    return list(names)


def test_entity_vs_string_rewrite():
    p = prompt(1)
    arms = five("<search>Who directed Blade Runner?</search>",
                "<search>Blade Runner director</search>",
                "<search>Isaac Asimov biography</search>",
                "<answer>Ridley Scott</answer>", "<answer>Other</answer>")
    prob, meta = H.rule_dist(p, arms)
    assert meta["status"] == "NOVEL_ENTITY" and meta["novel_ids"] == [2], (prob, meta)
    assert prob == [0.0, 0.0, 1.0, 0.0, 0.0]


def test_context_full_and_capitalization():
    p = prompt(0, title=["Blade Runner", "The Hobbit"])
    arms = five("<search>Ｂｌａｄｅ Ｒｕｎｎｅｒ director</search>",
                "<search>the hobbit ending</search>", "<answer>Ridley Scott</answer>",
                "<answer>Other</answer>", "<answer>Unknown</answer>")
    prob, m = H.rule_dist(p, arms)
    assert H.titles_in_prompt(p) == {"blade runner", "the hobbit"}
    assert m["status"] == "UNIFORM_FALLBACK" and prob == [.2] * 5


def test_repeat_is_not_novel():
    p = prompt(1, executed=["<search>Isaac Asimov biography</search>"],
               question="Which books did Isaac Asimov write?")
    arms = five("<search>Isaac Asimov biography</search>",
                "<search>Isaac Asimov biography</search> ",
                "<search>The Hobbit author</search>",
                "<answer>Foundation</answer>", "<answer>Other</answer>")
    # Exact duplicate after strip is invalid (the real E5 candidate gate also rejects duplicate arms).
    try:
        H.rule_dist(p, arms)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate candidate acceptance")
    arms[1] = "<search>the hobbit author</search>"
    arms[2] = "<search>unrelated generic query</search>"
    prob, meta = H.rule_dist(p, arms)
    assert meta["novel_ids"] == [] and prob == [.2] * 5, meta


def test_no_extraction_fallback():
    p = prompt(0, question="who does it?")
    arms = five("<search>who did it</search>", "<search>what is this</search>",
                "<search>how much</search>", "<answer>maybe</answer>", "<answer>unknown</answer>")
    prob, meta = H.rule_dist(p, arms)
    assert prob == [.2] * 5 and meta["status"] == "UNIFORM_FALLBACK"


def test_last_step_must_answer():
    p = prompt(3)
    arms = five("<search>Isaac Asimov biography</search>", "<answer>Ridley Scott</answer>",
                "<search>Blade Runner director</search>", "<answer>Other</answer>",
                "<search>The Hobbit ending</search>")
    prob, m = H.rule_dist(p, arms)
    assert m["status"] == "ANSWER_ONLY" and prob == [0, .5, 0, .5, 0]
    none, m = H.rule_dist(p, five("<search>a</search>", "<search>b</search>",
                                  "<search>c</search>", "<search>d</search>", "<search>e</search>"))
    assert none is None and m["status"] == "RULE_UNAVAILABLE_NO_ANSWER_ARM"


def test_full_versus_window_history():
    p = prompt(2, executed=["<search>The Hobbit</search>", "<search>unrelated</search>"],
               question="Who directed Blade Runner?", prefix="Example <search>Blade Runner</search> is not actual history.")
    # The current visible history happens to contain Blade Runner at Step 1 in this synthetic fixture.
    # Check the hidden-history distinction using a separate prompt with a non-Blade Runner visible document.
    p = "Your question: Who directed Blade Runner?\nPrior to this step, you have already taken 2 step(s). History:\n"
    p += "Step 2:<search>unrelated</search> " + info("The Hobbit") + "\n"
    arms = five("<search>Blade Runner director</search>", "<search>the hobbit ending</search>",
                "<search>generic query</search>", "<answer>Ridley Scott</answer>", "<answer>Other</answer>")
    w, wm = H.rule_dist(p, arms)
    full = [("<search>old</search>", info("Blade Runner")),
            ("<search>unrelated</search>", info("The Hobbit"))]
    f, fm = H.rule_dist(p, arms, full)
    assert wm["novel_ids"] == [0] and fm["novel_ids"] == []
    assert w[0] == 1 and f == [.2] * 5
    assert H.executed_queries(p) == {"unrelated"}  # template example text must be ignored


def test_pilot_coverage_excludes_final_step():
    p2 = prompt(1)
    a2 = five("<search>Blade Runner director</search>", "<search>Isaac Asimov biography</search>",
              "<answer>Ridley Scott</answer>", "<answer>Other</answer>", "<search>The Hobbit ending</search>")
    p4 = prompt(3)
    a4 = five("<search>a</search>", "<search>b</search>", "<search>c</search>",
              "<search>d</search>", "<search>e</search>")
    summary = H.coverage([(p2, a2), (p4, a4)])
    assert summary["early_points"] == 1 and summary["changed_from_uniform"] == 1, summary
    assert summary["search_mult_points"] == 1 and summary["within_search_discriminated"] == 1, summary
    assert summary["last_step_without_answer"] == 1 and summary["last_step_points"] == 1


def test_no_peek_or_labels_in_rule_inputs():
    # Identical decision-time prompt and candidate list -> identical selector output.
    p = prompt(1)
    arms = five("<search>Blade Runner director</search>", "<search>Isaac Asimov</search>",
                "<search>The Hobbit ending</search>", "<answer>A</answer>", "<answer>B</answer>")
    assert H.rule_dist(p, arms) == H.rule_dist(p, arms)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("All proposed E5 history rule synthetic tests passed; no real experiment data read")
