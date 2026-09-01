# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Train/eval split: disjointness assertion + deterministic auto-split."""
import json

from app.split import assert_disjoint, auto_split, full_row_key, prompt_key


def _jsonl(rows):
    return "\n".join(json.dumps(r) for r in rows)


def _row(q, a):
    return {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}


def test_disjoint_sets_pass():
    train = _jsonl([_row("q1", "a1"), _row("q2", "a2")])
    ev = _jsonl([_row("q3", "a3")])
    r = assert_disjoint(train, ev)
    assert r.ok
    assert r.overlap_count == 0
    assert r.train_rows == 2 and r.eval_rows == 1


def test_overlap_detected_and_blocks():
    shared = _row("dup", "same")
    train = _jsonl([_row("q1", "a1"), shared])
    ev = _jsonl([shared])
    r = assert_disjoint(train, ev)
    assert not r.ok
    assert r.overlap_count == 1


def test_invalid_row_blocks_split():
    # An invalid row (no assistant) must block the whole split, not silently drop.
    train = _jsonl([_row("q1", "a1")]) + '\n{"messages":[{"role":"user","content":"x"}]}'
    ev = _jsonl([_row("q2", "a2")])
    r = assert_disjoint(train, ev)
    assert not r.ok


def test_auto_split_deterministic():
    rows = [_row(f"q{i}", f"a{i}") for i in range(20)]
    text = _jsonl(rows)
    r1 = auto_split(text, eval_ratio=0.2, seed=42)
    r2 = auto_split(text, eval_ratio=0.2, seed=42)
    assert r1.ok and r2.ok
    # Same seed → identical membership (split_id is assigned at persist time, so
    # compare the actual eval rows instead).
    assert r1.eval_rows == r2.eval_rows == 4
    assert r1.train_rows == 16
    assert r1.eval_rows_full == r2.eval_rows_full


def test_auto_split_disjoint_by_construction():
    rows = [_row(f"q{i}", f"a{i}") for i in range(20)]
    r = auto_split(_jsonl(rows), eval_ratio=0.25, seed=7)
    assert r.ok
    assert r.overlap_count == 0


def test_different_seed_different_split():
    rows = [_row(f"q{i}", f"a{i}") for i in range(20)]
    a = auto_split(_jsonl(rows), eval_ratio=0.2, seed=1)
    b = auto_split(_jsonl(rows), eval_ratio=0.2, seed=2)
    # Different seed → very likely different eval membership.
    assert a.eval_rows_full != b.eval_rows_full


def test_keys_distinguish_full_vs_prompt():
    r1 = _row("same-q", "answer-A")
    r2 = _row("same-q", "answer-B")
    # Same prompt, different answer: prompt keys match, full keys differ.
    assert prompt_key(r1) == prompt_key(r2)
    assert full_row_key(r1) != full_row_key(r2)


# --- Optional validation set ---


def test_three_file_val_pairwise_disjoint():
    train = _jsonl([_row("q1", "a1"), _row("q2", "a2")])
    val = _jsonl([_row("v1", "b1")])
    ev = _jsonl([_row("e1", "c1")])
    r = assert_disjoint(train, ev, val_text=val)
    assert r.ok
    assert r.has_val and r.val_mode == "file" and r.val_rows == 1
    assert r.overlap_count == 0


def test_val_overlapping_train_blocks():
    shared = _row("dup", "same")
    train = _jsonl([_row("q1", "a1"), shared])
    val = _jsonl([shared])  # val row identical to a train row
    ev = _jsonl([_row("e1", "c1")])
    r = assert_disjoint(train, ev, val_text=val)
    assert not r.ok
    assert r.overlap_count >= 1


def test_assert_carve_val_from_train():
    train = _jsonl([_row(f"q{i}", f"a{i}") for i in range(10)])
    ev = _jsonl([_row("e1", "c1")])
    r = assert_disjoint(train, ev, val_ratio=0.2)
    assert r.ok
    assert r.has_val and r.val_mode == "carve" and r.val_ratio == 0.2
    # 20% of 10 train rows carved to val; rest stay train; eval untouched.
    assert r.val_rows == 2 and r.train_rows == 8 and r.eval_rows == 1
    # Carved val must be disjoint from the remaining train.
    train_keys = {full_row_key(x) for x in r.train_rows_full}
    assert all(full_row_key(x) not in train_keys for x in r.val_rows_full)


def test_auto_split_with_val_carve_three_way_disjoint():
    rows = [_row(f"q{i}", f"a{i}") for i in range(20)]
    r = auto_split(_jsonl(rows), eval_ratio=0.2, seed=7, val_ratio=0.25)
    assert r.ok and r.has_val and r.val_mode == "carve"
    tk = {full_row_key(x) for x in r.train_rows_full}
    vk = {full_row_key(x) for x in r.val_rows_full}
    ek = {full_row_key(x) for x in r.eval_rows_full}
    # Pairwise disjoint by construction.
    assert tk.isdisjoint(vk) and tk.isdisjoint(ek) and vk.isdisjoint(ek)
    # Counts add up to the original unique rows.
    assert len(tk) + len(vk) + len(ek) == 20


def test_no_val_unchanged_behaviour():
    train = _jsonl([_row("q1", "a1")])
    ev = _jsonl([_row("e1", "c1")])
    r = assert_disjoint(train, ev)
    assert r.ok and not r.has_val and r.val_mode == "" and r.val_rows == 0


# --- stratified sampling (proportional / mirror) ---------------------------

def _labeled(n):
    """n rows across 4 classes in a fixed imbalanced ratio (World heavy)."""
    classes = ["World"] * 50 + ["Sports"] * 20 + ["Business"] * 20 + ["Sci/Tech"] * 10
    rows = []
    for i in range(n):
        lab = classes[i % len(classes)]
        rows.append(_row(f"headline number {i}", lab))
    return rows


def _class_dist(rows):
    from collections import Counter
    c = Counter(r["messages"][-1]["content"] for r in rows)
    tot = sum(c.values())
    return {k: round(v / tot, 2) for k, v in c.items()}


def test_stratified_preserves_class_proportions():
    rows = _labeled(200)
    r = auto_split(_jsonl(rows), eval_ratio=0.25, seed=42, stratify=True)
    assert r.ok
    # every class present in BOTH splits, and proportions ~match between them
    train_d = _class_dist(r.train_rows_full)
    eval_d = _class_dist(r.eval_rows_full)
    assert set(train_d) == set(eval_d) == {"World", "Sports", "Business", "Sci/Tech"}
    for cls in train_d:
        assert abs(train_d[cls] - eval_d[cls]) <= 0.05  # mirror within 5pts


def test_stratified_deterministic():
    rows = _labeled(120)
    a = auto_split(_jsonl(rows), eval_ratio=0.2, seed=7, stratify=True)
    b = auto_split(_jsonl(rows), eval_ratio=0.2, seed=7, stratify=True)
    assert [full_row_key(x) for x in a.eval_rows_full] == [full_row_key(x) for x in b.eval_rows_full]


def test_stratified_with_val_three_way():
    rows = _labeled(200)
    r = auto_split(_jsonl(rows), eval_ratio=0.2, seed=7, val_ratio=0.1, stratify=True)
    assert r.has_val and r.val_rows > 0
    # all three splits carry every class
    assert set(_class_dist(r.val_rows_full)) == {"World", "Sports", "Business", "Sci/Tech"}
    # disjoint by construction
    keys = lambda L: {full_row_key(x) for x in L}
    assert not (keys(r.train_rows_full) & keys(r.eval_rows_full))
    assert not (keys(r.train_rows_full) & keys(r.val_rows_full))
    assert not (keys(r.val_rows_full) & keys(r.eval_rows_full))


def test_stratify_falls_back_for_freeform():
    # long unique sentences = freeform text (not a label task) → stratify no-ops
    rows = [_row(f"q{i}", f"This is a long unique free-form answer number {i} with detail.") for i in range(40)]
    r = auto_split(_jsonl(rows), eval_ratio=0.25, seed=42, stratify=True)
    assert r.ok
    assert any("not applied" in m for m in r.messages)  # noted the fallback


def test_auto_split_val_ratio_is_whole_fraction():
    """val_ratio is a fraction of the WHOLE dataset (so train/val/test sum to 1,
    matching the UI's 3-way split), not a fraction of the post-eval train slice."""
    import json
    from app.split import auto_split
    rows = "\n".join(
        json.dumps({"messages": [{"role": "user", "content": f"q{i}"},
                                 {"role": "assistant", "content": f"a{i}"}]})
        for i in range(100)
    )
    r = auto_split(rows, eval_ratio=0.1, seed=42, val_ratio=0.1)
    assert r.eval_rows == 10
    assert 9 <= r.val_rows <= 11   # ~10% of the WHOLE, not 10% of train (~9)
    assert 78 <= r.train_rows <= 82
    # disjoint + complete
    assert r.train_rows + r.val_rows + r.eval_rows == 100
