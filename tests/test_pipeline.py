from atp.config import LeanState, ProofCandidate
from atp.pipeline import (
    apply_verifier_feedback,
    choose_next_node,
    create_root_node,
    expand_with_candidates,
    record_candidates,
    record_tree_nodes,
    score_candidate,
)


def test_score_candidate_combines_signals():
    score = score_candidate(
        0.6,
        retrieval_hits=2,
        verifier_bonus=0.1,
        error_penalty=0.05,
    )
    assert score == 0.95


def test_expand_with_candidates_creates_rankable_children():
    root = create_root_node("theorem demo : True")
    candidates = [
        ProofCandidate(
            candidate_id="c1",
            theorem_statement="goal 1",
            tactic_script="exact True.intro",
            score=0.9,
        ),
        ProofCandidate(
            candidate_id="c2",
            theorem_statement="goal 2",
            tactic_script="simp",
            score=0.3,
        ),
    ]

    children = expand_with_candidates(root, candidates)

    assert [child.node_id for child in children] == ["root.1", "root.2"]
    assert children[0].priority > children[1].priority
    assert children[0].parent_id == "root"


def test_choose_next_node_prefers_best_priority():
    state = LeanState(query="demo")
    root = create_root_node("root goal")
    children = expand_with_candidates(
        root,
        [
            ProofCandidate(candidate_id="a", theorem_statement="A", score=0.2),
            ProofCandidate(candidate_id="b", theorem_statement="B", score=0.8),
        ],
    )
    record_tree_nodes(state, [root, *children])

    best = choose_next_node(state)

    assert best is not None
    assert best.node_id == "root"

    root.status = "expanded"
    best = choose_next_node(state)
    assert best is not None
    assert best.node_id == "root.2"


def test_apply_verifier_feedback_marks_terminal_branch():
    candidate = ProofCandidate(candidate_id="ok", theorem_statement="goal", score=0.7)
    node = expand_with_candidates(create_root_node("goal"), [candidate])[0]

    updated = apply_verifier_feedback(
        node,
        solved=True,
        note="Lean accepted the script.",
        priority_delta=0.2,
    )

    assert updated.status == "proved"
    assert updated.visits == 1
    assert updated.priority == 0.9
    assert updated.candidate is not None
    assert updated.candidate.notes == ["Lean accepted the script."]


def test_record_candidates_updates_shared_state():
    state = LeanState(query="demo")
    candidates = [
        ProofCandidate(candidate_id="1", theorem_statement="goal A"),
        ProofCandidate(candidate_id="2", theorem_statement="goal B"),
    ]

    record_candidates(state, candidates)

    assert [candidate.candidate_id for candidate in state.candidate_proofs] == ["1", "2"]
