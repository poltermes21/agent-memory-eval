"""Graph-projection tests: pure functions, no API calls, no database.

The invalidation path fires only 3 times on the whole corpus, so these tests are
most of its coverage. normalize_predicate() is lowercase+trim only -- predicates
are lemmatized at extraction time -- so what matters here is that
INVALIDATING_PREDICATES entries survive it and the invalidation logic is right.
"""
from src.build_graph import INVALIDATING_PREDICATES, compute_invalidations, normalize_predicate


def test_normalize_predicate_is_lowercase_trim_only():
    assert normalize_predicate("Live_In") == "live_in"
    assert normalize_predicate("  work_at  ") == "work_at"


def test_invalidating_predicates_are_normalized():
    """Every entry must be a fixed point of normalize_predicate().

    Guards the bug the old stemmed version of this list shipped with: entries
    written in surface form ("live_in") never matched the stemmed predicates the
    lookup actually used ("liv_in"), so invalidation silently never fired. The
    stemmer is gone now, but the invariant is still worth asserting so a typo in
    this list (e.g. accidental trailing space or mixed case) fails loudly.
    """
    for predicate in INVALIDATING_PREDICATES:
        assert normalize_predicate(predicate) == predicate, predicate


def _fact(key, subject, predicate, obj, date, subject_type="Person", object_type="Place"):
    # (fact_key, subject, subject_type, predicate, object, object_type, fact, source_turn_id, session_date, valid_from)
    return (key, subject, subject_type, predicate, obj, object_type, f"{subject} {predicate} {obj}", "t", date, None)


def test_single_valued_predicate_invalidates():
    facts = [
        _fact("k1", "Caroline", "live_in", "Sweden", "2023-01-01"),
        _fact("k2", "Caroline", "live_in", "New York", "2023-05-01"),
    ]
    assert compute_invalidations(facts) == {"k1": ("2023-05-01", "k2")}


def test_restating_the_same_value_does_not_invalidate():
    facts = [
        _fact("k1", "Caroline", "live_in", "New York", "2023-01-01"),
        _fact("k2", "Caroline", "live_in", "new york", "2023-05-01"),  # case-only difference
    ]
    assert compute_invalidations(facts) == {}


def test_accumulating_predicates_never_invalidate():
    """The measured failure mode: "joanna feels X" across 12 different feelings is
    a history of states, not a correction. Same for owning several things.
    """
    facts = [
        _fact("k1", "Joanna", "feel", "relieved", "2023-01-01", object_type="Concept"),
        _fact("k2", "Joanna", "feel", "anxious", "2023-05-01", object_type="Concept"),
        _fact("k3", "Evan", "own", "old Prius", "2023-01-01", object_type="Object"),
        _fact("k4", "Evan", "own", "new Prius", "2023-05-01", object_type="Object"),
    ]
    assert compute_invalidations(facts) == {}


def test_chain_of_moves_invalidates_each_by_its_immediate_successor():
    facts = [
        _fact("k1", "Ann", "live_in", "Madrid", "2023-01-01"),
        _fact("k2", "Ann", "live_in", "Berlin", "2023-05-01"),
        _fact("k3", "Ann", "live_in", "Tokyo", "2023-09-01"),
    ]
    assert compute_invalidations(facts) == {
        "k1": ("2023-05-01", "k2"),
        "k2": ("2023-09-01", "k3"),
    }


def test_facts_are_ordered_chronologically_not_by_input_order():
    facts = [
        _fact("k2", "Ann", "live_in", "Berlin", "2023-05-01"),
        _fact("k1", "Ann", "live_in", "Madrid", "2023-01-01"),
    ]
    assert compute_invalidations(facts) == {"k1": ("2023-05-01", "k2")}
