"""Tests for issue #1: archive quora-insincere-classification and
multilingual-sentiment-analysis once this suite has feature parity.

These exercise the archival-detection logic against small fixtures, not the
real sibling `portfolio-archived-repos` checkout or `PORTFOLIO_INDEX.md` —
so they run the same in any checkout, including CI. See
`scripts/verify_archival.py` for the one-off check against the real repos.
"""
from __future__ import annotations

from nlp_suite.archival import (
    ArchivalStatus,
    check_archival,
    index_lists_original_as_active_project,
    index_references_suite,
    repo_readme_is_archived,
)

ARCHIVED_README = """# Insincere Question Classification

![status](https://img.shields.io/badge/status-archived-lightgrey)
"""

ACTIVE_README = """# Insincere Question Classification

![status](https://img.shields.io/badge/status-in%20progress-yellow)
"""

GOOD_INDEX = """
- nlp-classification-suite -- combines the originals
  (`quora-insincere-classification`, `multilingual-sentiment-analysis`),
  both moved to portfolio-archived-repos.
"""

STALE_INDEX_STILL_ACTIVE = """
- nlp-classification-suite
- quora-insincere-classification
"""


def test_repo_readme_is_archived_true_for_archived_badge():
    assert repo_readme_is_archived(ARCHIVED_README) is True


def test_repo_readme_is_archived_false_for_active_badge():
    assert repo_readme_is_archived(ACTIVE_README) is False


def test_index_references_suite():
    assert index_references_suite(GOOD_INDEX, "nlp-classification-suite") is True
    assert index_references_suite(GOOD_INDEX, "some-other-suite") is False


def test_index_does_not_flag_original_mentioned_only_in_archival_note():
    assert (
        index_lists_original_as_active_project(
            GOOD_INDEX, "quora-insincere-classification"
        )
        is False
    )


def test_index_flags_original_still_listed_as_standalone_active_entry():
    assert (
        index_lists_original_as_active_project(
            STALE_INDEX_STILL_ACTIVE, "quora-insincere-classification"
        )
        is True
    )


def test_check_archival_complete_when_readme_archived_and_index_clean():
    status = check_archival(
        suite_name="nlp-classification-suite",
        original_name="quora-insincere-classification",
        original_readme_text=ARCHIVED_README,
        index_text=GOOD_INDEX,
    )
    assert isinstance(status, ArchivalStatus)
    assert status.complete is True


def test_check_archival_incomplete_when_readme_not_archived():
    status = check_archival(
        suite_name="nlp-classification-suite",
        original_name="quora-insincere-classification",
        original_readme_text=ACTIVE_README,
        index_text=GOOD_INDEX,
    )
    assert status.complete is False


def test_check_archival_incomplete_when_index_still_lists_original_as_active():
    status = check_archival(
        suite_name="nlp-classification-suite",
        original_name="quora-insincere-classification",
        original_readme_text=ARCHIVED_README,
        index_text=STALE_INDEX_STILL_ACTIVE,
    )
    assert status.complete is False
