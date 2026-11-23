"""Tests for dataset preprocessing and text cleaning."""

import pytest

from forgelm.data.clean import clean_text


def test_clean_text_normalizes_whitespace():
    raw = "  Once upon a time,   there lived   a puppy.  \n\n  The puppy loved playing!  "
    cleaned = clean_text(raw)
    assert cleaned == "Once upon a time, there lived a puppy. The puppy loved playing!"


def test_clean_text_rejects_short_strings():
    assert clean_text("Too short") == ""
    assert clean_text("") == ""
    assert clean_text("   ") == ""


def test_clean_text_handles_non_strings():
    assert clean_text(None) == ""
    assert clean_text(123) == ""


def test_clean_text_respects_custom_min_length():
    text = "Short valid text"
    assert clean_text(text, min_length=10) == "Short valid text"
    assert clean_text(text, min_length=30) == ""
