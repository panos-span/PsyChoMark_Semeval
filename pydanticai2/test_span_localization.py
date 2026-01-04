#!/usr/bin/env python3
"""Quick test for span localization."""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pydanticai2.psycomark_agents import find_best_span


def test_find_best_span():
    text = "The government and NASA are hiding aliens. NASA knows the truth."

    # Test 1: Find first occurrence
    start, end = find_best_span(text, "NASA", nth=0)
    print(f"Test 1 - First 'NASA': [{start}:{end}] = '{text[start:end]}'")
    assert text[start:end] == "NASA", f"Expected 'NASA', got '{text[start:end]}'"

    # Test 2: Find second occurrence
    start2, end2 = find_best_span(text, "NASA", nth=1)
    print(f"Test 2 - Second 'NASA': [{start2}:{end2}] = '{text[start2:end2]}'")
    assert start2 > start, "Second occurrence should be after first"

    # Test 3: Not found
    start3, end3 = find_best_span(text, "aliens from Mars", nth=0)
    print(f"Test 3 - Not found: [{start3}:{end3}]")
    assert start3 == -1 and end3 == -1, "Should return (-1, -1) when not found"

    # Test 4: Fuzzy match
    start4, end4 = find_best_span(text, "the government", nth=0)  # lowercase
    print(
        f"Test 4 - Fuzzy 'the government': [{start4}:{end4}] = '{text[start4:end4] if start4 != -1 else 'NOT FOUND'}'"
    )

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_find_best_span()
