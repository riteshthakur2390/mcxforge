import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.llm import normalize_llm_sentences


def test_normalize_llm_sentences_completes_missing_final_period():
    text = "Bias is mildly bullish. Expect mixed but tradeable moves. Wait for ORB before entries"
    normalized = normalize_llm_sentences(text, expected_sentences=3)
    assert normalized == (
        "Bias is mildly bullish. Expect mixed but tradeable moves. Wait for ORB before entries."
    )


def test_normalize_llm_sentences_flattens_numbered_output():
    text = "1. Trend bias is neutral.\n2. Choppy conditions are likely.\n3. Keep SL tight today"
    normalized = normalize_llm_sentences(text, expected_sentences=3)
    assert normalized == (
        "Trend bias is neutral. Choppy conditions are likely. Keep SL tight today."
    )
