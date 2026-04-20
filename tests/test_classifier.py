"""
tests/test_classifier.py
========================
Unit tests for src/classifier.py using pytest.

These tests verify the *behaviour* of the predict() function — that it returns
the correct data structure, valid labels, and well-formed probabilities — without
downloading the 500MB model from HuggingFace. This is achieved through mocking.

What is mocking?
----------------
Mocking replaces a real dependency (here: the HuggingFace model) with a fake
stand-in that returns a predetermined output. This lets us test the logic of
predict() in isolation, without network calls or GPU requirements. Think of it
as a stunt double — the real model doesn't need to show up for tests.

Run tests from the project root with:
    pytest tests/ -v
"""

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# We need to import classifier.py from src/, but since we're running pytest
# from the project root, src/ is not automatically on the Python path.
# The sys.path manipulation below adds it so the import works correctly.
# ---------------------------------------------------------------------------
sys.path.insert(0, "src")

# ---------------------------------------------------------------------------
# classifier.py imports torch and transformers at the top level. If for any
# reason these aren't installed in the test environment, we stub them out so
# the import doesn't fail. In practice, they should always be installed.
# ---------------------------------------------------------------------------
import classifier  # noqa: E402  (import after sys.path manipulation is intentional)


# ---------------------------------------------------------------------------
# FIXTURES
# A pytest fixture is a reusable piece of setup code. Instead of repeating
# "create a fake model output" in every test, we define it once here and
# pytest injects it automatically into any test that asks for it by name.
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_model_output():
    """
    Returns a fake tensor output that mimics what the real model would return.
    The logits are set so that index 1 (Anxiety) has the highest value,
    meaning predict() should return label "Anxiety".
    """
    mock_output      = MagicMock()
    # Shape: [1, 4] — one sample, four class logits
    # Anxiety (index 1) gets the highest value so softmax picks it
    mock_output.logits = __import__("torch").tensor([[0.1, 3.5, 0.8, 0.6]])
    return mock_output


@pytest.fixture
def mock_tokenizer_output():
    """
    Returns a fake tokenizer output that mimics what AutoTokenizer produces.
    The actual tensor values don't matter here — we just need the right shape
    so the model call doesn't crash.
    """
    mock_encoding = MagicMock()
    mock_encoding.items.return_value = [
        ("input_ids",      [[1, 2, 3]]),
        ("attention_mask", [[1, 1, 1]]),
    ]
    # .to(device) must return the same object so the chain doesn't break
    mock_encoding.__getitem__ = lambda self, k: MagicMock()
    return mock_encoding


# ---------------------------------------------------------------------------
# HELPER — patches both the model and tokeniser so predict() runs without
# downloading anything, then returns the result for inspection.
# ---------------------------------------------------------------------------

def _run_predict_with_mocks(text, mock_model_output, mock_tokenizer_output):
    """
    Runs predict(text) with the model and tokeniser replaced by mocks.
    Returns the result dict so individual tests can make assertions on it.
    """
    # Reset the module-level cache so the mock gets loaded fresh each time
    classifier._model     = None
    classifier._tokenizer = None

    with patch.object(classifier, "_tokenizer", create=True) as mock_tok, \
         patch.object(classifier, "_model",     create=True) as mock_mod:

        # Wire up the mock tokeniser
        mock_tok.return_value = mock_tokenizer_output
        mock_tokenizer_output.to = MagicMock(return_value=mock_tokenizer_output)

        # Wire up the mock model
        mock_mod.return_value          = mock_model_output
        mock_mod.config.num_labels     = 4
        mock_mod.eval                  = MagicMock()

        # Patch _load_model so it sets the mocks instead of downloading
        def fake_load():
            classifier._model     = mock_mod
            classifier._tokenizer = mock_tok

        with patch.object(classifier, "_load_model", side_effect=fake_load):
            return classifier.predict(text)


# ---------------------------------------------------------------------------
# TESTS
# Each function starting with test_ is automatically discovered and run by
# pytest. The assert statements are the actual checks — pytest will show
# a helpful diff if any assertion fails.
# ---------------------------------------------------------------------------

class TestPredictOutputStructure:
    """
    Tests that verify predict() always returns a dictionary with exactly
    the right keys, regardless of what the model predicts.
    """

    def test_returns_dict(self, mock_model_output, mock_tokenizer_output):
        """predict() must return a dict, not a string, list, or None."""
        result = _run_predict_with_mocks(
            "I feel anxious", mock_model_output, mock_tokenizer_output
        )
        assert isinstance(result, dict), \
            f"Expected dict, got {type(result)}"

    def test_has_required_keys(self, mock_model_output, mock_tokenizer_output):
        """predict() must return exactly the keys: label, label_id, scores."""
        result = _run_predict_with_mocks(
            "I feel anxious", mock_model_output, mock_tokenizer_output
        )
        assert "label"    in result, "Missing key: 'label'"
        assert "label_id" in result, "Missing key: 'label_id'"
        assert "scores"   in result, "Missing key: 'scores'"

    def test_scores_has_all_four_classes(self, mock_model_output, mock_tokenizer_output):
        """The scores dict must contain an entry for every class."""
        result = _run_predict_with_mocks(
            "I feel anxious", mock_model_output, mock_tokenizer_output
        )
        expected_classes = {"Normal", "Anxiety", "Depression", "Suicidal"}
        assert set(result["scores"].keys()) == expected_classes, \
            f"scores keys {set(result['scores'].keys())} != {expected_classes}"


class TestPredictOutputValues:
    """
    Tests that verify the values inside the output dictionary are valid
    and internally consistent.
    """

    def test_label_is_valid_class(self, mock_model_output, mock_tokenizer_output):
        """label must be one of the four recognised class names."""
        result = _run_predict_with_mocks(
            "I feel anxious", mock_model_output, mock_tokenizer_output
        )
        valid_labels = {"Normal", "Anxiety", "Depression", "Suicidal"}
        assert result["label"] in valid_labels, \
            f"label '{result['label']}' is not a valid class"

    def test_label_id_is_valid_integer(self, mock_model_output, mock_tokenizer_output):
        """label_id must be an integer between 0 and 3 inclusive."""
        result = _run_predict_with_mocks(
            "I feel anxious", mock_model_output, mock_tokenizer_output
        )
        assert isinstance(result["label_id"], int), \
            f"label_id must be int, got {type(result['label_id'])}"
        assert 0 <= result["label_id"] <= 3, \
            f"label_id {result['label_id']} is out of range [0, 3]"

    def test_label_and_label_id_are_consistent(self, mock_model_output, mock_tokenizer_output):
        """label and label_id must agree — they must refer to the same class."""
        label_map = {"Normal": 0, "Anxiety": 1, "Depression": 2, "Suicidal": 3}
        result = _run_predict_with_mocks(
            "I feel anxious", mock_model_output, mock_tokenizer_output
        )
        assert label_map[result["label"]] == result["label_id"], \
            f"label '{result['label']}' maps to id {label_map[result['label']]} " \
            f"but label_id is {result['label_id']}"

    def test_scores_are_probabilities(self, mock_model_output, mock_tokenizer_output):
        """Every score must be a float between 0 and 1 inclusive."""
        result = _run_predict_with_mocks(
            "I feel anxious", mock_model_output, mock_tokenizer_output
        )
        for cls, score in result["scores"].items():
            assert isinstance(score, float), \
                f"Score for '{cls}' must be float, got {type(score)}"
            assert 0.0 <= score <= 1.0, \
                f"Score for '{cls}' is {score}, outside [0, 1]"

    def test_scores_sum_to_one(self, mock_model_output, mock_tokenizer_output):
        """
        Softmax outputs must sum to 1.0. We allow a tiny floating point
        tolerance (1e-5) because floating point arithmetic is never exact.
        """
        result = _run_predict_with_mocks(
            "I feel anxious", mock_model_output, mock_tokenizer_output
        )
        total = sum(result["scores"].values())
        assert abs(total - 1.0) < 1e-5, \
            f"Scores sum to {total}, expected 1.0 (±1e-5)"

    def test_predicted_label_has_highest_score(self, mock_model_output, mock_tokenizer_output):
        """
        The label field must be the class with the highest softmax score.
        If predict() returns label='Anxiety', then scores['Anxiety'] must be
        the largest value in the scores dict.
        """
        result = _run_predict_with_mocks(
            "I feel anxious", mock_model_output, mock_tokenizer_output
        )
        predicted_score = result["scores"][result["label"]]
        max_score       = max(result["scores"].values())
        assert predicted_score == max_score, \
            f"Predicted label '{result['label']}' has score {predicted_score} " \
            f"but max score is {max_score}"


class TestPredictEdgeCases:
    """
    Tests that verify predict() handles unusual inputs without crashing.
    Robustness to unexpected inputs is important in production systems.
    """

    def test_very_short_input(self, mock_model_output, mock_tokenizer_output):
        """predict() must not crash on a single word."""
        result = _run_predict_with_mocks(
            "help", mock_model_output, mock_tokenizer_output
        )
        assert "label" in result

    def test_empty_string(self, mock_model_output, mock_tokenizer_output):
        """predict() must not crash on an empty string."""
        result = _run_predict_with_mocks(
            "", mock_model_output, mock_tokenizer_output
        )
        assert "label" in result

    def test_very_long_input_does_not_crash(self, mock_model_output, mock_tokenizer_output):
        """
        predict() must handle inputs longer than MAX_LENGTH=512 tokens gracefully.
        The tokeniser's truncation=True setting should handle this silently.
        """
        long_text = "I feel really anxious and overwhelmed. " * 200
        result = _run_predict_with_mocks(
            long_text, mock_model_output, mock_tokenizer_output
        )
        assert "label" in result
