"""Tests for single image prediction pipeline."""

from pathlib import Path
import pandas as pd
import pytest

from src.predict import Predictor, predict_single_image

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_CSV = DATA_DIR / "processed" / "labels.csv"
BEST_MODEL_PATH = BASE_DIR / "models" / "best_model.pth"


def test_predictor_pipeline_on_sample_image():
    """Verify that Predictor loads and returns structured multi-label outputs."""
    if not BEST_MODEL_PATH.exists() or not PROCESSED_CSV.exists():
        pytest.skip("Model checkpoint or processed data not found.")

    df = pd.read_csv(PROCESSED_CSV)
    test_rows = df[df["split"] == "test"]
    assert len(test_rows) > 0

    sample_rel_path = test_rows.iloc[0]["image_path"]
    predictor = Predictor()
    results = predictor.predict(sample_rel_path)

    assert isinstance(results, dict)
    assert len(results) == len(predictor.labels)

    for class_name, res in results.items():
        assert "probability" in res
        assert "threshold" in res
        assert "detected" in res
        assert "percentage" in res
        assert 0.0 <= res["probability"] <= 1.0
        assert 0.0 <= res["threshold"] <= 1.0
        assert isinstance(res["detected"], bool)
        assert res["detected"] == (res["probability"] >= res["threshold"])


def test_helper_predict_single_image():
    """Verify high-level helper predict_single_image."""
    if not BEST_MODEL_PATH.exists() or not PROCESSED_CSV.exists():
        pytest.skip("Model checkpoint or processed data not found.")

    df = pd.read_csv(PROCESSED_CSV)
    sample_rel_path = df.iloc[0]["image_path"]
    results = predict_single_image(sample_rel_path)

    assert isinstance(results, dict)
    assert len(results) > 0
