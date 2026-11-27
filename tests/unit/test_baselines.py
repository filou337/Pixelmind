import numpy as np
import pytest
from PIL import Image
from sklearn.metrics import average_precision_score, f1_score, jaccard_score

from pixelmind.data.manifest import build_manifest
from pixelmind.evaluation.metrics import ScoreHistogram, counts_at, metrics_from_counts
from pixelmind.evaluation.protocol import bucket_of, evaluate
from pixelmind.models.baselines import gbm
from pixelmind.models.baselines.image_diff import SCORERS

rng = np.random.default_rng(0)


# --- metrics: cross-checked against scikit-learn ----------------------------


def random_case(n=5000):
    mask = rng.random(n) < 0.1
    scores = np.clip(rng.normal(80, 40, n) + mask * 60, 0, 255).astype(np.uint8)
    return scores, mask


def test_histogram_metrics_match_sklearn():
    scores, mask = random_case()
    h = ScoreHistogram()
    h.add(scores[:2000], mask[:2000])  # streamed in two chunks
    h.add(scores[2000:], mask[2000:])
    t = 140
    m = h.metrics_at(t)
    pred = scores >= t
    assert m["f1"] == pytest.approx(f1_score(mask, pred))
    assert m["iou"] == pytest.approx(jaccard_score(mask, pred))
    assert h.average_precision() == pytest.approx(average_precision_score(mask, scores))
    assert m == metrics_from_counts(*counts_at(scores, mask, t))


def test_best_threshold_maximizes_f1():
    scores, mask = random_case()
    h = ScoreHistogram()
    h.add(scores, mask)
    best = h.best_threshold()
    f1s = [f1_score(mask, scores >= t, zero_division=0) for t in range(1, 256)]
    assert h.metrics_at(best)["f1"] == pytest.approx(max(f1s))


def test_metrics_undefined_without_positives():
    m = metrics_from_counts(tp=0, fp=0, fn=0, tn=10)
    assert m["f1"] is None and m["precision"] is None and m["fpr"] == 0.0


def test_histogram_rejects_float_scores():
    with pytest.raises(TypeError):
        ScoreHistogram().add(np.zeros(3, np.float32), np.zeros(3, bool))


def test_bucket_of():
    assert [bucket_of(x) for x in (0, 0.005, 0.03, 0.2)] == ["no change", "< 1 %", "1-5 %", "> 5 %"]


# --- scorers / features -----------------------------------------------------


def test_scorers_shape_dtype_and_zero_on_identical_images():
    t1 = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    for name, scorer in SCORERS.items():
        s = scorer(t1, t1)
        assert s.shape == (32, 32) and s.dtype == np.uint8, name
        assert s.max() == 0, name


def test_normdiff_ignores_global_brightness_shift():
    t1 = rng.integers(50, 150, (32, 32, 3), dtype=np.uint8)
    t2 = t1 + 60  # same scene, brighter
    assert SCORERS["absdiff"](t1, t2).mean() == 60
    assert SCORERS["normdiff"](t1, t2).max() <= 1


def test_pixel_features_shape():
    t1 = rng.integers(0, 256, (16, 20, 3), dtype=np.uint8)
    f = gbm.pixel_features(t1, t1)
    assert f.shape == (16 * 20, len(gbm.FEATURES)) and f.dtype == np.float32
    assert np.isfinite(f).all()


# --- protocol end-to-end on a tiny synthetic dataset ------------------------


@pytest.fixture
def synthetic_root(tmp_path):
    """Pairs where a bright square appears in T2: an easy change to detect."""
    for k, (split, n) in enumerate((("train", 4), ("val", 2), ("test", 3))):
        for i in range(n):
            # distinct seeds per split, otherwise the leakage check marks pairs FAIL
            t1 = np.random.default_rng(100 * k + i).integers(60, 120, (24, 24, 3), dtype=np.uint8)
            t2 = t1.copy()
            m = np.zeros((24, 24), np.uint8)
            if i % 3:  # some scenes without change
                t2[4:12, 4:12] = 250
                m[4:12, 4:12] = 255
            for sub, arr in (("A", t1), ("B", t2), ("label", m)):
                d = tmp_path / "levircd" / split / sub
                d.mkdir(parents=True, exist_ok=True)
                Image.fromarray(arr).save(d / f"{split}_{i}.png")
    return tmp_path


def test_protocol_on_easy_synthetic_changes(synthetic_root):
    records = build_manifest(synthetic_root)
    res = evaluate(SCORERS["absdiff"], records, synthetic_root)
    assert res["test"]["f1"] == pytest.approx(1.0)
    assert len(res["test_scenes"]) == 3
    assert res["error_analysis"]["by_change_bucket"]["no change"]["fp"] == 0


def test_gbm_trains_and_scores(synthetic_root):
    records = build_manifest(synthetic_root)
    train = [r for r in records if r["split"] == "train"]
    X, y = gbm.sample_training_set(synthetic_root, train, per_class=30, seed=0)
    assert X.shape[1] == len(gbm.FEATURES) and y.any() and (~y).any()
    model = gbm.train(X, y, seed=0)
    res = evaluate(gbm.make_scorer(model, chunk=100), records, synthetic_root)
    assert res["test"]["f1"] > 0.9


def test_error_analysis_ignores_no_change_scenes_in_f1_ranking():
    from pixelmind.evaluation.protocol import error_analysis

    def scene(pid, ratio, tp, fp, fn):
        m = metrics_from_counts(tp, fp, fn, 1000)
        return {"pair_id": pid, "changed_ratio": ratio, "bucket": bucket_of(ratio), **m}

    scenes = [scene("empty", 0.0, 0, 5, 0), scene("bad", 0.1, 1, 9, 9), scene("good", 0.1, 9, 1, 1)]
    ea = error_analysis(scenes)
    assert [s["pair_id"] for s in ea["worst_scenes"]] == ["bad", "good"]
    assert ea["by_change_bucket"]["no change"]["fp"] == 5
