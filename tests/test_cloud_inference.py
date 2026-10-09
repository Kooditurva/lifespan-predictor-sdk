"""predict() returns survival curves from a deployed model.

The container computes survival curves on every request and puts them on the
wire beside the mean times. These assert that they arrive, that they are the
right way round, and that they describe the same model the mean times come from.

This module ships in the published SDK, so its cases are the ones a buyer's own
account can run — and its docstrings say what each case checks rather than which
document asked for it. The case that compares against a second backend is not
here; it cannot run where there is one backend, so it lives in
tests/test_predict_backend_contract.py, which stays in this repository.

Most cases run a real batch transform and cost money, so they share one via the
`cloud_curves` fixture. The cases at the end need no AWS at all: they fake
the transform, which puts the response contract on every push rather than
only on cloud runs.
"""

import os

import numpy as np
import pandas as pd
import pytest

from conftest import assert_is_a_survival_frame  # noqa: E402


@pytest.mark.cloud
def test_predict_returns_a_curve_per_sample(cloud_curves):
    """One column per input unit, values in [0, 1], non-increasing with time."""
    sample, curves = cloud_curves
    assert_is_a_survival_frame(curves, sample["unit"].nunique())


@pytest.mark.cloud
def test_curves_and_mean_times_agree(cloud_curves, deployed_survival_model):
    """The two prediction paths read the same model.

    Without this, either could quietly return something plausible on its own.
    The mean of a survival curve is the area under it, so integrate and compare.
    """
    sample, curves = cloud_curves
    mean_times = deployed_survival_model.predict_mean(sample)

    times = curves.index.to_numpy(dtype=float)
    implied = np.trapezoid(curves.to_numpy(dtype=float), x=times, axis=0)

    assert len(implied) == len(mean_times)
    # Generous: the container's own mean may use a different quadrature, and the
    # grid is truncated at the last time point either way. A factor-of-two
    # disagreement still fails, which is what a mismatched model would give.
    np.testing.assert_allclose(implied, np.asarray(mean_times), rtol=0.25, atol=5.0)


@pytest.mark.cloud
def test_time_points_narrows_the_index(cloud_curves, deployed_survival_model):
    """`time_points` selects rows of the container's own grid.

    Nothing is interpolated, so the returned index is a subset of the full one.
    """
    sample, full = cloud_curves

    requested = list(full.index[:: max(1, len(full.index) // 4)][:3])
    narrowed = deployed_survival_model.predict(sample, time_points=requested)

    assert list(narrowed.index) == requested
    assert narrowed.shape[1] == full.shape[1]


def _fake_transform(monkeypatch, response):
    """A deployed SageMaker-backed model whose batch transform returns `response`.

    Everything AWS-facing is replaced, so the real `predict_mean` runs end to end:
    upload, transform, download, parse. Returns the model and a dict recording
    what the transform was asked for.
    """
    import json

    import boto3

    from lifespan_predictor_sdk.survival_model import SurvivalModel

    recorded = {}

    class Transformer:
        def transform(self, **kwargs):
            pass

    class Model:
        def transformer(self, **kwargs):
            recorded.update(kwargs)
            return Transformer()

    class Session:
        def upload_data(self, path, bucket, key_prefix):
            return f"s3://{bucket}/{key_prefix}/input.csv"

    class S3:
        def list_objects_v2(self, Bucket, Prefix):
            return {"Contents": [{"Key": f"{Prefix}input.csv.out"}]}

        def download_file(self, bucket, key, path):
            with open(path, "w") as f:
                json.dump(response, f)

    class Boto3Session:
        def client(self, name, **kwargs):
            return S3()

    monkeypatch.setattr(boto3, "Session", Boto3Session)

    model = SurvivalModel({"backend": "sagemaker", "bucket": "b"})
    model._is_fitted = True
    model._backend._sagemaker_session = Session()
    model._backend._sagemaker_model = Model()
    return model, recorded


# A transform response as a certified 1.0.0 container sends it: with the
# training config attached. Its values differ from the caller's, so copying them
# over would show.
RESPONSE_FROM_A_1_0_0_CONTAINER = {
    "predictions": [50.0, 80.0],
    "survival_curves": [[1.0, 0.6, 0.2], [1.0, 0.8, 0.5]],
    "time_grid": [0.0, 50.0, 100.0],
    "config": {"algorithm_arn": None, "epochs": 999, "max_rul": 7},
}


@pytest.mark.local
def test_predict_mean_returns_the_predictions_alone(monkeypatch):
    """`predict_mean` returns the mean times as an array, never a tuple."""
    model, _ = _fake_transform(monkeypatch, RESPONSE_FROM_A_1_0_0_CONTAINER)

    mean_times = model.predict_mean(pd.DataFrame({"a": [1, 2]}))

    assert isinstance(mean_times, np.ndarray)
    np.testing.assert_array_equal(mean_times, [50.0, 80.0])


@pytest.mark.local
def test_the_transform_leaves_the_callers_config_alone(monkeypatch):
    """The container's config does not reach the buyer's model.

    A 1.0.0 container still sends one, with a null algorithm ARN. Copied into
    the model, it would silently change what the next `fit` trains against.
    """
    model, _ = _fake_transform(monkeypatch, RESPONSE_FROM_A_1_0_0_CONTAINER)
    before = dict(model.config)

    model.predict_mean(pd.DataFrame({"a": [1, 2]}))
    model.evaluate(
        pd.DataFrame({"a": [1, 2]}), np.array([40.0, 90.0]), np.array([1, 1])
    )

    assert model.config == before


@pytest.mark.local
def test_predict_asks_for_the_json_path(monkeypatch):
    """Curves come back only on the JSON accept path.

    `text/csv` carries mean times alone, so a CSV request cannot return curves no
    matter what the client does with the response.
    """
    model, recorded = _fake_transform(monkeypatch, RESPONSE_FROM_A_1_0_0_CONTAINER)

    curves = model.predict(pd.DataFrame({"a": [1, 2]}))

    assert recorded.get("accept") == "application/json"
    assert_is_a_survival_frame(curves, 2)


@pytest.mark.local
def test_missing_curves_raise_rather_than_return_empty():
    """Absent curves are an error, not an empty answer.

    Returning an empty frame would look like a model that predicts nothing.
    """
    from lifespan_predictor_sdk.sagemaker_backend import SageMakerBackend

    backend = SageMakerBackend()
    backend.predict_mean = lambda model, X, **kwargs: None
    backend.get_survival_curves = lambda model, X: (None, None, None)

    with pytest.raises(RuntimeError, match="no survival curves"):
        backend.predict(object(), pd.DataFrame({"a": [1]}))
