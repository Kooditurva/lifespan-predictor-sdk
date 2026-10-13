"""What predict() and predict_mean() make of a batch transform's response.

No AWS: the transform is faked, so the response contract is checked on every
push rather than only on cloud runs. This module ships in the published SDK, so
its docstrings say what each case checks rather than which document asked for it.
"""

import os

import numpy as np
import pandas as pd
import pytest

from conftest import assert_is_a_survival_frame  # noqa: E402


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
    # The means of the returned curves, not the container's own.
    np.testing.assert_allclose(mean_times, [60.0, 77.5])


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
