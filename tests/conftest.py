"""Shared fixtures for tests."""

import json
import os
import re
from pathlib import Path

import matplotlib
import pytest

# Plotting must never open a window. The default backend on a developer Mac is
# ``macosx``, whose ``plt.show()`` blocks on a window that never appears under pytest,
# so any test reaching a plot hangs until the run is killed.
#
# This used to be set inside one test in test_survival_model.py, which made the backend
# depend on collection order: because pytest shares a process, that call leaked to
# whatever ran after it. The full suite passed while ``pytest tests/test_deephit.py``
# on its own hung indefinitely — and so would any single-file run, which is exactly how
# someone iterates on a failing test.
#
# Set here so it holds for every test regardless of order or selection. conftest is
# imported before test modules, so this wins over the backend matplotlib would
# otherwise choose on first use.
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Algorithm ARN resolution
#
# Public-repo cloud CI resolves which SageMaker Algorithm to test against by
# looking up the SDK's own major.minor in SSM. This keeps the published X.Y
# compatibility contract enforced in CI instead of relying on a
# manually-rotated GitHub secret. The SAGEMAKER_ALGORITHM_ARN env var is
# still honoured as an explicit override, for pinning a run to one algorithm.
# ---------------------------------------------------------------------------


def ssm_param_name_for(version: str) -> str:
    """Return the SSM parameter name for the given SDK/algorithm version."""
    match = re.match(r"(\d+\.\d+)", version)
    major_minor = match.group(1) if match else version
    channel = "rc" if "rc" in version else "stable"
    return f"/lifespan-predictor/algorithm-arn/{channel}/{major_minor}"


# Credentials that are missing, stale or malformed. Distinct from an
# authorization failure, which says the credentials are real and the permission
# is not — that has to surface, so it is deliberately absent here.
EXPIRED_CREDENTIAL_CODES = frozenset(
    {
        "UnrecognizedClientException",
        "InvalidClientTokenId",
        "ExpiredToken",
        "ExpiredTokenException",
    }
)


def resolve_algorithm_arn(version: str, env=None, boto_session=None):
    """Resolve an algorithm ARN for the given SDK version.

    Returns a tuple (arn, skip_reason). Exactly one element is non-None:
    - (arn, None) when the ARN is resolved (via env override or SSM).
    - (None, reason) when no ARN is available; the caller should skip.

    "No ARN available" covers having no usable credentials to ask with. It has
    to: this is the first AWS call a cloud run makes, and a machine that is
    simply logged out should get the same skip as one with nothing registered,
    not a traceback.
    """
    env = env if env is not None else os.environ
    override = env.get("SAGEMAKER_ALGORITHM_ARN")
    if override:
        return override, None

    import boto3
    import botocore.exceptions

    session = boto_session or boto3
    client = session.client("ssm")
    param_name = ssm_param_name_for(version)

    try:
        response = client.get_parameter(Name=param_name)
        return response["Parameter"]["Value"], None
    except client.exceptions.ParameterNotFound:
        return (
            None,
            (
                f"No algorithm ARN registered at {param_name}. "
                "Either run the private-repo release.py for this SDK's "
                "major.minor, or set SAGEMAKER_ALGORITHM_ARN."
            ),
        )
    except botocore.exceptions.BotoCoreError as error:
        # No usable credentials at all — an unconfigured or logged-out machine.
        return None, f"No AWS credentials for the algorithm lookup: {error}"
    except botocore.exceptions.ClientError as error:
        code = error.response.get("Error", {}).get("Code")
        if code not in EXPIRED_CREDENTIAL_CODES:
            # Anything else is a real answer from AWS — an authorization gap
            # most of all, which is the one thing here that must never pass
            # quietly as a skip.
            raise
        return (
            None,
            f"AWS credentials are not valid ({code}); refresh them to run this.",
        )


def _read_sdk_version() -> str:
    """The SDK version, from the installed distribution or from pyproject.toml.

    The fallback reads `pyproject.toml` rather than a `__version__` literal: this
    repository declares its version there, and the literal that used to live in
    `src/__init__.py` had already drifted from it. One source, and the fallback
    reads the same one the distribution was built from.
    """
    try:
        import importlib.metadata

        return importlib.metadata.version("lifespan-predictor")
    except Exception:
        pass

    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    if pyproject.exists():
        for line in pyproject.read_text().splitlines():
            match = re.match(r'\s*version\s*=\s*["\']([^"\']+)["\']', line)
            if match:
                return match.group(1)
    raise RuntimeError("Cannot determine SDK version for algorithm ARN lookup")


@pytest.fixture(scope="session")
def algorithm_arn():
    """SageMaker Algorithm ARN matching the SDK's major.minor."""
    arn, skip_reason = resolve_algorithm_arn(_read_sdk_version())
    if skip_reason:
        pytest.skip(skip_reason)
    return arn


@pytest.fixture(scope="session")
def sagemaker_trained_model(tmp_path_factory):
    """
    Fixture to share trained SageMaker model artifacts between tests.

    Test 1 (test_sagemaker_training_and_metrics) writes model info after training.
    Test 2 (test_sagemaker_inference) reads model info to skip training.

    Usage in Test 1 (save model info):
        model_info = {
            "model_data": model._model_data,
            "training_job_name": model._training_job_name,
            ...
        }
        sagemaker_trained_model["write"](model_info)

    Usage in Test 2 (load model info):
        model_info = sagemaker_trained_model["read"]()
        if model_info is None:
            pytest.skip("test_sagemaker_training_and_metrics must run first")
    """
    shared_dir = tmp_path_factory.mktemp("sagemaker_artifacts")
    model_info_file = shared_dir / "model_info.json"

    return {
        "file": model_info_file,
        "write": lambda info: model_info_file.write_text(
            json.dumps(info, indent=2), encoding="utf-8"
        ),
        "read": lambda: (
            json.loads(model_info_file.read_text(encoding="utf-8"))
            if model_info_file.exists()
            else None
        ),
        "clear": lambda: (
            model_info_file.unlink() if model_info_file.exists() else None
        ),
    }


# The RUL cap the session's model trains with. A model reattached by name has
# only the config its caller gives it, so a test that evaluates one
# must pass this too, or evaluate() scores the uncapped units as well.
TRAINED_MAX_RUL = 125


@pytest.fixture(scope="session")
def trained_and_deployed_model(algorithm_arn, sagemaker_trained_model):
    """One trained, deployed model for the whole session.

    Every cloud test that needs a model gets it from here, so the suite does not
    depend on which file pytest happens to collect first. Session-scoped, so the
    training job runs once no matter how many tests ask.

    Returns the artifacts the training test asserts over alongside the deployed
    name, because they all come out of the same run and re-deriving them would
    mean a second one.
    """
    from lifespan_predictor_sdk.data_loader import (
        download_nasa_dataset,
        get_nasa_test_dataframe,
        get_nasa_train_dataframe,
    )
    from lifespan_predictor_sdk.survival_model import SurvivalModel

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "nasa")
    download_nasa_dataset(data_dir, [1])

    train_df = get_nasa_train_dataframe(data_dir, datasets=[1])
    test_df, test_labels_df = get_nasa_test_dataframe(data_dir, datasets=[1])

    config = {
        "model_type": "deepsurv",
        "backend": "sagemaker",
        "algorithm_arn": algorithm_arn,
        "instance_type": "ml.m5.large",
        "num_nodes": [24],
        "dropout": 0.1,
        "learning_rate": 0.0001,
        "batch_size": 64,
        "epochs": 30,
        "max_rul": TRAINED_MAX_RUL,
        "use_delta": True,
        "rolling_means": [5, 10, 20, 30],
    }

    model = SurvivalModel(dict(config))
    model.fit(train_df, test_df=test_df, test_labels_df=test_labels_df)

    log = model.get_training_log()
    training_info = model._estimator.latest_training_job.describe()
    eval_result = model.get_evaluation_result()
    deployed_model_name = model.deploy()

    model_info = {
        "model_name": deployed_model_name,
        "model_data": model._model_data,
        "training_job_name": model._training_job_name,
        "algorithm_arn": algorithm_arn,
        "instance_type": config["instance_type"],
        "sagemaker_model_path": model._sagemaker_model_path,
        "config": config,
    }
    sagemaker_trained_model["write"](model_info)

    return {
        "model": model,
        "model_name": deployed_model_name,
        "log": log,
        "training_info": training_info,
        "eval_result": eval_result,
        "model_info": model_info,
    }


@pytest.fixture(scope="session")
def model_name(request):
    """The name of a deployed SageMaker model to run inference against.

    ``SAGEMAKER_MODEL_NAME`` names one that already exists; without it, one is
    trained and deployed.

    Resolved through ``request`` rather than declared as a parameter so that
    naming a model costs nothing. A plain dependency would train one first and
    then discard it, which is a twelve-minute bill for a value the caller had
    already supplied.
    """
    env_model_name = os.environ.get("SAGEMAKER_MODEL_NAME")
    if env_model_name:
        return env_model_name

    return request.getfixturevalue("trained_and_deployed_model")["model_name"]


def deployed_model_from_description(
    model_description, model_name, role=None, sagemaker_session=None
):
    """Rebuild a model object for a model that already exists in SageMaker.

    A cloud test given ``SAGEMAKER_MODEL_NAME`` skips training and deployment, but
    still needs something to call ``transformer()`` on. Rebuild it from the model
    package the deploy created, never from a container image URI: a model created
    through a Marketplace algorithm names only its package, and the image behind
    that package sits in a registry the buyer's account cannot pull from.


    ``name`` binds the object to the model already in the account, so the transform
    reuses it rather than minting a second one under a generated name.
    """
    from sagemaker.model import ModelPackage

    model_package_arn = model_description.get("PrimaryContainer", {}).get(
        "ModelPackageName"
    )
    if not model_package_arn:
        raise ValueError(
            f"Model '{model_name}' names no model package in its container "
            "definition, so it was not created through the algorithm. Falling back "
            "to the image URI it does carry is not a supported fallback — "
            "it resolves only inside the account that owns the registry. Deploy "
            "with SurvivalModel.deploy() and point SAGEMAKER_MODEL_NAME at that."
        )

    return ModelPackage(
        role=role if role is not None else os.environ.get("SAGEMAKER_ROLE_ARN"),
        model_package_arn=model_package_arn,
        name=model_name,
        sagemaker_session=sagemaker_session,
    )


@pytest.fixture(scope="session")
def deployed_survival_model(model_name, algorithm_arn):
    """A SurvivalModel wired to an already-deployed SageMaker model.

    Session-scoped because every test using it runs a batch transform, and the
    setup — describe the model, rebuild its package handle — is identical for all
    of them.

    Neither input has a "not available" branch here any more. ``model_name``
    always names a model, and ``algorithm_arn`` resolves the algorithm from the
    registry, falling back to ``SAGEMAKER_ALGORITHM_ARN`` only as an override.
    Both used to read as absent and skip: the first on an ordering accident, the
    second whenever the caller had not exported a variable the fixture could have
    looked up itself. A skip reports as success, so each one quietly removed
    these tests from the run.
    """
    import boto3
    import sagemaker

    from lifespan_predictor_sdk.survival_model import SurvivalModel

    client = boto3.client("sagemaker", region_name="eu-west-1")
    try:
        description = client.describe_model(ModelName=model_name)
    except client.exceptions.ClientError as error:
        if error.response["Error"]["Code"] == "ValidationException":
            pytest.skip(f"Model '{model_name}' does not exist in AWS SageMaker")
        raise

    model = SurvivalModel(
        {
            "backend": "sagemaker",
            "model_type": "deepsurv",
            "algorithm_arn": algorithm_arn,
            "instance_type": "ml.m5.large",
        }
    )
    model._sagemaker_model_name = model_name
    model._is_fitted = True
    model._sagemaker_model = deployed_model_from_description(
        description,
        model_name,
        sagemaker_session=sagemaker.Session(boto_session=boto3.Session()),
    )
    return model


# ---------------------------------------------------------------------------
# Survival curves from a deployed model
#
# Here rather than in one test module because two of them need the same curves,
# and each batch transform costs a job. Session-scoped so the second caller gets
# the first caller's answer: comparing curves from one transform against mean
# times from another compares two answers rather than two views of one.
# ---------------------------------------------------------------------------

# Enough rows to be a real transform, few enough to keep the job short.
SAMPLE_UNITS = 3


@pytest.fixture(scope="session")
def cloud_sample_df():
    """A few units of NASA test data, shared by everything that runs a transform."""
    from lifespan_predictor_sdk.data_loader import download_nasa_dataset, get_nasa_test_dataframe

    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "nasa"
    download_nasa_dataset(str(data_dir), [1])

    test_df, _ = get_nasa_test_dataframe(str(data_dir), datasets=[1])
    units = test_df["unit"].unique()[:SAMPLE_UNITS]
    return test_df[test_df["unit"].isin(units)].reset_index(drop=True)


@pytest.fixture(scope="session")
def cloud_curves(deployed_survival_model, cloud_sample_df):
    """One transform's survival curves, reused by every case that needs them."""
    return cloud_sample_df, deployed_survival_model.predict(cloud_sample_df)


def assert_is_a_survival_frame(curves, expected_columns):
    """The contract ``predict`` returns, asserted in one place.

    Monotonicity is the cheap assertion that catches a transposed reshape — the
    failure most likely to survive a shape check, because the shapes still match
    and only the numbers are wrong.
    """
    import numpy as np
    import pandas as pd

    assert isinstance(curves, pd.DataFrame)
    assert curves.shape[1] == expected_columns
    assert len(curves.index) > 1, "a single time point is not a curve"

    # Asserted here so every case that reads a curve checks it, rather than only
    # the one that compares two backends — which cannot run where there is one.
    assert curves.index.name == "duration", (
        f"the time index is named {curves.index.name!r}, not 'duration'; "
        "frames from different backends would not compare or concatenate"
    )

    values = curves.to_numpy(dtype=float)
    assert np.all(values >= -1e-9) and np.all(values <= 1 + 1e-9), (
        f"survival probabilities outside [0, 1]: "
        f"min={values.min()}, max={values.max()}"
    )

    assert curves.index.is_monotonic_increasing, "time index is not sorted"
    diffs = np.diff(values, axis=0)
    assert np.all(diffs <= 1e-6), (
        "survival probability rises with time in at least one column, which is "
        "what a transposed reshape looks like"
    )
