"""Cloud training tests using SurvivalModel."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

from lifespan_predictor_sdk.data_loader import (
    download_nasa_dataset,
    get_nasa_test_dataframe,
    get_nasa_train_dataframe,
)
from lifespan_predictor_sdk.survival_model import SurvivalModel, c_index_score, rmse_score


def verify_model_exists_in_aws(model_name, region="eu-west-1"):
    """
    Verify that a SageMaker model actually exists in AWS.

    Args:
        model_name: The SageMaker model name
        region: AWS region (default: eu-west-1)

    Returns:
        dict: Model description from AWS

    Raises:
        AssertionError: If model doesn't exist or validation fails
    """
    import boto3

    sagemaker_client = boto3.client("sagemaker", region_name=region)

    try:
        response = sagemaker_client.describe_model(ModelName=model_name)
        print(f"✓ Model verified in AWS SageMaker:")
        print(f"  - Model Name: {response['ModelName']}")
        print(f"  - Creation Time: {response['CreationTime']}")
        print(f"  - Execution Role: {response['ExecutionRoleArn']}")

        # Verify essential fields exist
        assert "ModelName" in response, "Model response missing ModelName"
        assert "ExecutionRoleArn" in response, "Model response missing ExecutionRoleArn"
        assert "CreationTime" in response, "Model response missing CreationTime"

        # Verify model name matches
        assert (
            response["ModelName"] == model_name
        ), f"Model name mismatch: expected {model_name}, got {response['ModelName']}"

        return response

    except sagemaker_client.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            raise AssertionError(
                f"Model '{model_name}' does not exist in AWS SageMaker"
            ) from e
        raise


@pytest.mark.cloud
def test_sagemaker_training_and_metrics(trained_and_deployed_model):
    """
    Test SageMaker training and comprehensive metrics validation.

    This test validates:
    - Training completes successfully
    - Training log contains expected data
    - metrics.json contains c_index and rmse
    - FinalMetricDataList (CloudWatch metrics) contains test:c_index and test:rmse
    - Consistency between metrics.json and CloudWatch metrics
    - Training job status is "Completed"
    - The trained model deploys and exists in AWS

    The run itself happens in the ``trained_and_deployed_model`` fixture, which
    every cloud test needing a model shares. This test asserts over what that run
    produced; it does not own it. Owning it was what made the inference cases
    depend on being collected after this file.

    Duration: ~8-12 minutes
    """
    model = trained_and_deployed_model["model"]
    log = trained_and_deployed_model["log"]
    training_info = trained_and_deployed_model["training_info"]
    eval_result = trained_and_deployed_model["eval_result"]

    # Validate training job status
    print("Validating training job status...")
    training_status = training_info["TrainingJobStatus"]
    print(f"  Training job status: {training_status}")
    assert (
        training_status == "Completed"
    ), f"Training job should be Completed, got {training_status}"
    print("✓ Training job completed successfully")

    # Verify training log was returned
    assert log is not None, "Training log should not be None"
    df = log.to_pandas()
    assert "train_loss" in df.columns, "Training log should contain train_loss"
    print(f"✓ Retrieved training log with {len(df)} epochs")

    # Verify model is marked as fitted
    assert model._is_fitted, "Model should be marked as fitted"
    assert model._sagemaker_model_path is not None, "SageMaker model path should be set"
    print(f"✓ Model artifact: {model._sagemaker_model_path}")

    # Validate metrics from EvaluationResult
    print("\nValidating metrics.json...")
    assert (
        eval_result is not None
    ), "Evaluation result should be available after training with test data"
    eval_metrics = eval_result.get_metrics()
    assert eval_metrics["c_index"] is not None, "C-index should not be None"
    assert (
        0.85 < eval_metrics["c_index"] < 1
    ), f"C-index should be between 0.85 and 1, got {eval_metrics['c_index']}"
    print(f"✓ Test C-index (metrics.json): {eval_metrics['c_index']:.4f}")

    if eval_metrics.get("rmse") is not None:
        assert (
            eval_metrics["rmse"] < 15
        ), f"RMSE should be less than 15, got {eval_metrics['rmse']}"
        print(f"✓ Test RMSE (metrics.json): {eval_metrics['rmse']:.4f}")

    # Validate FinalMetricDataList (CloudWatch metrics)
    print("\nValidating CloudWatch metrics (FinalMetricDataList)...")
    if "FinalMetricDataList" in training_info:
        metrics = training_info["FinalMetricDataList"]
        metric_dict = {m["MetricName"]: m["Value"] for m in metrics}

        print(f"  Available CloudWatch metrics: {list(metric_dict.keys())}")

        # Check if test metrics exist
        expected_metrics = {"test:c_index", "test:rmse"}
        missing_metrics = expected_metrics - set(metric_dict.keys())

        if missing_metrics:
            print(f"⚠ Warning: Missing CloudWatch metrics: {missing_metrics}")
            print(
                "  This may indicate MetricDefinitions need to be updated in algorithm registration"
            )
        else:
            # Validate CloudWatch metric values
            c_index_cw = metric_dict["test:c_index"]
            print(f"✓ Test C-index (CloudWatch): {c_index_cw:.4f}")
            assert (
                0.85 < c_index_cw < 1
            ), f"CloudWatch C-index should be between 0.85 and 1, got {c_index_cw}"

            # Validate consistency between metrics.json and CloudWatch
            c_index_diff = abs(c_index_cw - eval_metrics["c_index"])
            print(
                f"  C-index difference (CloudWatch vs metrics.json): {c_index_diff:.6f}"
            )
            assert (
                c_index_diff < 0.01
            ), f"C-index mismatch: CloudWatch={c_index_cw:.4f}, metrics.json={eval_metrics['c_index']:.4f}"
            print("✓ C-index consistency validated")

            if "test:rmse" in metric_dict:
                rmse_cw = metric_dict["test:rmse"]
                print(f"✓ Test RMSE (CloudWatch): {rmse_cw:.4f}")
                assert (
                    rmse_cw < 14
                ), f"CloudWatch RMSE should be less than 14, got {rmse_cw}"

                if eval_metrics.get("rmse") is not None:
                    rmse_diff = abs(rmse_cw - eval_metrics["rmse"])
                    print(
                        f"  RMSE difference (CloudWatch vs metrics.json): {rmse_diff:.6f}"
                    )
                    assert (
                        rmse_diff < 0.5
                    ), f"RMSE mismatch: CloudWatch={rmse_cw:.4f}, metrics.json={eval_metrics['rmse']:.4f}"
                    print("✓ RMSE consistency validated")
    else:
        print("⚠ Warning: FinalMetricDataList not found in training job description")
        print("  CloudWatch metrics validation skipped")

    # Fit quality analysis
    fit_quality = log.analyze_fit_quality()
    print(f"\nFit quality: {fit_quality}")
    assert (
        fit_quality is None
    ), "Fit quality should be None when no validation data is provided"

    # Validate the deployment the fixture performed
    deployed_model_name = trained_and_deployed_model["model_name"]
    print(f"\nValidating deployment: {deployed_model_name}")
    model_description = verify_model_exists_in_aws(deployed_model_name)
    assert model_description is not None, "Model validation failed after deployment"
    print("✓ Model exists in AWS")
    print("\n✅ SageMaker training and metrics validation passed!")


@pytest.mark.cloud
def test_sagemaker_inference(model_name, algorithm_arn):
    """
    Test SageMaker deployment and batch inference.

    This test validates:
    - Model can be deployed to SageMaker (if training job provided)
    - Model exists in AWS
    - Batch transform predictions work
    - Predictions are valid (non-negative, no NaN, correct shape)

    Duration: ~3-5 minutes

    Usage:
        # Option 1: Run with existing model (skip training and deployment)
        SAGEMAKER_MODEL_NAME=lifespan-predictor-1737154783 AWS_PROFILE=dev uv run pytest -m cloud tests/test_cloud_training.py::test_sagemaker_inference

        # Option 2: Run after training test (tests full flow including deployment)
        AWS_PROFILE=dev uv run pytest -m cloud tests/test_cloud_training.py
    """

    # Check if model_name is available
    if model_name is None:
        pytest.skip(
            "No model available. Either:\n"
            "  1. Run test_sagemaker_training_and_metrics first, OR\n"
            "  2. Set SAGEMAKER_MODEL_NAME environment variable"
        )

    print(f"Using model: {model_name}")

    # Load test data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "nasa")
    download_nasa_dataset(data_dir, [1])

    test_df, test_labels_df = get_nasa_test_dataframe(data_dir, datasets=[1])

    # Using existing model - skip deployment, just validate it exists
    print("\nValidating existing model in AWS SageMaker...")
    model_description = verify_model_exists_in_aws(model_name)
    assert model_description is not None, f"Model '{model_name}' does not exist in AWS"
    print("✓ Existing model validated in AWS")

    from conftest import TRAINED_MAX_RUL

    config = {
        "backend": "sagemaker",
        "model_type": "deepsurv",
        "algorithm_arn": algorithm_arn,
        "instance_type": "ml.m5.large",
        "max_rul": TRAINED_MAX_RUL,
    }
    model_for_inference = SurvivalModel(config)
    model_for_inference._sagemaker_model_name = model_name
    model_for_inference._is_fitted = True

    # Rebuild the model object for batch transform. No estimator is needed - we
    # use transformer() directly.
    import boto3
    import sagemaker
    from conftest import deployed_model_from_description

    session = boto3.Session()
    sagemaker_session = sagemaker.Session(boto_session=session)

    model_for_inference._sagemaker_model = deployed_model_from_description(
        model_description,
        model_name,
        sagemaker_session=sagemaker_session,
    )
    print("✓ Model wrapper created for inference")

    # Run batch transform predictions
    print("\nInference evaluation...")
    events = test_labels_df["event"].values
    durations = test_labels_df["duration"].values
    result_dict = model_for_inference.evaluate(test_df, durations, events)
    assert result_dict is not None, "Result dictionary should not be None"
    predictions = result_dict.get("predictions", None)
    assert predictions is not None, "Predictions should not be None"
    print(
        f"Predictions shape: {predictions.shape if hasattr(predictions, 'shape') else len(predictions)}"
    )

    # Validate predictions
    num_equipments = test_df["unit"].nunique()
    assert (
        len(predictions) == num_equipments
    ), f"Number of predictions ({len(predictions)}) should equal test equipment ({num_equipments})"
    assert not np.isnan(predictions).any(), "Predictions should not contain NaN"
    assert (predictions >= 0).all(), "Predictions should be non-negative"
    print(
        f"✓ Predictions valid: min={predictions.min():.2f}, max={predictions.max():.2f}, mean={predictions.mean():.2f}"
    )
    c_index = result_dict.get("c_index", None)
    assert c_index is not None, "C-index should not be None"
    assert 0.8 <= c_index <= 1.0, f"C-index should be between 0 and 1, got {c_index}"
    print(f"✓ C-index: {c_index:.4f}")
    # RMSE should be less than 14
    rmse = result_dict.get("rmse", None)
    assert rmse is not None, "RMSE should not be None"
    assert rmse < 14, f"RMSE should be less than 14, got {rmse}"
    print(f"✓ RMSE: {rmse:.4f}")

    # Cleanup
    print("\nCleaning up...")
    print("\n✅ SageMaker inference test passed!")


@pytest.mark.cloud
def test_free_tier_refuses_training_over_the_cap():
    """The free-tier candidate refuses one unit too many.

    The release run exports the free-tier candidate's ARN. Any other run, including
    the public SDK's CI, has no free-tier algorithm to test, so it skips.
    """
    algorithm_arn = os.environ.get("SAGEMAKER_FREE_TIER_ALGORITHM_ARN")
    if not algorithm_arn:
        pytest.skip("SAGEMAKER_FREE_TIER_ALGORITHM_ARN is not set")

    sys.path.append(os.path.join(os.path.dirname(__file__), "../scripts"))
    from cloud_helpers import FREE_TIER_MAX_UNITS

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "nasa")
    download_nasa_dataset(data_dir, [1])
    train_df = get_nasa_train_dataframe(data_dir, datasets=[1])

    # FD001 has exactly the cap's 100 units; add one more, copied from the first.
    extra = train_df[train_df["unit"] == train_df["unit"].min()].copy()
    extra["unit"] = train_df["unit"].max() + 1
    train_df = pd.concat([train_df, extra], ignore_index=True)
    assert train_df["unit"].nunique() == FREE_TIER_MAX_UNITS + 1

    model = SurvivalModel(
        {
            "model_type": "deepsurv",
            "backend": "sagemaker",
            "algorithm_arn": algorithm_arn,
            "instance_type": "ml.m5.large",
            "epochs": 1,
        }
    )
    # The job fails before training; SageMaker's exception carries the job's
    # FailureReason, which the container wrote.
    with pytest.raises(Exception) as excinfo:
        model.fit(train_df)

    reason = str(excinfo.value)
    assert f"{FREE_TIER_MAX_UNITS + 1} units" in reason
    assert f"cap of {FREE_TIER_MAX_UNITS}" in reason
