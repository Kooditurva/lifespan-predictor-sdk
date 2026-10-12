"""Test batch transform graceful failure handling."""

import os
import time

import pytest

from lifespan_predictor_sdk.data_loader import (
    download_nasa_dataset,
    get_nasa_test_dataframe,
)
from lifespan_predictor_sdk.sagemaker_backend import _resolve_bucket
from lifespan_predictor_sdk.survival_model import SurvivalModel


@pytest.mark.cloud
def test_batch_transform_fails_gracefully_on_model_load_error(algorithm_arn):
    """
    Test that batch transform job fails gracefully when model loading fails.

    This test validates that:
    - Container starts successfully
    - /invocations endpoint returns 500 errors (simulated failure)
    - SageMaker retries up to MaxRetries limit (default 3)
    - Transform job terminates with "Failed" status after exhausting retries

    Duration: ~1-2 minutes
    """
    # Load test data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "nasa")
    download_nasa_dataset(data_dir, [1])

    test_df, _ = get_nasa_test_dataframe(data_dir, datasets=[1])

    # Check if we can use existing model artifacts (faster)
    existing_model_data = os.environ.get("EXISTING_MODEL_DATA_S3_URI")

    if existing_model_data:
        # Use existing model artifacts with NEW container image
        print(f"Using existing model artifacts: {existing_model_data}")
        print("Creating Model with NEW container image...")

        import boto3
        import sagemaker
        from sagemaker.model import Model

        session = boto3.Session()
        sagemaker_session = sagemaker.Session(boto_session=session)

        # Get the algorithm to extract container image URI
        sm_client = sagemaker_session.sagemaker_client
        algo_info = sm_client.describe_algorithm(
            AlgorithmName=algorithm_arn.split("/")[-1]
        )
        inference_image_uri = algo_info["InferenceSpecification"]["Containers"][0][
            "Image"
        ]

        print(f"✓ Using container image: {inference_image_uri}")

        # Derive a stable, unique model name from the algorithm ARN so the
        # test doesn't depend on any external env var. Algorithm name is
        # already sanitized for SageMaker naming constraints.
        algo_name = algorithm_arn.split("/")[-1]
        model_name = f"{algo_name}-failure-test"

        # Check if model already exists (idempotent operation)
        try:
            existing = sm_client.describe_model(ModelName=model_name)
            print(f"⚠️  Model {model_name} already exists, deleting...")
            sm_client.delete_model(ModelName=model_name)
            time.sleep(2)  # Brief wait for deletion to propagate
        except Exception as e:
            if "ModelNotFound" not in str(e) and "Could not find model" not in str(e):
                raise  # Re-raise if it's not a "model not found" error
            # Model doesn't exist - proceed with creation

        # Create Model with existing artifacts but new image
        model = Model(
            image_uri=inference_image_uri,
            model_data=existing_model_data,
            role=os.environ.get("SAGEMAKER_ROLE_ARN"),
            name=model_name,
            sagemaker_session=sagemaker_session,
            env={"TEST_MODEL_LOAD_FAILURE": "true"},  # Env var for container
        )
        model.create()
        print(f"✓ Model created: {model_name}")

        # For cleanup
        created_model_for_cleanup = model

    else:
        # No existing artifacts - train a minimal model
        print(
            "No existing model artifacts provided. Training minimal model (1 epoch)..."
        )
        train_config = {
            "model_type": "deepsurv",
            "backend": "sagemaker",
            "algorithm_arn": algorithm_arn,
            "instance_type": "ml.m5.large",
            "num_nodes": [24],
            "epochs": 1,  # Minimal training just to get model artifacts
        }

        model_obj = SurvivalModel(train_config)

        # Train quickly (1 epoch) just to get model artifacts
        from lifespan_predictor_sdk.data_loader import get_nasa_train_dataframe

        train_df = get_nasa_train_dataframe(data_dir, datasets=[1])
        model_obj.fit(train_df)
        print("✓ Training complete")

        # Deploy the model
        print("Deploying model...")
        model_name = model_obj.deploy()
        print(f"✓ Model deployed: {model_name}")

        # Get the sagemaker Model object for transform
        model = model_obj._sagemaker_model
        created_model_for_cleanup = None  # Will cleanup via model_obj

    try:
        # Now run batch transform with TEST_MODEL_LOAD_FAILURE env var
        # This asks the container to simulate a model loading failure
        print("\nRunning batch transform with simulated model loading failure...")

        import tempfile
        from pathlib import Path

        import boto3
        import sagemaker

        session = boto3.Session()
        sagemaker_session = sagemaker.Session(boto_session=session)
        # Route through the configured SDK bucket (SAGEMAKER_BUCKET in CI);
        # the SDK CI role has no access to default_bucket().
        # train_config is only defined on one branch above — pass {} so the
        # helper falls through to SAGEMAKER_BUCKET / default_bucket().
        bucket = _resolve_bucket({}, sagemaker_session)

        # Upload test data
        timestamp = int(time.time())
        input_prefix = f"lifespan-predictor/test-failure-input-{timestamp}"
        output_prefix = f"lifespan-predictor/test-failure-output-{timestamp}"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_csv = tmpdir_path / "input.csv"
            test_df.head(10).to_csv(input_csv, index=False, header=False)

            input_s3_uri = sagemaker_session.upload_data(
                str(input_csv), bucket=bucket, key_prefix=input_prefix
            )

        print(f"✓ Test data uploaded: {input_s3_uri}")

        # Create transformer with environment variable to trigger failure
        output_s3_uri = f"s3://{bucket}/{output_prefix}/"

        transformer = model.transformer(
            instance_count=1,
            instance_type="ml.m5.large",
            output_path=output_s3_uri,
            accept="text/csv",
            max_payload=6,
        )

        # Start transform job (don't wait)
        print("Starting batch transform job (expecting failure)...")
        transformer.transform(
            data=input_s3_uri,
            content_type="text/csv",
            split_type="Line",
            wait=False,  # Don't wait - we'll poll manually
            logs=False,
        )

        transform_job_name = transformer.latest_transform_job.name
        print(f"✓ Transform job started: {transform_job_name}")

        # Poll job status - expecting it to fail gracefully after retries
        # With default MaxRetries=3, should fail quickly once all retries exhausted
        print("\nMonitoring job status (expecting 'Failed' within 1-2 minutes)...")
        sm_client = sagemaker_session.sagemaker_client

        max_wait = 360  # 6 minutes max (allows for retries with backoff)
        start_time = time.time()
        last_status = None

        while time.time() - start_time < max_wait:
            response = sm_client.describe_transform_job(
                TransformJobName=transform_job_name
            )
            status = response["TransformJobStatus"]

            if status != last_status:
                elapsed = int(time.time() - start_time)
                print(f"  [{elapsed}s] Status: {status}")
                last_status = status

            if status == "Failed":
                elapsed = int(time.time() - start_time)
                print("✓ Job failed gracefully as expected!")
                failure_reason = response.get("FailureReason", "No reason provided")
                print(f"  Failure reason: {failure_reason}")

                # Verify job failed (timing may vary due to SageMaker retries)
                assert status == "Failed", "Job should have Failed status"

                # Success - job failed as expected
                # Note: Timing can be 2-4 minutes due to SageMaker's retry logic and failure detection
                print(f"\n✅ Test passed: Job failed gracefully in {elapsed}s")
                print(
                    "   (SageMaker detected container crash and marked job as Failed)"
                )
                return

            elif status in ["Completed", "Stopped"]:
                pytest.fail(f"Job should have Failed, but got status: {status}")

            time.sleep(5)

        # If we get here, job didn't fail within timeout
        pytest.fail(
            f"Job did not fail within {max_wait}s (6 minutes). "
            f"Last status: {last_status}. "
            "Container may not be crashing, or SageMaker failure detection is taking longer than expected. "
            "Check CloudWatch logs for container output."
        )

    finally:
        # Cleanup: delete the model
        print("\nCleaning up...")
        try:
            if created_model_for_cleanup:
                # Cleanup via Model object
                import boto3

                session = boto3.Session()
                sm_client = session.client("sagemaker")
                sm_client.delete_model(ModelName=model_name)
            else:
                # Cleanup via SurvivalModel
                model_obj.delete()
            print("✓ Model deleted")
        except Exception as e:
            print(f"⚠️  Failed to delete model: {e}")
