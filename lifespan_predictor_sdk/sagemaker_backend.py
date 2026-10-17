"""
SageMaker backend for SurvivalModel.

Encapsulates all AWS SageMaker-specific logic: training, deployment,
batch transform inference, and calibration.
"""

import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _resolve_bucket(model_config, sagemaker_session):
    """Resolve the S3 bucket for SDK uploads and training output artifacts.

    Precedence: model.config['bucket'] > SAGEMAKER_BUCKET env var >
    sagemaker.Session.default_bucket(). The public SDK CI role is scoped
    to a dedicated bucket only, so the env var must be honored everywhere
    the SDK touches S3 — including AlgorithmEstimator output_path.
    """
    return (
        model_config.get("bucket")
        or os.environ.get("SAGEMAKER_BUCKET")
        or sagemaker_session.default_bucket()
    )


def load_tuning_trials(s3, model_data_uri: str, config: dict):
    """The tuning results of a training job, or None if it didn't tune.

    The container writes them to /opt/ml/output/data, which SageMaker uploads as
    output.tar.gz beside model.tar.gz. A job that wrote nothing there has no
    output.tar.gz.

    `config` is the one the job was submitted with. Its ranges name the tuned
    settings, which become each trial's `params`: the container records only a
    trial's whole config.
    """
    import tarfile

    import botocore.exceptions

    from lifespan_predictor_sdk.hpo_io import TrialResult, TuningTrials, without_internal_keys
    from lifespan_predictor_sdk.survival_model import CategoricalRange, ContinuousRange, IntegerRange

    tuned = [
        key
        for key, value in config.items()
        if isinstance(value, (ContinuousRange, IntegerRange, CategoricalRange))
    ]

    bucket = model_data_uri.split("/")[2]
    key = "/".join(model_data_uri.split("/")[3:]).rsplit("/", 1)[0] + "/output.tar.gz"

    with tempfile.TemporaryDirectory() as tmpdir:
        archive = os.path.join(tmpdir, "output.tar.gz")
        try:
            s3.download_file(bucket, key, archive)
        except botocore.exceptions.ClientError as error:
            if error.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
                return None
            raise
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(path=tmpdir, filter="data")

        trials_path = os.path.join(tmpdir, "tuning_trials.json")
        if not os.path.exists(trials_path):
            return None
        with open(trials_path) as f:
            recorded = json.load(f)
        results = []
        for t in recorded:
            trial_config = without_internal_keys(t.get("config", {}))
            results.append(
                TrialResult(
                    number=t["number"],
                    value=t["value"],
                    params={k: trial_config[k] for k in tuned if k in trial_config},
                    config=trial_config,
                    state=t["state"],
                    fold_scores_list=t.get("fold_scores"),
                )
            )
        trials = TuningTrials(results)

        best_config_path = os.path.join(tmpdir, "best_config.json")
        if os.path.exists(best_config_path):
            with open(best_config_path) as f:
                trials.best_config = without_internal_keys(json.load(f))
        return trials


# The only ModelPackageStatus values worth waiting on. Anything else is either
# terminal or unknown, and waiting on an unknown status is waiting forever.
_MODEL_PACKAGE_PENDING_STATUSES = frozenset({"Pending", "InProgress", "Deleting"})


def _validate_sagemaker_role():
    """
    Validate that SAGEMAKER_ROLE_ARN environment variable is set.

    This function should be called at the beginning of any SageMaker operation
    (training, deployment) to ensure the role is properly configured.

    Raises:
        ValueError: If SAGEMAKER_ROLE_ARN environment variable is not set or is empty

    Returns:
        str: The validated role ARN
    """
    role = os.environ.get("SAGEMAKER_ROLE_ARN")
    if not role or not role.strip():
        raise ValueError(
            "Environment variable 'SAGEMAKER_ROLE_ARN' is required for SageMaker operations. "
            "Set it before running your code:\n"
            "  export SAGEMAKER_ROLE_ARN='arn:aws:iam::ACCOUNT:role/service-role/AmazonSageMaker-ExecutionRole-...'\n"
            "\n"
            "To find your role ARN:\n"
            "  aws iam list-roles --query 'Roles[?contains(RoleName, `SageMaker`)].Arn' --output text\n"
            "\n"
            "See: https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html"
        )
    return role.strip()


def _log_training_job_info(estimator):
    """Log training job information and metrics."""
    training_info = estimator.latest_training_job.describe()

    print(f"\nTraining Job Summary:")
    print(f"  Name: {training_info['TrainingJobName']}")
    print(f"  Status: {training_info['TrainingJobStatus']}")
    if "TrainingTimeInSeconds" in training_info:
        print(f"  Training Time: {training_info['TrainingTimeInSeconds']} seconds")
    if "BillableTimeInSeconds" in training_info:
        print(f"  Billable Time: {training_info['BillableTimeInSeconds']} seconds")
    if "ResourceConfig" in training_info:
        print(f"  Instance Type: {training_info['ResourceConfig']['InstanceType']}")
    print(f"  Model artifacts: {training_info['ModelArtifacts']['S3ModelArtifacts']}")

    # Optional fields - check before accessing
    if (
        "DebugHookConfig" in training_info
        and "S3OutputPath" in training_info["DebugHookConfig"]
    ):
        print(f"  Debug output: {training_info['DebugHookConfig']['S3OutputPath']}")

    # Log hyperparameters
    if "HyperParameters" in training_info:
        print(f"  Hyperparameters: {len(training_info['HyperParameters'])} configured")

    # Log metrics
    if "FinalMetricDataList" in training_info:
        metrics = training_info["FinalMetricDataList"]
        if metrics:
            print(f"\n  Final Metrics:")
            for metric in metrics:
                metric_name = metric["MetricName"]
                metric_value = metric["Value"]
                print(f"    {metric_name}: {metric_value:.4f}")


class SageMakerBackend:
    """Backend that runs training and inference on AWS SageMaker."""

    # What this backend calls itself in config. SurvivalModel reads it back rather
    # than hard-coding the name.
    name = "sagemaker"

    def init_state(self, config):
        """Initialize SageMaker-specific state and merge defaults into config.

        Args:
            config: The shared config dict (mutated in place for defaults).
        """
        self._sagemaker_model_path = None
        self._sagemaker_session = None
        self._sagemaker_model_name = None
        self._endpoint_name = None
        self._estimator = None
        self._training_job_name = None
        self._model_data = None
        self._sagemaker_model = None  # Model object for batch transform
        # Cache for survival curves from predict_mean
        self._cached_survival_curves = None
        self._cached_time_grid = None

        # Merge SageMaker-specific defaults (only if not already set by user)
        if "instance_type" not in config or config["instance_type"] is None:
            config["instance_type"] = "ml.m5.large"

    def fit(self, model, train_df, val_df=None, test_df=None, test_labels_df=None):
        """
        Train on AWS SageMaker with raw DataFrame input.

        Args:
            model: The SurvivalModel instance (for shared state access).
            train_df: Training data (raw DataFrame)
            val_df: Optional validation data for early stopping
            test_df: Optional test samples
            test_labels_df: Optional test labels (columns: unit, duration, event)

        Returns:
            TrainingLog with c_index and rmse if test data provided

        Sends raw CSV to SageMaker - the container handles all preprocessing.
        This keeps preprocessing logic hidden from clients.
        """
        # Raw data goes out as-is; the container does every transformation
        # Container will load CSV, initialize preprocessor, and preprocess during training

        try:
            import boto3
            import sagemaker
            from sagemaker import AlgorithmEstimator
        except ImportError:
            raise ImportError(
                "SageMaker training requires boto3 and sagemaker packages. "
                "Install with: uv add boto3 sagemaker"
            )

        from lifespan_predictor_sdk.training_log import EvaluationResult, TrainingLog

        print("🚀 Starting SageMaker training...")

        # Setup session
        session = boto3.Session()
        sagemaker_session = sagemaker.Session(boto_session=session)
        region = session.region_name
        account_id = session.client("sts").get_caller_identity()["Account"]

        # Get or create bucket (see _resolve_bucket for precedence)
        bucket = _resolve_bucket(model.config, sagemaker_session)
        print(f"📦 S3 bucket: {bucket}")

        # Validate and get role from environment variable
        role = _validate_sagemaker_role()
        print(f"🔑 IAM Role: {role}")

        # Get Algorithm ARN (required for AlgorithmEstimator)
        algorithm_arn = model.config.get("algorithm_arn")
        if not algorithm_arn:
            # Check for SAGEMAKER_ALGORITHM_ARN environment variable
            algorithm_arn = os.environ.get("SAGEMAKER_ALGORITHM_ARN")

        if not algorithm_arn:
            raise ValueError(
                "algorithm_arn is required for SageMaker training. "
                "Provide it in config or set SAGEMAKER_ALGORITHM_ARN environment variable.\n"
                "\n"
                "For development, register your Docker image as an algorithm:\n"
                "  bash scripts/register_algorithm.sh latest\n"
                "\n"
                "For marketplace clients, use the Algorithm ARN from your subscription."
            )
        print(f"🔖 Algorithm ARN: {algorithm_arn}")

        # Prepare and upload RAW data as CSV
        print("📤 Uploading raw CSV data to S3...")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Save RAW training DataFrame as CSV
            train_csv_path = tmpdir_path / "train.csv"
            train_df.to_csv(train_csv_path, index=False)
            print(f"  ✓ Saved train.csv ({len(train_df)} rows)")

            # Save RAW validation DataFrame if provided
            if val_df is not None:
                val_csv_path = tmpdir_path / "val.csv"
                val_df.to_csv(val_csv_path, index=False)
                print(f"  ✓ Saved val.csv ({len(val_df)} rows)")

            # Save test files if BOTH provided
            if test_df is not None and test_labels_df is not None:
                test_csv_path = tmpdir_path / "test.csv"
                test_df.to_csv(test_csv_path, index=False)
                print(f"  ✓ Saved test.csv ({len(test_df)} rows)")

                test_labels_csv_path = tmpdir_path / "test_labels.csv"
                test_labels_df.to_csv(test_labels_csv_path, index=False)
                print(f"  ✓ Saved test_labels.csv ({len(test_labels_df)} rows)")

            # Upload to S3 (uploads tmpdir contents to train channel)
            # Container will receive these files at /opt/ml/input/data/train/
            prefix = f"lifespan-predictor/training-{int(time.time())}"
            train_input = sagemaker_session.upload_data(
                str(tmpdir_path), bucket=bucket, key_prefix=prefix
            )
            print(f"✓ Raw CSV uploaded to: {train_input}")

        # Create estimator
        print("⚙️  Configuring training job...")

        # The training container only needs model/training config, not AWS
        # infrastructure config.
        from lifespan_predictor_sdk.hpo_io import serialize_config, without_internal_keys

        training_config = without_internal_keys(model.config)
        serialized_config = serialize_config(training_config)
        # Include SDK version so the container can check compatibility. Omitted
        # entirely when it cannot be determined: the container skips the check on
        # an absent key, and sending a placeholder would fail the comparison
        # instead.
        sdk_version = model._get_sdk_version()
        if sdk_version:
            serialized_config["sdk_version"] = sdk_version
        print(f"📋 Training config keys: {list(serialized_config.keys())}")
        print(f"📋 Config size: {len(json.dumps(serialized_config))} chars")

        # output_path is required: without it the SageMaker SDK falls back
        # to sagemaker_session.default_bucket(), which the public SDK CI
        # role cannot access (HeadBucket 403).
        output_path = f"s3://{bucket}/lifespan-predictor/output"
        estimator = AlgorithmEstimator(
            algorithm_arn=algorithm_arn,
            role=role,
            instance_count=1,
            instance_type=model.config["instance_type"],
            sagemaker_session=sagemaker_session,
            base_job_name="survival-model",
            output_path=output_path,
            hyperparameters={
                "config": json.dumps(serialized_config),
            },
        )

        # Start training
        print(f"🏃 Starting training on {model.config['instance_type']}...")
        print("   (This will take a few minutes)")
        estimator.fit({"train": train_input}, wait=True, logs=True)

        # Check status
        status = estimator.latest_training_job.describe()["TrainingJobStatus"]
        if status != "Completed":
            raise RuntimeError(f"Training job failed with status: {status}")

        print("✓ Training completed!")

        _log_training_job_info(estimator)

        # Store estimator for deploy() and predict_mean() methods
        self._estimator = estimator

        # Store training job info for backward compatibility and other methods
        training_job_name = estimator.latest_training_job.name
        model_data_uri = estimator.model_data

        self._training_job_name = training_job_name
        self._model_data = model_data_uri
        self._sagemaker_model_path = (
            model_data_uri  # Store model path for save() method
        )
        self._sagemaker_session = sagemaker_session
        self._sagemaker_model_name = None  # Will be set by deploy()

        print(f"✓ Training job: {training_job_name}")
        print(f"✓ Model artifact: {model_data_uri}")

        # Download model artifact and extract training log
        print("📥 Downloading training metrics...")

        # Download and extract training log
        import tarfile

        s3 = session.client("s3")
        bucket_name = model_data_uri.split("/")[2]
        key = "/".join(model_data_uri.split("/")[3:])

        train_loss = []
        val_loss = []
        learning_rate = []
        epochs = model.config["epochs"]

        with tempfile.TemporaryDirectory() as tmpdir:
            local_tar_path = os.path.join(tmpdir, "model.tar.gz")
            s3.download_file(bucket_name, key, local_tar_path)

            # Extract training log
            with tarfile.open(local_tar_path, "r:gz") as tar:
                tar.extractall(path=tmpdir)

            # Read training log if it exists
            log_path = os.path.join(tmpdir, "training_log.json")
            if os.path.exists(log_path):
                with open(log_path, "r") as f:
                    log_data = json.load(f)
                    train_loss = log_data.get("train_loss", [])
                    val_loss = log_data.get("val_loss", [])
                    learning_rate = log_data.get("learning_rate", [])
                    epochs = log_data.get("epochs", model.config["epochs"])
            else:
                print("⚠ Training log not found in model artifact")
                # Fallback values if log not found
                learning_rate = (
                    [model.config["learning_rate"]] * len(train_loss)
                    if train_loss
                    else []
                )

            # Read metrics (now includes c_index/rmse/calibration if test.csv was provided)
            metrics_path = os.path.join(tmpdir, "metrics.json")
            test_c_index = None
            test_rmse = None
            test_calibration_bias = None
            test_calibration_mean_cdf = None
            test_calibration_classification = None
            test_ibs = None
            test_ibs_classification = None
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)

                    # Store evaluation metrics if present
                    if "c_index" in metrics:
                        test_c_index = metrics["c_index"]
                        test_rmse = metrics.get("rmse")
                        print(f"✓ Test C-index: {test_c_index:.4f}")
                        if test_rmse:
                            print(f"✓ Test RMSE: {test_rmse:.4f}")

                    test_calibration_bias = metrics.get("calibration_bias")
                    test_calibration_mean_cdf = metrics.get("calibration_mean_cdf")
                    test_calibration_classification = metrics.get(
                        "calibration_classification"
                    )
                    test_ibs = metrics.get("ibs")
                    test_ibs_classification = metrics.get("ibs_classification")
                    if test_calibration_bias is not None:
                        print(
                            f"✓ Calibration bias: {test_calibration_bias:.4f} ({test_calibration_classification})"
                        )
                    if test_ibs is not None:
                        print(f"✓ IBS: {test_ibs:.4f} ({test_ibs_classification})")

            # Read calibration curve artifact if present
            cal_path = os.path.join(tmpdir, "calibration_curve.npz")
            cal_curve_data = None
            if os.path.exists(cal_path):
                data = np.load(cal_path, allow_pickle=True)
                cal_curve_data = (data["sorted_cdf"], data["empirical_cdf"])

        tuning_trials = load_tuning_trials(s3, model_data_uri, model.config)
        if tuning_trials is not None:
            print(f"✓ tuning_trials loaded: {len(tuning_trials)} trials")

        print("✓ Training complete!")
        model._is_fitted = True

        # Create TrainingLog with evaluation metrics
        training_log = TrainingLog(
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=learning_rate,
            epochs=epochs,
        )
        if tuning_trials is not None:
            model._tuning_trials = tuning_trials

        # Store evaluation result if metrics were available
        if test_c_index is not None:
            sorted_cdf, empirical_cdf = (
                cal_curve_data if cal_curve_data else (None, None)
            )
            model._evaluation_results = {
                "after": EvaluationResult(
                    c_index=test_c_index,
                    rmse=test_rmse,
                    calibration_bias=test_calibration_bias,
                    calibration_mean_cdf=test_calibration_mean_cdf,
                    calibration_classification=test_calibration_classification,
                    calibration_sorted_cdf=sorted_cdf,
                    calibration_empirical_cdf=empirical_cdf,
                    ibs=test_ibs,
                    ibs_classification=test_ibs_classification,
                )
            }

        return training_log

    def predict_mean(
        self,
        model,
        X,
        model_name=None,
        instance_type="ml.m5.large",
        instance_count=1,
        wait=True,
        max_payload=6,
    ):
        """
        Predict mean survival time using SageMaker batch transform.

        Args:
            model: The SurvivalModel instance.
            X: Features DataFrame.
            model_name: SageMaker model name (uses last deployed if None).
            instance_type: Instance type for batch transform (default: 'ml.m5.large').
            instance_count: Number of instances (default: 1).
            wait: Wait for batch transform to complete (default: True).
            max_payload: Max payload size in MB (default: 6).

        Returns:
            numpy array of mean survival times, one per input row. The survival
            curves from the same response are cached for get_survival_curves().
        """
        if not model._is_fitted:
            raise ValueError(
                "Model must be fitted before prediction. Call fit() first."
            )

        # Check if model has been deployed (we need _sagemaker_model for transformer)
        if not hasattr(self, "_sagemaker_model") or self._sagemaker_model is None:
            raise ValueError(
                "Model must be deployed first. Call deploy() before predict_mean()."
            )

        if getattr(self, "_endpoint_name", None):
            return self._predict_through_endpoint(model, X)

        try:
            import boto3
            import sagemaker
        except ImportError:
            raise ImportError(
                "Batch transform requires boto3 and sagemaker packages. "
                "Install with: pip install boto3 sagemaker"
            )

        print("🔮 Running batch transform for mean survival predictions...")
        print(f"   Input: {len(X)} samples")

        # Get session (prefer existing session from estimator or stored session)
        session = boto3.Session()
        if self._sagemaker_session:
            sagemaker_session = self._sagemaker_session
        elif self._estimator and hasattr(self._estimator, "sagemaker_session"):
            sagemaker_session = self._estimator.sagemaker_session
        else:
            sagemaker_session = sagemaker.Session(boto_session=session)
        bucket = _resolve_bucket(model.config, sagemaker_session)

        # Upload input data as CSV to S3
        timestamp = int(time.time())
        input_prefix = f"lifespan-predictor/batch-input-{timestamp}"
        output_prefix = f"lifespan-predictor/batch-output-{timestamp}"

        print("📤 Uploading input data to S3...")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_csv = tmpdir_path / "input.csv"
            # Drop label columns if present — container expects only feature columns
            _label_cols = ["duration", "event"]
            X_upload = X.drop(columns=[c for c in _label_cols if c in X.columns])
            X_upload.to_csv(input_csv, index=False, header=True)

            # Upload to S3
            input_s3_uri = sagemaker_session.upload_data(
                str(input_csv), bucket=bucket, key_prefix=input_prefix
            )

        print(f"✓ Input uploaded: {input_s3_uri}")

        # Create transformer from the deployed model (not estimator)
        # This avoids creating a duplicate model with timestamp suffix
        output_s3_uri = f"s3://{bucket}/{output_prefix}/"

        # Use the Model object we created in deploy(). JSON, because the
        # survival curves come back only on that path (text/csv carries mean
        # times alone).
        transformer = self._sagemaker_model.transformer(
            instance_count=instance_count,
            instance_type=instance_type,
            output_path=output_s3_uri,
            accept="application/json",
            max_payload=max_payload,
        )

        # Run batch transform
        print(f"🏃 Starting batch transform job...")
        print(f"   Instance: {instance_type} (count: {instance_count})")
        transformer.transform(
            data=input_s3_uri,
            content_type="text/csv",
            split_type="Line",
            wait=wait,
            logs=True if wait else False,
        )

        if not wait:
            print("⏳ Batch transform job started (not waiting)")
            print(f"   Job name: {transformer.latest_transform_job.name}")
            print(f"   Output will be at: {output_s3_uri}")
            return None

        print("✅ Batch transform complete!")

        # Download and parse results
        print("📥 Downloading results from S3...")
        s3_client = session.client("s3")

        # List output files
        output_bucket = bucket
        output_key_prefix = output_prefix + "/"

        response = s3_client.list_objects_v2(
            Bucket=output_bucket, Prefix=output_key_prefix
        )

        if "Contents" not in response:
            raise RuntimeError(f"No output found at {output_s3_uri}")

        # Download all output files (batch transform may split into multiple files)
        output_files = [obj["Key"] for obj in response["Contents"]]
        if not output_files:
            raise RuntimeError(f"No output files found at {output_s3_uri}")

        print(f"   Found {len(output_files)} output file(s)")

        responses = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, output_file in enumerate(output_files):
                local_output = os.path.join(tmpdir, f"output_{i}.json")
                s3_client.download_file(output_bucket, output_file, local_output)

                with open(local_output, "r") as f:
                    responses.append(json.load(f))

        predictions = self._collect(model, responses)
        print(f"✓ Results downloaded: {len(predictions)} predictions")

        return predictions

    def _predict_through_endpoint(self, model, X):
        """One request to the real-time endpoint, answered like a transform."""
        print(f"🔮 Invoking endpoint {self._endpoint_name} for {len(X)} samples...")
        features = X.drop(columns=[c for c in ("duration", "event") if c in X.columns])
        runtime = self._estimator.sagemaker_session.sagemaker_runtime_client
        response = runtime.invoke_endpoint(
            EndpointName=self._endpoint_name,
            ContentType="text/csv",
            Accept="application/json",
            Body=features.to_csv(index=False, header=True),
        )
        return self._collect(model, [json.loads(response["Body"].read())])

    def _collect(self, model, responses):
        """Mean times from the container's JSON responses, caching their curves."""
        all_predictions = []
        all_survival_curves = []
        time_grid = None
        for response_data in responses:
            all_predictions.append(np.array(response_data["predictions"]))
            if "survival_curves" in response_data:
                all_survival_curves.append(np.array(response_data["survival_curves"]))
            if time_grid is None and "time_grid" in response_data:
                time_grid = np.array(response_data["time_grid"])

        # Cache survival curves for get_survival_curves()
        self._cached_survival_curves = (
            np.concatenate(all_survival_curves, axis=0) if all_survival_curves else None
        )
        self._cached_time_grid = time_grid

        return self._mean_times(
            model.config["model_type"],
            np.concatenate(all_predictions),
            self._cached_survival_curves,
            time_grid,
        )

    @staticmethod
    def _mean_times(model_type, container_means, curves, time_grid):
        """Mean times computed here from the returned curves, from t = 0.

        Computing them from the curves keeps them consistent with ``predict()``
        whatever the algorithm version returns as its own mean. WTTE-RNN's mean
        is analytical, not integrated, so the returned value stands.
        """
        if model_type == "wtte_rnn" or curves is None or time_grid is None:
            return container_means
        from lifespan_predictor_sdk.calibration import mean_survival_times

        return mean_survival_times(
            pd.DataFrame(np.asarray(curves).T, index=np.asarray(time_grid))
        )

    def get_survival_curves(self, model, X):
        """Return cached survival curves from the last predict_mean call.

        Args:
            model: The SurvivalModel instance.
            X: Features (unused, kept for interface consistency).

        Returns:
            Tuple of (survival_curves, time_grid, None). The third element is a
            DataFrame only where the Antolini concordance is computed in-process;
            here the curves and their time grid are the whole answer.
        """
        return (self._cached_survival_curves, self._cached_time_grid, None)

    def _after_fit(self, model, val_df=None, test_df=None, test_labels_df=None):
        """Nothing to do after a training job.

        The training job calibrates during training when the config asks for it,
        so the calibrator is already inside the model artifact by the time this
        returns. Defined rather than omitted because ``SurvivalModel.fit`` calls
        it, and a public method that reaches a backend attribute this class lacks
        would fail with an AttributeError the caller can do nothing about.
        """
        return None

    def predict(self, model, X, time_points=None, **transform_kwargs):
        """Survival curves for new samples, via batch transform.

        The transform already returns curves beside the mean times, and
        ``predict_mean`` already parses and caches them. So this runs one job and
        reshapes what came back — it does not open a second inference path.

        Args:
            model: The SurvivalModel instance.
            X: Features DataFrame.
            time_points: Optional time points to return. These *select* rows of
                the container's own grid; values are never interpolated.
            **transform_kwargs: Passed through to the batch transform
                (``instance_type``, ``instance_count``, ``max_payload``, ...).

        Returns:
            pandas DataFrame with survival probabilities
            - Index: time points
            - Columns: one per unit (0, 1, 2, ...)
            - Values: S(t) = probability of survival beyond time t

            The shape every backend returns, so callers do not need to know
            which one answered.
        """
        self.predict_mean(model, X, **transform_kwargs)
        curves, time_grid, _ = self.get_survival_curves(model, X)

        if curves is None or time_grid is None:
            raise RuntimeError(
                "The batch transform returned no survival curves. The container "
                "supplies them only on the JSON accept path; a text/csv response "
                "carries mean times alone."
            )

        # Curves arrive as (n_samples, n_time_points); the returned shape is the
        # transpose of that.
        surv_df = pd.DataFrame(np.asarray(curves).T, index=np.asarray(time_grid))

        if time_points is not None:
            surv_df = surv_df.reindex(np.asarray(time_points), method="nearest")

        return surv_df

    def get_competing_risk_metrics(self, model, X, durations, events, mask_cindex=None):
        """Not available server-side.

        Returns:
            ``(None, None, None)`` — the cause-specific and Antolini diagnostics, and the
            headline predictive figure.
        """
        return (None, None, None)

    @staticmethod
    def _wait_for_model_package(
        client, model_package_arn, poll_seconds=10, timeout_seconds=1800
    ):
        """Block until a model package leaves InProgress.

        Creating a model that names a package still in progress fails, so this
        gate is not optional. Raises on Failed rather than letting the model
        creation report the confusing downstream error, and gives up rather than
        polling forever if the package never settles.
        """
        deadline = time.monotonic() + timeout_seconds
        while True:
            status = client.describe_model_package(ModelPackageName=model_package_arn)[
                "ModelPackageStatus"
            ]
            if status == "Completed":
                return
            if status == "Failed":
                raise RuntimeError(
                    f"Model package {model_package_arn} failed to create. "
                    "The algorithm subscription or the model artifacts may be "
                    "unusable for inference."
                )
            # Only ever wait on a status that can actually change. Polling
            # anything else means polling forever — a mocked client returning a
            # sentinel hung the suite for the full timeout before this guard
            # existed, and a status AWS adds later would do the same.
            if status not in _MODEL_PACKAGE_PENDING_STATUSES:
                raise RuntimeError(
                    f"Model package {model_package_arn} reported unrecognised "
                    f"status {status!r}."
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Model package {model_package_arn} still {status} after "
                    f"{timeout_seconds}s."
                )
            time.sleep(poll_seconds)

    def deploy(self, model, model_name=None, endpoint_instance_type=None):
        """
        Create SageMaker Model for batch transform inference, and optionally host
        it on a real-time endpoint.

        Without ``endpoint_instance_type`` nothing is hosted, and prediction
        runs as batch transform jobs. With it, an endpoint of the model's name
        is created on one instance of that type, and ``predict_mean()`` and
        ``predict()`` answer through it until ``delete()``. An endpoint is
        billed for as long as it exists.

        Args:
            model: The SurvivalModel instance.
            model_name: Optional name for the model (auto-generated if None)
            endpoint_instance_type: Optional instance type for a real-time
                endpoint, e.g. ``"ml.m5.large"``.

        Returns:
            Model name (str)

        Raises:
            ValueError: If model hasn't been trained yet
            ImportError: If sagemaker package not installed

        Example:
            >>> model.fit(train_df)
            >>> model_name = model.deploy()
            >>> # Model is now ready for batch transform
            >>> mean_times = model.predict_mean(test_df)
        """
        if not hasattr(self, "_estimator") or self._estimator is None:
            raise ValueError("Model must be trained first. Call fit() before deploy().")

        print("🚀 Creating SageMaker Model for batch transform...")

        # Validate and get role from environment variable
        role = _validate_sagemaker_role()
        print(f"🔑 IAM Role: {role}")

        algorithm_arn = self._estimator.algorithm_arn

        # Auto-generate model name from the algorithm name embedded in the
        # ARN (last segment, e.g. "lifespan-predictor-1-0-4"). That value is
        # already the sanitized version string, so it satisfies SageMaker's
        # naming rules and stays in lockstep with the algorithm being deployed.
        # Fine as a *name*; never as an algorithm identifier — see below.
        if model_name is None:
            model_name = algorithm_arn.split("/")[-1]

        # Identify the algorithm by its full ARN, never by the name split out of
        # it. A bare name resolves only inside the caller's account, so it works
        # in cloud tests — where seller and buyer are the same account — and
        # fails for every Marketplace buyer. The same shape as any other
        # cross-account identifier mistake: correct-looking, and only ever
        # exercised on the side of the boundary where it happens to work.
        algo_client = self._estimator.sagemaker_session.sagemaker_client
        algo_client.describe_algorithm(AlgorithmName=algorithm_arn)

        # Check if model already exists (idempotent operation)
        try:
            existing_model = algo_client.describe_model(ModelName=model_name)
            print(f"⚠️  Model {model_name} already exists, deleting and recreating...")
            self._delete_model_and_package(algo_client, existing_model, model_name)
            time.sleep(2)  # Brief wait for deletion to propagate
        except algo_client.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if (
                error_code != "ValidationException"
                or "Could not find model" not in str(e)
            ):
                raise  # Re-raise if it's not a "model not found" error
            # Model doesn't exist - proceed with creation

        # A model from a Marketplace algorithm is created *through* the algorithm,
        # not from an image URI lifted out of its inference specification. The
        # buyer's account has no access to the seller's registry; SageMaker
        # resolves the image itself, and that indirection is also what makes the
        # subscription enforceable. This mirrors the SageMaker Python SDK's own
        # path — AlgorithmEstimator.create_model() returns a
        # ModelPackage(algorithm_arn=...), which creates a model package and then
        # names it in the container definition.
        # An unversioned model package has to be named: SageMaker builds the
        # package ARN out of that name, and a request carrying neither
        # ModelPackageName nor its versioned counterpart ModelPackageGroupName
        # is rejected with "ValidationException: Missing ARN". The timestamped
        # name is the SDK's own convention (`name_from_base`, via
        # Session.create_model_package_from_algorithm) and it matters: a name
        # stable across deploys collides on the second one, and SageMaker
        # settles that collision by keeping the package already there — which
        # points at the previous training run's artifacts.
        from sagemaker.utils import name_from_base

        model_package = algo_client.create_model_package(
            ModelPackageName=name_from_base(algorithm_arn.split("/")[-1]),
            ModelPackageDescription=f"Created from training with {algorithm_arn}",
            SourceAlgorithmSpecification={
                "SourceAlgorithms": [
                    {
                        "AlgorithmName": algorithm_arn,
                        "ModelDataUrl": self._estimator.model_data,
                    }
                ]
            },
        )
        model_package_arn = model_package["ModelPackageArn"]

        # The package must reach Completed before a model can name it.
        self._wait_for_model_package(algo_client, model_package_arn)

        # Create the model directly using boto3 client to avoid timestamp appending
        # The sagemaker.Model class appends timestamps, but boto3 client uses exact name
        # Isolated always. SageMaker refuses a Marketplace product without it
        # ("EnableNetworkIsolation must be set to true for using a product from
        # AWS Marketplace") and never asks in the seller's own account, so a
        # branch on the ARN would only ever be tested on one side. The container
        # makes no network calls.
        algo_client.create_model(
            ModelName=model_name,
            PrimaryContainer={"ModelPackageName": model_package_arn},
            ExecutionRoleArn=role,
            EnableNetworkIsolation=True,
        )

        # Store model name for predict_mean()
        self._sagemaker_model_name = model_name
        self._sagemaker_model_package_arn = model_package_arn

        # Create a Model object for potential future use (but don't call create())
        from sagemaker.model import ModelPackage

        # `name` is carried over from the previous Model(...) construction: it binds
        # this object to the model created just above, so predict_mean()'s
        # transformer() does not mint a second one under a generated name.
        self._sagemaker_model = ModelPackage(
            role=role,
            model_package_arn=model_package_arn,
            name=model_name,
            sagemaker_session=self._estimator.sagemaker_session,
        )

        print(f"📦 Model name: {model_name}")
        print(f"✅ Model created successfully!")

        self._endpoint_name = None
        if endpoint_instance_type:
            self._create_endpoint(algo_client, model_name, endpoint_instance_type)
            print(f"   Ready for real-time inference on endpoint {model_name}")
        else:
            print("   Ready for batch transform inference")
        print("")
        print("Next steps:")
        print(f"  mean_times = model.predict_mean(test_df)")
        print("")

        return model_name

    def _create_endpoint(
        self, client, name, instance_type, poll_seconds=30, timeout_seconds=1800
    ):
        """Host the model on an endpoint of its name and wait until it serves."""
        print(f"🌐 Creating endpoint {name} on {instance_type}...")
        client.create_endpoint_config(
            EndpointConfigName=name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": name,
                    "InstanceType": instance_type,
                    "InitialInstanceCount": 1,
                }
            ],
        )
        client.create_endpoint(EndpointName=name, EndpointConfigName=name)
        # Recorded before the wait, so delete() removes it even if it never serves.
        self._endpoint_name = name

        deadline = time.monotonic() + timeout_seconds
        while True:
            description = client.describe_endpoint(EndpointName=name)
            status = description["EndpointStatus"]
            if status == "InService":
                return
            if status not in ("Creating", "Updating"):
                raise RuntimeError(
                    f"Endpoint {name} is {status}: "
                    f"{description.get('FailureReason', 'no reason given')}. "
                    "Call delete() to remove it."
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Endpoint {name} still {status} after {timeout_seconds}s."
                )
            time.sleep(poll_seconds)

    def delete(self, model):
        """
        Delete what deploy() created: its endpoint and endpoint config, if any,
        then the SageMaker model and its model package.

        Args:
            model: The SurvivalModel instance.
        """
        if not getattr(self, "_sagemaker_model_name", None):
            raise ValueError("No model name found. Training may have failed.")

        try:
            if getattr(self, "_estimator", None) is not None:
                client = self._estimator.sagemaker_session.sagemaker_client
            else:
                import boto3

                client = boto3.client("sagemaker")
            if getattr(self, "_endpoint_name", None):
                client.delete_endpoint(EndpointName=self._endpoint_name)
                client.delete_endpoint_config(EndpointConfigName=self._endpoint_name)
                self._endpoint_name = None
            description = client.describe_model(ModelName=self._sagemaker_model_name)
            self._delete_model_and_package(
                client, description, self._sagemaker_model_name
            )
        except Exception as e:
            raise ValueError(f"Failed to delete model: {e}")

    @staticmethod
    def _delete_model_and_package(client, description, model_name):
        """Delete a model, then its model package if it is in the model's account.

        deploy() creates that package; one in another account is someone else's.
        """
        client.delete_model(ModelName=model_name)
        package = description.get("PrimaryContainer", {}).get("ModelPackageName", "")
        account = description.get("ModelArn", "").split(":")[4:5]
        if package and account and package.split(":")[4:5] == account:
            client.delete_model_package(ModelPackageName=package)

    def print_training_job_summary(self, model):
        """Print a summary of the most recent SageMaker training job.

        Args:
            model: The SurvivalModel instance.

        Raises:
            ValueError: If no SageMaker training job has been run yet.
        """
        if not self._estimator:
            raise ValueError(
                "No SageMaker training job found. Call fit() with backend='sagemaker' first."
            )
        _log_training_job_info(self._estimator)
