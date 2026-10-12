# Lifespan Predictor SDK

Survival analysis with deep learning via AWS SageMaker. This SDK provides a
Python interface to train and deploy DeepSurv survival models using
SageMaker's managed training infrastructure. Additional model types are
coming soon.

## Which product?

The algorithm is sold on AWS Marketplace as two products. **The same SDK works
with both**; only the algorithm ARN you pass it differs.

- **Free Tier Lifespan Predictor**: public, with no software charge. Training is
  capped at **100 units**, counted as distinct values of the `unit` column in
  your training data. Validation data, test data and inference are not capped.
  A training job over the cap fails before training starts, with a
  `FailureReason` naming the limit.
- **Lifespan Predictor**: no cap, hourly pricing with a free trial. It is
  offered privately: contact **info@kooditurva.fi**.

Subscribe to a product on AWS Marketplace, then train against its algorithm ARN
for your region. Every subscriber in a region uses the same ARN; your
subscription is what grants access to it.

| Product | Region | Algorithm ARN |
|---|---|---|
| Free Tier Lifespan Predictor | `eu-west-1` | `arn:aws:sagemaker:eu-west-1:985815980388:algorithm/free-tier-lifespan-predictor-m-227e5356082e3b6ea8f8e5e080c1ef12` |

To find the ARN in the AWS console instead: open **AWS Marketplace → Manage
subscriptions**, choose the product, then **Continue to configuration**, and
pick your region. The page shows the algorithm ARN.

You pay AWS for the SageMaker instances your jobs run on, with either product.

## Installation

```bash
# Install the wheel directly
pip install "https://github.com/kooditurva/lifespan-predictor-sdk/releases/download/v1.0.2/lifespan_predictor-1.0.2-py3-none-any.whl"

# Or add to an existing uv project
uv add "lifespan-predictor @ git+https://github.com/kooditurva/lifespan-predictor-sdk@v1.0.2"
```

See the [Releases page](https://github.com/kooditurva/lifespan-predictor-sdk/releases)
for other versions, and [CHANGELOG.md](./CHANGELOG.md) for what each one changed.

**Install a release.** Every tag has passed the SDK's tests and its sample
notebook against the free tier of the same algorithm version. One still marked
as a pre-release has passed them, but has not been released yet.

## Version Alignment

SDK version `X.Y.Z` is the algorithm version it was tested against, and works
with algorithm version `X.Y.*`. A mismatched major.minor produces a warning at
training time.

## Quick Start

Subscribe to a product (see [Which product?](#which-product)), set your
environment variables, then train a DeepSurv model on the NASA Turbofan Engine
dataset. The ARN below is the free tier's in `eu-west-1`; use the one for your
product and region.

```bash
export SAGEMAKER_ALGORITHM_ARN='arn:aws:sagemaker:eu-west-1:985815980388:algorithm/free-tier-lifespan-predictor-m-227e5356082e3b6ea8f8e5e080c1ef12'
export SAGEMAKER_ROLE_ARN='arn:aws:iam::ACCOUNT:role/service-role/YOUR-SageMaker-ExecutionRole'
# Optional: the S3 bucket for training data and model artifacts. Without it,
# SageMaker's default bucket for your account and region is used.
export SAGEMAKER_BUCKET='your-bucket'
```

> **Need a SageMaker role?** Run
> `scripts/setup-sagemaker-role.sh [ROLE_NAME]` to create one with the
> required permissions. If you bring your own, it must be able to read and
> write your bucket and call `sagemaker:DescribeAlgorithm` and
> `sagemaker:DescribeModelPackage`: SageMaker reads a subscribed algorithm as
> this role.

```python
import os
from lifespan_predictor_sdk import SurvivalModel, download_nasa_dataset, get_nasa_train_dataframe, get_nasa_test_dataframe

# Download NASA Turbofan Engine dataset (FD001 has 100 units, within the free tier's cap)
data_dir = "data/nasa"
download_nasa_dataset(data_dir)
train_df = get_nasa_train_dataframe(data_dir, datasets=[1])
test_df, test_labels_df = get_nasa_test_dataframe(data_dir, datasets=[1])

# Train a DeepSurv model on SageMaker
model = SurvivalModel({
    "model_type": "deepsurv",
    "algorithm_arn": os.environ["SAGEMAKER_ALGORITHM_ARN"],
    "instance_type": "ml.m5.large",
    "num_nodes": [24],
    "epochs": 30,
    "rolling_means": [5, 10, 20, 30],
    "max_rul": 125,
    "calibration": None,
})

model.fit(train_df, val_split=0.2, test_df=test_df, test_labels_df=test_labels_df)

# View results
model.get_evaluation_result().report()
```

### Predicting

`deploy()` creates a model for batch transform; each prediction runs a batch
transform job. Pass `endpoint_instance_type` to host it on a real-time endpoint
instead, which answers in seconds but is billed for as long as it exists.
`delete()` removes whatever `deploy()` created.

```python
model.deploy()                                     # batch transform
mean_times = model.predict_mean(test_df)           # expected time to event
curves = model.predict(test_df)                    # survival curves, one column per row
model.delete()

model.deploy(endpoint_instance_type="ml.m5.large") # real-time endpoint
curves = model.predict(test_df)
model.delete()                                     # stops the endpoint's billing
```

> An endpoint needs your credentials to allow `sagemaker:CreateEndpointConfig`,
> `CreateEndpoint`, `DescribeEndpoint`, `InvokeEndpoint`, `DeleteEndpoint` and
> `DeleteEndpointConfig`. A real-time request is limited to 6 MB; use batch
> transform for larger inputs.

## Examples

[`sagemaker_training_demo.ipynb`](examples/sagemaker_training_demo.ipynb) runs the
whole workflow on NASA's turbofan engine data: training, evaluation with
calibration curves, a SageMaker model, batch transform, scoring, and a real-time
endpoint, then cleanup. It fits the free tier, needs the environment variables
from the [Quick Start](#quick-start), and submits billed SageMaker jobs.

[See it with the output of the run that tested
v1.0.2](https://github.com/Kooditurva/lifespan-predictor-sdk/blob/v1.0.2/examples/sagemaker_training_demo.ipynb)
before running it yourself.

## Running the Tests

The tests that need no AWS account run on every push. To run them, and the cloud
tests against a real algorithm, from your own clone:

```bash
export SAGEMAKER_ROLE_ARN='arn:aws:iam::ACCOUNT:role/service-role/YOUR-SageMaker-ExecutionRole'
export SAGEMAKER_ALGORITHM_ARN='...'   # optional: your subscription's ARN
export SAGEMAKER_BUCKET='your-bucket'  # optional
python scripts/run_cloud_tests.py
```

It builds the wheel, installs it in a fresh virtual environment, runs the local
tests, and then the cloud tests against the installed wheel. The cloud tests
start SageMaker training and batch transform jobs in your account, which are
billed. With `--notebook` it then runs `examples/sagemaker_training_demo.ipynb` too,
as a release does. Without `SAGEMAKER_ALGORITHM_ARN` it uses the free tier's ARN in
`eu-west-1`. Arguments after `--` go to pytest, for example
`python scripts/run_cloud_tests.py -- -k inference`.

The `SDK Cloud Tests` workflow runs the same script. To run it in a fork, set the
`AWS_ROLE_ARN` secret to a role GitHub may assume through OIDC, the
`SAGEMAKER_ROLE_ARN` secret, and optionally the `SAGEMAKER_ALGORITHM_ARN` and
`SAGEMAKER_BUCKET` repository variables.

## Release Process

The `SDK Cloud Tests` workflow, run by hand on `main`, is the only thing that
tags. It runs the local tests, the cloud tests and the sample notebook against
the free tier, and when all pass it tags `vX.Y.Z` and publishes a pre-release
with the tested wheel. The tag's commit adds the notebook's recorded outputs to
the tested one. A maintainer then marks the pre-release as a release. A released
version never moves.

If the tests pass but publishing fails, use "Re-run failed jobs" on the run. It
reruns only the publish, with the wheel and the notebook the tests produced.

## License

MIT - see [LICENSE](LICENSE) for details.
