# Lifespan Predictor SDK

Survival analysis with deep learning via AWS SageMaker. This SDK provides a
Python interface to train and deploy DeepSurv survival models using
SageMaker's managed training infrastructure. Additional model types are
coming soon.

## Installation

```bash
# Install the wheel directly
pip install "https://github.com/kooditurva/lifespan-predictor-sdk/releases/download/vX.Y.Z/lifespan_predictor-X.Y.Z-py3-none-any.whl"

# Or add to an existing uv project
uv add "lifespan-predictor @ git+ssh://git@github.com/kooditurva/lifespan-predictor-sdk@vX.Y.Z"
```

Replace `X.Y.Z` with the desired version. See the
[Releases page](https://github.com/kooditurva/lifespan-predictor-sdk/releases)
for available versions, and [CHANGELOG.md](./CHANGELOG.md) for what each one
changed.

**Install a plain `vX.Y.Z` tag.** A tag ending in `-rc` — `v1.49.2-rc1` — is a
release candidate: it is built and installable, but it has not yet been run
against a live algorithm. Only a plain tag has, and each one is marked as a
full release rather than a pre-release.

## Version Alignment

SDK version `X.Y.*` is guaranteed to work with algorithm version `X.Y.*`.
Using a mismatched major.minor version will produce a warning at training time.

## Quick Start

Set your environment variables, then train a DeepSurv model on the NASA
Turbofan Engine dataset:

```bash
export SAGEMAKER_ALGORITHM_ARN='arn:aws:sagemaker:eu-west-1:926554707418:algorithm/lifespan-predictor-1-0-1'
export SAGEMAKER_ROLE_ARN='arn:aws:iam::ACCOUNT:role/service-role/YOUR-SageMaker-ExecutionRole'
```

> **Need a SageMaker role?** Run
> `scripts/setup-sagemaker-role.sh [ROLE_NAME]` to create one with the
> required permissions.

```python
import os
from lifespan_predictor_sdk import SurvivalModel, download_nasa_dataset, get_nasa_train_dataframe, get_nasa_test_dataframe

# Download NASA Turbofan Engine dataset
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

## Examples

The [`examples/`](examples/) directory contains a Jupyter notebook with a more
complete walkthrough including calibration curves, training plots, and job
details. See [`examples/README.md`](examples/README.md) for setup instructions.

## Release Process

Every merge to `main` is tested against SageMaker algorithm versions via cloud
tests. SDK releases are only tagged and published after all tests pass. This
guarantees that any installable version has been validated against the
corresponding algorithm release.

## License

MIT - see [LICENSE](LICENSE) for details.
