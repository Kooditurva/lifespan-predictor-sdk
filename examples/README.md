# Examples

## sagemaker_training_demo.ipynb

End-to-end demo of training a DeepSurv survival model on AWS SageMaker using the NASA Turbofan Engine dataset.

### Prerequisites

1. **Marketplace subscription** to Lifespan Predictor (note your Algorithm ARN)
2. **AWS credentials** configured (`aws configure` or `aws sso login`)
3. **Python dependencies** installed:
   ```bash
   pip install git+ssh://git@github.com/kooditurva/lifespan-predictor-sdk@vX.Y.Z
   pip install jupyter
   ```

### Set environment variables

```bash
export SAGEMAKER_ALGORITHM_ARN='arn:aws:sagemaker:eu-west-1:926554707418:algorithm/lifespan-predictor-1-0-1'
export SAGEMAKER_ROLE_ARN='arn:aws:iam::ACCOUNT:role/service-role/YOUR-SageMaker-ExecutionRole'
```

> **Need a SageMaker role?** Run `../scripts/setup-sagemaker-role.sh` to
> create one automatically.

### Run

```bash
jupyter notebook examples/sagemaker_training_demo.ipynb
```

Then execute cells in order. The training job takes 2-3 minutes and incurs SageMaker compute costs.

### What the notebook covers

- Loading the NASA Turbofan Engine dataset
- Configuring a DeepSurv model with hyperparameters
- Submitting a SageMaker training job
- Viewing evaluation metrics and calibration plots
- Inspecting training job details
