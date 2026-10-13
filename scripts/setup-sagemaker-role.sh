#!/usr/bin/env bash
set -euo pipefail

ROLE_NAME="${1:-LifespanPredictorSageMakerRole}"

if aws iam get-role --role-name "$ROLE_NAME" &>/dev/null; then
  ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text)
  echo "Role '$ROLE_NAME' already exists."
else
  echo "Creating SageMaker execution role: $ROLE_NAME"

  aws iam create-role \
    --role-name "$ROLE_NAME" \
    --assume-role-policy-document '{
      "Version": "2012-10-17",
      "Statement": [
        {
          "Effect": "Allow",
          "Principal": { "Service": "sagemaker.amazonaws.com" },
          "Action": "sts:AssumeRole"
        }
      ]
    }' \
    --query 'Role.Arn' \
    --output text

  aws iam attach-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

  ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text)
  echo "Done."
fi

echo ""
echo "Add this to your environment:"
echo ""
echo "  export SAGEMAKER_ROLE_ARN='$ROLE_ARN'"
