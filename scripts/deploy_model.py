"""
Register a propensity model artifact in AWS SageMaker Model Registry.

Usage:
    python scripts/deploy_model.py --experiment-id exp_A

Steps:
1. Upload model artifact (.pkl) to S3
2. Create model package group if it doesn't exist
3. Create a model package with metadata
4. Approve the model package for production use
"""
import argparse
import json
import os
import pickle
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

MODELS_PATH = Path(os.getenv("MODELS_PATH", "models"))
S3_BUCKET = os.getenv("S3_BUCKET", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
SAGEMAKER_ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN", "")
MODEL_GROUP_NAME = os.getenv("SAGEMAKER_MODEL_GROUP", "ab-testing-propensity-models")


def register_model(
    model_path: Path,
    experiment_id: str,
    cv_auc: float,
    feature_names: list[str],
) -> str:
    """
    Upload model to S3 and register in SageMaker Model Registry.

    Returns the ModelPackageArn of the registered model.
    Raises if AWS credentials or config are missing.
    """
    import boto3
    import tarfile
    import tempfile

    if not S3_BUCKET:
        raise EnvironmentError("S3_BUCKET env var not set")
    if not SAGEMAKER_ROLE_ARN:
        raise EnvironmentError("SAGEMAKER_ROLE_ARN env var not set")

    sm_client = boto3.client("sagemaker", region_name=AWS_REGION)
    s3_client = boto3.client("s3", region_name=AWS_REGION)

    # --- 1. Package model as tar.gz for SageMaker ---
    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = Path(tmpdir) / "model.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(model_path, arcname=model_path.name)

        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        s3_key = f"propensity-models/{experiment_id}/{timestamp}/model.tar.gz"
        s3_client.upload_file(str(tar_path), S3_BUCKET, s3_key)
        model_uri = f"s3://{S3_BUCKET}/{s3_key}"

    print(f"Model uploaded to {model_uri}")

    # --- 2. Ensure model package group exists ---
    try:
        sm_client.create_model_package_group(
            ModelPackageGroupName=MODEL_GROUP_NAME,
            ModelPackageGroupDescription="Propensity models for A/B testing IPW adjustment",
        )
        print(f"Created model package group: {MODEL_GROUP_NAME}")
    except sm_client.exceptions.ClientError as e:
        if "already exists" in str(e) or "ConflictException" in str(e.__class__.__name__):
            print(f"Model package group '{MODEL_GROUP_NAME}' already exists")
        else:
            raise

    # --- 3. Create model package ---
    metadata = {
        "experiment_id": experiment_id,
        "cv_auc": cv_auc,
        "feature_names": feature_names,
        "training_date": datetime.utcnow().isoformat(),
    }

    response = sm_client.create_model_package(
        ModelPackageGroupName=MODEL_GROUP_NAME,
        ModelPackageDescription=f"Propensity model for {experiment_id} | CV AUC={cv_auc:.3f}",
        InferenceSpecification={
            "Containers": [
                {
                    "Image": f"683313688378.dkr.ecr.{AWS_REGION}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
                    "ModelDataUrl": model_uri,
                    "Environment": {"MODEL_METADATA": json.dumps(metadata)},
                }
            ],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
        },
        ModelApprovalStatus="Approved",
        CustomerMetadataProperties={k: str(v) for k, v in metadata.items()},
    )

    arn = response["ModelPackageArn"]
    print(f"Model package registered: {arn}")
    return arn


def main():
    parser = argparse.ArgumentParser(description="Register propensity model in SageMaker")
    parser.add_argument("--experiment-id", required=True, help="Experiment identifier")
    args = parser.parse_args()

    model_path = MODELS_PATH / f"propensity_{args.experiment_id}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run training first."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    arn = register_model(
        model_path=model_path,
        experiment_id=args.experiment_id,
        cv_auc=model._cv_auc or 0.0,
        feature_names=model._feature_names or [],
    )
    print(f"Done. ARN: {arn}")


if __name__ == "__main__":
    main()
