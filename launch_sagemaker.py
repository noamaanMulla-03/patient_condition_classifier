# %%
"""
Launch a SageMaker training job for the Patient Condition Classifier.

Reads secrets from .env and all tunable configuration from .config.yaml.

Prerequisites:
  1. Copy .env.example → .env and fill in your AWS credentials.
  2. Install: pip install sagemaker python-dotenv pyyaml
  3. Upload the two TSV files to your S3 bucket:
       aws s3 cp data/drugsComTrain_raw.tsv s3://<bucket>/data/
       aws s3 cp data/drugsComTest_raw.tsv  s3://<bucket>/data/

Run this script directly (not as a notebook — the # %% markers are
for VS Code interactive mode if you prefer that).
"""

import os

import sagemaker
import yaml
from dotenv import load_dotenv
from sagemaker.huggingface import HuggingFace

# ------------------------------------------------------------------
# Load secrets from .env
# ------------------------------------------------------------------
load_dotenv()

S3_BUCKET = os.environ["SAGEMAKER_S3_BUCKET"]
ROLE = os.environ["SAGEMAKER_ROLE"]
REGION = os.environ.get("AWS_REGION", "us-east-1")

# ------------------------------------------------------------------
# Load tunable configuration from .config.yaml
# ------------------------------------------------------------------
with open(".config.yaml") as f:
    cfg = yaml.safe_load(f)

# ------------------------------------------------------------------
# SageMaker session
# ------------------------------------------------------------------
session = sagemaker.Session()
print(f"SageMaker session in region: {session.boto_region_name}")

# ------------------------------------------------------------------
# Create the estimator
# ------------------------------------------------------------------
# This packages main.py, src/*, sagemaker_entry.py, and all project
# files into a tar.gz and uploads it to S3.
estimator = HuggingFace(
    entry_point="sagemaker_entry.py",
    source_dir=".",
    instance_type=cfg["instance_type"],
    instance_count=cfg["instance_count"],
    role=ROLE,
    py_version=cfg["py_version"],
    pytorch_version=cfg["pytorch_version"],
    transformers_version=cfg["transformers_version"],
    hyperparameters=cfg.get("hyperparameters", {}),
    environment=cfg.get("environment", {}),
    volume_size=cfg["volume_size"],
    keep_alive_period_in_seconds=0,
)

# ------------------------------------------------------------------
# Launch training
# ------------------------------------------------------------------
# SageMaker copies the S3 data to /opt/ml/input/data/training on the
# instance. sagemaker_entry.py reads it from there.
estimator.fit(
    {
        "training": f"s3://{S3_BUCKET}/{cfg['s3_prefix']}/",
    }
)

print("\nTraining job submitted! Monitor at:")
print(f"  https://{REGION}.console.aws.amazon.com/sagemaker/home#/jobs")
print(f"\nThe trained model will be at: s3://{session.default_bucket()}/")
