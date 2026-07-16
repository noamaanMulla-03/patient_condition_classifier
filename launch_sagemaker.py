# %%
"""
Launch a SageMaker training job for the Patient Condition Classifier.

Reads secrets from .env and all tunable configuration from .config.yaml.

Requires SageMaker SDK v2 (pip install "sagemaker<3").

Prerequisites:
  1. Copy .env.example → .env and fill in your AWS credentials.
  2. Install: pip install "sagemaker<3" python-dotenv pyyaml
  3. Upload the two TSV files to your S3 bucket:
       aws s3 cp data/drugsComTrain_raw.tsv s3://<bucket>/data/
       aws s3 cp data/drugsComTest_raw.tsv  s3://<bucket>/data/
"""

import os

import sagemaker
import yaml
from dotenv import load_dotenv
from sagemaker.huggingface import HuggingFace
from sagemaker.image_uris import retrieve

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
# Resolve the container image URI explicitly (avoids hang)
# ------------------------------------------------------------------
# Pre-resolving the image URI outside the HuggingFace constructor
# skips a known internal hang during ECR queries.
image_uri = retrieve(
    "huggingface",
    region=REGION,
    version=cfg["transformers_version"],
    base_framework_version=f"pytorch{cfg['pytorch_version']}",
    py_version=cfg["py_version"],
    image_scope="training",
    instance_type=cfg["instance_type"],
)
print(f"Using image: {image_uri}")

# ------------------------------------------------------------------
# Stage only the needed files into a temp directory
# ------------------------------------------------------------------
# source_dir="." packages EVERYTHING (including .venv, data/, results/)
# which is hundreds of MB and hangs on upload.  Instead we create a
# minimal staging dir with only the files SageMaker needs.
import shutil
import tempfile

staging = tempfile.mkdtemp(prefix="sm-stage-")
print(f"Staging source in: {staging}")
shutil.copy("sagemaker_entry.py", staging)
shutil.copy("main.py", staging)
shutil.copytree("src", f"{staging}/src")
print("Staging complete.")

# ------------------------------------------------------------------
# Create the HuggingFace estimator
# ------------------------------------------------------------------
print("Creating estimator...")
estimator = HuggingFace(
    entry_point="sagemaker_entry.py",
    source_dir=staging,
    image_uri=image_uri,
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
print("Estimator created.")

# ------------------------------------------------------------------
# Launch training (non-blocking — don't wait for completion)
# ------------------------------------------------------------------
print("Submitting training job...")
estimator.fit(
    {"training": f"s3://{S3_BUCKET}/{cfg['s3_prefix']}/"},
    wait=False,
)

print(f"\nTraining job submitted: {estimator.latest_training_job.name}")
print("Monitor at:")
print(f"  https://{REGION}.console.aws.amazon.com/sagemaker/home#/jobs")
print(f"\nThe trained model will be at: s3://{session.default_bucket()}/")
