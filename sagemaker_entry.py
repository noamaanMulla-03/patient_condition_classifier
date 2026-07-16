"""
SageMaker entry point — bridges SageMaker environment variables to
the main.py pipeline via command-line arguments.

SageMaker sets these env vars automatically:
  - SM_MODEL_DIR  → /opt/ml/model   (final model saved here)
  - SM_OUTPUT_DIR → /opt/ml/output  (intermediate outputs)
  - SM_INPUT_DIR  → /opt/ml/input/data/training  (input TSV files)

See: https://sagemaker.readthedocs.io/en/stable/overview.html
"""

import os
import subprocess
import sys


def main():
    # ------------------------------------------------------------------
    # Read SageMaker-provided paths (with local fallbacks for testing)
    # ------------------------------------------------------------------
    data_dir = os.environ.get("SM_INPUT_DIR", "/opt/ml/input/data/training")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    output_dir = os.environ.get("SM_OUTPUT_DIR", "/opt/ml/output")

    # ------------------------------------------------------------------
    # Build the command to run main.py
    # ------------------------------------------------------------------
    cmd = [
        sys.executable,  # uses the same Python interpreter
        "main.py",
        "--data_dir",
        data_dir,
        "--model_dir",
        model_dir,
        "--output_dir",
        f"{output_dir}/drug-dataset",
    ]

    print(f"SageMaker entry point running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
