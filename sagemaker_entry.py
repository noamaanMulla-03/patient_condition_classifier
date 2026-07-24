"""
SageMaker entry point — bridges SageMaker environment variables to
the main.py pipeline via direct function call (no subprocess overhead).

SageMaker sets these env vars automatically:
  - SM_MODEL_DIR          → /opt/ml/model        (final model saved here)
  - SM_OUTPUT_DIR         → /opt/ml/output       (intermediate outputs)
  - SM_CHANNEL_TRAINING   → /opt/ml/input/data/training  (input TSV files)

Note: SM_INPUT_DIR points to the parent /opt/ml/input, NOT the
training channel directly. Use SM_CHANNEL_TRAINING for actual data.

See: https://sagemaker.readthedocs.io/en/stable/overview.html
"""

import os
import sys

# Ensure the current directory is on the path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import main as run_pipeline


def main():
    # ------------------------------------------------------------------
    # Read SageMaker-provided paths (with local fallbacks for testing)
    # ------------------------------------------------------------------
    data_dir = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    output_dir = os.environ.get("SM_OUTPUT_DIR", "/opt/ml/output")

    print(
        f"SageMaker entry point: data_dir={data_dir}, "
        f"model_dir={model_dir}, output_dir={output_dir}"
    )
    run_pipeline(
        data_dir=data_dir,
        model_dir=model_dir,
        output_dir=f"{output_dir}/drug-dataset",
    )


if __name__ == "__main__":
    main()
