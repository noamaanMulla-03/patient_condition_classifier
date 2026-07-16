"""
Entry point for the Patient Condition Classifier pipeline.

This script orchestrates the full workflow:
  1. Loads raw TSV data from the UCI Drug Reviews dataset via
     src.data_loader.load_data().
  2. Cleans and preprocesses the dataset via src.data_cleaner.clean_data().
  3. Loads the DeBERTa-v3 tokenizer (once, shared across steps).
  4. Tokenizes the reviews via src.tokenizer.tokenize_data().
  5. Saves the tokenized dataset to disk for reuse.
  6. Fine-tunes a DeBERTa-v3-base classifier via src.fine_tune.fine_tune().
  7. Displays summary information at each stage.

CLI arguments (for SageMaker or custom paths):
  --data_dir   : directory with raw TSV files (default: ./data)
  --model_dir  : output directory for checkpoints & final model
                 (default: ./results)
  --output_dir : directory for intermediate outputs like the
                 tokenized dataset (default: ./drug-dataset)
"""

import argparse

from transformers import AutoTokenizer, DataCollatorWithPadding
from src.data_loader import load_data
from src.data_cleaner import clean_data
from src.tokenizer import tokenize_data
from src.fine_tune import fine_tune


def main(
    data_dir: str = "data",
    model_dir: str = "./results",
    output_dir: str = "drug-dataset",
) -> None:
    """
    Run the full pipeline — load, clean, tokenize, save, and fine-tune.

    Parameters
    ----------
    data_dir : str
        Directory containing the raw TSV files.
    model_dir : str
        Directory for training checkpoints and the final saved model.
    output_dir : str
        Directory for saving the tokenized dataset (Arrow files).

    Returns
    -------
    None — results are printed to stdout; model saved to disk.
    """
    # ------------------------------------------------------------------
    # Step 1: Load the raw dataset
    # ------------------------------------------------------------------
    dataset = load_data(data_dir)

    # Print a quick summary of the raw dataset sizes.
    print(f"Loaded train and test datasets ({dataset['train']}) ({dataset['test']})")

    # ------------------------------------------------------------------
    # Step 2: Clean and preprocess the dataset
    # ------------------------------------------------------------------
    cleaned_dataset = clean_data(dataset)
    print(f"Cleaned dataset: {cleaned_dataset}")

    # ------------------------------------------------------------------
    # Step 3: Load the tokeniser and data collator (once)
    # ------------------------------------------------------------------
    print("\nLoading tokenizer...")
    checkpoint = "microsoft/deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    print("Tokenizer loaded.")

    # ------------------------------------------------------------------
    # Step 4: Tokenize the review text
    # ------------------------------------------------------------------
    tokenized_dataset, label2id, id2label, num_labels = tokenize_data(
        cleaned_dataset, tokenizer
    )
    print(f"Tokenized dataset: {tokenized_dataset}")

    # ------------------------------------------------------------------
    # Step 5: Save the tokenized dataset to disk
    # ------------------------------------------------------------------
    print("Saving tokenized dataset to disk...")
    tokenized_dataset.save_to_disk(output_dir)
    print(f"Tokenized dataset saved to '{output_dir}/'.")

    # ------------------------------------------------------------------
    # Step 6: Fine-tune the classifier
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Starting fine-tuning...")
    print("=" * 60)
    fine_tune(
        tokenized_dataset,
        tokenizer,
        data_collator,
        checkpoint,
        label2id,
        id2label,
        num_labels,
        model_dir=model_dir,
    )
    print("\nPipeline complete.")


# ------------------------------------------------------------------
# Script entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Patient Condition Classifier — full pipeline"
    )
    parser.add_argument(
        "--data_dir",
        default="data",
        help="Directory with raw TSV files (default: data)",
    )
    parser.add_argument(
        "--model_dir",
        default="./results",
        help="Output dir for checkpoints & final model (default: ./results)",
    )
    parser.add_argument(
        "--output_dir",
        default="drug-dataset",
        help="Dir for tokenized Arrow dataset (default: drug-dataset)",
    )
    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
    )
