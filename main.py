"""
Entry point for the Patient Condition Classifier pipeline.

This script orchestrates the full data preprocessing workflow:
  1. Loads raw TSV data from the UCI Drug Reviews dataset via
     src.data_loader.load_data().
  2. Cleans and preprocesses the dataset via src.data_cleaner.clean_data().
  3. Tokenizes the reviews via src.tokenizer.tokenize_data().
  4. Displays summary information about the final tokenized dataset.
"""

from src.data_loader import load_data
from src.data_cleaner import clean_data
from src.tokenizer import tokenize_data


def main() -> None:
    """
    Run the full data pipeline — load, clean, tokenize, and report.

    Steps
    -----
    1. Load: Read the raw 'train' and 'test' TSV files from disk
       using Hugging Face's datasets library.
    2. Clean: Apply filtering, normalisation, HTML unescaping, feature
       engineering, and train/validation/test splitting.
    3. Tokenize: Convert review text to token IDs with overflow
       handling for sequences exceeding 128 tokens.
    4. Report: Print a summary of the final tokenized dataset splits
       so the user can verify the pipeline ran correctly.

    Returns
    -------
    None — results are printed to stdout.
    """
    # ------------------------------------------------------------------
    # Step 1: Load the raw dataset
    # ------------------------------------------------------------------
    # load_data() reads the two TSV files (drugsComTrain_raw.tsv and
    # drugsComTest_raw.tsv) from the data/ directory and returns a
    # DatasetDict with 'train' and 'test' keys.
    dataset = load_data()

    # Print a quick summary of the raw dataset sizes.
    print(f"Loaded train and test datasets ({dataset['train']}) ({dataset['test']})")

    # ------------------------------------------------------------------
    # Step 2: Clean and preprocess the dataset
    # ------------------------------------------------------------------
    # clean_data() applies all cleaning steps (see its docstring for
    # details) and returns a DatasetDict with three splits:
    #   - train:      80% of the original training rows
    #   - validation: 20% of the original training rows
    #   - test:       the original test split (preserved as-is)
    cleaned_dataset = clean_data(dataset)
    print(f"Cleaned dataset: {cleaned_dataset}")

    # ------------------------------------------------------------------
    # Step 3: Tokenize the review text
    # ------------------------------------------------------------------
    # tokenize_data() uses the DeBERTa-v3 tokenizer to convert each
    # review into input_ids and attention_mask. Reviews longer than
    # 128 tokens are split into multiple chunks, with non-review
    # columns duplicated to maintain alignment.
    tokenized_dataset = tokenize_data(cleaned_dataset)
    print(f"Tokenized dataset: {tokenized_dataset}")


# ------------------------------------------------------------------
# Script entry point
# ------------------------------------------------------------------
# When run directly (python main.py), this block executes main().
# It also allows the module to be imported without running the
# pipeline, keeping the interface clean for future use.
if __name__ == "__main__":
    main()
