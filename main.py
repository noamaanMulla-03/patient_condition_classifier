"""
Entry point for the Patient Condition Classifier pipeline.

This script orchestrates the data loading and cleaning workflow:
  1. Loads raw TSV data from the UCI Drug Reviews dataset via
     src.data_loader.load_data().
  2. Cleans and preprocesses the dataset via src.data_cleaner.clean_data().
  3. Displays summary information about the resulting splits.
"""

from src.data_loader import load_data
from src.data_cleaner import clean_data


def main() -> None:
    """
    Run the full data pipeline — load, clean, and report results.

    Steps
    -----
    1. Load: Read the raw 'train' and 'test' TSV files from disk
       using Hugging Face's datasets library.
    2. Clean: Apply filtering, normalisation, HTML unescaping, feature
       engineering, and train/validation/test splitting.
    3. Report: Print a summary of the final dataset splits so the user
       can verify the pipeline ran correctly.

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

    # ------------------------------------------------------------------
    # Step 3: Report the final result
    # ------------------------------------------------------------------
    # Print the cleaned DatasetDict summary, showing the number of rows
    # and features in each split. This serves as a sanity check before
    # moving on to model training.
    print(f"Cleaned dataset: {cleaned_dataset}")


# ------------------------------------------------------------------
# Script entry point
# ------------------------------------------------------------------
# When run directly (python main.py), this block executes main().
# It also allows the module to be imported without running the
# pipeline, keeping the interface clean for future use.
if __name__ == "__main__":
    main()
