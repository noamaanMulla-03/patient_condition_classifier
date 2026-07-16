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
"""

from transformers import AutoTokenizer, DataCollatorWithPadding
from src.data_loader import load_data
from src.data_cleaner import clean_data
from src.tokenizer import tokenize_data
from src.fine_tune import fine_tune


def main() -> None:
    """
    Run the full pipeline — load, clean, tokenize, save, and fine-tune.

    Steps
    -----
    1. Load: Read the raw 'train' and 'test' TSV files from disk
       using Hugging Face's datasets library.
    2. Clean: Apply filtering, normalisation, HTML unescaping, feature
       engineering, and train/validation/test splitting.
    3. Load tokenizer: Initialise the DeBERTa-v3 tokeniser once and
       share it between tokenization and fine-tuning steps.
    4. Tokenize: Convert review text to token IDs with overflow
       handling for sequences exceeding 128 tokens.
    5. Save: Persist the tokenized dataset to disk so it can be
       loaded later without re-running preprocessing.
    6. Fine-tune: Train a DeBERTa-v3-base classifier on the tokenized
       dataset and save the best model to ./results/final-model/.
    7. Report: Print summaries at each stage for verification.

    Returns
    -------
    None — results are printed to stdout; model saved to disk.
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
    # Step 3: Load the tokeniser and data collator (once)
    # ------------------------------------------------------------------
    # The tokeniser is loaded here in main.py and passed down to both
    # tokenize_data() and fine_tune(). This ensures the model is
    # downloaded only once and configuration is centralised.
    print("\nLoading tokenizer...")
    checkpoint = "microsoft/deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    print("Tokenizer loaded.")

    # ------------------------------------------------------------------
    # Step 4: Tokenize the review text
    # ------------------------------------------------------------------
    # tokenize_data() uses the DeBERTa-v3 tokenizer to convert each
    # review into input_ids and attention_mask. Reviews longer than
    # 128 tokens are split into multiple chunks, with non-review
    # columns duplicated to maintain alignment.
    tokenized_dataset, label2id, id2label, num_labels = tokenize_data(
        cleaned_dataset, tokenizer
    )
    print(f"Tokenized dataset: {tokenized_dataset}")

    # ------------------------------------------------------------------
    # Step 5: Save the tokenized dataset to disk
    # ------------------------------------------------------------------
    # save_to_disk() persists the DatasetDict as a directory of Arrow
    # files on disk. This avoids re-running the entire preprocessing
    # pipeline on subsequent runs — just use load_from_disk() instead.
    #
    # The dataset is saved to "drug-dataset/" in the project root.
    print("Saving tokenized dataset to disk...")
    tokenized_dataset.save_to_disk("drug-dataset")
    print("Tokenized dataset saved to 'drug-dataset/'.")

    # ------------------------------------------------------------------
    # Step 6: Fine-tune the classifier
    # ------------------------------------------------------------------
    # fine_tune() takes the tokenized dataset, label mappings,
    # tokenizer, data collator, and checkpoint, then trains a
    # DeBERTa-v3-base sequence classifier for 3 epochs and saves the
    # best model to ./results/final-model/.
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
    )
    print("\nPipeline complete.")


# ------------------------------------------------------------------
# Script entry point
# ------------------------------------------------------------------
# When run directly (python main.py), this block executes main().
# It also allows the module to be imported without running the
# pipeline, keeping the interface clean for future use.
if __name__ == "__main__":
    main()
