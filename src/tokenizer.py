"""
Tokenization module for the Patient Condition Classifier.

Handles conversion of raw review text into model-ready token IDs
using the Hugging Face Transformers library. Includes overflow
handling for long reviews that exceed the model's maximum context
length.
"""

from transformers import AutoTokenizer

# ------------------------------------------------------------------
# Model checkpoint selection
# ------------------------------------------------------------------
# We use Microsoft's DeBERTa-v3-base, a modern Transformer model that
# uses disentangled attention (separate content and position vectors)
# and is pre-trained on a large English corpus. It has a max sequence
# length of 512 tokens and uses a SentencePiece tokeniser.
checkpoint = "microsoft/deberta-v3-base"

# ------------------------------------------------------------------
# Load the pre-trained tokeniser
# ------------------------------------------------------------------
# AutoTokenizer automatically selects the correct tokeniser class
# from the Hugging Face Hub based on the checkpoint name. The
# tokeniser splits text into sub-word tokens (BPE / SentencePiece)
# and maps them to integer IDs for model consumption.
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_and_split(data):
    """
    Tokenise a batch of reviews and handle sequences that overflow.

    The DeBERTa model has a maximum input length of 128 tokens (as
    configured below). Reviews that exceed this limit are split into
    multiple non-overlapping chunks, each of length <= max_length.
    This function duplicates the corresponding non-review columns
    (e.g. condition, rating) so that each chunk becomes its own row
    in the dataset — preserving alignment between inputs and labels.

    Parameters
    ----------
    data : dict of list
        A batch of rows from the dataset, where each key is a column
        name and each value is a list of entries (one per row).
        Expected keys include at least 'review' plus any label/
        metadata columns to propagate.

    Returns
    -------
    dict of list
        A dictionary with tokenizer outputs (input_ids, attention_mask)
        plus all original non-review columns, now expanded to match
        the number of overflow chunks.
    """
    # ------------------------------------------------------------------
    # Step 1: Run the tokeniser on the batch of reviews
    # ------------------------------------------------------------------
    # When return_overflowing_tokens=True, the tokeniser splits long
    # sequences into chunks of max_length tokens. It also returns
    # an 'overflow_to_sample_mapping' array that records which
    # original row each chunk came from.
    #
    # truncation=True ensures we don't silently keep tokens beyond
    # max_length — they are split off into overflow chunks instead.
    result = tokenizer(
        data["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )

    # ------------------------------------------------------------------
    # Step 2: Extract the overflow mapping
    # ------------------------------------------------------------------
    # 'overflow_to_sample_mapping' is a list where index i holds the
    # index of the original row that produced overflow chunk i.
    # Example: if row 0 splits into 3 chunks, the mapping starts
    # [0, 0, 0, 1, 1, ...].
    #
    # We pop it from 'result' because we only needed it temporarily
    # — it is not a model input and should not be passed to training.
    sample_map = result.pop("overflow_to_sample_mapping")

    # ------------------------------------------------------------------
    # Step 3: Duplicate non-review columns to match overflow chunks
    # ------------------------------------------------------------------
    # For each key (e.g. 'condition', 'rating', 'drugName'), we use
    # the sample_map to look up which original row value each overflow
    # chunk should inherit. This keeps every chunk aligned with its
    # original label, even after splitting long reviews.
    for key, values in data.items():
        result[key] = [values[i] for i in sample_map]

    return result


def tokenize_data(dataset):
    """
    Apply tokenization with overflow handling to the entire dataset.

    This function is designed to be called after data_cleaner.clean_data()
    so the dataset has already been filtered, normalised, and split into
    train/validation/test. It adds 'input_ids' and 'attention_mask'
    columns to each split, ready for model training.

    Parameters
    ----------
    dataset : DatasetDict
        A Hugging Face DatasetDict with three splits ('train',
        'validation', 'test'), each containing at least a 'review'
        column and label columns to propagate.

    Returns
    -------
    DatasetDict
        The same structure but with tokenized fields added and rows
        potentially expanded due to overflow splitting.
    """
    # ------------------------------------------------------------------
    # Apply the tokenizer to every split in the dataset
    # ------------------------------------------------------------------
    # Using batched=True processes rows in batches, which is more
    # memory-efficient and faster than one row at a time. The
    # tokenise_and_split function handles both the tokenization and
    # the overflow-aware duplication of non-review columns.
    tokenized_dataset = dataset.map(
        tokenize_and_split,
        batched=True,
    )

    return tokenized_dataset
