"""
Tokenization module for the Patient Condition Classifier.

Handles conversion of raw review text into model-ready token IDs
using the Hugging Face Transformers library. Includes overflow
handling for long reviews that exceed the model's maximum context
length.

The tokenizer is passed in from main.py to keep it as the single
source of configuration and avoid duplicate loading.
"""


def tokenize_and_split(data, tokenizer):
    """
    Tokenise a batch of reviews, truncating to max_model_length.

    Uses a 512-token limit — the maximum DeBERTa-v3-base supports.
    Reviews longer than this are truncated to the first 512 tokens
    (no overflow splitting). This keeps every review as a single
    sample and avoids noisy partial-text chunks being trained with
    the same label.

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
        plus all original non-review columns.
    """
    # ------------------------------------------------------------------
    # Run the tokeniser on the batch of reviews
    # ------------------------------------------------------------------
    # truncation=True clips to max_length (no overflow — one review,
    # one sample). 512 tokens covers ~95% of reviews in full.
    result = tokenizer(
        data["review"],
        truncation=True,
        padding=False,
        max_length=512,
    )

    # ------------------------------------------------------------------
    # Propagate original columns (condition, drugName, rating, etc.)
    # into the tokenized output. Since there's no overflow splitting,
    # it's a straight 1:1 copy — no need for overflow_to_sample_mapping.
    # ------------------------------------------------------------------
    for key, values in data.items():
        result[key] = values

    return result


def tokenize_data(dataset, tokenizer):
    """
    Tokenize reviews and encode condition labels as integers.

    This function is designed to be called after data_cleaner.clean_data()
    so the dataset has already been filtered, normalised, and split into
    train/validation/test. It performs two steps:
      1. Tokenizes review text → input_ids & attention_mask (with
         overflow handling for sequences > 128 tokens).
      2. Builds label mappings from the training split and converts
         condition strings to integer label IDs.

    Parameters
    ----------
    dataset : DatasetDict
        A Hugging Face DatasetDict with three splits ('train',
        'validation', 'test'), each containing at least a 'review'
        column and label columns to propagate.
    tokenizer : PreTrainedTokenizer
        The tokenizer to use for converting review text to token IDs.
        Passed from main.py to keep configuration centralised.

    Returns
    -------
    tuple[DatasetDict, dict[str, int], dict[int, str], int]
        - tokenized_dataset: DatasetDict with input_ids, attention_mask,
          token_type_ids, and labels columns added; rows may be expanded
          due to overflow splitting.
        - label2id: dict mapping each condition string to an integer ID.
        - id2label: dict mapping each integer ID back to a condition string.
        - num_labels: total number of unique conditions (class count).
    """
    # ------------------------------------------------------------------
    # Stage 1: Tokenize review text for every split
    # ------------------------------------------------------------------
    # Using batched=True processes rows in batches, which is more
    # memory-efficient and faster than one row at a time. The
    # tokenise_and_split function handles both the tokenization and
    # the overflow-aware duplication of non-review columns. The
    # tokenizer is passed via fn_kwargs so tokenize_and_split can
    # access it without being a module-level global.
    tokenized_dataset = dataset.map(
        tokenize_and_split,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
    )

    # ------------------------------------------------------------------
    # Stage 2: Build label mappings and convert conditions → integer IDs
    # ------------------------------------------------------------------
    # The model outputs integer logits, but our labels are strings
    # (e.g. "birth control", "depression"). We create bidirectional
    # mappings so we can:
    #   - Convert conditions to integer IDs for the loss function.
    #   - Convert predicted IDs back to human-readable strings.
    # Collect conditions from ALL splits, not just train. Otherwise
    # conditions unique to validation or test cause a KeyError.
    all_conditions = set()
    for split in tokenized_dataset.values():
        all_conditions.update(split["condition"])
    unique_conditions = sorted(all_conditions)
    num_labels = len(unique_conditions)
    label2id = {cond: i for i, cond in enumerate(unique_conditions)}
    id2label = {i: cond for cond, i in label2id.items()}
    print(f"\n  Unique conditions: {num_labels}")
    print(
        f"  Most common: {unique_conditions[0]}, {unique_conditions[1]}, {unique_conditions[2]} ..."
    )
    print(f"  Least common: {unique_conditions[-1]} ({num_labels} total)")

    # Map each condition string to its integer ID across ALL splits.
    # The Trainer expects a column named "labels" containing integer
    # class IDs for the cross-entropy loss function.
    def add_labels(example):
        example["labels"] = label2id[example["condition"]]
        return example

    tokenized_dataset = tokenized_dataset.map(add_labels)

    return tokenized_dataset, label2id, id2label, num_labels
