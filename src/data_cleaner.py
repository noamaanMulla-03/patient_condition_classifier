import re
from datasets import DatasetDict
import html


def clean_data(dataset: DatasetDict) -> DatasetDict:
    """
    Clean and preprocess the drug review dataset.

    This function performs the following cleaning steps:
    1. Renames the unnamed index column to 'patient_id' for clarity.
    2. Filters out rows where 'drugName' or 'condition' are null.
    3. Converts 'drugName' and 'condition' to lowercase for consistency.
    4. Unescapes HTML entities (e.g., &amp;, &lt;) in the review text.
    5. Strips HTML tags from condition labels.
    6. Computes a 'reviewLength' field (word count of the review).
    7. Removes very short reviews (≤30 words) as they lack useful signal.
    8. Splits the training data into train (80%) and validation (20%),
       while preserving the original test split for final evaluation.

    Args:
        dataset: A DatasetDict with 'train' and 'test' splits, each
                 containing columns like 'Unnamed: 0', 'drugName',
                 'condition', 'review', 'rating', 'date', 'usefulCount'.

    Returns:
        A DatasetDict with three splits:
            - 'train':      80% of the original training data.
            - 'validation': 20% of the original training data.
            - 'test':       The original test split (unchanged).
    """

    # ------------------------------------------------------------------
    # Step 1: Rename the anonymous index column
    # ------------------------------------------------------------------
    # The TSV files were saved from a pandas DataFrame, which included
    # the default integer index as a column named "Unnamed: 0". We give
    # it a meaningful name for downstream use.
    dataset = dataset.rename_column(
        original_column_name="Unnamed: 0", new_column_name="patient_id"
    )

    # ------------------------------------------------------------------
    # Step 2: Remove rows with missing key fields
    # ------------------------------------------------------------------
    # Reviews where the drugName or condition are null are not useful
    # for classification, so we drop them.
    dataset = dataset.filter(lambda x: x["drugName"] is not None)
    dataset = dataset.filter(lambda x: x["condition"] is not None)

    # ------------------------------------------------------------------
    # Step 3: Normalise drug name and condition to lowercase
    # ------------------------------------------------------------------
    # Using batched=True means the lambda receives a dict of lists
    # (one list per column), so we use a list comprehension to process
    # each element in the batch.
    dataset = dataset.map(
        lambda x: {**x, "drugName": [n.lower() for n in x["drugName"]]},
        batched=True,
    )
    dataset = dataset.map(
        lambda x: {**x, "condition": [c.lower() for c in x["condition"]]},
        batched=True,
    )

    # ------------------------------------------------------------------
    # Step 4: Unescape HTML entities in the review text
    # ------------------------------------------------------------------
    # Some reviews contain HTML-encoded characters (e.g. &amp; for &,
    # &lt; for <). The html.unescape() function converts these back to
    # their original characters for clean text.
    dataset = dataset.map(
        lambda x: {**x, "review": [html.unescape(r) for r in x["review"]]},
        batched=True,
    )

    # ------------------------------------------------------------------
    # Step 5: Strip HTML tags from condition labels
    # ------------------------------------------------------------------
    # Some rows have HTML markup in the condition field (e.g.
    # "0</span> users found this comment helpful."), likely from
    # scraping artifacts. We strip any HTML tags so the labels are
    # clean text. Rows are kept — the review text is still useful even
    # if the condition label has residual noise.
    dataset = dataset.map(
        lambda x: {
            **x,
            "condition": [re.sub(r"<[^>]+>", "", c) for c in x["condition"]],
        },
        batched=True,
    )

    # Filter out rows whose condition is still obvious scraper junk
    # after the HTML stripping — e.g. "0 users found this comment
    # helpful." These have no learnable relationship to review text
    # and cause NaN gradients during training.
    print("  Filtering junk labels (non-medical conditions)...")
    for split in list(dataset.keys()):
        before = len(dataset[split])
        dataset[split] = dataset[split].filter(
            lambda x: "user" not in x["condition"]
            and "found" not in x["condition"]
            and len(x["condition"]) <= 100
            and any(c.isalpha() for c in x["condition"])
        )
        after = len(dataset[split])
        if before > after:
            print(
                f"    {split}: removed {before - after} junk label rows"
                f" ({before} → {after})"
            )

    # ------------------------------------------------------------------
    # Step 6: Compute review length (word count)
    # ------------------------------------------------------------------
    # reviewLength stores the number of words per review, which can be
    # used as a feature or filtering criterion.
    dataset = dataset.map(
        lambda x: {**x, "reviewLength": [len(r.split()) for r in x["review"]]},
        batched=True,
    )

    # ------------------------------------------------------------------
    # Step 7: Remove very short reviews
    # ------------------------------------------------------------------
    # Reviews with 30 or fewer words are unlikely to contain meaningful
    # patient condition information, so we filter them out.
    dataset = dataset.filter(lambda x: x["reviewLength"] > 30)

    # ------------------------------------------------------------------
    # Step 8: Create a validation split from the training data
    # ------------------------------------------------------------------
    # The original dataset has only 'train' and 'test' splits. We split
    # the training data further: 80% for training, 20% for validation.
    # The original test split is preserved for final evaluation.
    original_test = dataset["test"]
    splits = dataset["train"].train_test_split(train_size=0.8, seed=42)
    dataset["train"] = splits["train"]
    dataset["validation"] = splits["test"]
    dataset["test"] = original_test

    return dataset
