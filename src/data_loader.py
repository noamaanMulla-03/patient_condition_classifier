"""
Data loading module for the Patient Condition Classifier.

Responsible for reading the raw UCI Drug Reviews dataset from local
TSV files via Hugging Face's datasets.load_dataset() function.
"""

from datasets import load_dataset, DatasetDict

# ------------------------------------------------------------------
# Paths to the raw TSV files (relative to the project root)
# ------------------------------------------------------------------
# These files are extracted from the UCI Drug Reviews Dataset
# (drugsCom_raw.zip) and placed inside the data/ directory.
#   - drugsComTrain_raw.tsv:  ~161k training reviews
#   - drugsComTest_raw.tsv:   ~54k test reviews
DATA_FILES = {
    "train": "data/drugsComTrain_raw.tsv",
    "test": "data/drugsComTest_raw.tsv",
}


def load_data(data_dir: str = "data") -> DatasetDict:
    """
    Load the raw drug review dataset from local TSV files.

    Uses Hugging Face's datasets.load_dataset() with the 'csv'
    configuration and tab delimiter to read the TSV files into a
    DatasetDict with two keys:
        - 'train': training reviews
        - 'test':  test reviews

    Each split contains the following columns:
        - Unnamed: 0  — integer index (carried over from pandas)
        - drugName    — name of the drug being reviewed
        - condition   — patient condition the drug is for
        - review      — full text of the review
        - rating      — patient rating (1–10 scale)
        - date        — date the review was posted
        - usefulCount — number of users who found the review helpful

    Parameters
    ----------
    data_dir : str
        Path to the directory containing drugsComTrain_raw.tsv and
        drugsComTest_raw.tsv. Defaults to "data" for local use.
        On SageMaker, set to /opt/ml/input/data/training.

    Returns
    -------
    DatasetDict
        A dictionary-like object with 'train' and 'test' Dataset keys.
    """
    data_files = {
        "train": f"{data_dir}/drugsComTrain_raw.tsv",
        "test": f"{data_dir}/drugsComTest_raw.tsv",
    }
    # load_dataset with "csv" and delimiter="\t" reads tab-separated
    # files. The library infers column types automatically from the
    # header row and data.
    return load_dataset("csv", data_files=data_files, delimiter="\t")
