from datasets import load_dataset, DatasetDict

DATA_FILES = {"train": "data/drugsComTrain_raw.tsv", "test": "data/drugsComTest_raw.tsv"}

def load_data() -> DatasetDict:
    return load_dataset("csv", data_files=DATA_FILES, delimiter="\t")