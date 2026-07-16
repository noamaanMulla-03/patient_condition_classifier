# Patient Condition Classifier

Preprocesses the [UCI Drug Reviews dataset](https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com) for patient condition classification using a **DeBERTa-v3** tokenizer.

This pipeline loads raw TSV files, cleans and normalises the text, tokenizes reviews with overflow handling, and saves the result to disk for downstream model training.

## Pipeline Overview

```
drugsComTrain_raw.tsv ─┐
                       ├──> load_data() ──> clean_data() ──> tokenize_data() ──> save_to_disk()
drugsComTest_raw.tsv  ─┘
```

| Step | Module                | Description                                                                                                      |
| ---- | --------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 1    | `src/data_loader.py`  | Reads the raw TSV files using Hugging Face `datasets`                                                            |
| 2    | `src/data_cleaner.py` | Renames columns, filters nulls, lowercases text, unescapes HTML, removes short reviews, creates validation split |
| 3    | `src/tokenizer.py`    | Tokenizes with `microsoft/deberta-v3-base` (max 128 tokens, overflow chunking)                                   |
| 4    | `main.py`             | Orchestrates the pipeline and saves to `drug-dataset/`                                                           |

## Dataset

The [UCI Drug Review Dataset](https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com) contains ~215k patient drug reviews with the following columns:

| Column        | Description                                  |
| ------------- | -------------------------------------------- |
| `drugName`    | Name of the drug                             |
| `condition`   | Patient condition (target label)             |
| `review`      | Full text of the review                      |
| `rating`      | Patient rating (1–10)                        |
| `date`        | Date of the review                           |
| `usefulCount` | Number of users who found the review helpful |

### Cleaning steps

1. **Rename** `Unnamed: 0` → `patient_id`
2. **Filter** rows with null `drugName` or `condition`
3. **Lowercase** `drugName` and `condition`
4. **Unescape** HTML entities in reviews (`html.unescape`)
5. **Compute** `reviewLength` (word count)
6. **Remove** reviews ≤ 30 words
7. **Split** training data 80/20 → train / validation (original test set preserved)

### Final splits

| Split        | Approx. rows | Purpose               |
| ------------ | ------------ | --------------------- |
| `train`      | ~111k        | Model training        |
| `validation` | ~28k         | Hyperparameter tuning |
| `test`       | ~46k         | Final evaluation      |

## Setup

### Prerequisites

- Python ≥ 3.13

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd patient-condition-classifier

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Additional tokenizer dependencies

The DeBERTa-v3 tokenizer requires `sentencepiece` and `protobuf`:

```bash
pip install sentencepiece protobuf
```

## Usage

Place the TSV files in the `data/` directory, then run:

```bash
python main.py
```

This will:

1. Load the raw TSV files
2. Clean and preprocess the data
3. Tokenize reviews using DeBERTa-v3
4. Save the tokenized dataset to `drug-dataset/`

### Loading the saved dataset

```python
from datasets import load_from_disk

dataset = load_from_disk("drug-dataset")
print(dataset)
# DatasetDict({
#     train:       Dataset({ features: [...], num_rows: 110811 })
#     validation:  Dataset({ features: [...], num_rows: 27703 })
#     test:        Dataset({ features: [...], num_rows: 46108 })
# })
```

## Project Structure

```
patient-condition-classifier/
├── main.py                  # Pipeline entry point
├── pyproject.toml           # Project configuration
├── requirements.txt         # Pinned dependencies
├── data/                    # Raw TSV files (gitignored)
├── drug-dataset/            # Tokenized output (gitignored)
└── src/
    ├── data_loader.py       # Loads raw TSV files
    ├── data_cleaner.py      # Cleans and preprocesses
    └── tokenizer.py         # Tokenizes with overflow handling
```

## License

MIT — see [LICENSE](LICENSE).
