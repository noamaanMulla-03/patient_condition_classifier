# Patient Condition Classifier

Fine-tunes **DeBERTa-v3-large** on the [UCI Drug Reviews dataset](https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com) to predict patient conditions from drug review text.

The pipeline loads raw TSV files, cleans and normalises the text, tokenizes reviews (512 tokens, simple truncation), fine-tunes a transformer classifier with early stopping and test-set evaluation, and saves the trained model to disk.

## Pipeline Overview

```
drugsComTrain_raw.tsv ─┐
                       ├──> load_data() ──> clean_data() ──> tokenize_data() ──> fine_tune() ──> saved model
drugsComTest_raw.tsv  ─┘
```

| Step | Module                | Description                                                                                                                                                             |
| ---- | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `src/data_loader.py`  | Reads the raw TSV files using Hugging Face `datasets`                                                                                                                   |
| 2    | `src/data_cleaner.py` | Renames columns, filters nulls, lowercases text, unescapes HTML, strips junk labels, removes short reviews, filters low-rated rants, creates validation split (seed=42) |
| 3    | `src/tokenizer.py`    | Prepends drug name to reviews, tokenizes with `microsoft/deberta-v3-large` (max 512 tokens), builds label mappings                                                      |
| 4    | `src/fine_tune.py`    | Fine-tunes DeBERTa-v3-large with focal loss, label smoothing, increased dropout, early stopping, and test-set evaluation                                                |
| 5    | `main.py`             | Orchestrates the full pipeline end-to-end (caches tokenized dataset to disk)                                                                                            |

## Dataset

The [UCI Drug Review Dataset](https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com) contains ~215k patient drug reviews with the following columns:

| Column        | Description                                  |
| ------------- | -------------------------------------------- |
| `drugName`    | Name of the drug                             |
| `condition`   | Patient condition (target label)             |
| `review`      | Full text of the review                      |
| `rating`      | Patient rating (1–10)                        |
| `date`        | Date the review was posted                   |
| `usefulCount` | Number of users who found the review helpful |

### Cleaning steps

1. **Rename** `Unnamed: 0` → `patient_id`
2. **Filter** rows with null `drugName` or `condition`
3. **Lowercase** `drugName` and `condition`
4. **Unescape** HTML entities in reviews (`html.unescape`)
5. **Strip** HTML tags from condition labels
6. **Filter** junk labels (scraper artifacts like "0 users found this comment helpful.")
7. **Compute** `reviewLength` (word count)
8. **Remove** reviews ≤ 30 words
9. **Filter** low-quality reviews — drops ratings 1–3 unless other users found them helpful (`usefulCount ≥ 5`). Removes rants with zero clinical signal.
10. **Split** training data 80/20 → train / validation (original test set preserved)

### Final splits

| Split        | Approx. rows | Purpose               |
| ------------ | ------------ | --------------------- |
| `train`      | ~90k         | Model training        |
| `validation` | ~23k         | Hyperparameter tuning |
| `test`       | ~38k         | Final evaluation      |

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

### Option A: Run locally

Place the TSV files in the `data/` directory, then run:

```bash
python main.py
```

This will:

1. Load the raw TSV files
2. Clean and preprocess the data
3. Tokenize reviews using DeBERTa-v3
4. Save the tokenized dataset to `drug-dataset/`
5. Fine-tune the classifier (5 epochs)
6. Save the trained model to `results/final-model/`

CLI arguments for custom paths:

```bash
python main.py --data_dir data --model_dir ./results --output_dir drug-dataset
```

### Option B: Train on AWS SageMaker (cloud GPU)

1. Copy `.env.example` → `.env` and fill in your AWS credentials:

    ```
    SAGEMAKER_S3_BUCKET=your-bucket-name
    SAGEMAKER_ROLE=arn:aws:iam::<account-id>:role/sagemaker-execution-role
    AWS_REGION=us-east-1
    ```

2. Upload the TSV files to S3:

    ```bash
    aws s3 cp data/drugsComTrain_raw.tsv s3://<bucket>/data/
    aws s3 cp data/drugsComTest_raw.tsv  s3://<bucket>/data/
    ```

3. Launch training:

    ```bash
    python launch_sagemaker.py
    ```

4. Monitor at the [SageMaker console](https://console.aws.amazon.com/sagemaker/home#/jobs).

Instance type and container versions are configured in `.config.yaml`.

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

### Inference

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("results/final-model")
tokenizer = AutoTokenizer.from_pretrained("results/final-model")

review = "I've been taking this for 3 months and my anxiety has significantly reduced."
inputs = tokenizer(review, truncation=True, max_length=512, return_tensors="pt")
outputs = model(**inputs)
predicted_id = outputs.logits.argmax(dim=-1).item()
print(model.config.id2label[predicted_id])  # e.g. "anxiety"
```

Or use the `pipeline` API:

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="results/final-model", tokenizer="results/final-model")
result = classifier("I've been taking this for 3 months and my anxiety has significantly reduced.")
print(result[0]["label"])  # predicted condition
```

### Evaluation

The training script reports metrics on **two** held-out sets:

- **Validation**: used during training for early stopping and checkpoint selection
- **Test**: evaluated after training completes — gives an unbiased estimate of real-world performance

## Training Configuration

| Parameter            | Value                                                   |
| -------------------- | ------------------------------------------------------- |
| Model                | `microsoft/deberta-v3-large` (304M params)              |
| Max sequence length  | 512 tokens                                              |
| Batch size           | 8 per GPU (effective 32 with gradient accumulation × 4) |
| Learning rate        | 2e-5                                                    |
| LR scheduler         | Cosine with 10% linear warmup                           |
| Epochs               | 5 (max, with early stopping)                            |
| Weight decay         | 0.01                                                    |
| Dropout              | 0.2 hidden, 0.2 attention, 0.3 classifier               |
| Loss function        | Focal loss (γ=2) with label smoothing (0.1)             |
| Class weights        | Inverse-frequency per class                             |
| Seed                 | 42 (reproducible)                                       |
| Mixed precision      | bf16 (if available) / fp32                              |
| Evaluation           | Accuracy + weighted F1                                  |
| Early stopping       | Patience of 3 (1,500 steps)                             |
| Best model selection | Highest validation accuracy                             |
| Checkpoint retention | Last 2 checkpoints kept                                 |

### Model Variants

The default pipeline uses DeBERTa-v3-large for maximum accuracy on cloud GPUs. For local training, you can swap to a lighter model by changing the `checkpoint` variable in `main.py`:

| Model                         | Params | VRAM (bf16) | Accuracy (est.) | Best for                        |
| ----------------------------- | ------ | ----------- | --------------- | ------------------------------- |
| `microsoft/deberta-v3-base`   | 184M   | ~8 GB       | ~73%            | T4, laptop GPU, fast iteration  |
| `microsoft/deberta-v3-large`  | 304M   | ~14 GB      | ~77%            | A10G, L4, best accuracy         |
| `distilbert-base-uncased`     | 67M    | ~4 GB       | ~68%            | CPU training, Colab free tier   |
| `microsoft/deberta-v3-xsmall` | 22M    | ~2 GB       | ~62%            | Quick prototyping, edge devices |

To switch models locally:

```python
# In main.py, change line ~71:
checkpoint = "microsoft/deberta-v3-base"  # or distilbert-base-uncased, etc.
```

> **Note:** `distilbert-base-uncased` uses a different tokenizer — update `from_pretrained(checkpoint)` calls accordingly. DeBERTa variants share the same tokenizer.

> **Note:** DeBERTa-v3 does not support fp16 due to its disentangled attention mechanism. The pipeline automatically uses bf16 on Ampere+ GPUs (A100, A10G, L4) and falls back to fp32 on T4 and Apple Silicon.

### Key techniques

- **Drug name prepended to review** — `"Drug: Levora. Review: ..."` gives the model a free predictive signal (Levora → birth control)
- **Focal loss (γ=2.0)** — down-weights easy/common classes so the model focuses on rare/hard conditions
- **Label smoothing (0.1)** — prevents overconfidence on frequent classes, reserving probability for rare ones
- **Increased dropout** — forces the model to read the full review instead of pattern-matching surface-level words
- **Rating-based quality filter** — removes reviews rated 1–3 that are pure rants with no clinical content

## Results

| Experiment                                        | Accuracy    | Weighted F1 |
| ------------------------------------------------- | ----------- | ----------- |
| DeBERTa-v3-base, 128 tokens, basic CE (T4)        | 67.4%       | 63.9%       |
| DeBERTa-v3-base, 512 tokens, basic CE (L4)        | 73.3%       | 69.6%       |
| **DeBERTa-v3-large, 512 tokens, focal loss (L4)** | **running** | **running** |

### SageMaker Instance

Default instance: **`ml.g6.xlarge`** (NVIDIA L4, 24 GB VRAM). This enables bf16 training at ~$0.80/hr.

## Project Structure

```
patient-condition-classifier/
├── main.py                  # Pipeline entry point (local)
├── sagemaker_entry.py       # SageMaker entry point (bridges env vars to main.py)
├── launch_sagemaker.py      # Launches a SageMaker training job
├── pyproject.toml           # Project configuration
├── requirements.txt         # Pinned dependencies
├── .config.yaml             # SageMaker instance & container config
├── .env.example             # Template for AWS secrets (copy to .env)
├── .sagemakerignore         # Excludes large files from SageMaker upload
├── data/                    # Raw TSV files (gitignored)
├── drug-dataset/            # Tokenized Arrow dataset (gitignored)
├── results/                 # Trained model checkpoints (gitignored)
└── src/
    ├── data_loader.py       # Loads raw TSV files
    ├── data_cleaner.py      # Cleans and preprocesses (8 steps)
    ├── tokenizer.py         # Tokenizes with truncation (512 tokens) + label mapping
    ├── fine_tune.py         # Fine-tunes with focal loss, label smoothing, dropout
```

## License

MIT — see [LICENSE](LICENSE).
