"""
Fine-tuning module for the Patient Condition Classifier.

Takes a tokenized DatasetDict from the pipeline and fine-tunes a
DeBERTa-v3-base model for sequence classification on patient conditions.

Called by main.py as the final step of the pipeline.
"""

import torch
import numpy as np
import evaluate
from datasets import DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


def fine_tune(
    tokenized_dataset: DatasetDict,
    tokenizer,
    data_collator,
    checkpoint: str,
    label2id: dict,
    id2label: dict,
    num_labels: int,
    model_dir: str = "./results",
) -> None:
    """
    Fine-tune DeBERTa-v3-base on the tokenized drug review dataset.

    The tokenizer, data_collator, checkpoint, and label mappings are
    passed in from main.py to keep configuration centralised.

    Steps
    -----
    1. Strip all columns except the ones the model needs.
    2. Load DeBERTa-v3-base with a classification head.
    3. Define accuracy and weighted F1 as evaluation metrics.
    4. Configure training hyperparameters.
    5. Create the Hugging Face Trainer.
    6. Train the model.
    7. Evaluate on the validation set.
    8. Save the final model and tokenizer to disk.

    Parameters
    ----------
    tokenized_dataset : DatasetDict
        A DatasetDict with 'train', 'validation', and 'test' splits,
        each containing 'input_ids', 'attention_mask',
        'token_type_ids', and 'labels' columns (as produced by
        tokenize_data).
    tokenizer : PreTrainedTokenizer
        The DeBERTa-v3 tokenizer, loaded once in main.py.
    data_collator : DataCollatorWithPadding
        A data collator configured with the above tokenizer.
    checkpoint : str
        The Hugging Face model checkpoint name (e.g.
        'microsoft/deberta-v3-base'), passed from main.py.
    label2id : dict[str, int]
        Mapping from condition strings to integer class IDs.
    id2label : dict[int, str]
        Mapping from integer class IDs back to condition strings.
    num_labels : int
        Total number of unique conditions (classification classes).
    model_dir : str
        Directory for checkpoints during training and final model.
        Defaults to "./results". On SageMaker, set to
        /opt/ml/model (where SageMaker expects the final artifact).

    Returns
    -------
    None — the trained model and tokenizer are saved to disk.
    """
    print(f"\n  Train:      {len(tokenized_dataset['train']):>7} samples")
    print(f"  Validation: {len(tokenized_dataset['validation']):>7} samples")
    print(f"  Test:       {len(tokenized_dataset['test']):>7} samples")

    # ------------------------------------------------------------------
    # Automatically detect device for mixed precision
    # ------------------------------------------------------------------
    # DeBERTa-v3 doesn't support fp16 (disentangled attention breaks).
    # We use bf16 instead if available (Ampere+ GPUs), otherwise fp32.
    use_fp16 = False  # kept for reference; bf16 is used instead

    # ------------------------------------------------------------------
    # Step 1: Remove unnecessary columns
    # ------------------------------------------------------------------
    # The model only needs input_ids, attention_mask, and labels.
    # token_type_ids is included for compatibility (DeBERTa ignores it
    # for single-sequence classification but it doesn't hurt).
    # All other columns (patient_id, drugName, review, rating, etc.)
    # are dropped to save memory and avoid confusion during training.
    columns_to_keep = ["input_ids", "attention_mask", "token_type_ids", "labels"]
    columns_to_remove = [
        c for c in tokenized_dataset["train"].column_names if c not in columns_to_keep
    ]
    tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)
    print(f"\nColumns kept for training: {tokenized_dataset['train'].column_names}")

    # ------------------------------------------------------------------
    # Step 2: Load the model
    # ------------------------------------------------------------------
    # AutoModelForSequenceClassification wraps DeBERTa-v3-base with a
    # linear classification head on top of the [CLS] token embedding.
    # The label mappings are baked into the model config so predictions
    # carry human-readable labels automatically.
    print(f"Loading model '{checkpoint}' with {num_labels} classes...")
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    # ------------------------------------------------------------------
    # Step 3: Define evaluation metrics
    # ------------------------------------------------------------------
    # We track two metrics:
    #   - accuracy: % of predictions that exactly match the true label.
    #   - weighted F1: harmonic mean of precision & recall, weighted by
    #     class support (accounts for class imbalance).
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        """Calculate accuracy and weighted F1 from model predictions."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        acc = accuracy_metric.compute(predictions=predictions, references=labels)
        f1 = f1_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )
        return {**acc, **f1}

    # ------------------------------------------------------------------
    # Step 4: Configure training arguments
    # ------------------------------------------------------------------
    # These hyperparameters are sensible defaults for fine-tuning a
    # BERT-style model on a moderate-sized dataset (~160k samples).
    # Key choices:
    #   - learning_rate=2e-5: standard for fine-tuning transformers.
    #   - batch_size=16: fits most GPUs with 8-16 GB VRAM.
    #   - num_train_epochs=3: enough to converge without overfitting.
    #   - eval_strategy="epoch": evaluate once per epoch.
    #   - load_best_model_at_end: keeps the epoch with best accuracy.
    #   - bf16: used instead of fp16 because DeBERTa-v3's disentangled
    #     attention breaks with fp16 gradients. bf16 is available on
    #     Ampere+ GPUs (A100, A10G). T4 and MPS fall back to fp32.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print("\nConfiguring training arguments...")
    training_args = TrainingArguments(
        output_dir=model_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=100,
        logging_first_step=True,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=False,
        bf16=use_bf16,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    # ------------------------------------------------------------------
    # Step 5: Create the Trainer
    # ------------------------------------------------------------------
    # The Trainer orchestrates the training loop, evaluation, logging,
    # checkpointing, and metric computation.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # ------------------------------------------------------------------
    # Step 6: Train the model
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    trainer.train()

    # ------------------------------------------------------------------
    # Step 7: Evaluate on the validation set
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Evaluating on validation set...")
    print("=" * 60)
    eval_results = trainer.evaluate()
    print(f"\nValidation results: {eval_results}")

    # ------------------------------------------------------------------
    # Step 8: Save the final model
    # ------------------------------------------------------------------
    # On SageMaker, model_dir points to /opt/ml/model and output_dir
    # to /opt/ml/output. These are set from main.py via CLI args.
    final_dir = f"{model_dir}/final-model"
    print(f"\nSaving final model to '{final_dir}/'...")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("Done!")
