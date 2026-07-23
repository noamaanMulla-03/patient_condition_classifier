"""
Fine-tuning module for the Patient Condition Classifier.

Takes a tokenized DatasetDict from the pipeline and fine-tunes a
DeBERTa-v3-base model for sequence classification on patient conditions.

Called by main.py as the final step of the pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import evaluate
from collections import Counter
from datasets import DatasetDict
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
)

# ------------------------------------------------------------------
# Module-level metric loaders — avoids downloading from Hugging Face
# Hub on every SageMaker training job startup (~30s saved).
# ------------------------------------------------------------------
ACCURACY_METRIC = evaluate.load("accuracy")
F1_METRIC = evaluate.load("f1")


# ------------------------------------------------------------------
# Focal Loss — down-weights easy examples so the model focuses on
# rare / hard-to-classify conditions instead of being dominated by
# the top 20 common classes.
# ------------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss with label smoothing for imbalanced multi-class classification.

    Combines two techniques:
    1. **Focal loss** — down-weights easy examples (high confidence) to
       focus learning on rare/hard classes.
    2. **Label smoothing** — replaces one-hot targets with a mixture of
       the true label and a uniform distribution, preventing overconfidence
       on common classes and giving rare classes more gradient signal.

    Parameters
    ----------
    gamma : float
        Focusing parameter. Higher = more focus on hard examples.
        gamma=2.0 is the standard recommended value.
    alpha : torch.Tensor or None
        Optional class weights for per-class re-weighting.
    label_smoothing : float
        Smoothing factor. 0.0 = no smoothing, 0.1 = 10% uniform.
    """

    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        n_classes = inputs.size(-1)

        # Apply label smoothing: one-hot → (1-smooth)*one_hot + smooth/num_classes
        if self.label_smoothing > 0:
            smooth_targets = torch.full_like(
                inputs, self.label_smoothing / (n_classes - 1)
            )
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            log_probs = F.log_softmax(inputs, dim=-1)
            ce_loss = -(smooth_targets * log_probs).sum(dim=-1)
        else:
            ce_loss = F.cross_entropy(
                inputs, targets, weight=self.alpha, reduction="none"
            )

        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ------------------------------------------------------------------
# Custom Trainer that uses Focal Loss instead of standard CE.
# ------------------------------------------------------------------
class FocalLossTrainer(Trainer):
    """
    Trainer subclass that replaces cross-entropy with Focal Loss.
    Accepts pre-computed class weights for imbalanced datasets.
    """

    def __init__(
        self, *args, focal_gamma=2.0, class_weights=None, label_smoothing=0.1, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss(
            gamma=focal_gamma,
            alpha=class_weights,
            label_smoothing=label_smoothing,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.focal_loss(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


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
        'microsoft/deberta-v3-large'), passed from main.py.
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
    # Step 2: Load the model with tuned dropout & label smoothing
    # ------------------------------------------------------------------
    # Default dropout (0.1) is too low for 700+ classes — the model
    # memorizes frequent classes instead of learning rare ones.
    # Bumping dropout forces the model to read the full review rather
    # than relying on surface-level pattern matching.
    # Label smoothing (0.1) prevents overconfidence on common classes,
    # reserving probability mass for rare classes.
    print(f"Loading model '{checkpoint}' with {num_labels} classes...")
    config = AutoConfig.from_pretrained(checkpoint)
    config.num_labels = num_labels
    config.label2id = label2id
    config.id2label = id2label
    config.hidden_dropout_prob = 0.2
    config.attention_probs_dropout_prob = 0.2
    config.classifier_dropout = 0.3
    config.label_smoothing = 0.1

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        config=config,
        ignore_mismatched_sizes=True,
    )

    # ------------------------------------------------------------------
    # Step 3: Define evaluation metrics
    # ------------------------------------------------------------------
    # We track two metrics:
    #   - accuracy: % of predictions that exactly match the true label.
    #   - weighted F1: harmonic mean of precision & recall, weighted by
    #     class support (accounts for class imbalance).
    # Metric loaders are module-level constants to avoid re-downloading
    # from HF Hub on every training run.

    def compute_metrics(eval_pred):
        """Calculate accuracy and weighted F1 from model predictions."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        acc = ACCURACY_METRIC.compute(predictions=predictions, references=labels)
        f1 = F1_METRIC.compute(
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
    #   - batch_size=8, gradient_accumulation_steps=4: effective batch
    #     of 32 while fitting large model in 24 GB L4 VRAM.
    #   - num_train_epochs=5: extended training with cosine LR decay
    #     for further convergence without overfitting.
    #   - evaluation_strategy="steps": evaluate every eval_steps for finer-
    #     grained early-stopping control (~16 evals/epoch at 500 steps).
    #   - load_best_model_at_end: loads the checkpoint with best accuracy
    #     (not necessarily the last one) after training finishes.
    #   - EarlyStoppingCallback: stops training if validation accuracy
    #     doesn't improve for patience * eval_steps consecutive steps,
    #     saving GPU time and preventing overfitting.
    #   - bf16: used instead of fp16 because DeBERTa-v3's disentangled
    #     attention breaks with fp16 gradients. bf16 is available on
    #     Ampere+ GPUs (A100, A10G). T4 and MPS fall back to fp32.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print("\nConfiguring training arguments...")
    training_args = TrainingArguments(
        output_dir=model_dir,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        gradient_accumulation_steps=4,
        logging_steps=100,
        logging_first_step=True,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=False,
        bf16=use_bf16,
        seed=42,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    # ------------------------------------------------------------------
    # Step 5: Create the Trainer with focal loss + class weights
    # ------------------------------------------------------------------
    # Compute per-class weights: rare classes get higher weight so
    # the model pays attention to them instead of being dominated
    # by the top-20 frequent conditions.
    label_counts = Counter(tokenized_dataset["train"]["labels"])
    total = sum(label_counts.values())
    class_weights = torch.zeros(num_labels, dtype=torch.float)
    for i in range(num_labels):
        count = label_counts.get(i, 1)
        class_weights[i] = total / (num_labels * count)

    trainer = FocalLossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        focal_gamma=2.0,
        class_weights=class_weights,
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
    # Step 7b: Evaluate on the held-out test set
    # ------------------------------------------------------------------
    # The test split was never seen during training or hyperparameter
    # tuning — this gives an unbiased estimate of real-world performance.
    print("\n" + "=" * 60)
    print("Evaluating on held-out test set...")
    print("=" * 60)
    test_results = trainer.evaluate(tokenized_dataset["test"])
    print(f"\nTest results: {test_results}")

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
