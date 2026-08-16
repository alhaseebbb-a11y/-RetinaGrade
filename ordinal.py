"""Ordinal (CORAL) building blocks for DR severity grading.

DR grades 0-4 are ordinal: predicting grade 2 when the truth is grade 4 is
far worse than predicting grade 3.  CORAL (Cumulative Ordinal Regression
with Logits) models K-1 binary thresholds P(grade >= k) and optimises them
with sigmoid binary cross-entropy, which respects label ordering instead of
treating all misclassifications equally (plain categorical cross-entropy).
"""

import numpy as np
import tensorflow as tf

__all__ = [
    "threshold_probs",
    "logits_to_class",
    "thresholds_to_class_probs",
    "CORALLoss",
    "OrdinalAccuracy",
    "QWKCallback",
    "compute_qwk",
]


def threshold_probs(logits):
    """Sigmoid threshold probabilities P(grade >= k), shape [batch, K-1]."""
    return tf.sigmoid(tf.cast(logits, tf.float32))


def logits_to_class(logits):
    """Canonical CORAL prediction: class = count of thresholds above 0.5.

    y_pred = 1 + sum_k 1[P(grade >= k) > 0.5]  (0-indexed for grades 0..K-1).
    """
    p = threshold_probs(logits)
    return tf.reduce_sum(tf.cast(p > 0.5, tf.int64), axis=-1)


def thresholds_to_class_probs(p):
    """Map threshold probs [..., K-1] -> valid class probabilities [..., K].

    Display/confidence only: class mass P(grade=k) = P(>=k) - P(>=k+1) from
    the raw threshold probabilities, clamped to [0, 1] and renormalised so it
    is a valid distribution.  Use logits_to_class() for the actual prediction.
    """
    p = np.asarray(p, dtype=np.float64)
    upper = np.concatenate([np.ones_like(p[..., :1]), p], axis=-1)   # P(>=0..K-1)
    lower = np.concatenate([p, np.zeros_like(p[..., :1])], axis=-1)  # P(>=1..K)
    cp = np.clip(upper - lower, 0.0, None)
    return cp / np.maximum(cp.sum(axis=-1, keepdims=True), 1e-12)


@tf.keras.utils.register_keras_serializable(package="DRGrade")
class CORALLoss(tf.keras.losses.Loss):
    """Binary cross-entropy over the K-1 cumulative thresholds.

    y_true: one-hot [batch, K];  y_pred: ordinal logits [batch, K-1].
    """

    def __init__(self, num_classes=5, name="coral_loss",
                 reduction="sum_over_batch_size"):
        super().__init__(name=name, reduction=reduction)
        self.num_classes = num_classes

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        logits = tf.cast(y_pred, tf.float32)
        labels = tf.argmax(y_true, axis=-1)                   # [batch]
        thresholds = tf.range(1, self.num_classes, dtype=tf.int64)
        targets = tf.cast(labels[..., None] >= thresholds[None, ...], tf.float32)
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=targets)
        )

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config


@tf.keras.utils.register_keras_serializable(package="DRGrade")
class OrdinalAccuracy(tf.keras.metrics.Metric):
    """Accuracy computed on CORAL class probabilities (argmax)."""

    def __init__(self, name="ordinal_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self._acc = tf.keras.metrics.Mean()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        true_cls = tf.argmax(y_true, axis=-1)
        pred_cls = logits_to_class(y_pred)
        match = tf.cast(tf.equal(true_cls, pred_cls), tf.float32)
        if sample_weight is not None:
            match = match * tf.cast(sample_weight, tf.float32)
        self._acc.update_state(match)

    def result(self):
        return self._acc.result()

    def reset_state(self):
        self._acc.reset_state()


def compute_qwk(y_true, y_pred):
    """Quadratic Weighted Kappa (NaN-safe)."""
    from sklearn.metrics import cohen_kappa_score

    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0
    val = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return float(val) if np.isfinite(val) else 0.0


class QWKCallback(tf.keras.callbacks.Callback):
    """Compute validation QWK after every epoch so it can drive
    ModelCheckpoint / EarlyStopping / ReduceLROnPlateau.

    Place this BEFORE the checkpoint callback in the callbacks list.
    """

    def __init__(self, val_ds, num_classes=5, name="val_qwk"):
        super().__init__()
        self.val_ds = val_ds
        self.num_classes = num_classes
        self.name = name

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Distributed batched predict is far faster than calling self.model()
        # per batch on the main thread (which stalled epochs ~3-4x). model.predict
        # routes inference through the strategy (both GPUs) and one graph per batch.
        logits = self.model.predict(self.val_ds, verbose=0)
        y_true = np.concatenate(
            [tf.argmax(tf.cast(labels, tf.float32), axis=-1).numpy()
             for _, labels in self.val_ds]
        )
        y_pred = logits_to_class(logits)
        logs[self.name] = compute_qwk(np.asarray(y_true), np.asarray(y_pred))
