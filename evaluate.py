#!/usr/bin/env python
"""DR-Grade — evaluate the trained model on the test split.

Computes accuracy, macro/weighted P/R/F1, Quadratic Weighted Kappa (QWK),
confusion matrices (count + row-normalised), per-class charts and a
misclassified-sample grid.  Predictions use test-time augmentation (TTA:
original, H-flip, V-flip, HV-flip) by default.

Usage:
  source setenv.sh
  python evaluate.py --data-root split_dataset_cropped --model outputs/best_model.keras
"""

import argparse
import json
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from ordinal import (
    CORALLoss,
    OrdinalAccuracy,
    compute_qwk,
    logits_to_class,
    threshold_probs,
    thresholds_to_class_probs,
)

SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description="DR-Grade evaluation")
    p.add_argument("--data-root", required=True, help="Folder with train/ val/ test/")
    p.add_argument("--model", required=True, help="Path to best_model.keras")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--image-size", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--tta", action="store_true", default=True,
                   help="Test-time augmentation (default on)")
    p.add_argument("--no-tta", dest="tta", action="store_false")
    p.add_argument("--max-errors", type=int, default=20)
    return p.parse_args()


TTA_TRANSFORMS = [
    ("original", lambda x: x),
    ("hflip", tf.image.flip_left_right),
    ("vflip", tf.image.flip_up_down),
    ("hflip+vflip", lambda x: tf.image.flip_left_right(tf.image.flip_up_down(x))),
]


def predict_all(model, test_ds, tta):
    """Return y_true (ints) and mean threshold probs P(grade >= k) over TTA."""
    y_true, probs_agg = [], []
    for images, labels in test_ds:
        y_true.extend(tf.argmax(tf.cast(labels, tf.float32), axis=-1).numpy())
        if not tta:
            p = threshold_probs(model(images, training=False)).numpy()
        else:
            acc = None
            for _, fn in TTA_TRANSFORMS:
                cur = threshold_probs(model(fn(images), training=False)).numpy()
                acc = cur if acc is None else acc + cur
            p = acc / len(TTA_TRANSFORMS)
        probs_agg.append(p)
    return np.asarray(y_true), np.concatenate(probs_agg, axis=0)


def save_figures(y_true, y_pred, probs, class_names, test_ds, out_dir, max_errors, model):
    n = len(class_names)

    # Confusion matrices
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    labels_cm = [f"Class {c}" for c in class_names]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Confusion Matrix — Test Set", fontweight="bold")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels_cm, yticklabels=labels_cm, ax=axes[0])
    axes[0].set_title("Counts")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=labels_cm, yticklabels=labels_cm, ax=axes[1])
    axes[1].set_title("Row-Normalised (recall per class)")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    with open(os.path.join(out_dir, "confusion_matrix.json"), "w") as f:
        json.dump(cm.tolist(), f, indent=2)

    # Per-class P/R/F1
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n)), zero_division=0
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Per-Class Performance on Test Set", fontweight="bold")
    for ax, values, title, color in zip(
        axes, [prec, rec, f1], ["Precision", "Recall", "F1-Score"],
        ["#4C72B0", "#55A868", "#C44E52"],
    ):
        bars = ax.bar(range(n), values, color=color, edgecolor="white")
        ax.set_xticks(range(n)); ax.set_xticklabels(labels_cm, rotation=20)
        ax.set_ylim(0, 1.05); ax.set_title(title); ax.set_ylabel(title)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax.axhline(np.mean(values), color="gray", ls="--", lw=1, label=f"Mean={np.mean(values):.3f}")
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "per_class_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Misclassified samples (re-walk test_ds for image tensors)
    errors = []
    for images, labels in test_ds:
        pb = threshold_probs(model(images, training=False)).numpy()
        true = tf.argmax(tf.cast(labels, tf.float32), axis=-1).numpy()
        pred = (pb > 0.5).sum(axis=1).astype(int)
        cls_probs = thresholds_to_class_probs(pb)
        conf = cls_probs[np.arange(len(pb)), pred]
        for img, t, pc, c in zip(images.numpy(), true, pred, conf):
            if t != pc:
                errors.append((img.astype("uint8"), int(t), int(pc), float(c)))
                if len(errors) >= max_errors:
                    break
        if len(errors) >= max_errors:
            break
    if errors:
        cols = 5
        rows = int(np.ceil(len(errors) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))
        fig.suptitle("Misclassified Test Samples — Actual vs Predicted", fontweight="bold")
        for i, ax in enumerate(np.atleast_2d(axes).reshape(-1)):
            ax.axis("off")
            if i < len(errors):
                img, t, pc, c = errors[i]
                ax.imshow(img)
                ax.set_title(f"True: {class_names[t]} | Pred: {class_names[pc]} ({c:.1%})",
                             fontsize=9, color="red")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "error_analysis.png"), dpi=150, bbox_inches="tight")
        plt.close()
    else:
        print("No misclassifications found on test set.")

    print(f"\nTop confusion pairs (excluding diagonal):")
    cm_nd = cm.copy(); np.fill_diagonal(cm_nd, 0)
    flat_idx = np.argsort(cm_nd.flatten())[::-1][:10]
    for idx in flat_idx:
        r, c = divmod(int(idx), n)
        if cm_nd[r, c] > 0:
            print(f"  Actual Class {class_names[r]} -> Predicted Class {class_names[c]}: {int(cm_nd[r, c])}")

    print("\nWeakest classes by F1:")
    for cls_idx, f1_val in sorted(enumerate(f1), key=lambda x: x[1]):
        print(f"  Class {class_names[cls_idx]}: F1={f1_val:.4f} | P={prec[cls_idx]:.4f} | R={rec[cls_idx]:.4f}")

    return dict(zip(class_names, prec.tolist())), dict(zip(class_names, rec.tolist())), dict(
        zip(class_names, f1.tolist())
    )


def inference_demo(model, class_names, test_dir, image_size):
    """Quick single-image demo (no TTA)."""
    import json
    from tensorflow.keras.utils import load_img, img_to_array

    candidates = []
    for cls in class_names:
        d = os.path.join(test_dir, cls)
        if os.path.isdir(d):
            imgs = [f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if imgs:
                candidates.append((os.path.join(d, random.choice(imgs)), cls))
    if not candidates:
        print("No test images found for demo."); return None
    path, true_label = random.choice(candidates)
    img = load_img(path, target_size=(image_size, image_size))
    arr = img_to_array(img)[None]
    tp = threshold_probs(model(arr, training=False)).numpy()[0]
    pred_idx = int((tp > 0.5).sum())
    class_probs = thresholds_to_class_probs(tp)
    print("=" * 60)
    print(" INFERENCE TEST")
    print("=" * 60)
    print(f"  Image      : {os.path.basename(path)}")
    print(f"  True label : Class {true_label}")
    print(f"  Predicted  : Class {class_names[pred_idx]}")
    print(f"  Confidence : {class_probs[pred_idx]:.4f} ({class_probs[pred_idx] * 100:.2f}%)")
    print("\n  All class probabilities:")
    for cls, p in zip(class_names, class_probs):
        bar = chr(9608) * int(p * 30)
        print(f"    Class {cls}: {p:.4f}  {bar}")
    return {"image": os.path.basename(path), "true": true_label,
            "pred": class_names[pred_idx],
            "probs": {c: float(p) for c, p in zip(class_names, class_probs)}}


def main():
    args = parse_args()
    random.seed(SEED); np.random.seed(SEED)
    os.makedirs(args.output_dir, exist_ok=True)

    model = tf.keras.models.load_model(args.model)
    class_names = sorted(os.listdir(os.path.join(args.data_root, "test")))

    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(args.data_root, "test"),
        labels="inferred", label_mode="categorical",
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size, shuffle=False,
    ).cache().prefetch(tf.data.AUTOTUNE)

    y_true, probs = predict_all(model, test_ds, tta=args.tta)
    y_pred = (probs > 0.5).sum(axis=1).astype(int)

    acc = np.mean(y_true == y_pred)
    qwk = compute_qwk(y_true, y_pred)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("=" * 60)
    print(" TEST SET EVALUATION")
    print("=" * 60)
    print(f"  TTA            : {args.tta}")
    print(f"  Test Accuracy  : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"  QWK            : {qwk:.4f}")
    print(f"  Macro Precision: {macro_prec:.4f}")
    print(f"  Macro Recall   : {macro_rec:.4f}")
    print(f"  Macro F1       : {macro_f1:.4f}")
    print(f"  Weighted F1    : {weighted_f1:.4f}")
    print()
    print(classification_report(
        y_true, y_pred, target_names=[f"Class {c}" for c in class_names], digits=4
    ))

    per_class_prec, per_class_rec, per_class_f1 = save_figures(
        y_true, y_pred, probs, class_names, test_ds, args.output_dir, args.max_errors, model
    )

    metrics = {
        "test_accuracy": float(acc),
        "qwk": float(qwk),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "tta": args.tta,
        "per_class_precision": per_class_prec,
        "per_class_recall": per_class_rec,
        "per_class_f1": per_class_f1,
        "n_test": int(len(y_true)),
    }
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Metrics saved -> {os.path.join(args.output_dir, 'test_metrics.json')}")

    inference_demo(model, class_names, os.path.join(args.data_root, "test"), args.image_size)


if __name__ == "__main__":
    main()
