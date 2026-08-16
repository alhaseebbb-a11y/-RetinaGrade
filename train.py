#!/usr/bin/env python
"""DR-Grade — EfficientNet-B3 ordinal DR grading, HPC production training.

Runs the two-phase transfer-learning recipe from the original Colab notebook
(05_EfficientNetB3_Colab_Training_Pipeline.ipynb), upgraded with:
  * ordinal (CORAL) loss that respects DR grade ordering,
  * Quadratic Weighted Kappa (QWK) as the monitored / headline metric,
  * data-race-free multi-GPU training via tf.distribute.MirroredStrategy,
  * optional mixed precision (fp16) on the V100 Tensor Cores.

Usage:
  source setenv.sh
  python train.py --data-root split_dataset_cropped --output-dir outputs
  python train.py --data-root smoke_dataset --output-dir outputs/smoke \
         --head-epochs 1 --finetune-epochs 1 --gpus 1
"""

import argparse
import json
import os
import random
import time

import numpy as np

SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description="DR-Grade EfficientNet-B3 training")
    p.add_argument("--data-root", required=True,
                   help="Folder containing train/ val/ test/ sub-folders")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--image-size", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16,
                   help="Batch per GPU replica (global batch = batch_size * gpus)")
    p.add_argument("--gpus", type=int, default=2, choices=[1, 2])
    p.add_argument("--mixed-precision", action="store_true",
                   help="Enable fp16 on V100 Tensor Cores")
    p.add_argument("--cache", action="store_true",
                   help="Cache datasets in RAM (needs ~16 GB; you have plenty)")
    p.add_argument("--head-epochs", type=int, default=20, help="Phase 1 max epochs")
    p.add_argument("--finetune-epochs", type=int, default=30, help="Phase 2 max epochs")
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--finetune-lr", type=float, default=1e-5)
    p.add_argument("--unfreeze-from", type=int, default=200,
                   help="Unfreeze backbone layers from this index in Phase 2")
    p.add_argument("--patience", type=int, default=6, help="EarlyStopping patience")
    p.add_argument("--resume", default=None, metavar="MODEL",
                   help="Skip Phase 1: load this .keras checkpoint and continue "
                        "with Phase 2 fine-tuning only (used to recover from a "
                        "failed run without retraining the head).")
    return p.parse_args()


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def make_strategy(gpus):
    import tensorflow as tf

    visible = tf.config.list_physical_devices("GPU")
    if len(visible) < gpus:
        raise RuntimeError(f"Requested {gpus} GPUs but only {len(visible)} visible")
    for gpu in visible:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    if gpus == 1:
        return tf.distribute.OneDeviceStrategy("/gpu:0")
    return tf.distribute.MirroredStrategy()


def build_model(image_size, num_classes, dropout_1=0.4, dropout_2=0.3):
    """EfficientNet-B3 backbone + CORAL ordinal head."""
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras import layers
    from tensorflow.keras.applications import EfficientNetB3
    from tensorflow.keras.applications.efficientnet import preprocess_input

    base_model = EfficientNetB3(
        include_top=False, weights="imagenet", input_shape=(image_size, image_size, 3)
    )
    base_model.trainable = False

    aug = Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.10),
        ],
        name="data_augmentation",
    )

    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="input_images")
    x = preprocess_input(inputs)
    x = aug(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout_1, name="dropout_1")(x)
    x = layers.Dense(256, activation="relu", name="dense_head")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(dropout_2, name="dropout_2")(x)
    # float32 on purpose: under mixed_float16 the head logits are fp16 and can
    # overflow to inf once the model gets confident (loss=inf -> permanent NaN,
    # which dynamic loss scaling cannot recover from). fp32 logits keep the
    # ordinal thresholds numerically safe; the fp16 backbone still saves memory
    # and flops. CORALLoss / OrdinalAccuracy already cast to float32.
    x = layers.Dense(num_classes - 1, dtype="float32", name="ordinal_logits")(x)
    model = tf.keras.Model(inputs, outputs=x, name="EfficientNetB3_Ordinal")
    return model, base_model


def load_datasets(data_root, image_size, batch_size, seed, cache, num_classes, strategy):
    import tensorflow as tf
    from tensorflow.keras.utils import image_dataset_from_directory

    autotune = tf.data.AUTOTUNE

    def make_plain(phase, shuffle):
        return image_dataset_from_directory(
            os.path.join(data_root, phase),
            labels="inferred",
            label_mode="categorical",
            image_size=(image_size, image_size),
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
        )

    def make_raw(phase, shuffle):
        return image_dataset_from_directory(
            os.path.join(data_root, phase),
            labels="inferred",
            label_mode="categorical",
            image_size=(image_size, image_size),
            batch_size=None,
            shuffle=shuffle,
            seed=seed,
        )

    raw_train = make_raw("train", shuffle=True)
    class_names = raw_train.class_names
    assert len(class_names) == num_classes, f"Expected {num_classes} classes, got {class_names}"

    # val/test stay plain (single) datasets: QWKCallback iterates val_ds
    # directly, and the eval cost over ~7k images is negligible.
    val_ds = make_plain("val", shuffle=False)
    test_ds = make_plain("test", shuffle=False)
    if cache:
        val_ds = val_ds.cache().prefetch(autotune)
        test_ds = test_ds.cache().prefetch(autotune)
    else:
        val_ds = val_ds.prefetch(autotune)
        test_ds = test_ds.prefetch(autotune)

    # Train set: explicit per-replica sharding via distribute_datasets_from_function.
    # MirroredStrategy does NOT auto-shard image_dataset_from_directory pipelines
    # in TF 2.16: fit() would silently DUPLICATE the train set on every replica
    # (2x redundant compute, effective batch = per-replica batch). Sharding each
    # replica before batching gives the true global batch (batch_size * replicas).
    def train_fn(input_context):
        ds = image_dataset_from_directory(
            os.path.join(data_root, "train"),
            labels="inferred",
            label_mode="categorical",
            image_size=(image_size, image_size),
            batch_size=None,
            shuffle=True,
            seed=seed,
        )
        ds = ds.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
        ds = ds.shuffle(1024, seed=seed, reshuffle_each_iteration=True)
        if cache:
            ds = ds.cache()
        ds = ds.batch(batch_size, drop_remainder=True)
        return ds.repeat().prefetch(autotune)

    train_ds = strategy.distribute_datasets_from_function(train_fn)
    n_train = tf.data.experimental.cardinality(raw_train).numpy()
    train_steps = n_train // (batch_size * strategy.num_replicas_in_sync)
    return train_ds, val_ds, test_ds, class_names, train_steps


def make_callbacks(output_dir, val_ds, num_classes, patience):
    import tensorflow as tf
    from ordinal import QWKCallback

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(output_dir, "best_model.keras"),
        monitor="val_qwk",
        mode="max",
        save_best_only=True,
        verbose=1,
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_qwk", mode="max", patience=patience, restore_best_weights=True, verbose=1
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_qwk", mode="max", factor=0.5, patience=3, min_lr=1e-6, verbose=1
    )
    csv_logger = tf.keras.callbacks.CSVLogger(
        os.path.join(output_dir, "training_history.csv"), append=False
    )
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(output_dir, "tensorboard"), update_freq="epoch"
    )
    qwk_cb = QWKCallback(val_ds, num_classes=num_classes)
    # QWKCallback must run before the checkpoint/early-stop so logs['val_qwk']
    # exists when they fire.
    return [qwk_cb, checkpoint, early_stop, reduce_lr, csv_logger, tensorboard]


def run_phase(model, train_ds, val_ds, callbacks, epochs, lr, label, train_steps):
    import tensorflow as tf
    from ordinal import CORALLoss, OrdinalAccuracy

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=CORALLoss(num_classes=5),
        metrics=[OrdinalAccuracy()],
    )
    print("=" * 70)
    print(f" {label}  |  LR={lr}  |  max epochs={epochs}  |  steps/epoch={train_steps}")
    print("=" * 70)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=train_steps,
        callbacks=callbacks,
        verbose=2,
    )
    return history


def main():
    args = parse_args()
    set_seed(SEED)

    import tensorflow as tf
    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision: mixed_float16")

    strategy = make_strategy(args.gpus)
    print(f"Distribute strategy: {type(strategy).__name__} "
          f"({strategy.num_replicas_in_sync} replicas)")

    os.makedirs(args.output_dir, exist_ok=True)

    with strategy.scope():
        model, base_model = build_model(args.image_size, num_classes=5)
        if args.resume:
            model.load_weights(args.resume)
            print(f"Resumed weights from {args.resume} (skipping Phase 1)")
        else:
            model.summary(line_length=100)
            print(f"Backbone layers: {len(base_model.layers)}")

    train_ds, val_ds, test_ds, class_names, train_steps = load_datasets(
        args.data_root, args.image_size, args.batch_size, SEED, args.cache, 5, strategy
    )
    with open(os.path.join(args.output_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)

    config = vars(args)
    config["num_classes"] = 5
    config["num_replicas"] = strategy.num_replicas_in_sync
    config["global_batch_size"] = args.batch_size * strategy.num_replicas_in_sync
    config["class_names"] = class_names
    with open(os.path.join(args.output_dir, "train_config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    callbacks = make_callbacks(args.output_dir, val_ds, num_classes=5,
                               patience=args.patience)

    # ---- Phase 2: fine-tuning ----
    with strategy.scope():
        base_model.trainable = True
        for layer in base_model.layers[: args.unfreeze_from]:
            layer.trainable = False
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        trainable = sum(1 for l in base_model.layers if l.trainable)
        print(f"Phase 2: unfrozen backbone layers {trainable}/{len(base_model.layers)}")

    if not args.resume:
        t0 = time.time()
        with strategy.scope():
            run_phase(model, train_ds, val_ds, callbacks, args.head_epochs, args.head_lr,
                      "PHASE 1 — TRAINING CLASSIFICATION HEAD (backbone frozen)", train_steps)
        print(f"Phase 1 done in {(time.time() - t0) / 60:.1f} min")

        # ---- Phase 2: fine-tuning ----
        with strategy.scope():
            base_model.trainable = True
            for layer in base_model.layers[: args.unfreeze_from]:
                layer.trainable = False
            for layer in base_model.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False
            trainable = sum(1 for l in base_model.layers if l.trainable)
            print(f"Phase 2: unfrozen backbone layers {trainable}/{len(base_model.layers)}")

    t0 = time.time()
    with strategy.scope():
        run_phase(model, train_ds, val_ds, callbacks, args.finetune_epochs, args.finetune_lr,
                  "PHASE 2 — FINE-TUNING (top layers, BN frozen)", train_steps)
    print(f"Phase 2 done in {(time.time() - t0) / 60:.1f} min")

    print(f"\n✅ Training finished. Artifacts in {args.output_dir}")
    print("   best_model.keras  |  training_history.csv  |  tensorboard/  |  class_names.json")


if __name__ == "__main__":
    main()
