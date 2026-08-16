# HPC Notes — DR-Grade Training

Operational notes for running `train.py` on a 2× Tesla V100-16GB node (no scheduler,
bare-metal GPU box, TF 2.16.1 in a venv).

## Environment

```bash
source setenv.sh          # activates .venv + adds NVIDIA/CUDA libs to LD_LIBRARY_PATH
./run_train.sh ...        # launches train.py in a detached tmux session "drgrade"
tmux attach -t drgrade    # watch; logs also tee'd to train.log
```

Hardware: 40 cores, 187 GB RAM, 2× V100-SXM2-16GB. Node is shared — another user's
process ("chuk") runs concurrently on GPU 0 and its footprint fluctuates (5–8.6 GB),
so memory headroom on GPU 0 is the binding constraint.

## Run commands

Final production run (fp32, the recommended config):

```bash
./run_train.sh --data-root split_dataset_cropped --output-dir outputs \
               --cache --batch-size 8
```

Resume an interrupted run (skips Phase 1, continues fine-tuning from the checkpoint):

```bash
./run_train.sh --data-root split_dataset_cropped --output-dir outputs \
               --cache --batch-size 8 --resume outputs/best_model.keras
```

Smoke test on a tiny dataset: `make_smoke_dataset.sh` then
`./run_train.sh --data-root smoke_dataset --output-dir /tmp/opencode/out_smoke --cache --head-epochs 2 --finetune-epochs 2`.

## Dataset pipeline (correct multi-GPU sharding)

TF 2.16's `image_dataset_from_directory` is **not auto-sharded** by MirroredStrategy —
running it directly yields the *same* images on every replica (2× redundant compute).
Fix: build the raw pipeline inside `strategy.distribute_datasets_from_function` and
`.shard()` per replica explicitly:

```python
def train_fn(ctx):
    ds = image_dataset_from_directory(..., shuffle=False)
    if ctx.num_replicas_in_sync > 1:
        ds = ds.shard(ctx.num_replicas_in_sync, ctx.input_pipeline_id)
    return (ds.shuffle(BUF).cache().batch(B, drop_remainder=True).repeat()
            .map(augment, num_parallel_calls=AUTO).prefetch(AUTO))

train_ds = strategy.distribute_datasets_from_function(train_fn)
```

- `steps_per_epoch` must be set explicitly: `total_train // (batch * replicas)`
  (29831 // 16 = **1864** at batch 8×2). A `.repeat()`-less / auto-sharded dataset
  instead reports 3728 steps (or hangs on `cardinality`).
- `val_ds`/`test_ds` stay **plain** (undistributed) — the QWK callback and evaluation
  consume them directly; `model.predict(plain_ds)` returns correctly aggregated
  predictions (shape `(N, classes)`), verified against batch-shape expectations.
- Use `--cache` on this box: 29,831×300×300×3×4 B ≈ 31 GB fits in the 187 GB RAM and
  drops epoch-1→later cost from ~360 s to ~100–140 s/epoch.

## fp16 vs fp32 — two NaN incidents (important)

`--mixed-precision` (fp16) is ~1.5× faster on V100 Tensor Cores but was **unstable for
this ordinal model**:

1. **Phase 1 (head only, fp16): NaN at epoch 7.** Root cause: the final logits layer ran
   in fp16; once predictions got confident, logits overflowed to `inf` → `loss=inf` →
   forward-inf → permanent NaN (dynamic loss scaling only rescales *gradients*; it cannot
   recover a forward `inf`). **Fix:** build the last Dense as `dtype="float32"` — this
   alone got past epoch 7 (val QWK 0.730 → 0.834).
2. **Phase 2 (unfrozen backbone, fp16): NaN at fine-tune epoch 18** despite the fp32
   logits fix (best val QWK 0.834). Root cause: intermediate *backbone* activations
   overflow in fp16 once ImageNet weights are fine-tuned toward larger values. The fp32
   logits layer can't help — the overflow is upstream.
3. **Resolution:** train the whole model in **fp32** (drop `--mixed-precision`).
   Worked flawlessly to completion (val QWK 0.8710, test QWK 0.8718). Because the fp16
   run's best checkpoint was clean (saved before the NaN), we **resumed from it** in fp32
   with `--resume` — skipping Phase 1 entirely and continuing fine-tuning.

**Guidance:** for this workload prefer fp32 on V100. If fp16 is required, keep the fp32
logits layer **and** add gradient clipping — but note clipping does *not* fix a forward
`inf`; it only caps large-but-finite gradients.

## Timings (2× V100, batch 8/replica, fp32)

| Phase | steps/epoch | ms/step | s/epoch | notes |
|-------|------------|---------|---------|-------|
| Phase 1 head (fp16, batch 16/GPU) | 932 | ~110 | ~103 | cache warm |
| Phase 2 finetune (fp32, batch 8/GPU) | 1864 | ~138 | ~255 | heavier compute |

- 2× faster than a single V100; the sharding fix made both GPUs genuinely work
  (~30–55% util each; before the fix the replicas did 2× redundant compute).
- fp16 batch 16/GPU showed memory ≈ fp32 batch 8/GPU — use batch 8/GPU in fp32 to keep
  GPU 0 safe from the concurrent user's memory spikes.

## QWK callback (epoch-stall gotcha)

The first version evaluated validation batches on the *main thread* during
`on_epoch_end`, inflating each epoch 3–4× (392 ms/step → 129 ms/step). Fix:
`self.model.predict(self.val_ds, verbose=0)` inside the callback — one fast, correctly
distributed pass instead of a Python per-batch loop.

## Evaluation

```bash
python evaluate.py --data-root split_dataset_cropped --model outputs/best_model.keras --output-dir outputs
```

- TTA on by default (`--no-tta` to disable): 2 flips + original, averaged.
- Batch 32 fp32 does **not** fit GPU 0 under the concurrent user's memory; run on the
  free card with `CUDA_VISIBLE_DEVICES=1 python evaluate.py ...` (or lower `--batch-size`).

## Data quality

- `preprocess.py` detected 15 all-black (corrupt) source images; they were removed from
  `split_dataset_cropped/` and logged in `corrupt_black_images.txt`. Final split:
  train 29,831 / val 6,795 / test 8,686 (45,312 total).
- Crop verification is numeric (no human-in-the-loop viewing): mean black fraction
  dropped 20.4% → 18.0%; dark images are preserved faithfully (65% → 65%), i.e. the
  largest-connected-component circular crop keeps genuine fundus content.

## Results summary

| Metric | Value |
|--------|-------|
| Best val QWK | 0.8710 (epoch 30/30, still improving) |
| Test QWK (TTA) | **0.8718** |
| Test accuracy (TTA) | **0.7026** |
