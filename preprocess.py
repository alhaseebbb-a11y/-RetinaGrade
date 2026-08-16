#!/usr/bin/env python
"""DR-Grade — fundus preprocessing: circular crop + square pad + resize.

Removes the black borders / side annotations around the retina (shown in the
literature to help DR grading), square-pads the result and resizes to the
model input size.  Outputs a parallel directory tree so the original data is
never modified.

Usage:
  python preprocess.py --src split_dataset --dst split_dataset_cropped --size 300
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image
from scipy import ndimage

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")


def parse_args():
    p = argparse.ArgumentParser(description="Fundus circular crop preprocessing")
    p.add_argument("--src", default="split_dataset")
    p.add_argument("--dst", default="split_dataset_cropped")
    p.add_argument("--size", type=int, default=300, help="Output square side in px")
    p.add_argument("--quality", type=int, default=95)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--threshold", type=int, default=10,
                   help="Grayscale threshold separating retina from background")
    return p.parse_args()


def find_fundus_bbox(img_arr, threshold):
    """Return tight square bbox (x0, y0, side) around the retina circle.

    Uses the largest connected bright component to ignore side text/annotations.
    """
    gray = img_arr.mean(axis=2).astype(np.uint8)
    mask = gray > threshold
    labels, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    biggest = int(np.argmax(sizes) + 1)
    ys, xs = np.nonzero(labels == biggest)
    if len(xs) < 200:  # degenerate / all-dark image
        return None
    cy, cx = ys.mean(), xs.mean()
    radius = float(np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2).max())
    half = max(radius, 8.0)
    x0 = int(max(0, cx - half))
    y0 = int(max(0, cy - half))
    side = int(min(img_arr.shape[1], img_arr.shape[0], 2 * half))
    return x0, y0, side


def process_one(job):
    src_path, dst_path, size, quality, threshold = job
    try:
        with Image.open(src_path) as im:
            im = im.convert("RGB")
            arr = np.asarray(im)
        h, w = arr.shape[:2]

        bbox = find_fundus_bbox(arr, threshold)
        if bbox is not None:
            x0, y0, side = bbox
            x0 = min(x0, w - side)
            y0 = min(y0, h - side)
            crop = arr[y0:y0 + side, x0:x0 + side]
        else:
            side = max(h, w)  # fallback: square-pad full image
            crop = arr

        ch, cw = crop.shape[:2]
        pad = abs(ch - cw)
        top, bottom = (pad // 2, pad - pad // 2) if ch < cw else (0, 0)
        left, right = (0, 0) if ch < cw else (pad // 2, pad - pad // 2)
        square = np.pad(crop, ((top, bottom), (left, right), (0, 0)), mode="constant")

        out = Image.fromarray(square).resize((size, size), Image.BILINEAR)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        out.save(dst_path, "JPEG", quality=quality)
        return 1, None
    except Exception as e:  # noqa: BLE001
        return 0, f"{src_path}: {e!r}"


def main():
    args = parse_args()
    if not os.path.isdir(args.src):
        sys.exit(f"Source dir not found: {args.src}")

    jobs = []
    for split in ("train", "val", "test"):
        src_split = os.path.join(args.src, split)
        if not os.path.isdir(src_split):
            print(f"  (skip) {split}: not found")
            continue
        for cls in sorted(os.listdir(src_split)):
            cls_dir = os.path.join(src_split, cls)
            if not os.path.isdir(cls_dir):
                continue
            dst_cls = os.path.join(args.dst, split, cls)
            for fname in os.listdir(cls_dir):
                if not fname.lower().endswith(IMG_EXTS):
                    continue
                stem = os.path.splitext(fname)[0]
                jobs.append((
                    os.path.join(cls_dir, fname),
                    os.path.join(dst_cls, f"{stem}.jpg"),
                    args.size, args.quality, args.threshold,
                ))

    print(f"Processing {len(jobs):,} images with {args.workers} workers "
          f"-> {args.dst} (size {args.size})")
    ok, failures = 0, []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            res, err = fut.result()
            ok += res
            if err:
                failures.append(err)
            if i % 5000 == 0 or i == len(jobs):
                print(f"  {i:,}/{len(jobs):,} done ({ok:,} ok)")
    print(f"✅ Done: {ok:,}/{len(jobs):,} images processed.")
    if failures:
        print(f"  {len(failures)} failures (first 10):")
        for f in failures[:10]:
            print("   ", f)


if __name__ == "__main__":
    main()
