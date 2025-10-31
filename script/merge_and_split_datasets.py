"""Merge several fridge datasets, tag source, split into train/test, and report counts.

Usage:
    python scripts/merge_and_split_datasets.py 

Options:
    --inputs FILE LABEL [FILE LABEL ...]  : pairs of filepath and label (optional)
    --test-size FLOAT (default 0.1)
    --seed INT (default 42)
    --outdir DIR (default ./outputs)

The script will search common locations for the expected filenames if not provided.
"""
from pathlib import Path
import json
import argparse
import random
import sys
from collections import Counter, defaultdict

# default filenames (basename -> default label)
DEFAULT_FILES = {
    "fridge_dataset_boolq_cleaned_g.json": "boolq",
    "fridge_dataset_piqa_cleaned.json": "piqa",
    "fridge_dataset_hellaswag_cleaned_g.json": "hellaswag",
    "fridge_dataset_v3.1_clean.json": "base",
}

SEARCH_DIRS = [
    Path("."),
    Path("datasets"),
    Path("datasets/open_chat/generative/categorical"),
    Path("OpenChat"),
]


def find_file_candidates(basename):
    candidates = []
    for d in SEARCH_DIRS:
        p = d / basename
        if p.exists():
            candidates.append(p)
    return candidates


def load_json_file(p: Path):
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading {p}: {e}")
        return []
    # normalize to list of dicts
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # common wrappers
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
        # maybe mapping of id->entry
        # if values are dicts, return list(values)
        vals = [v for v in data.values() if isinstance(v, dict)]
        if vals:
            return vals
        # otherwise return empty
        return []
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="*", help="Optional pairs: filepath label filepath label ...")
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # build list of (path,label)
    inputs = []
    if args.inputs and len(args.inputs) >= 2:
        # expect pairs
        if len(args.inputs) % 2 != 0:
            print("--inputs expects pairs: filepath label ...")
            sys.exit(1)
        it = iter(args.inputs)
        for p, l in zip(it, it):
            inputs.append((Path(p), l))
    else:
        # try to find defaults
        for name, label in DEFAULT_FILES.items():
            cand = find_file_candidates(name)
            if cand:
                # pick first candidate
                print(f"Found {cand[0]} for {name} -> label {label}")
                inputs.append((cand[0], label))
            else:
                print(f"Warning: {name} not found in search dirs")

    if not inputs:
        print("No input datasets found. Provide --inputs or place files in search dirs.")
        sys.exit(1)

    combined = []
    source_counts = Counter()

    for p, label in inputs:
        items = load_json_file(p)
        print(f"Loaded {len(items)} entries from {p} (label={label})")
        for item in items:
            # ensure dict
            if not isinstance(item, dict):
                continue
            # do not overwrite existing source if present
            item.setdefault("source", label)
            # add source anyway to be explicit
            item["source"] = label
            combined.append(item)
        source_counts[label] += len(items)

    print(f"Total combined entries: {len(combined)}")

    # deterministic shuffle
    random.seed(args.seed)
    random.shuffle(combined)

    n = len(combined)
    test_n = max(1, int(n * args.test_size))
    test = combined[:test_n]
    train = combined[test_n:]

    # write files
    merged_path = outdir / "merged_dataset.json"
    train_path = outdir / "train.json"
    test_path = outdir / "test.json"

    with merged_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    with train_path.open("w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with test_path.open("w", encoding="utf-8") as f:
        json.dump(test, f, ensure_ascii=False, indent=2)

    print(f"Wrote merged -> {merged_path}")
    print(f"Wrote train -> {train_path} ({len(train)} items)")
    print(f"Wrote test -> {test_path} ({len(test)} items)")

    # analyze test set: tag counts and source x tag counts
    tag_counts = Counter()
    source_tag_counts = defaultdict(Counter)
    for item in test:
        tag = item.get("tag") or item.get("label") or "unknown"
        tag_counts[tag] += 1
        source_tag_counts[item.get("source", "unknown")][tag] += 1

    print("\nTest set summary:")
    for tag, cnt in tag_counts.items():
        print(f"  tag={tag}: {cnt}")

    print("\nBreakdown by source:")
    for src, c in source_tag_counts.items():
        print(f"  source={src}")
        for tag, cnt in c.items():
            print(f"    {tag}: {cnt}")

    # print specifically relevant vs irrelevant
    irrelevant = tag_counts.get("irrelevant", 0)
    relevant = sum(cnt for t, cnt in tag_counts.items() if t != "irrelevant")
    print(f"\nTest set: irrelevant={irrelevant}, relevant={relevant}, total={len(test)}")

    print("Source counts discovered:")
    for src, cnt in source_counts.items():
        print(f"  {src}: {cnt}")

    print("Done.")

if __name__ == "__main__":
    main()
