"""Export scripted demos as a Hugging Face-ready dataset — demos only.

Reads <demos>.npz + <demos>.json from scripts/collect_demos.py and
writes a self-contained dataset folder: episode JSONL split(s),
dataset_info.json, README dataset card, and checksums. No model
weights, episodes, GIFs, VLM calls, or secrets are included.

The export mirrors training-visible arrays exactly: lengths include
collect_demos.py's +16 hold-last extension, and actions remain
normalized joint targets in [-1, 1].

Upload is opt-in and dry-run by default. With --upload, the script
uses huggingface_hub (already locked) and a token supplied via
--token or HF_TOKEN; the token is never written to the export.

Usage:
    python scripts/export_hf_dataset.py --demos data/demos_cage_fc \
        --out data/hf_demos_cage_fc --val-frac 0.1
    python scripts/export_hf_dataset.py --demos data/demos_cage_fc \
        --out data/hf_demos_cage_fc --repo-id USER/grasping-demos \
        --upload
"""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import subprocess

import numpy as np


FORMAT_VERSION = "1.0"
TARGET_INDEX = 0
TARGET_COLOR = "red"


def obs_names(n_objects, contact_obs):
    """Flat-obs field names in GraspGymWrapper order."""
    names = ["qpos_0", "qpos_1", "qpos_2",
             "qvel_0", "qvel_1", "qvel_2",
             "target_x", "target_y",
             "target_half_x", "target_half_y"]
    for i in range(max(0, n_objects - 1)):
        names += [f"distractor_sorted_{i}_dx",
                  f"distractor_sorted_{i}_dy"]
    if contact_obs:
        names += ["links_touching_norm", "enclosed"]
    return names


def load_demos(demos):
    """Load demo arrays + collection metadata from a base path."""
    base = pathlib.Path(demos)
    npz_path = base.with_suffix(".npz")
    json_path = base.with_suffix(".json")
    if not npz_path.exists():
        raise FileNotFoundError(f"missing demos array: {npz_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"missing demos metadata: {json_path}")
    data = np.load(npz_path)
    meta = json.loads(json_path.read_text())
    return data, meta, npz_path, json_path


def validate_demos(data, meta):
    """Check shapes, lengths, ranges, and metadata consistency."""
    for key in ("obs", "actions", "lengths"):
        if key not in data:
            raise ValueError(f"demos .npz missing '{key}'")
    obs, actions = data["obs"], data["actions"]
    lengths = np.asarray(data["lengths"], dtype=int)
    fc_steps = (np.asarray(data["fc_steps"], dtype=int)
                if "fc_steps" in data else None)
    if obs.ndim != 3 or actions.ndim != 3:
        raise ValueError("obs/actions must be (episodes, steps, dim)")
    if len(obs) != len(actions) or len(obs) != len(lengths):
        raise ValueError("episode count differs across obs/actions/lengths")
    if len(obs) == 0:
        raise ValueError("no episodes to export")
    if actions.shape[2] != 3:
        raise ValueError("actions must have 3 joint-target dims")
    if np.any(lengths <= 0) or np.any(lengths > obs.shape[1]):
        raise ValueError("lengths fall outside the padded arrays")
    if not np.all(np.isfinite(obs)) or not np.all(np.isfinite(actions)):
        raise ValueError("obs/actions contain non-finite values")
    for i, length in enumerate(lengths):
        seg = actions[i, :int(length)]
        if seg.min() < -1.0 or seg.max() > 1.0:
            raise ValueError(f"episode {i} has actions outside [-1, 1]")
    if fc_steps is not None:
        if len(fc_steps) != len(obs):
            raise ValueError("fc_steps length differs from episode count")
        for i, (fs, length) in enumerate(zip(fc_steps, lengths)):
            if fs != -1 and not 0 <= fs < length:
                raise ValueError(f"episode {i} has fc_step outside length")
    for key in ("demonstrator", "keep", "n_objects", "contact_obs"):
        if key not in meta:
            raise ValueError(f"demos .json missing '{key}'")
    if int(meta.get("episodes_kept", len(obs))) != len(obs):
        raise ValueError("metadata episodes_kept differs from .npz episodes")
    expected_dim = len(obs_names(int(meta["n_objects"]),
                                 bool(meta["contact_obs"])))
    if obs.shape[2] != expected_dim:
        raise ValueError("obs dim does not match n_objects/contact_obs")
    return obs, actions, lengths, fc_steps


def split_episodes(n, val_frac, split_seed):
    """Deterministic train/validation episode indices."""
    if not 0.0 <= val_frac < 1.0:
        raise ValueError("--val-frac must be in [0, 1)")
    if val_frac == 0.0:
        return {"train": list(range(n))}
    n_val = int(n * val_frac)
    if n < 2 or n_val == 0 or n_val >= n:
        raise ValueError(f"cannot split {n} episodes with val-frac={val_frac}")
    order = np.random.RandomState(split_seed).permutation(n).tolist()
    return {"train": sorted(order[:n - n_val]),
            "validation": sorted(order[n - n_val:])}


def git_commit():
    """Best-effort source commit for provenance; None if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            timeout=10, check=True)
    except Exception:
        return None
    return out.stdout.strip() or None


def sha256_file(path):
    """SHA-256 hex digest for an export file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def size_category(n_episodes):
    """HF size bucket for episode-row counts."""
    if n_episodes < 1000:
        return "n<1K"
    if n_episodes < 10000:
        return "1K<n<10K"
    if n_episodes < 100000:
        return "10K<n<100K"
    return "100K<n<1M"


def write_split(path, episodes):
    """Write one episode-per-line JSONL split."""
    with open(path, "w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")


def dataset_card(name, info, splits, meta):
    """README dataset card: scope, schema, provenance, limits."""
    lines = [
        "---",
        "license: mit",
        "task_categories: [robotics]",
        ("tags: [robotics, simulation, behavior-cloning, "
         "diffusion-policy, whole-arm-grasping]"),
        f"size_categories: [{size_category(info['episodes'])}]",
        "---",
        f"# {name}",
        "",
        "Simulation-only scripted demonstrations for planar whole-arm",
        "grasping. Each JSONL row is one episode of flat simulator",
        "observations and normalized joint-target actions.",
        "",
        "## Intended use",
        "",
        ("- Train/evaluate closed-loop state-based cloning policies "
         "(BC, diffusion) in this repo's MuJoCo task."),
        "- Not robot hardware data; no images, language labels, rewards,",
        "  model weights, or human subjects.",
        "",
        "## Splits",
        "",
    ]
    for split, idx in splits.items():
        lines.append(f"- `{split}.jsonl`: {len(idx)} episodes")
    lines += [
        "",
        "## Row schema",
        "",
        ("- `episode_id`, `split`, `demonstrator`, `keep`, `n_objects`, "
         "`contact_obs`, `target_index`, `target_color`"),
        "- `length`: true training-visible steps (includes +16 hold-last",
        "  extension from `collect_demos.py`)",
        "- `fc_step`: force-closure step index, or -1",
        "- `obs_names`: flat observation field order",
        "- `obs`: `length x obs_dim` floats (raw simulator values)",
        "- `actions`: `length x 3` joint targets normalized to [-1, 1]",
        "  (multiply by 2.6 for simulator joint targets)",
        "",
        "## Provenance",
        "",
        f"- Source demos: `{info['source_npz']}` + `{info['source_json']}`",
        (f"- Collection: {meta['demonstrator']} demonstrator, keep="
         f"{meta['keep']}, episodes_run={meta.get('episodes_run')}, "
         f"episodes_kept={meta.get('episodes_kept')}, "
         f"fc_count={meta.get('fc_count')}"),
        f"- Format version: {FORMAT_VERSION}",
        f"- Exported (UTC): {info['created_utc']}",
        f"- Source commit: {info.get('git_commit') or 'unknown'}",
        "",
        "## Load",
        "",
        "```python",
        'from datasets import load_dataset',
        ("ds = load_dataset('json', "
         "data_files={'train': 'train.jsonl'})"),
        "```",
        "",
    ]
    return "\n".join(lines)


def export_dataset(demos, out, name=None, val_frac=0.0, split_seed=0,
                   force=False):
    """Build the dataset folder and return its dataset_info dict."""
    data, meta, npz_path, json_path = load_demos(demos)
    obs, actions, lengths, fc_steps = validate_demos(data, meta)
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / "train.jsonl"
    if train_path.exists() and not force:
        raise FileExistsError(f"{train_path} exists (pass --force to redo)")

    splits = split_episodes(len(lengths), val_frac, split_seed)
    names = obs_names(int(meta["n_objects"]), bool(meta["contact_obs"]))
    files = {}
    for split, idx in splits.items():
        rows = [{
            "episode_id": int(i),
            "split": split,
            "demonstrator": meta["demonstrator"],
            "keep": meta["keep"],
            "n_objects": int(meta["n_objects"]),
            "contact_obs": bool(meta["contact_obs"]),
            "target_index": TARGET_INDEX,
            "target_color": TARGET_COLOR,
            "length": int(lengths[i]),
            "fc_step": (int(fc_steps[i]) if fc_steps is not None else -1),
            "obs_names": names,
            "obs": obs[i, :int(lengths[i])].tolist(),
            "actions": actions[i, :int(lengths[i])].tolist(),
        } for i in idx]
        path = out / f"{split}.jsonl"
        write_split(path, rows)
        files[path.name] = {"episodes": len(rows), "bytes": path.stat().st_size}

    info = {
        "name": name or npz_path.stem,
        "format_version": FORMAT_VERSION,
        "created_utc": (datetime.datetime.now(datetime.timezone.utc)
                        .replace(microsecond=0).isoformat()),
        "source_npz": npz_path.name,
        "source_json": json_path.name,
        "collection": meta,
        "episodes": int(len(lengths)),
        "steps": int(lengths.sum()),
        "obs_dim": int(obs.shape[2]),
        "action_dim": 3,
        "obs_names": names,
        "splits": {k: {"episodes": len(v), "episode_ids": v}
                   for k, v in splits.items()},
        "split_seed": int(split_seed),
        "val_frac": float(val_frac),
        "files": files,
        "git_commit": git_commit(),
    }
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n")
    (out / "README.md").write_text(dataset_card(info["name"], info, splits,
                                                meta))
    sums = []
    for fname in sorted(p.name for p in out.iterdir()
                        if p.is_file() and p.name != "checksums.sha256"):
        sums.append(f"{sha256_file(out / fname)}  {fname}\n")
    (out / "checksums.sha256").write_text("".join(sums))
    return info


def resolve_token(token, env=None):
    """Upload token from --token or HF_TOKEN (never written to disk)."""
    env = os.environ if env is None else env
    value = token or env.get("HF_TOKEN")
    if not value:
        raise SystemExit("[export] --upload needs --token or HF_TOKEN")
    return value


def upload_dataset(folder, repo_id, token, private=False):
    """Create/update an HF dataset repo and upload the export folder."""
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise SystemExit("[export] huggingface_hub is required for --upload") \
            from e
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private,
                    exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=str(folder),
                      repo_type="dataset")
    print(f"[export] uploaded {folder} -> https://huggingface.co/datasets/{repo_id}")


def main(argv=None):
    """CLI: local export by default; --upload publishes with a token."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--val-frac", type=float, default=0.0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--token", default=None)
    args = ap.parse_args(argv)

    info = export_dataset(args.demos, args.out, name=args.name,
                          val_frac=args.val_frac,
                          split_seed=args.split_seed, force=args.force)
    print(f"[export] {info['episodes']} episodes, {info['steps']} steps -> "
          f"{args.out}")
    if not args.upload:
        print("[export] dry-run: no upload (add --repo-id/--upload)")
        return info
    if not args.repo_id:
        raise SystemExit("[export] --upload needs --repo-id")
    token = resolve_token(args.token)
    upload_dataset(args.out, args.repo_id, token, private=args.private)
    return info


if __name__ == "__main__":
    main()
