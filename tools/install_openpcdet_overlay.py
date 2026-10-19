#!/usr/bin/env python3
"""Check or install the M300 compatibility patch/configs into OpenPCDet.

Dry-run by default. Never downloads, builds, resets Git, or starts training.
Use a dedicated checkout of the pinned upstream commit, not a working project
with unrelated edits. Existing different configs are never overwritten.
"""
import argparse
from pathlib import Path
import shutil
import subprocess

PROJECT = Path(__file__).resolve().parents[1]
PIN = "233f849829b6ac19afb8af8837a0246890908755"


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args],
                          text=True, capture_output=True, check=False)


def install(root, apply=False):
    root = root.expanduser().resolve()
    if root == PROJECT or not (root / "pcdet").is_dir():
        raise ValueError("Select a separate OpenPCDet checkout with a pcdet directory")
    revision = git(root, "rev-parse", "HEAD")
    if revision.returncode or revision.stdout.strip() != PIN:
        raise ValueError(f"Expected OpenPCDet commit {PIN}; no files changed")
    patch = PROJECT / "patches/openpcdet-m300.patch"
    pending = []
    for source in sorted((PROJECT / "configs/openpcdet").glob("*/*.yaml")):
        target = root / "tools/cfgs" / source.parent.name / source.name
        if target.exists():
            if target.read_bytes() != source.read_bytes():
                raise FileExistsError(f"Different config exists: {target}; use a clean checkout")
        else:
            pending.append((source, target))
    # The pinned custom_dataset.py uses CRLF; patch transport uses LF.
    forward = git(root, "apply", "--ignore-space-change", "--check", str(patch))
    if forward.returncode:
        reverse = git(root, "apply", "--ignore-space-change", "--reverse", "--check", str(patch))
        if reverse.returncode:
            raise ValueError("Patch does not apply cleanly and is not already installed. "
                             "No reset/overwrite attempted.\n" + forward.stderr)
        patch_needed = False
    else:
        patch_needed = True
    print(f"OpenPCDet: {root}\nPinned commit: {PIN}")
    print(f"Patch: {'needed' if patch_needed else 'already installed'}; new configs: {len(pending)}")
    if not apply:
        print("DRY RUN: no changes. Add --apply to install.")
        return
    if patch_needed:
        result = git(root, "apply", "--ignore-space-change", str(patch))
        if result.returncode:
            raise RuntimeError(result.stderr)
    for source, target in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as destination, source.open("rb") as stream:
            shutil.copyfileobj(stream, destination)
    print("Overlay installed. No build, ROS node, or training was started.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpcdet", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        install(args.openpcdet, args.apply)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
