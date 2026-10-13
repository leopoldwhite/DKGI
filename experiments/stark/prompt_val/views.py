"""Build and verify corpus2/<dataset>/ of the rg-only variant (p2): hardlinks of corpus/<dataset>/, except
DATA_FORMAT.txt, which lacks its last line. The original corpus is not changed."""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STARK = ROOT.parent
DATASETS = ("prime", "mag", "amazon")
REMOVED_LINE = b"Use shell tools or Python's json/csv modules to read these files.\n"


def modified_format(source):
    data = Path(source).read_bytes()
    lines = data.splitlines(keepends=True)
    if len(lines) < 9 or lines[8] != REMOVED_LINE or data.count(REMOVED_LINE) != 1:
        raise ValueError(f"Unexpected DATA_FORMAT.txt line 9: {source}")
    return b"".join(lines[:8] + lines[9:])


def corpus_files(directory):
    files = sorted(path.relative_to(directory) for path in directory.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"Empty corpus: {directory}")
    for relative in files:
        path = directory / relative
        if path.is_symlink() or not (
            relative.as_posix() in ("DATA_FORMAT.txt", "schema.json", "candidate_ids.txt")
            or (len(relative.parts) == 2 and relative.parts[0] in ("nodes", "edges"))
        ):
            raise ValueError(f"Unexpected corpus entry: {path}")
    return files


def verify_view(dataset):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    source = STARK / "corpus" / dataset
    target = STARK / "corpus2" / dataset
    files = corpus_files(source)
    if files != corpus_files(target):
        raise ValueError(f"Source and view file lists differ: {dataset}")
    if (target / "DATA_FORMAT.txt").read_bytes() != modified_format(source / "DATA_FORMAT.txt"):
        raise ValueError(f"Unexpected DATA_FORMAT.txt difference: {dataset}")
    entries = []
    for relative in files:
        old, new = source / relative, target / relative
        old_stat, new_stat = old.stat(), new.stat()
        linked = (old_stat.st_dev, old_stat.st_ino) == (new_stat.st_dev, new_stat.st_ino)
        # A hardlink is the same inode on the same device, so its bytes are the source bytes;
        # the one unlinked file, DATA_FORMAT.txt, was compared byte for byte above.
        if linked != (relative.name != "DATA_FORMAT.txt"):
            raise ValueError(f"Incorrect hardlink status: {new}")
        entries.append({"path": relative.as_posix(), "hardlinked": linked,
                        "source": {"device": old_stat.st_dev, "inode": old_stat.st_ino,
                                   "size": old_stat.st_size},
                        "view": {"device": new_stat.st_dev, "inode": new_stat.st_ino,
                                 "size": new_stat.st_size}})
    return {"dataset": dataset, "source": str(source), "view": str(target),
            "file_count": len(files), "removed_line_number": 9,
            "removed_line": REMOVED_LINE.decode().rstrip("\n"), "files": entries}


def build_views():
    manifests = {}
    for dataset in DATASETS:
        source, target = STARK / "corpus" / dataset, STARK / "corpus2" / dataset
        expected_format = modified_format(source / "DATA_FORMAT.txt")
        for relative in corpus_files(source):
            old, new = source / relative, target / relative
            new.parent.mkdir(parents=True, exist_ok=True)
            if new.exists():
                continue
            if relative.name == "DATA_FORMAT.txt":
                with new.open("xb") as output:
                    output.write(expected_format)
            else:
                os.link(old, new)
        manifest = verify_view(dataset)
        destination = ROOT / "view_manifests" / f"{dataset}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        if destination.exists():
            if destination.read_text() != serialized:
                raise ValueError(f"Existing view manifest changed: {destination}")
        else:
            destination.write_text(serialized)
        manifests[dataset] = str(destination)
        print(f"Verified {dataset}: {manifest['file_count']} files; manifest {destination}", flush=True)
    return manifests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Verify existing views without creating files")
    args = parser.parse_args()
    if args.verify:
        for dataset in DATASETS:
            manifest = verify_view(dataset)
            destination = ROOT / "view_manifests" / f"{dataset}.json"
            if manifest != json.loads(destination.read_text()):
                raise ValueError(f"View no longer matches manifest: {dataset}")
            print(f"Verified {dataset}: {manifest['file_count']} files", flush=True)
    else:
        build_views()


if __name__ == "__main__":
    main()
