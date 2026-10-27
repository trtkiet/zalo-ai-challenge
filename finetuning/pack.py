"""Create a code/config-only portable archive; omit data, caches, environments and outputs."""
import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT.parent / "finetuning.zip"))
    args = parser.parse_args()
    destination = Path(args.output).expanduser().resolve()
    allowed_directories = {"configs", "requirements", "ftkit", "tests"}
    allowed_root = {"README.md", "SUBMISSION.md", "cli.py", "setup_env.py", "finetune.sh", "pack.py", ".gitignore", "VERIFICATION.md"}
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob("*")):
            relative = path.relative_to(ROOT)
            if not path.is_file() or (relative.name != ".gitignore" and any(part.startswith(".") or part == "__pycache__" for part in relative.parts)):
                continue
            if (len(relative.parts) == 1 and relative.name in allowed_root) or (relative.parts[0] in allowed_directories and path.suffix in (".py", ".yaml", ".txt")):
                archive.write(path, Path("finetuning") / relative)
    print(destination)


if __name__ == "__main__":
    main()
