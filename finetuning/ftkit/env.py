"""Load the toolkit's .ENV defaults without executing shell code or adding dependencies."""
import argparse
import os
from pathlib import Path
import re
import shlex
import sys


VARIABLE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")
ASSIGNMENT = re.compile(r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)")


def find_env_file(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    explicit = os.environ.get("LIFEMATE_ENV_FILE")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Environment file does not exist: {path}")
        return path
    return next((path for path in (root / ".ENV", root.parent / ".ENV") if path.is_file()), None)


def parse_value(raw, location):
    """Return a dotenv value and whether $VARIABLE expansion is allowed."""
    if not raw or raw[0] not in ("'", '"'):
        return re.split(r"\s+#", raw, maxsplit=1)[0].rstrip(), True
    quote = raw[0]
    chars, index = [], 1
    while index < len(raw):
        char = raw[index]
        if char == quote:
            tail = raw[index + 1:].strip()
            if tail and not tail.startswith("#"):
                raise ValueError(f"Unexpected text after quoted value at {location}")
            return "".join(chars), quote == '"'
        if quote == '"' and char == "\\" and index + 1 < len(raw):
            following = raw[index + 1]
            if following in ('"', "\\"):
                chars.append(following)
                index += 2
                continue
        chars.append(char)
        index += 1
    raise ValueError(f"Unclosed quoted value at {location}")


def load_env_file(root=None):
    """Set missing environment variables; preserve explicitly exported overrides.

    Supports KEY=value, optional export, comments, quotes and $VAR/${VAR}.
    Single quotes are literal. Expansion uses the existing environment and
    earlier assignments; command substitution is never executed.
    """
    path = find_env_file(root)
    if path is None:
        return {}
    values = dict(os.environ)
    if "HOME" not in values:
        try:
            values["HOME"] = str(Path.home())
        except RuntimeError:
            pass  # Only require HOME if a value actually references it.
    added = {}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        location = f"{path}:{number}"
        match = ASSIGNMENT.fullmatch(line)
        if match is None:
            raise ValueError(f"Expected KEY=value at {location}")
        key, raw = match.groups()
        value, expand = parse_value(raw, location)
        if key in os.environ:
            continue
        if expand:
            def replace(match):
                name = match.group(1) or match.group(2)
                if name not in values:
                    raise ValueError(f"Undefined environment variable {name} at {location}")
                return values[name]
            value = VARIABLE.sub(replace, value)
        values[key] = value
        added[key] = value
    # Apply atomically so a malformed file cannot leave partial settings behind.
    os.environ.update(added)
    return added


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shell", action="store_true", help="Emit safely quoted Bash exports for the launcher")
    args = parser.parse_args()
    try:
        added = load_env_file()
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.shell:
        for key, value in added.items():
            print(f"export {key}={shlex.quote(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
