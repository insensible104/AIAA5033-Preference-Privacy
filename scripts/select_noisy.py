"""Pure selector CLI: accepts ONLY a sanitized bundle, never clean data paths."""

import argparse
from pathlib import Path
from preference_privacy.experiment import read_json
from preference_privacy.runtime import sha256, write_json
from preference_privacy.selection import select_models


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if Path(a.output).exists():
        p.error("Selection output already exists")
    write_json(
        a.output,
        {
            "bundle_sha256": sha256(a.bundle),
            "decisions": select_models(read_json(a.bundle)),
        },
    )


if __name__ == "__main__":
    main()
