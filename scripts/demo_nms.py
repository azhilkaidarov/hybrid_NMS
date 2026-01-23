from __future__ import annotations

import sys
from pathlib import Path


def _add_src_to_path() -> None:
    """Позволяет запускать `scripts/demo_nms.py` без установки пакета."""
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> int:
    _add_src_to_path()
    from bounding_box_project import main as pkg_main

    return pkg_main()


if __name__ == "__main__":
    raise SystemExit(main())
