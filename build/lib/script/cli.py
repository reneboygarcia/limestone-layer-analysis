#!/usr/bin/env python3
"""
Console CLI for Limestone Layer Analysis.

Features
- Creative ASCII art banner
- Cross-platform Downloads folder detection
- Interactive CSV selection (Downloads, project input/, or browse any path)
- Prompts for site location and depth datum per geotech workflow
- Calls analyze_limestone.run_analysis()

No external dependencies.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

try:
    # Local import when installed as a package
    from .analyze_limestone import run_analysis, _banner
except ImportError:  # pragma: no cover
    # Fallback if run as a script directly
    sys.path.append(str(Path(__file__).resolve().parent))
    from analyze_limestone import run_analysis, _banner  # type: ignore


def _downloads_dir() -> Path:
    """Best-effort cross-platform Downloads directory detection."""
    # macOS/Linux default
    home = Path.home()
    candidates = [home / "Downloads"]
    # Windows common location
    if os.name == "nt":
        userprofile = os.environ.get("USERPROFILE")
        if userprofile:
            candidates.append(Path(userprofile) / "Downloads")
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    # Fallback to project input/ if exists else home
    proj_input = Path(__file__).resolve().parents[1] / "input"
    return proj_input if proj_input.exists() else home


def _project_default_io() -> tuple[Path, Path]:
    here = Path(__file__).resolve()
    project_root = here.parents[1]
    return project_root / "input", project_root / "output"


def _list_csvs(dir_path: Path) -> List[Path]:
    if not dir_path.exists() or not dir_path.is_dir():
        return []
    return sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() == ".csv"])


def _prompt(msg: str, default: Optional[str] = None) -> str:
    if default:
        val = input(f"{msg} [{default}]: ")
        return val.strip() or default
    return input(f"{msg}: ").strip()


def _choose_csv() -> Path:
    downloads = _downloads_dir()
    project_input, _project_output = _project_default_io()

    while True:
        print("\nWhere is your input CSV?")
        print(f"  1) Downloads folder  -> {downloads}")
        print(f"  2) Project input/    -> {project_input}")
        print("  3) Enter a file path")
        print("  4) Browse a directory path")
        print("  q) Quit")
        choice = input("> ").strip().lower()

        if choice == "1":
            options = _list_csvs(downloads)
            if not options:
                print("No CSV files found in Downloads.")
                continue
            return _select_from_list(options)
        elif choice == "2":
            options = _list_csvs(project_input)
            if not options:
                print("No CSV files found in project input/.")
                continue
            return _select_from_list(options)
        elif choice == "3":
            path = Path(_prompt("Enter CSV full path"))
            if path.exists() and path.is_file():
                return path
            print("Path not found or not a file. Try again.")
        elif choice == "4":
            dirp = Path(_prompt("Enter directory to browse", str(downloads)))
            if not dirp.exists() or not dirp.is_dir():
                print("Directory not found. Try again.")
                continue
            options = _list_csvs(dirp)
            if not options:
                print("No CSV files found in that directory.")
                continue
            return _select_from_list(options)
        elif choice in ("q", "quit", "exit"):
            print("Aborting.")
            sys.exit(1)
        else:
            print("Invalid choice. Try again.")


def _select_from_list(paths: List[Path]) -> Path:
    print("\nSelect CSV:")
    for i, p in enumerate(paths, 1):
        print(f"  {i:2d}) {p.name}")
    while True:
        sel = _prompt("Enter number")
        if sel.isdigit():
            idx = int(sel)
            if 1 <= idx <= len(paths):
                return paths[idx - 1]
        print("Invalid selection. Try again.")


def main(argv: Optional[List[str]] = None) -> int:
    _banner()
    argv = argv if argv is not None else sys.argv[1:]

    # Ask required geotechnical context
    location = _prompt("Site location (coords/site name/map ref)")
    datum = _prompt("Reference datum for depths (e.g., RL, MSL, benchmark)")

    # Pick input CSV interactively
    input_csv = _choose_csv()

    # Decide output directory (project output/ by default)
    _project_input, project_output = _project_default_io()
    out_dir_default = str(project_output)
    out_dir = Path(_prompt("Output directory", out_dir_default))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run analysis
    output_csv = run_analysis(str(input_csv), str(out_dir), location=location or None, datum=datum or None)

    print("\nDone.")
    print(f"Result: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
