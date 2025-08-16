#!/usr/bin/env python3
"""
Console CLI for Limestone Layer Analysis.

Features
- Creative ASCII art banner (retro style)
- Cross-platform Downloads folder detection
- Menu-driven UX with simple, friendly prompts (no external deps)
- Interactive CSV selection (Downloads, project input/, or browse any path)
- Prompts for site location and depth datum per geotech workflow
- Configurable default output directory (persisted in ~/.limestone_cli.json)
- Calls analyze_limestone.run_analysis()

No external dependencies.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict
import threading
import itertools
import time
import json

try:
    # Local import when installed as a package
    from .analyze_limestone import run_analysis, _banner
except ImportError:  # pragma: no cover
    # Fallback if run as a script directly
    sys.path.append(str(Path(__file__).resolve().parent))
    from analyze_limestone import run_analysis, _banner  # type: ignore


class Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"


def _c(txt: str, color: str) -> str:
    if not sys.stdout.isatty():
        return txt
    return f"{color}{txt}{Color.RESET}"


class Glyph:
    """Common glyphs/emojis for a friendlier UI (fallbacks are simple ASCII)."""
    PLAY = "▶"
    GEAR = "⚙"
    INFO = "ℹ"
    EXIT = "⏻"
    FOLDER = "📂"
    FILE = "📄"
    CHART = "📊"
    NOTE = "📝"
    CHECK = "✓"
    CROSS = "✖"
    BACK = "↩"
    ARROW = "➤"
    HOURGLASS = "⏳"

    @staticmethod
    def safe(txt: str) -> str:
        # For non-TTY, just return text (avoids control codes). Emojis are fine.
        return txt


class _Spinner:
    """Tiny terminal spinner context manager (no external deps)."""

    def __init__(self, text: str = "Working…", color: str = Color.CYAN, interval: float = 0.08):
        self.text = text
        self.color = color
        self.interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._enabled = sys.stdout.isatty()

    def _run(self) -> None:
        frames = list("⠋⠙⠹⠸⠼⠴⠦⠇⠏") or ["-", "\\", "|", "/"]
        for ch in itertools.cycle(frames):
            if self._stop.is_set():
                break
            msg = f"\r{_c(ch + ' ' + self.text, self.color)}"
            sys.stdout.write(msg)
            sys.stdout.flush()
            time.sleep(self.interval)
        # Clear the line
        sys.stdout.write("\r" + " " * (len(self.text) + 4) + "\r")
        sys.stdout.flush()

    def __enter__(self):
        if self._enabled:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._enabled:
            self._stop.set()
            if self._thread is not None:
                self._thread.join()

def _retro_box(title: str, lines: list[str]) -> str:
    """Render a borderless, left-indented retro section.

    - Left indent for breathing room
    - UPPERCASE title with underline
    - Simple list of lines beneath
    """
    indent = "  "  # 2-space left indent
    ttl = title.strip().upper()
    underline = "─" * len(ttl)
    out = [
        _c(indent + ttl, Color.GREEN),
        _c(indent + underline, Color.GREEN),
    ]
    for l in lines:
        out.append(indent + "  " + l)
    return "\n".join(out)


# -----------------------------
# Config handling (persist simple preferences)
# -----------------------------

_CONFIG_FILENAME = ".limestone_cli.json"


def _config_path() -> Path:
    home = Path.home()
    return home / _CONFIG_FILENAME


def _load_config() -> Dict[str, str]:
    p = _config_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_config(cfg: Dict[str, str]) -> None:
    p = _config_path()
    try:
        p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception:
        pass


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


def _confirm(msg: str, default: bool = True) -> bool:
    d = "Y/n" if default else "y/N"
    ans = input(f"{_c('▶ ' + msg, Color.GREEN)} [{_c(d, Color.DIM)}]: ").strip().lower()
    if ans == "":
        return default
    return ans in ("y", "yes")


def _list_csvs(dir_path: Path) -> List[Path]:
    if not dir_path.exists() or not dir_path.is_dir():
        return []
    return sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() == ".csv"])


def _prompt(msg: str, default: Optional[str] = None) -> str:
    if default:
        val = input(f"{_c('▶ ' + msg, Color.GREEN)} [{_c(default, Color.DIM)}]: ")
        return val.strip() or default
    return input(f"{_c('▶ ' + msg, Color.GREEN)}: ").strip()


def _choose_csv() -> Path:
    downloads = _downloads_dir()
    project_input, _project_output = _project_default_io()

    while True:
        lines = [
            f"1] {Glyph.FOLDER} Downloads       -> {str(downloads)}",
            f"2] {Glyph.FOLDER} Project input/  -> {str(project_input)}",
            f"3] {Glyph.ARROW} Enter a file path",
            f"4] {Glyph.ARROW} Browse a directory path",

            _c(f"q] {Glyph.BACK} Back/Quit", Color.YELLOW) if sys.stdout.isatty() else "q] Back/Quit",
        ]
        print()
        print(_retro_box(f"{Glyph.FOLDER} Input CSV Selection", lines))
        choice = input(_c("› ", Color.GREEN)).strip().lower()

        if choice == "1":
            options = _list_csvs(downloads)
            if not options:
                print(_c("! No CSV files found in Downloads.", Color.YELLOW))
                continue
            return _select_from_list(options)
        elif choice == "2":
            options = _list_csvs(project_input)
            if not options:
                print(_c("! No CSV files found in project input/.", Color.YELLOW))
                continue
            return _select_from_list(options)
        elif choice == "3":
            path = Path(_prompt("Enter CSV full path"))
            if path.exists() and path.is_file():
                return path
            print(_c("✖ Path not found or not a file. Try again.", Color.RED))
        elif choice == "4":
            dirp = Path(_prompt("Enter directory to browse", str(downloads)))
            if not dirp.exists() or not dirp.is_dir():
                print(_c("✖ Directory not found. Try again.", Color.RED))
                continue
            options = _list_csvs(dirp)
            if not options:
                print(_c("! No CSV files found in that directory.", Color.YELLOW))
                continue
            return _select_from_list(options)
        elif choice in ("q", "quit", "exit", "b", "back"):
            print(_c(f"{Glyph.BACK} Back.", Color.YELLOW))
            raise KeyboardInterrupt  # signal to caller to handle back
        else:
            print(_c("✖ Invalid choice. Try again.", Color.RED))


def _select_from_list(paths: List[Path]) -> Path:
    title = "Select CSV"
    items = [f"{i:2d}) {p.name}" for i, p in enumerate(paths, 1)]
    print("\n" + _retro_box(title, items))
    while True:
        sel = _prompt("Enter number")
        if sel.isdigit():
            idx = int(sel)
            if 1 <= idx <= len(paths):
                return paths[idx - 1]
        print(_c("✖ Invalid selection. Try again.", Color.RED))


def _select_from_menu(title: str, options: List[str], exit_label: str = "Exit") -> Optional[int]:
    items = [f"{i:2d}) {opt}" for i, opt in enumerate(options, 1)] + [f"  q) {exit_label}"]
    print("\n" + _retro_box(title, items))
    while True:
        sel = _prompt("Enter choice number or q to quit")
        if sel.lower() in ("q", "quit", "exit"):
            return None
        if sel.isdigit():
            idx = int(sel)
            if 1 <= idx <= len(options):
                return idx
        print(_c("✖ Invalid selection. Try again.", Color.RED))


def _run_analysis_flow(cfg: Dict[str, str]) -> None:
    # Ask required geotechnical context
    print()
    print(_retro_box(f"🧭 Geotechnical Context", [
        "Provide minimal site metadata before analysis.",
        "- Location: coords/site name/map ref",
        "- Datum: RL/MSL/benchmark",
    ]))
    location = _prompt("Site location (coords/site name/map ref)")
    datum = _prompt("Reference datum for depths (e.g., RL, MSL, benchmark)")

    # Pick input CSV interactively
    try:
        input_csv = _choose_csv()
    except KeyboardInterrupt:
        print(_c("↩ Back to menu.", Color.YELLOW))
        return

    # Decide output directory (defaults to config or Downloads)
    default_base = cfg.get("default_output_dir") or str(_downloads_dir())
    out_dir = Path(_prompt("Output directory (root for results)", default_base))
    # Create timestamped results folder
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    final_out = out_dir / f"results_{ts}"
    final_out.mkdir(parents=True, exist_ok=True)

    # Run analysis
    try:
        with _Spinner(text="Running analysis…"):
            output_csv = run_analysis(str(input_csv), str(final_out), location=location or None, datum=datum or None)

        # Summarize outputs nicely
        print()
        results = []
        output_csv_p = Path(output_csv)
        if output_csv_p.exists():
            results.append(f"{Glyph.FILE} {output_csv_p.name}")
        cv = final_out / "cv_summary.csv"
        if cv.exists():
            results.append(f"{Glyph.CHART} {cv.name}")
        rpt = final_out / "report.md"
        if rpt.exists():
            results.append(f"{Glyph.NOTE} {rpt.name}")
        results.append(f"{Glyph.FOLDER} {final_out}")

        print(_retro_box(f"{Glyph.CHECK} Done", results))
    except Exception as e:
        print(_c(f"✖ Analysis failed: {e}", Color.RED))


def _configure_defaults_flow(cfg: Dict[str, str]) -> Dict[str, str]:
    print()
    lines = [
        "Configure CLI defaults (stored in ~/.limestone_cli.json)",
        f"Current default output: {cfg.get('default_output_dir', '(none)')}",
    ]
    print(_retro_box(f"{Glyph.GEAR} Configuration", lines))
    new_default = _prompt("Set default output base directory", cfg.get("default_output_dir", str(_downloads_dir())))
    if new_default.strip():
        cfg["default_output_dir"] = new_default.strip()
        _save_config(cfg)
        print(_c("✓ Saved.", Color.GREEN))
    return cfg


def _show_info_flow() -> None:
    print()
    lines = [
        "Limestone Layer Analysis",
        "- Interpolation: IDW + Trend-anisotropic residuals",
        "- Validation: LOOCV + spatially buffered LOOCV",
        "- Outputs: filled CSV, cv_summary.csv, report.md",
        "- Tip: Keep coordinates in meters for best results",
    ]
    print(_retro_box(f"{Glyph.INFO} About", lines))


def main(argv: Optional[List[str]] = None) -> int:
    # Wrap banner in green for retro terminal look
    if sys.stdout.isatty():
        print(Color.GREEN, end="")
    _banner()
    if sys.stdout.isatty():
        print(Color.RESET, end="")
    argv = argv if argv is not None else sys.argv[1:]

    cfg = _load_config()

    while True:
        choice = _select_from_menu(
            f"Main Menu",
            [
                f"{Glyph.PLAY} Run analysis",
                f"{Glyph.GEAR} Configure defaults",
                f"{Glyph.INFO} Show information",
            ],
            exit_label=f"{Glyph.EXIT} Exit",
        )

        if choice is None:
            print(_c("👋 Goodbye!", Color.YELLOW))
            return 0

        if choice == 1:
            _run_analysis_flow(cfg)
        elif choice == 2:
            cfg = _configure_defaults_flow(cfg)
        elif choice == 3:
            _show_info_flow()
        else:
            print(_c("✖ Unknown option.", Color.RED))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
