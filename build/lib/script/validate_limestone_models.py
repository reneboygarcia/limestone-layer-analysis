#!/usr/bin/env python3
"""
Validate limestone interpolation models with stronger spatial validation.

- Input: limestone_layers_phase_1.csv (same as main analysis)
- Methods compared:
  * Baseline isotropic IDW (grid-search hyperparameters)
  * Enhanced: Trend plane + anisotropic IDW of residuals

- Validation schemes:
  * Standard LOOCV (baseline only, reference)
  * Spatially buffered LOOCV (sensitivity over multiple buffer radii)
  * Spatial block CV (3x3 grid blocks, nested tuning on training folds)

- Output:
  * Writes CSV summary: limestone_validation_results.csv
  * Prints compact tables to stdout

Note: Uses functions from analyze_limestone.py; no external deps required.
"""
from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

# Import core algorithms from the existing script
import analyze_limestone as al


@dataclass
class Point:
    x: float
    y: float
    value: float


def read_rows(csv_path: str) -> List[Dict[str, Any]]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            rows.append(r)
    return rows


def parse_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip().strip('"').strip("'")
    if s == "":
        return None
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def derive_points(rows: List[Dict[str, Any]]) -> Tuple[List[Point], List[Point]]:
    """Return (rock_points, thick_points) from raw CSV rows."""
    rock_points: List[Point] = []
    thick_points: List[Point] = []

    for r in rows:
        ph = parse_float(r.get("ph_elev"))
        pt = parse_float(r.get("pt_elev"))
        y = parse_float(r.get("northing_coord_y"))
        x = parse_float(r.get("easting_coord_x"))
        thick = parse_float(r.get("limestone_thickness"))
        sound = parse_float(r.get("sounding_beg_limestone"))

        # Derived rock top elevation (measured when available)
        if ph is not None and sound is not None:
            rock_top = ph - sound
        else:
            if ph is not None and pt is not None and (ph - pt) > 1e-6:
                rock_top = pt
            else:
                rock_top = None

        if x is None or y is None:
            continue
        if rock_top is not None:
            rock_points.append(Point(x=x, y=y, value=rock_top))
        if thick is not None:
            thick_points.append(Point(x=x, y=y, value=thick))

    return rock_points, thick_points


def metrics(trues: List[float], preds: List[float]) -> Dict[str, float]:
    if not trues or not preds:
        return {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")}
    n = min(len(trues), len(preds))
    trues = trues[:n]
    preds = preds[:n]
    mae = sum(abs(p - t) for p, t in zip(preds, trues)) / n
    rmse = math.sqrt(sum((p - t) ** 2 for p, t in zip(preds, trues)) / n)
    y_mean = sum(trues) / n
    ss_tot = sum((t - y_mean) ** 2 for t in trues)
    ss_res = sum((t - p) ** 2 for t, p in zip(trues, preds))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def span_and_buffer(points: List[Point], frac: float, min_abs: float = 5.0) -> float:
    if not points:
        return min_abs
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    return max(min_abs, frac * span)


def buffered_cv_sweep(points: List[Point], method: str) -> List[Dict[str, Any]]:
    """Run buffered LOOCV over buffer fractions for a method ('baseline'|'enhanced')."""
    results: List[Dict[str, Any]] = []
    k_grid = [3, 4, 5, 6, 8, 10, 12]
    p_grid = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
    theta_grid = [float(t) for t in range(0, 180, 10)]  # 0,10,...,170
    ratio_grid = [1.0, 1.5, 2.0, 3.0]
    for frac in [0.10, 0.20, 0.30, 0.40]:
        buf = span_and_buffer(points, frac)
        if method == "baseline":
            k_sel, p_sel, met = al.buffered_loocv_idw(points, k_grid, p_grid, buf)
            results.append({
                "scheme": f"buffered_loocv@{int(frac*100)}%",
                "k": k_sel, "p": p_sel, "theta": "", "ratio": "",
                "buffer": buf,
                "MAE": met["MAE"], "RMSE": met["RMSE"], "R2": met["R2"],
            })
        else:
            k_sel, p_sel, th_sel, ra_sel, met = al.buffered_loocv_trend_idw(points, k_grid, p_grid, theta_grid, ratio_grid, buf)
            results.append({
                "scheme": f"buffered_loocv@{int(frac*100)}%",
                "k": k_sel, "p": p_sel, "theta": th_sel, "ratio": ra_sel,
                "buffer": buf,
                "MAE": met["MAE"], "RMSE": met["RMSE"], "R2": met["R2"],
            })
    return results


def grid_cells(points: List[Point], nx: int, ny: int) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    x_edges = [xmin + (xmax - xmin) * i / nx for i in range(nx + 1)]
    y_edges = [ymin + (ymax - ymin) * j / ny for j in range(ny + 1)]
    return x_edges, y_edges


def block_cv(points: List[Point], method: str, nx: int = 3, ny: int = 3) -> Dict[str, Any]:
    """Spatial block CV with nested tuning on training folds.

    - method: 'baseline' or 'enhanced'
    - Returns aggregated metrics and average selected params.
    """
    if len(points) < 5:
        return {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan"), "k": "", "p": "", "theta": "", "ratio": "", "folds": 0}

    k_grid = [3, 4, 5, 6, 8, 10, 12]
    p_grid = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
    theta_grid = [float(t) for t in range(0, 180, 10)]
    ratio_grid = [1.0, 1.5, 2.0, 3.0]

    x_edges, y_edges = grid_cells(points, nx, ny)

    # Accumulate across blocks
    trues_all: List[float] = []
    preds_all: List[float] = []
    k_sel_list: List[float] = []
    p_sel_list: List[float] = []
    th_sel_list: List[float] = []
    ra_sel_list: List[float] = []
    folds_used = 0

    for ix in range(nx):
        for iy in range(ny):
            x0, x1 = x_edges[ix], x_edges[ix + 1]
            y0, y1 = y_edges[iy], y_edges[iy + 1]
            test = [p for p in points if (x0 <= p.x <= x1 and y0 <= p.y <= y1)]
            if len(test) == 0:
                continue
            train = [p for p in points if p not in test]
            if len(train) < 3:
                continue

            # Nested tuning on training set
            if method == "baseline":
                k_sel, p_sel, _ = al.loocv_idw(train, k_grid, p_grid)
                k_sel_list.append(float(k_sel))
                p_sel_list.append(float(p_sel))
                th_sel = ""; ra_sel = ""
            else:
                # Use buffered LOOCV on training to select enhanced params (20% span)
                buf = span_and_buffer(train, 0.20)
                k_sel, p_sel, th_sel, ra_sel, _ = al.buffered_loocv_trend_idw(train, k_grid, p_grid, theta_grid, ratio_grid, buf)
                k_sel_list.append(float(k_sel))
                p_sel_list.append(float(p_sel))
                th_sel_list.append(float(th_sel))
                ra_sel_list.append(float(ra_sel))

            # Evaluate on held-out block
            for q in test:
                if method == "baseline":
                    pred = al.idw_predict((q.x, q.y), train, k=min(k_sel, len(train)), p=p_sel)
                else:
                    pred = al.predict_trend_plus_residual(train, (q.x, q.y), k_sel, p_sel, th_sel, ra_sel)
                if pred is None:
                    continue
                preds_all.append(pred)
                trues_all.append(q.value)
            folds_used += 1

    met = metrics(trues_all, preds_all)
    out: Dict[str, Any] = {
        "MAE": met["MAE"], "RMSE": met["RMSE"], "R2": met["R2"],
        "k": sum(k_sel_list) / len(k_sel_list) if k_sel_list else "",
        "p": sum(p_sel_list) / len(p_sel_list) if p_sel_list else "",
        "theta": sum(th_sel_list) / len(th_sel_list) if th_sel_list else "",
        "ratio": sum(ra_sel_list) / len(ra_sel_list) if ra_sel_list else "",
        "folds": folds_used,
        "nx": nx,
        "ny": ny,
        "n_test_points": len(trues_all),
    }
    return out


def print_table(title: str, rows: List[Dict[str, Any]], cols: List[Tuple[str, str]]):
    print(f"\n{title}")
    headers = [hdr for hdr, _ in cols]
    keys = [key for _, key in cols]
    widths = [max(len(hdr), 10) for hdr in headers]
    # derive column widths from data
    for i, key in enumerate(keys):
        for r in rows:
            v = r.get(key, "")
            s = f"{v:.3f}" if isinstance(v, float) else str(v)
            widths[i] = max(widths[i], len(s))
    fmt = " | ".join(f"{{:{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for r in rows:
        vals = []
        for key in keys:
            v = r.get(key, "")
            if isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        print(fmt.format(*vals))


def write_summary_csv(path: str, all_rows: List[Dict[str, Any]]):
    if not all_rows:
        return
    fieldnames = list(all_rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)
    in_path = os.path.join(project_root, "input", "limestone_layers_phase_1_input.csv")
    # Timestamped output filename: mm-dd-yy-hh-mm
    ts = datetime.now().strftime("%m-%d-%y-%H-%M")
    out_path = os.path.join(project_root, "output", f"limestone_validation_results_{ts}.csv")

    rows = read_rows(in_path)
    rock_points, thick_points = derive_points(rows)

    print("Data summary:")
    print(f"  Rock points: {len(rock_points)} | Thick points: {len(thick_points)}")

    # Hyperparameter grids for reference
    k_grid = [3, 4, 5, 6, 8, 10, 12]
    p_grid = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]

    all_summary_rows: List[Dict[str, Any]] = []

    # 1) Standard LOOCV (baseline only)
    if len(rock_points) >= 3:
        k_r, p_r, m_r = al.loocv_idw(rock_points, k_grid, p_grid)
    else:
        k_r, p_r, m_r = 3, 2.0, {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")}

    if len(thick_points) >= 3:
        k_t, p_t, m_t = al.loocv_idw(thick_points, k_grid, p_grid)
    else:
        k_t, p_t, m_t = 3, 2.0, {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")}

    loocv_rows = [
        {"variable": "rock_head", "method": "baseline", "scheme": "loocv", "k": k_r, "p": p_r, "theta": "", "ratio": "", "buffer": "",
         "MAE": m_r["MAE"], "RMSE": m_r["RMSE"], "R2": m_r["R2"], "folds": len(rock_points)},
        {"variable": "thickness", "method": "baseline", "scheme": "loocv", "k": k_t, "p": p_t, "theta": "", "ratio": "", "buffer": "",
         "MAE": m_t["MAE"], "RMSE": m_t["RMSE"], "R2": m_t["R2"], "folds": len(thick_points)},
    ]

    # 2) Buffered LOOCV sensitivity (both methods)
    rock_buf_base = buffered_cv_sweep(rock_points, method="baseline")
    rock_buf_enh = buffered_cv_sweep(rock_points, method="enhanced")
    thick_buf_base = buffered_cv_sweep(thick_points, method="baseline")
    thick_buf_enh = buffered_cv_sweep(thick_points, method="enhanced")

    # 3) Spatial block CV (3x3 grid) with nested tuning
    rock_block_base = block_cv(rock_points, method="baseline", nx=3, ny=3)
    rock_block_enh = block_cv(rock_points, method="enhanced", nx=3, ny=3)
    thick_block_base = block_cv(thick_points, method="baseline", nx=3, ny=3)
    thick_block_enh = block_cv(thick_points, method="enhanced", nx=3, ny=3)

    # Aggregate all rows for CSV
    def add_rows(variable: str, method: str, items: List[Dict[str, Any]]):
        for it in items:
            row = {
                "variable": variable,
                "method": method,
                "scheme": it.get("scheme", ""),
                "k": it.get("k", ""),
                "p": it.get("p", ""),
                "theta": it.get("theta", ""),
                "ratio": it.get("ratio", ""),
                "buffer": it.get("buffer", ""),
                "grid": it.get("grid", ""),
                "MAE": it["MAE"],
                "RMSE": it["RMSE"],
                "R2": it["R2"],
                "folds": it.get("folds", ""),
                "n_test_points": it.get("n_test_points", ""),
            }
            all_summary_rows.append(row)

    add_rows("rock_head", "baseline", loocv_rows[0:1])
    add_rows("thickness", "baseline", loocv_rows[1:2])

    add_rows("rock_head", "baseline", rock_buf_base)
    add_rows("rock_head", "enhanced", rock_buf_enh)
    add_rows("thickness", "baseline", thick_buf_base)
    add_rows("thickness", "enhanced", thick_buf_enh)

    # Add block CV rows
    add_rows("rock_head", "baseline", [{"scheme": "block_cv_3x3", **rock_block_base, "grid": "3x3"}])
    add_rows("rock_head", "enhanced", [{"scheme": "block_cv_3x3", **rock_block_enh, "grid": "3x3"}])
    add_rows("thickness", "baseline", [{"scheme": "block_cv_3x3", **thick_block_base, "grid": "3x3"}])
    add_rows("thickness", "enhanced", [{"scheme": "block_cv_3x3", **thick_block_enh, "grid": "3x3"}])

    write_summary_csv(out_path, all_summary_rows)

    # Build compact console tables
    # Rock head table
    rock_rows = [r for r in all_summary_rows if r["variable"] == "rock_head"]
    rock_table = []
    for r in rock_rows:
        rock_table.append({
            "Method": r["method"],
            "Scheme": r["scheme"],
            "Params": f"k={r['k']}, p={r['p']}, th={r['theta']}, ra={r['ratio']}",
            "MAE": r["MAE"],
            "RMSE": r["RMSE"],
            "R2": r["R2"],
        })

    thick_rows = [r for r in all_summary_rows if r["variable"] == "thickness"]
    thick_table = []
    for r in thick_rows:
        thick_table.append({
            "Method": r["method"],
            "Scheme": r["scheme"],
            "Params": f"k={r['k']}, p={r['p']}, th={r['theta']}, ra={r['ratio']}",
            "MAE": r["MAE"],
            "RMSE": r["RMSE"],
            "R2": r["R2"],
        })

    print_table(
        "Rock head validation (all schemes)",
        rock_table,
        [("Method", "Method"), ("Scheme", "Scheme"), ("Params", "Params"), ("MAE", "MAE"), ("RMSE", "RMSE"), ("R2", "R2")],
    )
    print_table(
        "Thickness validation (all schemes)",
        thick_table,
        [("Method", "Method"), ("Scheme", "Scheme"), ("Params", "Params"), ("MAE", "MAE"), ("RMSE", "RMSE"), ("R2", "R2")],
    )

    print(f"\nWrote validation summary to: {out_path}")


if __name__ == "__main__":
    main()
