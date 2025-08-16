#!/usr/bin/env python3
"""
Hyperparameter grid exploration for limestone layer interpolation.

- Uses the existing utilities in analyze_limestone.py
- Scans the following grid (from user request):
  * k: 3, 5, 7, 10, 15
  * p: 1.0, 1.5, 2.0, 2.5, 3.0, 3.5
  * Buffer radius (m): 4×, 6×, 8× average point spacing (APS)
  * Trend: none, planar
  * Anisotropy θ (deg): 0, 30, 60
  * Anisotropy ratio (y/x): 1.0, 2.0

- Targets evaluated:
  * rock_top_elev_measured (derived)
  * limestone_thickness

- Validation: spatially buffered LOOCV

Outputs:
- CSV summary under output/: hypergrid_scan_{timestamp}.csv
- Console prints of top rows per target/trend by RMSE

No external dependencies.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

try:
    # When installed as a package/module
    from .analyze_limestone import (
        Point,
        read_rows,
        parse_float,
        euclidean,
        idw_predict,
        idw_predict_aniso,
        aniso_distance,
        fit_trend_plane,
        predict_trend,
        buffered_loocv_idw,
        buffered_loocv_trend_idw,
        write_rows,
    )
except Exception:  # pragma: no cover
    # Fallback if run directly as a script
    sys.path.append(os.path.dirname(__file__))
    from analyze_limestone import (  # type: ignore
        Point,
        read_rows,
        parse_float,
        euclidean,
        idw_predict,
        idw_predict_aniso,
        aniso_distance,
        fit_trend_plane,
        predict_trend,
        buffered_loocv_idw,
        buffered_loocv_trend_idw,
        write_rows,
    )


# -----------------------------
# Data loading helpers
# -----------------------------

def load_points_from_csv(csv_path: str) -> Tuple[List[Point], List[Point]]:
    """Load CSV and produce two point sets:
    - rock_points: using derived rock_top_elev_measured
    - thick_points: using limestone_thickness
    """
    rows = read_rows(csv_path)

    # Parse and derive
    for r in rows:
        r["ph_elev"] = parse_float(r.get("ph_elev"))
        r["pt_elev"] = parse_float(r.get("pt_elev"))
        r["northing_coord_y"] = parse_float(r.get("northing_coord_y"))
        r["easting_coord_x"] = parse_float(r.get("easting_coord_x"))
        r["limestone_thickness"] = parse_float(r.get("limestone_thickness"))
        r["sounding_beg_limestone"] = parse_float(r.get("sounding_beg_limestone"))

        if r["ph_elev"] is not None and r["sounding_beg_limestone"] is not None:
            r["rock_top_elev_measured"] = r["ph_elev"] - r["sounding_beg_limestone"]
        else:
            if (
                r["ph_elev"] is not None
                and r["pt_elev"] is not None
                and (r["ph_elev"] - r["pt_elev"]) > 1e-6
            ):
                r["rock_top_elev_measured"] = r["pt_elev"]
            else:
                r["rock_top_elev_measured"] = None

    rock_points: List[Point] = []
    thick_points: List[Point] = []
    for r in rows:
        x = r["easting_coord_x"]
        y = r["northing_coord_y"]
        if x is None or y is None:
            continue
        if r["rock_top_elev_measured"] is not None:
            rock_points.append(Point(x=float(x), y=float(y), value=float(r["rock_top_elev_measured"])) )
        if r["limestone_thickness"] is not None:
            thick_points.append(Point(x=float(x), y=float(y), value=float(r["limestone_thickness"])) )

    return rock_points, thick_points


def average_point_spacing(points: List[Point]) -> Optional[float]:
    """Estimate average nearest-neighbor spacing (meters)."""
    if len(points) < 2:
        return None
    dsum = 0.0
    cnt = 0
    coords = [(p.x, p.y) for p in points]
    for i, (xi, yi) in enumerate(coords):
        best = None
        for j, (xj, yj) in enumerate(coords):
            if i == j:
                continue
            d = euclidean((xi, yi), (xj, yj))
            if best is None or d < best:
                best = d
        if best is not None:
            dsum += best
            cnt += 1
    return dsum / max(1, cnt)


# -----------------------------
# CV routines for anisotropic no-trend IDW
# -----------------------------

def buffered_loocv_aniso_idw(
    points: List[Point],
    k_values: List[int],
    p_values: List[float],
    theta_values: List[float],
    ratio_values: List[float],
    buffer_radius: float,
) -> Tuple[int, float, float, float, Dict[str, float]]:
    """Spatially buffered LOOCV for anisotropic IDW without trend.
    Returns (k, p, theta, ratio, metrics).
    """
    if len(points) < 3:
        return 3, 2.0, 0.0, 1.0, {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")}

    xs = [pt.x for pt in points]
    ys = [pt.y for pt in points]
    vs = [pt.value for pt in points]

    best = None  # (mae, rmse, k, p, theta, ratio, metrics)

    for k in k_values:
        for p in p_values:
            for theta in theta_values:
                for ratio in ratio_values:
                    errors: List[float] = []
                    preds: List[float] = []
                    trues: List[float] = []
                    for i in range(len(points)):
                        xi, yi, zi = xs[i], ys[i], vs[i]
                        # Build training set excluding a buffer around the test point (anisotropic buffer)
                        train_idxs: List[int] = []
                        for j in range(len(points)):
                            if j == i:
                                continue
                            d = aniso_distance((xs[j], ys[j]), (xi, yi), theta, ratio)
                            if d >= buffer_radius:
                                train_idxs.append(j)
                        if len(train_idxs) < 1:
                            continue
                        train_pts = [Point(xs[j], ys[j], vs[j]) for j in train_idxs]
                        pred = idw_predict_aniso((xi, yi), train_pts, k=min(k, len(train_pts)), p=p, theta_deg=theta, ratio=ratio)
                        if pred is None:
                            continue
                        preds.append(pred)
                        trues.append(zi)
                        errors.append(pred - zi)
                    if not errors:
                        continue
                    mae = sum(abs(e) for e in errors) / len(errors)
                    rmse = (sum(e*e for e in errors) / len(errors)) ** 0.5
                    y_mean = sum(trues) / len(trues)
                    ss_tot = sum((y - y_mean) ** 2 for y in trues)
                    ss_res = sum((t - p_) ** 2 for t, p_ in zip(trues, preds))
                    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
                    key = (mae, rmse, k, p, theta, ratio, {"MAE": mae, "RMSE": rmse, "R2": r2})
                    if best is None or (mae < best[0]) or (mae == best[0] and rmse < best[1]):
                        best = key

    if best is None:
        return 3, 2.0, 0.0, 1.0, {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")}
    _, _, k_sel, p_sel, theta_sel, ratio_sel, metrics = best
    return k_sel, p_sel, theta_sel, ratio_sel, metrics


# -----------------------------
# Runner
# -----------------------------

def run_scan(input_csv: str, output_dir: str) -> str:
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_path = os.path.join(output_dir, f"hypergrid_scan_{ts}.csv")

    rock_points, thick_points = load_points_from_csv(input_csv)
    print(f"Loaded points: rock={len(rock_points)}, thickness={len(thick_points)}")

    # Grids from user
    k_grid = [3, 5, 7, 10, 15]
    p_grid = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    theta_grid = [0.0, 30.0, 60.0]
    ratio_grid = [1.0, 2.0]

    # Buffer radii from APS
    aps_rock = average_point_spacing(rock_points)
    aps_thick = average_point_spacing(thick_points)

    def _buffers(aps: Optional[float]) -> List[float]:
        if aps is None or not (aps > 0):
            # Fallback to 5 m as in main analysis
            return [5.0]
        return [4.0 * aps, 6.0 * aps, 8.0 * aps]

    buffers_rock = _buffers(aps_rock)
    buffers_thick = _buffers(aps_thick)

    print(f"Average spacing: rock≈{aps_rock:.2f} m, thickness≈{aps_thick:.2f} m" if (aps_rock and aps_thick) else "Average spacing: insufficient data for one or both targets")

    rows: List[Dict[str, Any]] = []

    def _append_rows(target: str, points: List[Point], buffers: List[float]):
        n_train = len(points)
        for buf in buffers:
            # Trend = none (anisotropic IDW without trend)
            k_a, p_a, th_a, ra_a, m_a = buffered_loocv_aniso_idw(points, k_grid, p_grid, theta_grid, ratio_grid, buf)
            rows.append({
                "target": target,
                "trend": "none",
                "k": k_a, "p": p_a, "theta": th_a, "ratio": ra_a,
                "buffer_radius": round(buf, 3),
                "MAE": m_a.get("MAE"), "RMSE": m_a.get("RMSE"), "R2": m_a.get("R2"),
                "n_train": n_train,
            })
            # Trend = planar (trend + anisotropic residual IDW)
            k_t, p_t, th_t, ra_t, m_t = buffered_loocv_trend_idw(points, k_grid, p_grid, theta_grid, ratio_grid, buf)
            rows.append({
                "target": target,
                "trend": "planar",
                "k": k_t, "p": p_t, "theta": th_t, "ratio": ra_t,
                "buffer_radius": round(buf, 3),
                "MAE": m_t.get("MAE"), "RMSE": m_t.get("RMSE"), "R2": m_t.get("R2"),
                "n_train": n_train,
            })

    if rock_points:
        _append_rows("rock_top_elevation", rock_points, buffers_rock)
    if thick_points:
        _append_rows("limestone_thickness", thick_points, buffers_thick)

    # Write results
    write_rows(out_path, [
        "target", "trend", "k", "p", "theta", "ratio", "buffer_radius", "MAE", "RMSE", "R2", "n_train"
    ], rows)

    # Print top configs per group (by RMSE asc)
    def _top_print(tt: str):
        grp = [r for r in rows if r["target"] == tt]
        if not grp:
            return
        print(f"\nTop configurations for {tt} (by RMSE):")
        top = sorted(grp, key=lambda r: (float('inf') if r["RMSE"] is None else r["RMSE"]))[:5]
        for r in top:
            print(f"  trend={r['trend']:<6} buf≈{r['buffer_radius']:<7} k={r['k']:<2} p={r['p']:<3} θ={r['theta']:<3} ratio={r['ratio']:<3} | RMSE={r['RMSE']:.3f}, MAE={r['MAE']:.3f}, R2={r['R2']:.3f}")

    _top_print("rock_top_elevation")
    _top_print("limestone_thickness")

    print(f"\nWrote hypergrid scan to: {out_path}")
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)
    input_csv = os.path.join(project_root, "input", "limestone_layers_phase_1_input.csv")
    output_dir = os.path.join(project_root, "output")

    try:
        run_scan(input_csv, output_dir)
    except Exception as e:
        print(f"✖ Scan failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
