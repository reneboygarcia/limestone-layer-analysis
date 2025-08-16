#!/usr/bin/env python3
"""
Analyze and fill limestone properties via spatial interpolation with validation.

Baseline:
- Interpolate rock head elevation (top of limestone) with isotropic IDW (grid-search LOOCV for k, p),
  then derive sounding_beg_limestone = ph_elev - rock_top.
- Interpolate limestone_thickness similarly.

Enhanced (default for filling):
- Fit a first-order trend plane z = a + b x + c y to measured values.
- Interpolate residuals with anisotropic IDW, tuning (k, p, orientation theta, anisotropy ratio)
  via spatially buffered LOOCV to mitigate spatial leakage.
- Combine trend + residual to predict.
- Bound predictions within observed ranges and enforce non-negativity.

Outputs:
- Writes an enhanced filled CSV with provenance flags (measured vs predicted_idw).

No external dependencies; uses only Python stdlib.
"""
from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any


@dataclass
class Point:
    x: float
    y: float
    value: float


def parse_float(v: Any) -> Optional[float]:
    """Parse a number possibly containing thousands separators and quotes.

    Returns None if v is empty/None.
    """
    if v is None:
        return None
    s = str(v).strip().strip('"').strip("'")
    if s == "":
        return None
    # Remove thousands separators if present
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def read_rows(csv_path: str) -> List[Dict[str, Any]]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            # Normalize/trim some fields
            if r.get("pile_type") is not None:
                r["pile_type"] = r["pile_type"].strip()
            rows.append(r)
    return rows


def write_rows(csv_path: str, fieldnames: List[str], rows: List[Dict[str, Any]]):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])


def idw_predict(
    xy: Tuple[float, float],
    known: List[Point],
    k: int,
    p: float,
) -> Optional[float]:
    """IDW prediction at location xy using up to k nearest neighbors and power p.

    - If a neighbor has zero distance, returns the average of coincident values.
    - Returns None if no known points are given.
    """
    n = len(known)
    if n == 0:
        return None
    # Distances to all
    dists = [(euclidean((pt.x, pt.y), xy), pt.value) for pt in known]
    dists.sort(key=lambda t: t[0])

    # Handle exact duplicates
    if dists and dists[0][0] == 0.0:
        # Average of all points at zero distance
        zeros = [v for (d, v) in dists if d == 0.0]
        return sum(zeros) / len(zeros)

    # Use up to k neighbors (exclude any with None values — shouldn't occur here)
    k_eff = max(1, min(k, n))
    neighbors = dists[:k_eff]

    # Weighted average
    num = 0.0
    den = 0.0
    for d, v in neighbors:
        # small epsilon to avoid div by zero (though zero handled above)
        w = 1.0 / ((d + 1e-12) ** p)
        num += w * v
        den += w
    return num / den if den > 0 else None


def loocv_idw(
    points: List[Point],
    k_values: List[int],
    p_values: List[float],
) -> Tuple[int, float, Dict[str, float]]:
    """LOOCV grid search for IDW. Returns best (k, p) and metrics for that choice.

    Metrics: MAE, RMSE, R2 (cross-validated)
    """
    if len(points) < 3:
        # Not enough for meaningful CV; fall back to defaults
        return 3, 2.0, {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")}

    # Precompute arrays
    xs = [pt.x for pt in points]
    ys = [pt.y for pt in points]
    vs = [pt.value for pt in points]

    best = None  # (mae, rmse, k, p, metrics)

    for k in k_values:
        # ensure k is valid for LOOCV (must be <= N-1)
        k_eff = min(k, max(1, len(points) - 1))
        for p in p_values:
            errors = []
            preds = []
            trues = []
            for i in range(len(points)):
                # Leave out i
                train = [Point(xs[j], ys[j], vs[j]) for j in range(len(points)) if j != i]
                pred = idw_predict((xs[i], ys[i]), train, k=k_eff, p=p)
                if pred is None:
                    continue
                preds.append(pred)
                trues.append(vs[i])
                errors.append(pred - vs[i])
            if not errors:
                continue
            # Metrics
            mae = sum(abs(e) for e in errors) / len(errors)
            rmse = math.sqrt(sum(e*e for e in errors) / len(errors))
            y_mean = sum(trues) / len(trues)
            ss_tot = sum((y - y_mean) ** 2 for y in trues)
            ss_res = sum((t - p_) ** 2 for t, p_ in zip(trues, preds))
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

            key = (mae, rmse, k_eff, p, {"MAE": mae, "RMSE": rmse, "R2": r2})
            if best is None:
                best = key
            else:
                # Primary minimize MAE, tie-breaker RMSE
                if (mae < best[0]) or (mae == best[0] and rmse < best[1]):
                    best = key

    if best is None:
        # Fallback
        return 3, 2.0, {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")}

    _, _, k_sel, p_sel, metrics = best
    return k_sel, p_sel, metrics


def round_if_number(x: Optional[float], ndigits: int = 2) -> Optional[float]:
    if x is None or isinstance(x, str):
        return x  # leave as is
    return round(float(x), ndigits)


# -----------------------------
# Enhanced: Trend + Anisotropic IDW residuals
# -----------------------------

def solve_3x3(a11: float, a12: float, a13: float,
              a21: float, a22: float, a23: float,
              a31: float, a32: float, a33: float,
              b1: float, b2: float, b3: float) -> Optional[Tuple[float, float, float]]:
    """Solve 3x3 linear system A x = b via Gaussian elimination. Returns (x1,x2,x3) or None.
    """
    # Augmented matrix
    M = [
        [a11, a12, a13, b1],
        [a21, a22, a23, b2],
        [a31, a32, a33, b3],
    ]
    # Forward elimination
    for i in range(3):
        # Pivot
        piv = i
        for r in range(i+1, 3):
            if abs(M[r][i]) > abs(M[piv][i]):
                piv = r
        if abs(M[piv][i]) < 1e-12:
            return None
        if piv != i:
            M[i], M[piv] = M[piv], M[i]
        # Normalize row
        div = M[i][i]
        for c in range(i, 4):
            M[i][c] /= div
        # Eliminate below
        for r in range(i+1, 3):
            factor = M[r][i]
            for c in range(i, 4):
                M[r][c] -= factor * M[i][c]
    # Back substitution
    x3 = M[2][3]
    x2 = M[1][3] - M[1][2] * x3
    x1 = M[0][3] - M[0][2] * x3 - M[0][1] * x2
    return x1, x2, x3


def fit_trend_plane(points: List[Point]) -> Optional[Tuple[float, float, float]]:
    """Fit z = a + b x + c y. Returns (a,b,c) or None if degenerate/insufficient."""
    if len(points) < 3:
        return None
    n = float(len(points))
    sx = sum(p.x for p in points)
    sy = sum(p.y for p in points)
    sz = sum(p.value for p in points)
    sxx = sum(p.x * p.x for p in points)
    syy = sum(p.y * p.y for p in points)
    sxy = sum(p.x * p.y for p in points)
    sxz = sum(p.x * p.value for p in points)
    syz = sum(p.y * p.value for p in points)

    sol = solve_3x3(
        n,   sx,  sy,
        sx,  sxx, sxy,
        sy,  sxy, syy,
        sz,  sxz, syz
    )
    return sol


def predict_trend(x: float, y: float, coeffs: Optional[Tuple[float, float, float]]) -> float:
    if coeffs is None:
        return 0.0
    a, b, c = coeffs
    return a + b * x + c * y


def rotate(dx: float, dy: float, theta_deg: float) -> Tuple[float, float]:
    th = math.radians(theta_deg)
    ct, st = math.cos(th), math.sin(th)
    xprime = ct * dx + st * dy
    yprime = -st * dx + ct * dy
    return xprime, yprime


def aniso_distance(a: Tuple[float, float], b: Tuple[float, float], theta_deg: float, ratio: float) -> float:
    """Elliptical distance with orientation theta (deg) and anisotropy ratio >= 1.
    ratio = major/minor axis scaling. ratio=1 => isotropic.
    """
    dx, dy = a[0] - b[0], a[1] - b[1]
    xpr, ypr = rotate(dx, dy, theta_deg)
    # Compress along minor axis by ratio to elongate influence along major axis
    return math.hypot(xpr, ypr / max(1.0, ratio))


def idw_predict_aniso(
    xy: Tuple[float, float],
    known: List[Point],
    k: int,
    p: float,
    theta_deg: float,
    ratio: float,
) -> Optional[float]:
    n = len(known)
    if n == 0:
        return None
    dists = [(aniso_distance((pt.x, pt.y), xy, theta_deg, ratio), pt.value) for pt in known]
    dists.sort(key=lambda t: t[0])
    if dists and dists[0][0] == 0.0:
        zeros = [v for (d, v) in dists if d == 0.0]
        return sum(zeros) / len(zeros)
    k_eff = max(1, min(k, n))
    neighbors = dists[:k_eff]
    num = 0.0
    den = 0.0
    for d, v in neighbors:
        w = 1.0 / ((d + 1e-12) ** p)
        num += w * v
        den += w
    return num / den if den > 0 else None


def buffered_loocv_idw(
    points: List[Point],
    k_values: List[int],
    p_values: List[float],
    buffer_radius: float,
) -> Tuple[int, float, Dict[str, float]]:
    """Spatially buffered LOOCV for isotropic IDW. Returns best (k, p) and metrics."""
    if len(points) < 3:
        return 3, 2.0, {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")}

    xs = [pt.x for pt in points]
    ys = [pt.y for pt in points]
    vs = [pt.value for pt in points]

    best = None  # (mae, rmse, k, p, metrics)

    for k in k_values:
        for p in p_values:
            errors = []
            preds = []
            trues = []
            for i in range(len(points)):
                xi, yi, zi = xs[i], ys[i], vs[i]
                # Build training set excluding a Euclidean buffer around the test point
                train_idxs = []
                for j in range(len(points)):
                    if j == i:
                        continue
                    d = euclidean((xs[j], ys[j]), (xi, yi))
                    if d >= buffer_radius:
                        train_idxs.append(j)
                if len(train_idxs) < 1:
                    continue
                train_pts = [Point(xs[j], ys[j], vs[j]) for j in train_idxs]
                pred = idw_predict((xi, yi), train_pts, k=min(k, len(train_pts)), p=p)
                if pred is None:
                    continue
                preds.append(pred)
                trues.append(zi)
                errors.append(pred - zi)
            if not errors:
                continue
            mae = sum(abs(e) for e in errors) / len(errors)
            rmse = math.sqrt(sum(e*e for e in errors) / len(errors))
            y_mean = sum(trues) / len(trues)
            ss_tot = sum((y - y_mean) ** 2 for y in trues)
            ss_res = sum((t - p_) ** 2 for t, p_ in zip(trues, preds))
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
            key = (mae, rmse, k, p, {"MAE": mae, "RMSE": rmse, "R2": r2})
            if best is None or (mae < best[0]) or (mae == best[0] and rmse < best[1]):
                best = key

    if best is None:
        return 3, 2.0, {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")}
    _, _, k_sel, p_sel, metrics = best
    return k_sel, p_sel, metrics


def buffered_loocv_trend_idw(
    points: List[Point],
    k_values: List[int],
    p_values: List[float],
    theta_values: List[float],
    ratio_values: List[float],
    buffer_radius: float,
) -> Tuple[int, float, float, float, Dict[str, float]]:
    """Spatially buffered LOOCV for trend + anisotropic IDW residuals.
    Returns best (k, p, theta, ratio) and metrics dict.
    """
    if len(points) < 5:
        # Fall back to isotropic simple model if too few points
        return 6, 1.0, 0.0, 1.0, {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")}

    xs = [pt.x for pt in points]
    ys = [pt.y for pt in points]
    vs = [pt.value for pt in points]

    best = None  # (mae, rmse, k, p, theta, ratio, metrics)

    for k in k_values:
        for p in p_values:
            for theta in theta_values:
                for ratio in ratio_values:
                    errors = []
                    preds = []
                    trues = []
                    for i in range(len(points)):
                        xi, yi, zi = xs[i], ys[i], vs[i]
                        # Build training set excluding a buffer around the test point
                        train_idxs = []
                        for j in range(len(points)):
                            if j == i:
                                continue
                            d = aniso_distance((xs[j], ys[j]), (xi, yi), theta, ratio)
                            if d >= buffer_radius:
                                train_idxs.append(j)
                        if len(train_idxs) < 3:
                            # Not enough training data when buffering; skip this fold
                            continue
                        train_pts = [Point(xs[j], ys[j], vs[j]) for j in train_idxs]
                        coeffs = fit_trend_plane(train_pts)
                        # Residuals on train
                        res_pts = [
                            Point(p_.x, p_.y, p_.value - predict_trend(p_.x, p_.y, coeffs))
                            for p_ in train_pts
                        ]
                        # Predict trend + residual for test
                        z_trend = predict_trend(xi, yi, coeffs)
                        z_res = idw_predict_aniso((xi, yi), res_pts, k=min(k, len(res_pts)), p=p, theta_deg=theta, ratio=ratio)
                        if z_res is None:
                            continue
                        z_hat = z_trend + z_res
                        preds.append(z_hat)
                        trues.append(zi)
                        errors.append(z_hat - zi)
                    if not errors:
                        continue
                    mae = sum(abs(e) for e in errors) / len(errors)
                    rmse = math.sqrt(sum(e*e for e in errors) / len(errors))
                    y_mean = sum(trues) / len(trues)
                    ss_tot = sum((y - y_mean) ** 2 for y in trues)
                    ss_res = sum((t - p_) ** 2 for t, p_ in zip(trues, preds))
                    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
                    key = (mae, rmse, k, p, theta, ratio, {"MAE": mae, "RMSE": rmse, "R2": r2})
                    if best is None or (mae < best[0]) or (mae == best[0] and rmse < best[1]):
                        best = key

    if best is None:
        return 6, 1.0, 0.0, 1.0, {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")}
    _, _, k_sel, p_sel, theta_sel, ratio_sel, metrics = best
    return k_sel, p_sel, theta_sel, ratio_sel, metrics


def predict_trend_plus_residual(
    all_points: List[Point],
    query_xy: Tuple[float, float],
    k: int,
    p: float,
    theta: float,
    ratio: float,
) -> Optional[float]:
    """Predict using trend plane fit on all points + anisotropic IDW of residuals."""
    if len(all_points) == 0:
        return None
    coeffs = fit_trend_plane(all_points)
    res_pts = [Point(p_.x, p_.y, p_.value - predict_trend(p_.x, p_.y, coeffs)) for p_ in all_points]
    z_tr = predict_trend(query_xy[0], query_xy[1], coeffs)
    z_res = idw_predict_aniso(query_xy, res_pts, k=min(k, len(res_pts)), p=p, theta_deg=theta, ratio=ratio)
    if z_res is None:
        return None
    return z_tr + z_res


def value_bounds(points: List[Point]) -> Tuple[float, float]:
    vals = [p.value for p in points]
    return min(vals), max(vals)


def _banner():
    art = r"""
██╗░░░░░██╗███╗░░░███╗███████╗░██████╗████████╗░█████╗░███╗░░██╗███████╗  ██╗░░░░░░█████╗░██╗░░░██╗███████╗██████╗░
██║░░░░░██║████╗░████║██╔════╝██╔════╝╚══██╔══╝██╔══██╗████╗░██║██╔════╝  ██║░░░░░██╔══██╗╚██╗░██╔╝██╔════╝██╔══██╗
██║░░░░░██║██╔████╔██║█████╗░░╚█████╗░░░░██║░░░██║░░██║██╔██╗██║█████╗░░  ██║░░░░░███████║░╚████╔╝░█████╗░░██████╔╝
██║░░░░░██║██║╚██╔╝██║██╔══╝░░░╚═══██╗░░░██║░░░██║░░██║██║╚████║██╔══╝░░  ██║░░░░░██╔══██║░░╚██╔╝░░██╔══╝░░██╔══██╗
███████╗██║██║░╚═╝░██║███████╗██████╔╝░░░██║░░░╚█████╔╝██║░╚███║███████╗  ███████╗██║░░██║░░░██║░░░███████╗██║░░██║
╚══════╝╚═╝╚═╝░░░░░╚═╝╚══════╝╚═════╝░░░░╚═╝░░░░╚════╝░╚═╝░░╚══╝╚══════╝  ╚══════╝╚═╝░░╚═╝░░░╚═╝░░░╚══════╝╚═╝░░╚═╝

░█████╗░███╗░░██╗░█████╗░██╗░░░░░██╗░░░██╗░██████╗██╗░██████╗
██╔══██╗████╗░██║██╔══██╗██║░░░░░╚██╗░██╔╝██╔════╝██║██╔════╝
███████║██╔██╗██║███████║██║░░░░░░╚████╔╝░╚█████╗░██║╚█████╗░
██╔══██║██║╚████║██╔══██║██║░░░░░░░╚██╔╝░░░╚═══██╗██║░╚═══██╗
██║░░██║██║░╚███║██║░░██║███████╗░░░██║░░░██████╔╝██║██████╔╝
╚═╝░░╚═╝╚═╝░░╚══╝╚═╝░░╚═╝╚══════╝░░░╚═╝░░░╚═════╝░╚═╝╚═════╝░
                                                                                   
    """
    print(art)
    print("Limestone Layer Analysis — IDW + Trend (CLI mode)\n")


def run_analysis(input_csv: str, output_dir: str, location: Optional[str] = None, datum: Optional[str] = None) -> str:
    """Run the limestone analysis.

    Returns the output CSV path.
    """
    # Timestamped output filename: mm-dd-yy-hh_mm
    ts = datetime.now().strftime("%m-%d-%y-%H_%M")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"limestone_layers_phase_idw_{ts}.csv")

    rows = read_rows(input_csv)

    # Parse numeric fields and derive rock head elevation where measured
    for r in rows:
        r["ph_elev"] = parse_float(r.get("ph_elev"))
        r["pt_elev"] = parse_float(r.get("pt_elev"))
        r["northing_coord_y"] = parse_float(r.get("northing_coord_y"))
        r["easting_coord_x"] = parse_float(r.get("easting_coord_x"))
        r["limestone_thickness"] = parse_float(r.get("limestone_thickness"))
        r["sounding_beg_limestone"] = parse_float(r.get("sounding_beg_limestone"))

        # Derived rock top elevation if sounding is present
        if r["ph_elev"] is not None and r["sounding_beg_limestone"] is not None:
            r["rock_top_elev_measured"] = r["ph_elev"] - r["sounding_beg_limestone"]
        else:
            # If pt_elev significantly below ph_elev, treat as measured
            if (
                r["ph_elev"] is not None
                and r["pt_elev"] is not None
                and (r["ph_elev"] - r["pt_elev"]) > 1e-6
            ):
                r["rock_top_elev_measured"] = r["pt_elev"]
            else:
                r["rock_top_elev_measured"] = None

    # Build known points for rock top and thickness
    rock_points: List[Point] = []
    thick_points: List[Point] = []
    for r in rows:
        x = r["easting_coord_x"]
        y = r["northing_coord_y"]
        if x is None or y is None:
            continue
        if r["rock_top_elev_measured"] is not None:
            rock_points.append(Point(x=x, y=y, value=r["rock_top_elev_measured"]))
        if r["limestone_thickness"] is not None:
            thick_points.append(Point(x=x, y=y, value=r["limestone_thickness"]))

    print("Data summary:")
    print(f"  Total rows: {len(rows)}")
    print(f"  Known rock head points: {len(rock_points)}")
    print(f"  Known thickness points: {len(thick_points)}")

    # Hyperparameter grids (baseline)
    k_grid = [3, 4, 5, 6, 7, 8, 10, 12]
    p_grid = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]

    # Baseline LOOCV (isotropic IDW)
    k_rock, p_rock, rock_metrics = loocv_idw(rock_points, k_grid, p_grid)
    print("\nBaseline isotropic IDW for rock head elevation:")
    print(f"  k = {k_rock}, p = {p_rock}")
    print(f"  CV Metrics: MAE = {rock_metrics['MAE']:.3f}, RMSE = {rock_metrics['RMSE']:.3f}, R2 = {rock_metrics['R2']:.3f}")

    k_thick, p_thick, thick_metrics = loocv_idw(thick_points, k_grid, p_grid)
    print("\nBaseline isotropic IDW for thickness:")
    print(f"  k = {k_thick}, p = {p_thick}")
    print(f"  CV Metrics: MAE = {thick_metrics['MAE']:.3f}, RMSE = {thick_metrics['RMSE']:.3f}, R2 = {thick_metrics['R2']:.3f}")

    # Enhanced buffered LOOCV (trend + anisotropic IDW residuals)
    # Determine a buffer radius from spatial span (meters, assuming local grid)
    if rock_points:
        rx = [p.x for p in rock_points]
        ry = [p.y for p in rock_points]
        span_r = max(max(rx) - min(rx), max(ry) - min(ry))
        buffer_r_rock = max(5.0, 0.10 * span_r)
    else:
        buffer_r_rock = 5.0
    if thick_points:
        tx = [p.x for p in thick_points]
        ty = [p.y for p in thick_points]
        span_t = max(max(tx) - min(tx), max(ty) - min(ty))
        buffer_r_thick = max(5.0, 0.10 * span_t)
    else:
        buffer_r_thick = 5.0

    theta_grid = [float(t) for t in range(0, 180, 10)]
    ratio_grid = [1.0, 1.5, 2.0, 3.0]

    # Baseline isotropic IDW with spatially buffered LOOCV (for fair comparison)
    k_rock_bbuf, p_rock_bbuf, rock_bbuf_metrics = buffered_loocv_idw(
        rock_points, k_grid, p_grid, buffer_r_rock
    )
    print("\nBaseline isotropic IDW (buffered CV) for rock head elevation:")
    print(f"  k = {k_rock_bbuf}, p = {p_rock_bbuf}")
    print(f"  Spatially buffered CV: MAE = {rock_bbuf_metrics['MAE']:.3f}, RMSE = {rock_bbuf_metrics['RMSE']:.3f}, R2 = {rock_bbuf_metrics['R2']:.3f}")

    k_thick_bbuf, p_thick_bbuf, thick_bbuf_metrics = buffered_loocv_idw(
        thick_points, k_grid, p_grid, buffer_r_thick
    )
    print("\nBaseline isotropic IDW (buffered CV) for thickness:")
    print(f"  k = {k_thick_bbuf}, p = {p_thick_bbuf}")
    print(f"  Spatially buffered CV: MAE = {thick_bbuf_metrics['MAE']:.3f}, RMSE = {thick_bbuf_metrics['RMSE']:.3f}, R2 = {thick_bbuf_metrics['R2']:.3f}")

    k_rock_e, p_rock_e, theta_rock, ratio_rock, rock_e_metrics = buffered_loocv_trend_idw(
        rock_points, k_grid, p_grid, theta_grid, ratio_grid, buffer_r_rock
    )
    print("\nEnhanced trend+anisotropic IDW for rock head elevation:")
    print(f"  k = {k_rock_e}, p = {p_rock_e}, theta = {theta_rock} deg, ratio = {ratio_rock}")
    print(f"  Spatially buffered CV: MAE = {rock_e_metrics['MAE']:.3f}, RMSE = {rock_e_metrics['RMSE']:.3f}, R2 = {rock_e_metrics['R2']:.3f}")

    k_thick_e, p_thick_e, theta_thick, ratio_thick, thick_e_metrics = buffered_loocv_trend_idw(
        thick_points, k_grid, p_grid, theta_grid, ratio_grid, buffer_r_thick
    )
    print("\nEnhanced trend+anisotropic IDW for thickness:")
    print(f"  k = {k_thick_e}, p = {p_thick_e}, theta = {theta_thick} deg, ratio = {ratio_thick}")
    print(f"  Spatially buffered CV: MAE = {thick_e_metrics['MAE']:.3f}, RMSE = {thick_e_metrics['RMSE']:.3f}, R2 = {thick_e_metrics['R2']:.3f}")

    # Bounds from measured values
    rock_min, rock_max = value_bounds(rock_points) if rock_points else (-float('inf'), float('inf'))
    thick_min, thick_max = value_bounds(thick_points) if thick_points else (0.0, float('inf'))

    # Predict for all rows (baseline and enhanced)
    for r in rows:
        x = r["easting_coord_x"]
        y = r["northing_coord_y"]
        if x is None or y is None:
            r["rock_top_elev_pred_base"] = None
            r["limestone_thickness_pred_base"] = None
            r["rock_top_elev_pred_enh"] = None
            r["limestone_thickness_pred_enh"] = None
            continue

        # Baseline isotropic predictions
        rock_pred_b = idw_predict((x, y), rock_points, k=k_rock, p=p_rock)
        thick_pred_b = idw_predict((x, y), thick_points, k=k_thick, p=p_thick)
        r["rock_top_elev_pred_base"] = rock_pred_b
        r["limestone_thickness_pred_base"] = thick_pred_b

        # Enhanced predictions (trend + anisotropic residual IDW)
        rock_pred_e = predict_trend_plus_residual(rock_points, (x, y), k=k_rock_e, p=p_rock_e, theta=theta_rock, ratio=ratio_rock)
        if rock_pred_e is not None and rock_pred_e == rock_pred_e:  # not NaN
            rock_pred_e = min(rock_max, max(rock_min, rock_pred_e))
        r["rock_top_elev_pred_enh"] = rock_pred_e

        thick_pred_e = predict_trend_plus_residual(thick_points, (x, y), k=k_thick_e, p=p_thick_e, theta=theta_thick, ratio=ratio_thick)
        if thick_pred_e is not None and thick_pred_e == thick_pred_e:
            thick_pred_e = min(thick_max, max(thick_min, thick_pred_e))
        r["limestone_thickness_pred_enh"] = thick_pred_e

    # Fill missing values and add provenance flags
    filled_rows: List[Dict[str, Any]] = []
    for r in rows:
        ph = r["ph_elev"]
        sounding = r["sounding_beg_limestone"]
        thick = r["limestone_thickness"]
        rock_meas = r["rock_top_elev_measured"]
        rock_pred_e = r["rock_top_elev_pred_enh"]
        rock_pred_b = r["rock_top_elev_pred_base"]
        thick_pred_e = r["limestone_thickness_pred_enh"]
        thick_pred_b = r["limestone_thickness_pred_base"]

        sounding_src = "measured"
        thickness_src = "measured"

        # Fill sounding_beg_limestone from predicted rock head (prefer enhanced) if missing
        if sounding is None:
            used_enh_r = (rock_pred_e is not None)
            cand = rock_pred_e if used_enh_r else rock_pred_b
            if ph is not None and cand is not None:
                sounding = max(0.0, ph - cand)
                sounding_src = "predicted_enhanced" if used_enh_r else "predicted_idw"
        else:
            sounding_src = "measured"

        # Fill thickness if missing (prefer enhanced)
        if thick is None:
            used_enh_t = (thick_pred_e is not None)
            cand_t = thick_pred_e if used_enh_t else thick_pred_b
            if cand_t is not None:
                # enforce bounds and non-negativity
                cand_t = min(thick_max, max(thick_min, cand_t))
                thick = max(0.0, cand_t)
                thickness_src = "predicted_enhanced" if used_enh_t else "predicted_idw"

        # Persist filled values
        r_out = dict(r)  # shallow copy
        r_out["sounding_beg_limestone"] = round_if_number(sounding, 2) if sounding is not None else ""
        r_out["limestone_thickness"] = round_if_number(thick, 2) if thick is not None else ""

        # Include rock head elevation fields and provenance
        # Final rock head elevation (prefer measured, else enhanced, else baseline)
        rock_top_final = (
            rock_meas if rock_meas is not None else (
                rock_pred_e if rock_pred_e is not None else rock_pred_b
            )
        )
        r_out["rock_top_elev_final"] = round_if_number(rock_top_final, 3) if rock_top_final is not None else ""
        if rock_meas is not None:
            rock_src = "measured"
        elif rock_pred_e is not None:
            rock_src = "predicted_enhanced"
        elif rock_pred_b is not None:
            rock_src = "predicted_idw"
        else:
            rock_src = ""
        r_out["rock_top_source"] = rock_src
        r_out["sounding_source"] = sounding_src
        r_out["thickness_source"] = thickness_src

        # Ensure numeric coords in output
        r_out["northing_coord_y"] = round_if_number(r.get("northing_coord_y"), 3) if r.get("northing_coord_y") is not None else ""
        r_out["easting_coord_x"] = round_if_number(r.get("easting_coord_x"), 3) if r.get("easting_coord_x") is not None else ""
        r_out["ph_elev"] = round_if_number(r.get("ph_elev"), 2) if r.get("ph_elev") is not None else ""
        r_out["pt_elev"] = round_if_number(r.get("pt_elev"), 2) if r.get("pt_elev") is not None else ""

        # Drop intermediate predictions in final output but keep provenance columns
        r_out.pop("rock_top_elev_measured", None)
        r_out.pop("rock_top_elev_pred_base", None)
        r_out.pop("limestone_thickness_pred_base", None)
        r_out.pop("rock_top_elev_pred_enh", None)
        r_out.pop("limestone_thickness_pred_enh", None)

        filled_rows.append(r_out)

    # Compose field order
    base_fields = [
        "pile_number",
        "pile_diameter",
        "pile_type",
        "ph_elev",
        "pt_elev",
        "northing_coord_y",
        "easting_coord_x",
        "limestone_thickness",
        "sounding_beg_limestone",
    ]
    extra_fields = [
        "rock_top_elev_final",
        "rock_top_source",
        "sounding_source",
        "thickness_source",
    ]
    # Keep any other original fields at the end if present
    other_fields = [k for k in filled_rows[0].keys() if k not in set(base_fields + extra_fields)]
    # But avoid duplicates
    fieldnames = base_fields + extra_fields + [k for k in other_fields if k not in base_fields and k not in extra_fields]

    write_rows(out_path, fieldnames, filled_rows)

    # Show a compact table of rows that were filled (either sounding or thickness)
    filled_preview = []
    for r in filled_rows:
        if (r.get("sounding_source") in ("predicted_idw", "predicted_enhanced")) or (r.get("thickness_source") in ("predicted_idw", "predicted_enhanced")):
            filled_preview.append({
                "pile_number": r.get("pile_number"),
                "X": r.get("easting_coord_x"),
                "Y": r.get("northing_coord_y"),
                "sounding_beg_limestone": r.get("sounding_beg_limestone"),
                "limestone_thickness": r.get("limestone_thickness"),
                "rock_top_elev_final": r.get("rock_top_elev_final"),
            })

    print("\nFilled values preview (predicted rows):")
    if not filled_preview:
        print("  No rows required filling.")
    else:
        # Pretty print as columns
        cols = ["pile_number", "X", "Y", "sounding_beg_limestone", "limestone_thickness", "rock_top_elev_final"]
        header = " | ".join(f"{c:>24}" for c in cols)
        print(header)
        print("-" * len(header))
        for r in filled_preview:
            line = " | ".join(f"{str(r.get(c, '')):>24}" for c in cols)
            print(line)

    # Print CV summary
    print("\nCross-validation summary:")
    print("  Baseline Rock head (IDW):        k = %d, p = %.2f, MAE = %.3f, RMSE = %.3f, R2 = %.3f" % (
        k_rock, p_rock, rock_metrics["MAE"], rock_metrics["RMSE"], rock_metrics["R2"],
    ))
    print("  Baseline Thickness (IDW):        k = %d, p = %.2f, MAE = %.3f, RMSE = %.3f, R2 = %.3f" % (
        k_thick, p_thick, thick_metrics["MAE"], thick_metrics["RMSE"], thick_metrics["R2"],
    ))
    print("  Baseline Rock head (IDW, buffered):  k = %d, p = %.2f, MAE = %.3f, RMSE = %.3f, R2 = %.3f" % (
        k_rock_bbuf, p_rock_bbuf, rock_bbuf_metrics["MAE"], rock_bbuf_metrics["RMSE"], rock_bbuf_metrics["R2"],
    ))
    print("  Baseline Thickness (IDW, buffered):  k = %d, p = %.2f, MAE = %.3f, RMSE = %.3f, R2 = %.3f" % (
        k_thick_bbuf, p_thick_bbuf, thick_bbuf_metrics["MAE"], thick_bbuf_metrics["RMSE"], thick_bbuf_metrics["R2"],
    ))
    print("  Enhanced Rock head (trend+aniso): k = %d, p = %.2f, theta = %.0f, ratio = %.1f, MAE = %.3f, RMSE = %.3f, R2 = %.3f" % (
        k_rock_e, p_rock_e, theta_rock, ratio_rock, rock_e_metrics["MAE"], rock_e_metrics["RMSE"], rock_e_metrics["R2"],
    ))
    print("  Enhanced Thickness (trend+aniso): k = %d, p = %.2f, theta = %.0f, ratio = %.1f, MAE = %.3f, RMSE = %.3f, R2 = %.3f" % (
        k_thick_e, p_thick_e, theta_thick, ratio_thick, thick_e_metrics["MAE"], thick_e_metrics["RMSE"], thick_e_metrics["R2"],
    ))

    # Provenance summary
    def _tally(key: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in filled_rows:
            v = r.get(key, "")
            counts[v] = counts.get(v, 0) + 1
        return counts

    rock_counts = _tally("rock_top_source")
    sounding_counts = _tally("sounding_source")
    thickness_counts = _tally("thickness_source")

    print("\nProvenance summary (counts):")
    def _fmt_counts(name: str, c: Dict[str, int]):
        total = sum(c.values())
        ordered = ["measured", "predicted_enhanced", "predicted_idw", ""]
        parts = []
        for k in ordered:
            if k in c:
                parts.append(f"{k or 'empty'}={c[k]}")
        for k, v in c.items():
            if k in ordered:
                continue
            parts.append(f"{k}={v}")
        print(f"  {name}: total={total}; " + ", ".join(parts))
    _fmt_counts("rock_top_source", rock_counts)
    _fmt_counts("sounding_source", sounding_counts)
    _fmt_counts("thickness_source", thickness_counts)

    print(f"\nWrote filled dataset to: {out_path}")
    if location:
        print(f"  Site location: {location}")
    if datum:
        print(f"  Depth datum: {datum}")
    return out_path


def main():
    """Fallback entry that preserves original defaults when invoked directly."""
    _banner()
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)
    default_input = os.path.join(project_root, "input", "limestone_layers_phase_1_input.csv")
    default_output_dir = os.path.join(project_root, "output")
    # Environment overrides (used by CLI wrapper if desired)
    input_csv = os.environ.get("LIMESTONE_INPUT", default_input)
    output_dir = os.environ.get("LIMESTONE_OUTPUT_DIR", default_output_dir)
    location = os.environ.get("LIMESTONE_LOCATION") or None
    datum = os.environ.get("LIMESTONE_DATUM") or None
    run_analysis(input_csv=input_csv, output_dir=output_dir, location=location, datum=datum)


if __name__ == "__main__":
    main()
