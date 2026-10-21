"""MuJoCo ray-cast LiDAR + landmark-based EKF-SLAM educational demo.

This first milestone is deliberately ROS 2 independent: it proves that MuJoCo
ray casting, feature extraction, and EKF-SLAM work together before adding ROS 2
nodes. The robot base follows a known kinematic ellipse; EKF receives only noisy
odometry and LiDAR-derived landmark observations. Ground truth is evaluator-only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer 
import numpy as np

ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "scene.xml"
OUT = ROOT / "outputs"

# ---- Simulation settings ----------------------------------------------------
DT = 0.1                      # EKF / sensor update period [s]
SIM_TIME = 60.0                # one full ellipse [s]
N_RAYS = 720                  # 0.5 degree resolution over 360 degrees
LIDAR_MAX_RANGE = 7.5         # [m]
LIDAR_Z = 0.35                # height in world frame [m]
LIDAR_RANGE_NOISE_STD = 0.004  # synthetic range noise [m]
CLUSTER_GAP_M = 0.28           # split adjacent scan points at larger gaps
MIN_CLUSTER_POINTS = 5
CIRCLE_RADIUS_MIN = 0.065
CIRCLE_RADIUS_MAX = 0.19
CIRCLE_FIT_RMSE_MAX = 0.035

# Landmark radius in MJCF is 0.12 m. The detector intentionally uses a range,
# not geom IDs, so EKF observations are derived from scan geometry.
LANDMARKS_TRUE = np.array([
    [-1.25, 1.05],
    [1.05, 1.15],
    [-1.15, -1.15],
    [1.20, -1.10],
    [0.05, 0.05],
], dtype=float)

# EKF measurement model for feature-centre observations: range and bearing.
R_MEAS = np.diag([0.055**2, np.deg2rad(2.2)**2])
Q_ODOM = np.diag([0.012**2, np.deg2rad(0.8)**2])
ASSOCIATION_GATE = 5.991  # chi-square gate, 2 dimensions, approx. 95%


def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def set_robot_pose(model: mujoco.MjModel, data: mujoco.MjData,
                   x: float, y: float, yaw: float, z: float = 0.20) -> None:
    """Kinematically set the freejoint pose and refresh MuJoCo kinematics."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot_free")
    if jid < 0:
        raise RuntimeError("Joint 'robot_free' tidak ditemukan di scene.xml")
    adr = int(model.jnt_qposadr[jid])
    data.qpos[adr:adr + 3] = [x, y, z]
    # MuJoCo quaternion order is w, x, y, z.
    data.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
    mujoco.mj_forward(model, data)


def ellipse_pose(t: float) -> np.ndarray:
    """Ground-truth trajectory, starting at x=2m, y=0, heading +pi/2."""
    omega = 2.0 * np.pi / SIM_TIME
    ax, ay = 2.0, 1.8
    x = ax * np.cos(omega * t)
    y = ay * np.sin(omega * t)
    dx = -ax * omega * np.sin(omega * t)
    dy = ay * omega * np.cos(omega * t)
    yaw = np.arctan2(dy, dx)
    return np.array([x, y, wrap_angle(yaw)], dtype=float)


def cast_lidar(model: mujoco.MjModel, data: mujoco.MjData,
               robot_body_id: int, pose: np.ndarray,
               rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Cast actual MuJoCo rays against scene geoms, excluding the robot body.

    Returns scan angles and ranges. NaN represents a ray with no valid return.
    This is MuJoCo's geometric ray casting, not a precomputed fake scan.
    """
    angles = np.linspace(-np.pi, np.pi, N_RAYS, endpoint=False, dtype=float)
    origin = np.array([pose[0], pose[1], LIDAR_Z], dtype=np.float64)
    ranges = np.full(N_RAYS, np.nan, dtype=float)
    geomid = np.zeros(1, dtype=np.int32)

    for i, a in enumerate(angles):
        theta = pose[2] + a
        direction = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float64)
        # mj_ray requires current MuJoCo kinematics, refreshed in set_robot_pose().
        distance = mujoco.mj_ray(
            model, data, origin, direction,
            None, 1, int(robot_body_id), geomid
        )
        if 0.02 < distance <= LIDAR_MAX_RANGE:
            noisy = distance + rng.normal(0.0, LIDAR_RANGE_NOISE_STD)
            ranges[i] = float(np.clip(noisy, 0.02, LIDAR_MAX_RANGE))
    return angles, ranges


def split_scan_clusters(angles: np.ndarray, ranges: np.ndarray) -> list[np.ndarray]:
    """Split local XY scan points using jump distance between adjacent rays."""
    valid = np.isfinite(ranges)
    indices = np.flatnonzero(valid)
    if len(indices) == 0:
        return []

    pts = np.column_stack((ranges[indices] * np.cos(angles[indices]),
                           ranges[indices] * np.sin(angles[indices])))
    clusters: list[np.ndarray] = []
    start = 0
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[i - 1]) > CLUSTER_GAP_M:
            cluster = pts[start:i]
            if len(cluster) >= MIN_CLUSTER_POINTS:
                clusters.append(cluster)
            start = i
    tail = pts[start:]
    if len(tail) >= MIN_CLUSTER_POINTS:
        clusters.append(tail)

    # The scan is circular. Merge first and last clusters if they touch across +/-pi.
    if len(clusters) >= 2 and len(pts) > 1:
        first_i, last_i = indices[0], indices[-1]
        if first_i == 0 and last_i == len(ranges) - 1:
            if np.linalg.norm(pts[0] - pts[-1]) <= CLUSTER_GAP_M:
                merged = np.vstack((clusters[-1], clusters[0]))
                clusters = [merged] + clusters[1:-1]
    return clusters


def fit_circle_kasa(points: np.ndarray) -> tuple[np.ndarray, float, float] | None:
    """Fit circle to a scan cluster. Returns centre, radius, radial RMSE."""
    if len(points) < MIN_CLUSTER_POINTS:
        return None
    x = points[:, 0]
    y = points[:, 1]
    A = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    b = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy, c = sol
    radius_sq = cx * cx + cy * cy + c
    if not np.isfinite(radius_sq) or radius_sq <= 0.0:
        return None
    radius = float(np.sqrt(radius_sq))
    residual = np.sqrt((x - cx)**2 + (y - cy)**2) - radius
    rmse = float(np.sqrt(np.mean(residual**2)))
    if not (CIRCLE_RADIUS_MIN <= radius <= CIRCLE_RADIUS_MAX):
        return None
    if rmse > CIRCLE_FIT_RMSE_MAX:
        return None
    return np.array([cx, cy], dtype=float), radius, rmse


def extract_landmark_observations(angles: np.ndarray,
                                  ranges: np.ndarray) -> list[np.ndarray]:
    """Extract circular landmark centres and return [range, bearing] observations.

    It does not use MuJoCo geom IDs or ground-truth landmark coordinates.
    The prototype assumes landmarks are cylindrical markers with known-size range.
    """
    observations: list[np.ndarray] = []
    for cluster in split_scan_clusters(angles, ranges):
        fit = fit_circle_kasa(cluster)
        if fit is None:
            continue
        centre = fit[0]  # centre in the LiDAR/robot local frame
        r = float(np.linalg.norm(centre))
        bearing = float(wrap_angle(np.arctan2(centre[1], centre[0])))
        if 0.15 < r < LIDAR_MAX_RANGE:
            observations.append(np.array([r, bearing], dtype=float))

    # Remove near-duplicate feature centres from occasional over-segmentation.
    deduped: list[np.ndarray] = []
    for z in observations:
        if all(abs(z[0] - zz[0]) > 0.10 or
               abs(wrap_angle(z[1] - zz[1])) > 0.07 for zz in deduped):
            deduped.append(z)
    return deduped


def observe_from_true_landmarks(pose: np.ndarray) -> list[np.ndarray]:
    """Evaluation/debug helper only; not used by the EKF. Ground truth only."""
    out = []
    for lm in LANDMARKS_TRUE:
        dx, dy = lm[0] - pose[0], lm[1] - pose[1]
        r = np.hypot(dx, dy)
        if r < LIDAR_MAX_RANGE:
            b = wrap_angle(np.arctan2(dy, dx) - pose[2])
            out.append(np.array([r, b]))
    return out


class EKFSLAM:
    """Compact landmark-based EKF-SLAM with differential-drive motion input."""
    ROBOT_DIM = 3
    LANDMARK_DIM = 2

    def __init__(self, initial_pose: np.ndarray):
        self.x = np.asarray(initial_pose, dtype=float).reshape(3, 1).copy()
        # Initial pose is assumed approximately known; uncertainty is not zero.
        self.P = np.diag([0.03**2, 0.03**2, np.deg2rad(3.0)**2]).astype(float)
        self.landmark_init_cov = np.diag([0.20**2, 0.20**2])
        self.landmark_ids_added = 0
        self.landmark_count_history: list[int] = []

    @property
    def landmark_count(self) -> int:
        return max(0, (self.x.shape[0] - self.ROBOT_DIM) // self.LANDMARK_DIM)

    def predict(self, ds: float, dtheta: float) -> None:
        theta = float(self.x[2, 0])
        half_yaw = theta + 0.5 * dtheta
        c, s = np.cos(half_yaw), np.sin(half_yaw)

        self.x[0, 0] += ds * c
        self.x[1, 0] += ds * s
        self.x[2, 0] = wrap_angle(theta + dtheta)

        n = self.x.shape[0]
        F = np.eye(n)
        F[0, 2] = -ds * s
        F[1, 2] = ds * c

        G3 = np.array([
            [c, -0.5 * ds * s],
            [s,  0.5 * ds * c],
            [0.0, 1.0],
        ])
        G = np.zeros((n, 2), dtype=float)
        G[:3, :] = G3
        self.P = F @ self.P @ F.T + G @ Q_ODOM @ G.T
        self.P = 0.5 * (self.P + self.P.T)

    def measurement_model(self, landmark_index: int) -> tuple[np.ndarray, np.ndarray]:
        idx = self.ROBOT_DIM + self.LANDMARK_DIM * landmark_index
        lx, ly = self.x[idx:idx + 2, 0]
        rx, ry, yaw = self.x[:3, 0]
        dx, dy = lx - rx, ly - ry
        q = dx * dx + dy * dy
        q = max(q, 1e-12)
        r = np.sqrt(q)
        pred = np.array([r, wrap_angle(np.arctan2(dy, dx) - yaw)], dtype=float)

        H = np.zeros((2, self.x.shape[0]), dtype=float)
        H[:, 0:3] = np.array([
            [-dx / r, -dy / r, 0.0],
            [ dy / q, -dx / q, -1.0],
        ])
        H[:, idx:idx + 2] = np.array([
            [dx / r, dy / r],
            [-dy / q, dx / q],
        ])
        return pred.reshape(2, 1), H

    def innovation(self, z: np.ndarray, landmark_index: int):
        pred, H = self.measurement_model(landmark_index)
        nu = np.asarray(z, dtype=float).reshape(2, 1) - pred
        nu[1, 0] = wrap_angle(float(nu[1, 0]))
        S = H @ self.P @ H.T + R_MEAS
        return nu, S, H

    def add_landmark(self, z: np.ndarray) -> None:
        r, bearing = float(z[0]), float(z[1])
        x, y, theta = self.x[:3, 0]
        angle = theta + bearing
        c, s = np.cos(angle), np.sin(angle)
        lm = np.array([[x + r * c], [y + r * s]])

        n_old = self.x.shape[0]
        Jx = np.zeros((2, n_old), dtype=float)
        Jx[0, 0] = 1.0
        Jx[0, 2] = -r * s
        Jx[1, 1] = 1.0
        Jx[1, 2] = r * c
        Jz = np.array([[c, -r * s], [s, r * c]], dtype=float)

        P_cross = Jx @ self.P
        P_lm = Jx @ self.P @ Jx.T + Jz @ R_MEAS @ Jz.T
        P_aug = np.zeros((n_old + 2, n_old + 2), dtype=float)
        P_aug[:n_old, :n_old] = self.P
        P_aug[n_old:, :n_old] = P_cross
        P_aug[:n_old, n_old:] = P_cross.T
        P_aug[n_old:, n_old:] = P_lm
        self.x = np.vstack((self.x, lm))
        self.P = P_aug
        self.landmark_ids_added += 1

    def associate(self, z: np.ndarray) -> int | None:
        """Return best landmark index; None means initialize a new landmark."""
        if self.landmark_count == 0:
            return None
        best_i, best_d2 = None, float("inf")
        for i in range(self.landmark_count):
            nu, S, _ = self.innovation(z, i)
            try:
                d2 = float((nu.T @ np.linalg.solve(S, nu))[0, 0])
            except np.linalg.LinAlgError:
                continue
            if d2 < best_d2:
                best_i, best_d2 = i, d2
        if best_i is None or best_d2 > ASSOCIATION_GATE:
            return None
        return best_i

    def update_one(self, z: np.ndarray) -> None:
        idx = self.associate(z)
        if idx is None:
            self.add_landmark(z)
            return
        nu, S, H = self.innovation(z, idx)
        try:
            # Solve rather than explicitly invert S.
            K = np.linalg.solve(S, H @ self.P).T
        except np.linalg.LinAlgError:
            return
        self.x = self.x + K @ nu
        self.x[2, 0] = wrap_angle(float(self.x[2, 0]))
        I_KH = np.eye(self.P.shape[0]) - K @ H
        # Joseph covariance update for better numerical stability.
        self.P = I_KH @ self.P @ I_KH.T + K @ R_MEAS @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    def update(self, observations: Iterable[np.ndarray]) -> None:
        for z in observations:
            self.update_one(z)
        self.landmark_count_history.append(self.landmark_count)


def plot_scan(angles: np.ndarray, ranges: np.ndarray, out_path: Path) -> None:
    valid = np.isfinite(ranges)
    x = ranges[valid] * np.cos(angles[valid])
    y = ranges[valid] * np.sin(angles[valid])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, s=4, label="MuJoCo LiDAR ray hits")
    ax.scatter([0], [0], marker="+", s=100, label="LiDAR origin")
    ax.set_title("2D LiDAR scan generated by MuJoCo mj_ray")
    ax.set_xlabel("x local (m)")
    ax.set_ylabel("y local (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_result(truth: np.ndarray, dead_reckoning: np.ndarray,
                estimate: np.ndarray, slam: EKFSLAM, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(truth[:, 0], truth[:, 1], "b-", linewidth=2, label="Ground truth (MuJoCo pose)")
    ax.plot(dead_reckoning[:, 0], dead_reckoning[:, 1], "k--", alpha=0.8, label="Noisy odometry only")
    ax.plot(estimate[:, 0], estimate[:, 1], "r-", linewidth=1.6, label="EKF-SLAM estimate")
    ax.scatter(LANDMARKS_TRUE[:, 0], LANDMARKS_TRUE[:, 1], marker="*", s=120,
               color="black", label="True landmark centres (evaluator only)")

    if slam.landmark_count:
        lms = slam.x[3:].reshape(-1, 2)
        ax.scatter(lms[:, 0], lms[:, 1], marker="x", s=65, color="green",
                   label="EKF landmark estimates")
    ax.set_title("Landmark-based EKF-SLAM using MuJoCo ray-cast LiDAR")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_csv(times: np.ndarray, truth: np.ndarray, dr: np.ndarray,
             est: np.ndarray, lm_counts: list[int]) -> None:
    path = OUT / "trajectory.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "gt_x", "gt_y", "gt_yaw_rad",
                    "odom_x", "odom_y", "odom_yaw_rad",
                    "ekf_x", "ekf_y", "ekf_yaw_rad", "landmark_count"])
        for i in range(len(times)):
            c = lm_counts[i] if i < len(lm_counts) else lm_counts[-1]
            w.writerow([times[i], *truth[i], *dr[i], *est[i], c])


def run_simulation(seed: int = 7, viewer: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    robot_body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot"))
    if robot_body_id < 0:
        raise RuntimeError("Body 'robot' tidak ditemukan pada scene.xml")

    times = np.arange(0.0, SIM_TIME + DT * 0.5, DT)
    truth = np.vstack([ellipse_pose(float(t)) for t in times])
    initial = truth[0].copy()
    slam = EKFSLAM(initial)
    dr_pose = initial.copy()
    dr_history: list[np.ndarray] = []
    est_history: list[np.ndarray] = []
    last_angles = np.zeros(N_RAYS)
    last_ranges = np.full(N_RAYS, np.nan)

    print("Memuat scene MuJoCo:", SCENE)
    print(f"Langkah simulasi: {len(times)} | rays/scan: {N_RAYS} | waktu: {SIM_TIME:.1f}s")
    print("Menjalankan ray casting MuJoCo + ekstraksi landmark + EKF-SLAM...")

    all_obs_counts = []
    for k in range(len(times)):
        pose = truth[k]
        set_robot_pose(model, data, pose[0], pose[1], pose[2])
        angles, ranges = cast_lidar(model, data, robot_body_id, pose, rng)
        obs = extract_landmark_observations(angles, ranges)
        all_obs_counts.append(len(obs))
        if k > 0:
            dx = truth[k, 0] - truth[k - 1, 0]
            dy = truth[k, 1] - truth[k - 1, 1]
            ds_true = float(np.hypot(dx, dy))
            dtheta_true = float(wrap_angle(truth[k, 2] - truth[k - 1, 2]))
            ds_odom = ds_true + rng.normal(0.0, 0.001 + 0.012 * ds_true)
            dtheta_odom = dtheta_true + rng.normal(0.0, np.deg2rad(0.25) + 0.03 * abs(dtheta_true))
            dr_pose[0] += ds_odom * np.cos(dr_pose[2] + 0.5 * dtheta_odom)
            dr_pose[1] += ds_odom * np.sin(dr_pose[2] + 0.5 * dtheta_odom)
            dr_pose[2] = wrap_angle(dr_pose[2] + dtheta_odom)
            slam.predict(ds_odom, dtheta_odom)
        slam.update(obs)
        dr_history.append(dr_pose.copy())
        est_history.append(slam.x[:3, 0].copy())
        last_angles, last_ranges = angles, ranges
        if (k + 1) % 100 == 0 or k == len(times) - 1:
            print(f"  step {k + 1:4d}/{len(times)} | LiDAR features: {len(obs):2d} | EKF landmarks: {slam.landmark_count:2d}")

    # Align array lengths and save outputs.
    truth_arr = truth
    dr_arr = np.asarray(dr_history[:len(times)])
    est_arr = np.asarray(est_history[:len(times)])
    position_error = np.linalg.norm(est_arr[:, :2] - truth_arr[:, :2], axis=1)
    yaw_error = np.array([abs(wrap_angle(a - b)) for a, b in zip(est_arr[:, 2], truth_arr[:, 2])])
    metrics = {
        "note": "Synthetic kinematic path; LiDAR ranges are obtained by actual MuJoCo mj_ray calls. Ground truth is evaluator-only.",
        "seed": seed,
        "duration_s": SIM_TIME,
        "update_period_s": DT,
        "rays_per_scan": N_RAYS,
        "total_detected_feature_observations": int(sum(all_obs_counts)),
        "mean_features_per_scan": float(np.mean(all_obs_counts)),
        "estimated_landmark_count": int(slam.landmark_count),
        "position_rmse_m": float(np.sqrt(np.mean(position_error**2))),
        "position_mean_error_m": float(np.mean(position_error)),
        "position_final_error_m": float(position_error[-1]),
        "yaw_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(yaw_error**2)))),
        "yaw_final_error_deg": float(np.rad2deg(yaw_error[-1])),
    }
    with (OUT / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    np.savetxt(OUT / "last_scan.csv", np.column_stack((last_angles, last_ranges)),
               delimiter=",", header="angle_rad,range_m", comments="")
    plot_scan(last_angles, last_ranges, OUT / "lidar_scan.png")
    plot_result(truth_arr, dr_arr, est_arr, slam, OUT / "ekf_slam_result.png")
    save_csv(times, truth_arr, dr_arr, est_arr, slam.landmark_count_history)

    print("\nSelesai. Output tersimpan pada:", OUT)
    print(json.dumps(metrics, indent=2))
    print("\nCatatan: cek ekf_slam_result.png. Bila jumlah landmark jauh dari 5, ubah parameter clustering/circle fitting.")

    if viewer:
        print("Membuka MuJoCo viewer. Tutup jendela untuk selesai.")
        try:
            from mujoco import viewer as mj_viewer
            with mj_viewer.launch_passive(model, data) as v:
                    time.sleep(0.015)
        except Exception as exc:
            print("Viewer gagal dibuka. File hasil tetap tersedia.")
            print(f"Detail: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MuJoCo ray-cast LiDAR + landmark EKF-SLAM")
    parser.add_argument("--seed", type=int, default=7, help="Seed noise agar eksperimen bisa diulang")
    parser.add_argument("--viewer", action="store_true", help="Buka MuJoCo viewer setelah kalkulasi selesai")
    args = parser.parse_args()
    run_simulation(seed=args.seed, viewer=args.viewer)


if __name__ == "__main__":
    main()
