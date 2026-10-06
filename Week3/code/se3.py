"""SE(3) / SO(3) 基础工具：旋转矩阵、欧拉角、四元数、齐次变换矩阵。

被 01_pose_transform.py、02_forward_kinematics.py、03_trajectory_planning.py 复用。

约定（全文统一，不要混用）：
- 四元数一律 [x, y, z, w]，与 scipy / ROS / pybullet 一致。
- 欧拉角一律 ZYX（roll-pitch-yaw），R = Rz(yaw) @ Ry(pitch) @ Rx(roll)。
- T_ij 表示“j 系在 i 系中的位姿”，把 j 系坐标映射到 i 系：p_i = T_ij @ p_j。
  因此链式法则是右乘：T_03 = T_01 @ T_12 @ T_23。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

EPS = 1e-12


# ------------------------------------------------------------- SO(3) 基本旋转 ----
def rot_x(a: float) -> np.ndarray:
    """绕 x 轴转 a 弧度的 3x3 旋转矩阵。"""
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_axis_angle(axis: ArrayLike, angle: float) -> np.ndarray:
    """罗德里格斯公式：任意轴角 -> 旋转矩阵。"""
    k = np.asarray(axis, dtype=float).reshape(3)
    k = k / np.linalg.norm(k)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def rotvec_to_matrix(rv: ArrayLike) -> np.ndarray:
    """旋转向量（轴 * 角度）-> 旋转矩阵。"""
    rv = np.asarray(rv, dtype=float).reshape(3)
    theta = np.linalg.norm(rv)
    if theta < 1e-12:
        return np.eye(3)
    return rot_axis_angle(rv, np.asarray(theta).item())


def matrix_to_rotvec(R: ArrayLike) -> np.ndarray:
    """旋转矩阵 -> 旋转向量，逆解由罗德里格斯公式导出。"""
    R = np.asarray(R, dtype=float)
    cos_t = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    theta = np.arccos(cos_t)
    if theta < 1e-9:
        return np.zeros(3)
    if abs(np.pi - theta) < 1e-6:  # 接近 180°，用对称部分
        A = (R + np.eye(3)) / 2
        axis = np.sqrt(np.clip(np.diag(A), 0, None))
        axis[1] = np.copysign(axis[1], R[0, 1] + R[1, 0])
        axis[2] = np.copysign(axis[2], R[0, 2] + R[2, 0])
        return axis / np.linalg.norm(axis) * theta
    return theta / (2 * np.sin(theta)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


# ------------------------------------------------------------- 姿态表示互转 ----
def euler_zyx_to_matrix(rpy: ArrayLike) -> np.ndarray:
    """ZYX 欧拉角（roll-pitch-yaw，RPY）-> 旋转矩阵。"""
    roll, pitch, yaw = np.asarray(rpy, dtype=float).reshape(3)
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def matrix_to_euler_zyx(R: ArrayLike) -> np.ndarray:
    """ZYX 欧拉角逆解，返回 [roll, pitch, yaw]。

    当 pitch = ±90° 时出现万向锁：此时 roll 与 yaw 不可分离，
    本函数把 roll 置 0，把等效转角全部并入 yaw。
    """
    R = np.asarray(R, dtype=float)
    sp = np.clip(-R[2, 0], -1.0, 1.0)
    pitch = np.arcsin(sp)
    if abs(sp) >= 1.0 - 1e-10:  # 万向锁分支
        return np.array([0.0, pitch, np.arctan2(-R[0, 1], R[1, 1])])
    return np.array([np.arctan2(R[2, 1], R[2, 2]), pitch, np.arctan2(R[1, 0], R[0, 0])])


def matrix_to_quat_xyzw(R: ArrayLike) -> np.ndarray:
    """旋转矩阵 -> 单位四元数 [x, y, z, w]。"""
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def quat_to_matrix(q_xyzw: ArrayLike) -> np.ndarray:
    """单位四元数 [x, y, z, w] -> 旋转矩阵。"""
    q = np.asarray(q_xyzw, dtype=float).reshape(4)
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def slerp_quat(q0: ArrayLike, q1: ArrayLike, t: float) -> np.ndarray:
    """四元数球面线性插值：姿态沿测地线匀速过渡。"""
    a = np.asarray(q0, dtype=float).reshape(4)
    b = np.asarray(q1, dtype=float).reshape(4)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    if np.dot(a, b) < 0:  # 取短弧
        b = -b
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    if dot > 1.0 - 1e-9:
        return a
    theta = np.arccos(dot)
    return (np.sin((1 - t) * theta) * a + np.sin(t * theta) * b) / np.sin(theta)


def rotate_vector(R: ArrayLike, vectors: ArrayLike) -> np.ndarray:
    """把若干三维向量按 R 旋转，支持单个 (3,) 或批量 (N, 3)。"""
    R = np.asarray(R, dtype=float)
    v = np.atleast_2d(np.asarray(vectors, dtype=float))
    return (R @ v.T).T


# ------------------------------------------------------------- SE(3) 齐次矩阵 ----
def make_transform(R: ArrayLike, p: ArrayLike) -> np.ndarray:
    """由旋转矩阵和平移向量拼成 4x4 齐次变换矩阵 [R | p; 0 1]。"""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(p, dtype=float).reshape(3)
    return T


def trans(x: float, y: float, z: float) -> np.ndarray:
    """纯平移的 4x4 齐次矩阵。"""
    return make_transform(np.eye(3), np.array([x, y, z], dtype=float))


def rot_z_h(a: float) -> np.ndarray:
    """绕 z 轴转 a 弧度的 4x4 齐次矩阵（无平移）。"""
    return make_transform(rot_z(a), np.zeros(3))


def invert_transform(T: ArrayLike) -> np.ndarray:
    """解析求逆：T^-1 = [[R^T, -R^T p], [0, 1]]，不必调 np.linalg.inv。"""
    T = np.asarray(T, dtype=float)
    R = T[:3, :3]
    p = T[:3, 3]
    return make_transform(R.T, -R.T @ p)


def apply_transform(T: ArrayLike, points: ArrayLike) -> np.ndarray:
    """用齐次矩阵变换点，`points` 形状为 (3,) 或 (N, 3)。"""
    T = np.asarray(T, dtype=float)
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


def transform_to_rpy(T: ArrayLike) -> np.ndarray:
    """齐次矩阵 -> [x, y, z, roll, pitch, yaw]，便于读数。"""
    T = np.asarray(T, dtype=float)
    return np.concatenate([T[:3, 3], matrix_to_euler_zyx(T[:3, :3])])
