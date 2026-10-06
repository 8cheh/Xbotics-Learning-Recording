"""复现 1：坐标系与位姿变换（齐次矩阵、欧拉角、四元数、链式法则）

对应周报第一部分。运行：

    python 01_pose_transform.py

脚本自带断言，全部通过后打印 "01_pose_transform: all assertions passed"，
并在 ../figures/ 下生成 fig_pose_transform.png。

底层数学函数在 se3.py 中，本脚本只做“约定验证 + 图示”。
"""

from __future__ import annotations

import numpy as np
from mpl_style import setup
from se3 import (
    EPS,
    apply_transform,
    euler_zyx_to_matrix,
    invert_transform,
    make_transform,
    matrix_to_euler_zyx,
    matrix_to_quat_xyzw,
    quat_to_matrix,
    rot_axis_angle,
    rot_x,
    rot_y,
    rot_z,
    rot_z_h,
    slerp_quat,
    trans,
)

plt = setup()


# ---------------------------------------------------------------- 自检 ----
def check_rotation_basics() -> None:
    a = 0.7
    rots = [rot_x(a), rot_y(a), rot_z(a), rot_axis_angle(np.array([1.0, 2.0, 3.0]), a)]
    for R in rots:
        assert np.allclose(R @ R.T, np.eye(3), atol=EPS), "旋转矩阵必须正交"
        assert np.isclose(np.linalg.det(R), 1.0, atol=EPS), "旋转矩阵行列式必须为 +1"
    # 轴角公式退化为基本旋转
    assert np.allclose(rot_axis_angle([0.0, 0.0, 1.0], 0.3), rot_z(0.3), atol=EPS)
    # 绕 z 轴 90°：x 轴 -> y 轴
    assert np.allclose(rot_z(np.pi / 2) @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=EPS)
    # 复合顺序：Rz@Rx 等价于“先绕 x 转，再绕 z 转”（定轴左乘）
    assert np.allclose((rot_z(0.4) @ rot_x(0.2)) @ np.array([0.3, -0.5, 0.8]),
                       rot_z(0.4) @ (rot_x(0.2) @ np.array([0.3, -0.5, 0.8])), atol=EPS)
    assert not np.allclose(rot_z(0.4) @ rot_x(0.2), rot_x(0.2) @ rot_z(0.4), atol=1e-6), "旋转不可交换"


def check_euler_quat_roundtrip() -> None:
    for rpy in ([0.1, -0.3, 1.2], [0.0, 0.0, np.pi / 3], [-1.0, 0.4, 2.5], [0.5, 0.25, -0.75]):
        R = euler_zyx_to_matrix(rpy)
        assert np.allclose(matrix_to_euler_zyx(R), rpy, atol=1e-10), f"欧拉角往返失败 {rpy}"
        q = matrix_to_quat_xyzw(R)
        assert np.isclose(np.linalg.norm(q), 1.0, atol=EPS), "四元数必须归一"
        assert np.allclose(quat_to_matrix(q), R, atol=1e-10), "四元数往返失败"
    # w 分量 = 旋转角一半的余弦
    assert np.isclose(matrix_to_quat_xyzw(rot_z(1.0))[3], np.cos(0.5), atol=1e-12)
    # 万向锁：pitch = ±90° 时 roll 被置零，姿态仍严格等价
    for pitch in (np.pi / 2, -np.pi / 2):
        R_lock = euler_zyx_to_matrix([0.9, pitch, -0.4])
        rpy_lock = matrix_to_euler_zyx(R_lock)
        assert np.isclose(rpy_lock[0], 0.0), "万向锁时 roll 应被置零"
        assert np.allclose(euler_zyx_to_matrix(rpy_lock), R_lock, atol=1e-9), "万向锁重构失败"


def check_homogeneous_chain() -> None:
    """链式法则：T_03 = T_01 @ T_12 @ T_23；且 T_ij^-1 = T_ji。

    约定：T_ij 是“j 系在 i 系中的位姿”，把 j 系坐标映射到 i 系。
    于是 p_i = T_ij @ p_j，连续变换就是右乘。
    """
    rng = np.random.default_rng(0)
    T01, T12, T23 = [make_transform(rot_axis_angle(rng.normal(size=3), rng.uniform(-1, 1)),
                                    rng.uniform(-1, 1, size=3)) for _ in range(3)]

    T03 = T01 @ T12 @ T23
    assert np.allclose(invert_transform(T03) @ T03, np.eye(4), atol=1e-12)
    assert np.allclose(invert_transform(T03), np.linalg.inv(T03), atol=1e-12)
    # 逐层右乘（分步变换）与整体矩阵一次乘法，结果完全一致
    p3 = np.array([0.3, -0.2, 0.5])
    step = apply_transform(T01, apply_transform(T12, apply_transform(T23, p3)))
    assert np.allclose(step, apply_transform(T03, p3), atol=1e-12), "链式右乘必须等于整体乘法"
    # 顺序敏感：交换两个变换结果就不同，所以连续右乘不能随便调序
    assert not np.allclose(apply_transform(T01 @ T12, p3), apply_transform(T12 @ T01, p3), atol=1e-6)
    # 分开做“旋转 + 加法”与齐次矩阵一次乘法等价 —— 这正是用 4x4 的理由
    split = T01[:3, :3] @ p3 + T01[:3, 3]
    assert np.allclose(split, apply_transform(T01, p3), atol=1e-12)
    # 齐次矩阵最后一行恒为 [0, 0, 0, 1]
    assert np.allclose(T03[3, :], np.array([0.0, 0.0, 0.0, 1.0]))
    # 逆变换就是反向链路
    assert np.allclose(invert_transform(T01 @ T12), invert_transform(T12) @ invert_transform(T01), atol=1e-12)


# T_AB 表示 B 系在 A 系中的位姿，把 B 系坐标映射到 A 系
REFERENCE_T_AB = make_transform(euler_zyx_to_matrix([0.0, 0.0, np.pi / 4]), [0.5, 0.0, 0.2])


def check_coordinate_conversion() -> np.ndarray:
    """把 B 系下的一个点换算到 A 系，并验证逆变换能原样还原。"""
    T_AB = REFERENCE_T_AB
    p_B = np.array([1.0, 0.0, 0.0])
    p_A = apply_transform(T_AB, p_B)[0]
    assert np.allclose(apply_transform(invert_transform(T_AB), p_A)[0], p_B, atol=1e-12)
    # 手算校验：B 系绕 z 转 45°、原点在 (0.5, 0, 0.2)
    expected = np.array([np.cos(np.pi / 4) + 0.5, np.sin(np.pi / 4), 0.2])
    assert np.allclose(p_A, expected, atol=1e-12)
    return p_A


def forward_kinematics_2link(q: np.ndarray, lengths: tuple[float, float] = (0.6, 0.4)) -> np.ndarray:
    """两连杆平面臂正运动学：逐关节右乘齐次矩阵。

    第 i 个关节的变换 = 绕 z 轴转 q_i（关节旋转）后沿当前 x 轴平移 l_i（连杆）。
    写成矩阵就是 Rot_z(q_i) @ Trans(l_i, 0, 0)，两者不可交换 ——
    Trans @ Rot 得到的是“先在父系平移再旋转”，物理上是另一个机械臂。
    """
    T = np.eye(4)
    for qi, li in zip(np.atleast_1d(q), lengths, strict=True):
        T = T @ rot_z_h(qi) @ trans(li, 0.0, 0.0)
    return T


def analytic_2link(q: np.ndarray, lengths: tuple[float, float] = (0.6, 0.4)) -> np.ndarray:
    """两连杆平面臂正运动学的解析解，用来交叉验证齐次矩阵链。"""
    l1, l2 = lengths
    x = l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1])
    y = l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1])
    return np.array([x, y, 0.0])


def check_two_link_fk() -> None:
    for q in ([0.0, 0.0], [np.pi / 6, np.pi / 3], [-0.4, 1.1], [1.0, -0.5]):
        q = np.array(q)
        T = forward_kinematics_2link(q)
        assert np.allclose(T[:3, 3], analytic_2link(q), atol=1e-12), f"FK 与解析解不一致 q={q}"
        # 末端姿态角 = 两关节角之和（平面问题，只有绕 z 的自由度）
        assert np.isclose(np.arctan2(T[1, 0], T[0, 0]), q[0] + q[1], atol=1e-12)
    # Rot @ Trans != Trans @ Rot：顺序写错，机械臂形状就完全变了
    first_link = rot_z_h(0.6) @ trans(0.5, 0.0, 0.0)
    swapped = trans(0.5, 0.0, 0.0) @ rot_z_h(0.6)
    assert np.allclose(first_link[:3, 3], rot_z(0.6) @ np.array([0.5, 0.0, 0.0]), atol=1e-12)
    assert np.allclose(swapped[:3, 3], np.array([0.5, 0.0, 0.0]), atol=1e-12)
    assert not np.allclose(first_link, swapped, atol=1e-6)


def check_slerp() -> None:
    q0 = matrix_to_quat_xyzw(np.eye(3))
    q1 = matrix_to_quat_xyzw(rot_z(np.pi / 2))
    mid = slerp_quat(q0, q1, 0.5)
    assert np.isclose(2 * np.arccos(np.clip(abs(mid[3]), -1.0, 1.0)), np.pi / 4, atol=1e-9)
    assert np.allclose(slerp_quat(q0, q1, 0.0), q0, atol=1e-12)
    assert np.allclose(np.abs(slerp_quat(q0, q1, 1.0)), np.abs(q1), atol=1e-12)
    # 等间隔采样时相邻转角相等 —— 即匀速转动
    angles = [2 * np.arccos(np.clip(abs(slerp_quat(q0, q1, t)[3]), -1.0, 1.0)) for t in np.linspace(0, 1, 11)]
    assert np.allclose(np.diff(angles), np.diff(angles)[0], atol=1e-9)


# ---------------------------------------------------------------- 绘图 ----
def make_figure(out_dir) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))

    # (a) 两连杆平面臂：右乘齐次矩阵链
    ax = axes[0]
    for q, color in (([0.5, 0.9], "tab:blue"), ([1.2, -0.6], "tab:orange")):
        T = np.eye(4)
        pts = [T[:2, 3]]
        for qi, li in zip(q, (0.6, 0.4), strict=True):
            T = T @ rot_z_h(qi) @ trans(li, 0.0, 0.0)
            pts.append(T[:2, 3])
        pts_arr = np.array(pts)
        ax.plot(pts_arr[:, 0], pts_arr[:, 1], "-o", color=color, label=f"q = {q}")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title("(a) 两连杆正运动学：右乘链式")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(fontsize=8)

    # (b) 万向锁：pitch 逼近 90° 时欧拉角逆解崩溃
    ax = axes[1]
    pitches = np.linspace(0, np.pi / 2 - 1e-6, 400)
    errs = [np.linalg.norm(matrix_to_euler_zyx(euler_zyx_to_matrix([0.4, p, 0.7])) - np.array([0.4, p, 0.7]))
            for p in pitches]
    ax.semilogy(np.rad2deg(pitches), np.maximum(errs, 1e-16))
    ax.set_ylim(1e-16, 1.0)
    ax.grid(alpha=0.3, which="both")
    ax.set_title("(b) 万向锁：pitch → 90° 时逆解失效")
    ax.set_xlabel("pitch [deg]")
    ax.set_ylabel("欧拉角逆解误差 [rad]")

    # (c) 姿态插值：SLERP 匀速，欧拉角分量线性插值走非测地线
    ax = axes[2]
    q0 = matrix_to_quat_xyzw(np.eye(3))
    rpy1 = np.array([0.7, np.pi / 2 - 0.05, 1.2])
    q1 = matrix_to_quat_xyzw(euler_zyx_to_matrix(rpy1))
    ts = np.linspace(0, 1, 200)
    sl = np.array([2 * np.arccos(np.clip(abs(slerp_quat(q0, q1, t)[3]), -1.0, 1.0)) for t in ts])
    eu = np.array([2 * np.arccos(np.clip(abs(matrix_to_quat_xyzw(euler_zyx_to_matrix(t * rpy1))[3]), -1.0, 1.0))
                   for t in ts])
    ax.plot(ts, np.rad2deg(sl), label="四元数 SLERP")
    ax.plot(ts, np.rad2deg(eu), "--", label="欧拉角分量线性插值")
    ax.grid(alpha=0.3)
    ax.set_title("(c) 姿态插值：相对起始姿态的转角")
    ax.set_xlabel("归一化时间 t")
    ax.set_ylabel("转角 [deg]")
    ax.legend(fontsize=8)
    # SLERP 的角速度恒定，欧拉角线性插值不是
    assert np.allclose(np.diff(sl), np.diff(sl)[0], atol=1e-9)
    assert not np.allclose(np.diff(eu), np.diff(eu)[0], atol=1e-3)

    fig.tight_layout()
    fig.savefig(out_dir / "fig_pose_transform.png", dpi=140)
    plt.close(fig)


def main() -> None:
    from pathlib import Path

    check_rotation_basics()
    check_euler_quat_roundtrip()
    check_homogeneous_chain()
    p_A = check_coordinate_conversion()
    check_two_link_fk()
    check_slerp()

    print("T_AB (B 系在 A 系中的位姿):\n", np.round(REFERENCE_T_AB, 4))
    print("p_B = [1, 0, 0] -> p_A =", np.round(p_A, 4))
    print("校验 T_AB @ T_BA = I:", np.allclose(REFERENCE_T_AB @ invert_transform(REFERENCE_T_AB), np.eye(4)))

    out_dir = Path(__file__).resolve().parent.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    make_figure(out_dir)
    print(f"[fig] {out_dir / 'fig_pose_transform.png'}")
    print("01_pose_transform: all assertions passed")


if __name__ == "__main__":
    main()
