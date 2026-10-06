"""复现 2：6 自由度机械臂正运动学、雅可比与数值逆运动学

对应周报第一部分的“多关节坐标系链式变换”作业题。
用标准 DH 参数建一个类 UR5 的 6 轴臂，验证三件事：

1. 末端位姿 = 各关节齐次矩阵连续右乘（T_06 = T_01 T_12 ... T_56）；
2. 解析几何雅可比 与 有限差分雅可比 一致；
3. 用阻尼最小二乘（DLS）迭代求逆运动学，能把末端位姿收敛到目标。

运行：

    python 02_forward_kinematics.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from mpl_style import setup
from se3 import (
    apply_transform,
    invert_transform,
    make_transform,
    matrix_to_quat_xyzw,
    rot_x,
    rot_z,
    rot_z_h,
    trans,
    transform_to_rpy,
)

plt = setup()


def rot_x_h(a: float) -> np.ndarray:
    """绕 x 轴转 a 弧度的 4x4 齐次矩阵（无平移）。"""
    return make_transform(rot_x(a), np.zeros(3))

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

# ---------------------------------------------------------------- 机器人模型 ----
# 标准 DH 参数：(a_i, alpha_i, d_i, theta_offset_i)，长度单位 m，角度单位 rad。
# 数值取自 UR5 官方规格。
# 记法：T_(i-1,i) = Rz(theta_i) Tz(d_i) Tx(a_i) Rx(alpha_i)
DH_TABLE = np.array([
    [0.0, np.pi / 2, 0.089159, 0.0],     # 关节 1
    [-0.425, 0.0, 0.0, 0.0],             # 关节 2
    [-0.39225, 0.0, 0.0, 0.0],           # 关节 3
    [0.0, np.pi / 2, 0.10915, 0.0],      # 关节 4
    [0.0, -np.pi / 2, 0.09465, 0.0],     # 关节 5
    [0.0, 0.0, 0.0823, 0.0],             # 关节 6
])

JOINT_LIMITS = np.array([
    [-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi],
    [-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi],
])


def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """标准 DH 关节变换 T = Rz(theta) Tz(d) Tx(a) Rx(alpha)。

    展开后平移列是 Rz(theta) @ [a, 0, d] —— 注意先平移再被 Rz(theta) 带走，
    漏掉这一步整个链路就错了。
    """
    R = rot_z(theta) @ rot_x(alpha)
    return make_transform(R, rot_z(theta) @ np.array([a, 0.0, d]))


def forward_kinematics(q: np.ndarray, table: np.ndarray = DH_TABLE) -> list[np.ndarray]:
    """正运动学。返回 [T_01, T_02, ..., T_0n]，即每个关节系在基座系中的位姿。"""
    frames = []
    T = np.eye(4)
    for theta_i, row in zip(q, table, strict=True):
        a, alpha, d, offset = row
        T = T @ dh_transform(a, alpha, d, theta_i + offset)
        frames.append(T.copy())
    return frames


def jacobian_geometric(q: np.ndarray, frames: list[np.ndarray]) -> np.ndarray:
    """几何雅可比 (6xn)：把关节速度映射到末端线速度/角速度（基坐标系表达）。

    线速度列 J_v = z_i x (p_end - p_i)，角速度列 J_w = z_i，其中 z_i 是第 i
    个关节轴在基座系中的方向。
    """
    p_end = frames[-1][:3, 3]
    J = np.zeros((6, len(q)))
    for i in range(len(q)):
        T_prev = frames[i - 1] if i > 0 else np.eye(4)
        z_i = T_prev[:3, 2]  # 关节 i 的旋转轴（DH 下每个关节都绕自身 z 转）
        p_i = T_prev[:3, 3]
        J[:3, i] = np.cross(z_i, p_end - p_i)
        J[3:, i] = z_i
    return J


def jacobian_finite_difference(q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """有限差分雅可比，只用来交叉验证解析雅可比，不用于控制。"""
    T0 = forward_kinematics(q)[-1]
    J = np.zeros((6, len(q)))
    for i in range(len(q)):
        dq = np.zeros_like(q)
        dq[i] = eps
        T1 = forward_kinematics(q + dq)[-1]
        J[:3, i] = (T1[:3, 3] - T0[:3, 3]) / eps
        # 姿态差的旋转向量 / eps，近似角速度
        dR = T1[:3, :3] @ T0[:3, :3].T
        J[3:, i] = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2 * eps)
    return J


def pose_error(T_current: np.ndarray, T_target: np.ndarray) -> np.ndarray:
    """位姿误差 [位置误差(3); 姿态误差(3)]，姿态部分用旋转向量表示，在基座系中。"""
    e_pos = T_target[:3, 3] - T_current[:3, 3]
    dR = T_target[:3, :3] @ T_current[:3, :3].T
    e_rot = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / 2
    return np.concatenate([e_pos, e_rot])


def inverse_kinematics_dls(
    T_target: np.ndarray,
    q_init: np.ndarray,
    max_iter: int = 400,
    tol: float = 1e-10,
    damping: float = 1e-3,
    step_limit: float = 0.2,
) -> tuple[np.ndarray, list[float]]:
    """阻尼最小二乘（Levenberg-Marquardt 风格）数值逆解。

    dq = J^T (J J^T + lambda^2 I)^-1 e
    阻尼项在奇异位形附近让解保持有界，代价是收敛略慢。
    """
    q = q_init.astype(float).copy()
    history = []
    for _ in range(max_iter):
        frames = forward_kinematics(q)
        err = pose_error(frames[-1], T_target)
        history.append(np.linalg.norm(err))
        if history[-1] < tol:
            break
        J = jacobian_geometric(q, frames)
        dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), err)
        # 单步限幅：避免在奇异点附近跳出一大步
        scale = np.max(np.abs(dq)) / step_limit
        if scale > 1.0:
            dq = dq / scale
        q = np.clip(q + dq, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
    return q, history


# ---------------------------------------------------------------- 自检 ----
def check_dh_transform_formula() -> None:
    """把 DH 变换与教科书展开式逐项对照，避免“自己验证自己”。"""
    a, alpha, d, theta = -0.425, np.pi / 2, 0.10915, 0.7
    T = dh_transform(a, alpha, d, theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    expected = np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ])
    assert np.allclose(T, expected, atol=1e-12), f"DH 变换与展开式不符\n{T}\n{expected}"
    # θ 必须影响平移列：否则只是“先平移再旋转”的另一种写法
    assert not np.allclose(dh_transform(a, alpha, d, 0.0)[:3, 3], T[:3, 3], atol=1e-6)


def check_dh_chain() -> None:
    """逐级手写矩阵乘（不用 forward_kinematics），与函数结果一致。"""
    q = np.array([0.3, -0.8, 1.1, -0.5, 1.4, 0.2])
    frames = forward_kinematics(q)
    assert len(frames) == 6

    # 按定义展开：Rz Tz Tx Rx 四个基本矩阵顺序相乘，不借助 dh_transform
    T = np.eye(4)
    for i, (theta_i, row) in enumerate(zip(q, DH_TABLE, strict=True)):
        a, alpha, d, offset = row
        T = T @ rot_z_h(theta_i + offset) @ trans(0.0, 0.0, d) @ trans(a, 0.0, 0.0) @ rot_x_h(alpha)
        assert np.allclose(T, frames[i], atol=1e-12), f"第 {i + 1} 级链路不一致"
        assert np.allclose(T[3, :], np.array([0.0, 0.0, 0.0, 1.0]))
        assert np.isclose(np.linalg.det(T[:3, :3]), 1.0, atol=1e-12)
    # T_0n 的逆 = T_n0，把它作用回末端点必须落在基座原点
    assert np.allclose(apply_transform(invert_transform(frames[-1]), frames[-1][:3, 3]), np.zeros(3), atol=1e-12)


def check_jacobian() -> None:
    """几何雅可比 vs 有限差分雅可比，两者必须在数值精度内一致。"""
    rng = np.random.default_rng(1)
    for _ in range(4):
        q = rng.uniform(-1.5, 1.5, size=6)
        frames = forward_kinematics(q)
        J_geo = jacobian_geometric(q, frames)
        J_fd = jacobian_finite_difference(q)
        # 姿态部分依赖差分步长，容差放宽到 1e-5
        assert np.allclose(J_geo[:3], J_fd[:3], atol=1e-6), "位置雅可比不匹配"
        assert np.allclose(J_geo[3:], J_fd[3:], atol=1e-5), "姿态雅可比不匹配"


def check_jacobian_velocity() -> None:
    """J @ dq 应等于末端位姿的瞬时变化率。"""
    q = np.array([0.2, 0.5, -0.7, 0.1, 0.9, -0.3])
    dq = np.array([0.4, -0.2, 0.3, 0.1, -0.5, 0.2])
    dt = 1e-7
    T0 = forward_kinematics(q)[-1]
    T1 = forward_kinematics(q + dq * dt)[-1]
    dR = T1[:3, :3] @ T0[:3, :3].T
    v_num = np.concatenate([
        (T1[:3, 3] - T0[:3, 3]) / dt,
        np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2 * dt),
    ])
    v_jac = jacobian_geometric(q, forward_kinematics(q)) @ dq
    assert np.allclose(v_num, v_jac, atol=1e-5), f"雅可比速度映射不符\n{v_num}\n{v_jac}"


def check_inverse_kinematics() -> None:
    """先正解出一个可达位姿，再逆解回去，关节角与位姿都要复原。"""
    q_true = np.array([0.4, -0.6, 0.9, -0.4, 1.2, 0.3])
    T_target = forward_kinematics(q_true)[-1]
    q_sol, hist = inverse_kinematics_dls(T_target, np.zeros(6))
    assert hist[-1] < 1e-8, f"IK 未收敛，末误差 {hist[-1]:.3e}"
    T_sol = forward_kinematics(q_sol)[-1]
    assert np.allclose(T_sol[:3, 3], T_target[:3, 3], atol=1e-7), "IK 位置不匹配"
    assert np.allclose(T_sol[:3, :3], T_target[:3, :3], atol=1e-7), "IK 姿态不匹配"
    # 初始猜测不同应收敛到同一末端位姿（关节角可能不同解，但位姿必须一致）
    q_sol2, hist2 = inverse_kinematics_dls(T_target, np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
    assert hist2[-1] < 1e-8
    assert np.allclose(forward_kinematics(q_sol2)[-1], T_target, atol=1e-7)


def check_singularity() -> None:
    """奇异位形：腕部对齐时雅可比秩亏，阻尼项是保证数值稳定的关键。"""
    q_sing = np.array([0.0, -np.pi / 2, 0.0, 0.0, 0.0, 0.0])
    J = jacobian_geometric(q_sing, forward_kinematics(q_sing))
    s = np.linalg.svd(J, compute_uv=False)
    assert s[-1] < 1e-6, f"该位形应为奇异位形，最小奇异值 {s[-1]:.3e}"
    # 无阻尼的最小二乘在奇异处给出巨大关节速度，阻尼后可控
    e = np.array([0.0, 0.0, -0.01, 0.0, 0.0, 0.0])
    dq_bare = np.linalg.pinv(J) @ e
    dq_damped = J.T @ np.linalg.solve(J @ J.T + 1e-3**2 * np.eye(6), e)
    assert np.linalg.norm(dq_damped) < np.linalg.norm(dq_bare)


# ---------------------------------------------------------------- 绘图 ----
def make_figure(out_dir: Path) -> None:
    q_true = np.array([0.4, -0.6, 0.9, -0.4, 1.2, 0.3])
    T_target = forward_kinematics(q_true)[-1]
    q_sol, hist = inverse_kinematics_dls(T_target, np.zeros(6))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))

    # (a) 机械臂在 xz 平面的投影（初始位形 vs 逆解位形）
    ax = axes[0]
    for q, color, label in ((np.zeros(6), "tab:gray", "初始位形"),
                            (q_sol, "tab:blue", "IK 解")):
        pts = np.array([[0, 0, 0]] + [T[:3, 3] for T in forward_kinematics(q)])
        ax.plot(pts[:, 0], pts[:, 2], "-o", color=color, label=label, ms=4)
    ax.plot(*T_target[[0, 2], 3], "r*", ms=14, label="目标点")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title("(a) IK 求解：基座系 xz 投影")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.legend(fontsize=8)

    # (b) 雅可比交叉验证：解析 vs 有限差分
    ax = axes[1]
    J_geo = jacobian_geometric(q_sol, forward_kinematics(q_sol))
    J_fd = jacobian_finite_difference(q_sol)
    idx = np.arange(6)
    ax.bar(idx - 0.2, J_geo[0], width=0.4, label="解析几何雅可比")
    ax.bar(idx + 0.2, J_fd[0], width=0.4, label="有限差分")
    ax.grid(alpha=0.3, axis="y")
    ax.set_title("(b) 雅可比第 1 行（末端 x 方向线速度）")
    ax.set_xlabel("关节序号")
    ax.set_ylabel("dJ/dq")
    ax.set_xticks(idx, [f"J{i + 1}" for i in idx])
    ax.legend(fontsize=8)

    # (c) IK 收敛曲线
    ax = axes[2]
    ax.semilogy(np.arange(len(hist)), np.maximum(hist, 1e-16))
    ax.grid(alpha=0.3, which="both")
    ax.set_title("(c) 阻尼最小二乘 IK 收敛")
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("位姿误差范数")
    # DLS 是线性收敛：误差单调下降
    assert np.all(np.diff(hist) <= 1e-12), "IK 误差应单调不增"

    fig.tight_layout()
    fig.savefig(out_dir / "fig_forward_kinematics.png", dpi=140)
    plt.close(fig)


def main() -> None:
    check_dh_transform_formula()
    check_dh_chain()
    check_jacobian()
    check_jacobian_velocity()
    check_inverse_kinematics()
    check_singularity()

    q = np.array([0.3, -0.8, 1.1, -0.5, 1.4, 0.2])
    T = forward_kinematics(q)[-1]
    print("关节角 q =", q)
    print("末端位姿 T_06:\n", np.round(T, 4))
    print("末端位姿 [x, y, z, roll, pitch, yaw] =", np.round(transform_to_rpy(T), 4))
    print("末端四元数 [x, y, z, w] =", np.round(matrix_to_quat_xyzw(T[:3, :3]), 4))
    s = np.linalg.svd(jacobian_geometric(q, forward_kinematics(q)), compute_uv=False)
    print("雅可比奇异值 =", np.round(s, 4), "-> 可操作度 =", round(np.prod(s), 6))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    make_figure(FIG_DIR)
    print(f"[fig] {FIG_DIR / 'fig_forward_kinematics.png'}")
    print("02_forward_kinematics: all assertions passed")


if __name__ == "__main__":
    main()
