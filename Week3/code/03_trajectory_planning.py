"""复现 3：轨迹规划（多项式插值、梯形速度剖面、笛卡尔直线 + 姿态 SLERP）

对应周报第二部分。回答两个作业题时用到的可运行依据：

1. 为什么必须限制速度和加速度 —— 打印每个剖面在 limt 内/外的实测峰值；
2. 规定 2 秒完成一个动作时怎么排 —— plan_with_fixed_duration 在给定 T 下
   搜索最优加速时间，并给出“是否可行、需要放宽哪个限制”的结论。

运行：

    python 03_trajectory_planning.py

输出：../figures/fig_trajectory.png 与 ../results/trajectory.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from mpl_style import setup
from se3 import euler_zyx_to_matrix, matrix_to_quat_xyzw, slerp_quat
from traj import lspb_eval, lspb_profile, plan_with_fixed_duration

plt = setup()

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
RESULT_DIR = ROOT / "results"


# ------------------------------------------------------ 关节空间多项式插值 ----
def cubic_coeffs(q0: float, q1: float, T: float) -> np.ndarray:
    """三次多项式 a0 + a1 t + a2 t^2 + a3 t^3，两端速度为零。

    四个边界条件恰好定出四个系数，解是闭式的：
    a0 = q0, a1 = 0, a2 = 3(q1-q0)/T^2, a3 = -2(q1-q0)/T^3
    """
    return np.array([q0, 0.0, 3 * (q1 - q0) / T**2, -2 * (q1 - q0) / T**3])


def quintic_coeffs(q0: float, q1: float, T: float,
                   v0: float = 0.0, v1: float = 0.0, a0: float = 0.0, a1: float = 0.0) -> np.ndarray:
    """五次多项式，可同时指定两端的位移/速度/加速度（6 个条件 -> 6 个系数）。"""
    A = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0],
        [1, T, T**2, T**3, T**4, T**5],
        [0, 1, 2 * T, 3 * T**2, 4 * T**3, 5 * T**4],
        [0, 0, 2, 6 * T, 12 * T**2, 20 * T**3],
    ], dtype=float)
    return np.linalg.solve(A, np.array([q0, v0, a0, q1, v1, a1], dtype=float))


def poly_eval(coeffs: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """同时算多项式的位置、速度、加速度（对系数求导，而不是数值差分）。"""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    powers = np.vstack([t**k for k in range(len(coeffs))])
    q = coeffs @ powers
    dq = np.array([k * coeffs[k] for k in range(1, len(coeffs))]) @ np.vstack([t**k for k in range(len(coeffs) - 1)])
    ddq = np.array([k * (k - 1) * coeffs[k] for k in range(2, len(coeffs))]) @ np.vstack(
        [t**k for k in range(len(coeffs) - 2)])
    return q, dq, ddq


# ------------------------------------------------------ 梯形速度剖面 (LSPB) ----
# lspb_profile / lspb_eval / plan_with_fixed_duration 实现在 traj.py，
# 04_rrt_planner.py 也用同一份，避免两处各写一遍造成不一致。


# ------------------------------------------------------ 笛卡尔空间轨迹 ----
def cartesian_line(p_start: np.ndarray, p_end: np.ndarray,
                   R_start: np.ndarray, R_end: np.ndarray, n: int) -> tuple[np.ndarray, list]:
    """笛卡尔直线：位置线性插值 + 姿态四元数 SLERP。

    位置线性保证末端走直线（关节空间线性插值做不到这点），
    姿态用 SLERP 保证角速度恒定、不经过奇怪的中间姿态。
    """
    ts = np.linspace(0, 1, n)
    positions = np.array([(1 - t) * p_start + t * p_end for t in ts])
    q0 = matrix_to_quat_xyzw(R_start)
    q1 = matrix_to_quat_xyzw(R_end)
    rotations = [slerp_quat(q0, q1, t) for t in ts]
    return positions, rotations


def cartesian_line_path_length(p_start: np.ndarray, p_end: np.ndarray, n: int = 400) -> float:
    """数值验证：笛卡尔直线插值得到的路径长度等于两点直线距离。"""
    ts = np.linspace(0, 1, n)
    pts = np.array([(1 - t) * p_start + t * p_end for t in ts])
    return np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))


# ------------------------------------------------------ 自检 ----
def check_polynomial_boundaries() -> None:
    q0, q1, T = 0.3, 1.7, 1.5
    for coeffs, name in ((cubic_coeffs(q0, q1, T), "三次"),
                         (quintic_coeffs(q0, q1, T, 0.1, -0.2, 0.0, 0.0), "五次")):
        q, dq, ddq = poly_eval(coeffs, np.array([0.0, T]))
        assert np.isclose(q[0], q0, atol=1e-12), f"{name}多项式起点错误"
        assert np.isclose(q[1], q1, atol=1e-12), f"{name}多项式终点错误"
        if name == "三次":
            assert np.allclose(dq, 0.0, atol=1e-12), "三次多项式两端速度应为 0"
        else:
            assert np.allclose(dq, [0.1, -0.2], atol=1e-9), "五次多项式端点速度未命中"
            assert np.allclose(ddq, [0.0, 0.0], atol=1e-9), "五次多项式端点加速度未命中"


def check_lspb_limits() -> None:
    """LSPB 剖面必须满足速度和加速度上限，且端点精确到位。"""
    for q0, q1, vmax, amax in ((0.0, 1.0, 0.6, 2.0), (0.0, 0.05, 0.6, 2.0), (0.5, -0.7, 0.4, 1.5)):
        prof = lspb_profile(q0, q1, vmax, amax)
        t = np.linspace(0, prof["T"], 5001)
        q, dq, ddq = lspb_eval(prof, t)
        assert np.allclose(q[0], q0, atol=1e-12) and np.allclose(q[-1], q1, atol=1e-12), "LSPB 端点错误"
        assert np.max(np.abs(dq)) <= vmax + 1e-9, f"速度超限 {np.max(np.abs(dq))} > {vmax}"
        assert np.max(np.abs(ddq)) <= amax + 1e-9, f"加速度超限 {np.max(np.abs(ddq))} > {amax}"
        # 位置必须单调（不会中途倒退）
        assert np.all(np.diff(q) >= -1e-12) or np.all(np.diff(q) <= 1e-12), "LSPB 位置非单调"
    # 短距离走三角形、长距离走梯形
    assert lspb_profile(0.0, 0.05, 0.6, 2.0)["kind"] == "triangle"
    assert lspb_profile(0.0, 5.0, 0.6, 2.0)["kind"] == "trapezoid"


def check_cubic_violates_limits() -> None:
    """三次多项式只约束端点，中间段的峰值会随 T 变小而超限 ——
    这正是“必须显式限制速度/加速度”的算术依据。"""
    q0, q1, vmax, amax = 0.0, 1.0, 0.6, 2.0
    peaks = []
    for T in (2.0, 1.0, 0.5):
        coeffs = cubic_coeffs(q0, q1, T)
        _, dq, ddq = poly_eval(coeffs, np.linspace(0, T, 2001))
        peaks.append((T, np.max(np.abs(dq)), np.max(np.abs(ddq))))
    # T 越小峰值越大
    assert peaks[0][1] < peaks[1][1] < peaks[2][1]
    # 1.5 * D / T 是三次多项式峰值速度的解析值
    T = 1.0
    assert np.isclose(peaks[1][1], 1.5 * (q1 - q0) / T, atol=1e-9)
    assert np.isclose(peaks[1][2], 6.0 * (q1 - q0) / T**2, atol=1e-9)
    # 短时间下确实超限：说明必须主动检查，不能只给端点条件
    assert peaks[2][1] > vmax
    assert peaks[2][2] > amax


def check_fixed_duration() -> None:
    """固定时长 T 下排轨迹：最优加速时间恰好是 vmax/amax，
    于是“T 秒内能否完成”等价于“自然 LSPB 时长是否 <= T”。"""
    vmax, amax = 0.6, 2.0

    for D, T in ((0.8, 2.0), (1.2, 2.0), (1.2, 0.5), (0.1, 0.5)):
        plan = plan_with_fixed_duration(0.0, D, T, vmax, amax)
        # 峰值速度不可能低于平均速度 D/T，也不可能高于 2D/T（三角形极限）
        assert D / T - 1e-9 <= plan["v_peak"] <= 2 * D / T + 1e-9
        # 最优加速时间：交点被截断在 (0, T/2] 内
        assert np.isclose(plan["t_acc"], min(vmax / amax, T / 2), rtol=1e-12)
        if vmax / amax <= T / 2:  # 交点在区间内 -> 两个限制同时顶满
            assert np.isclose(plan["v_peak"] / vmax, plan["a_peak"] / amax, rtol=1e-12)
        # 解析最优解不差于任何网格搜索
        grid = [(max(D / (T - ta) / vmax, D / (T - ta) / ta / amax), ta)
                for ta in np.linspace(1e-4, T / 2, 2001)]
        best_load, _ = min(grid)
        assert plan["load"] <= best_load + 1e-9, "解析最优解应不差于网格搜索"
        # 可行性判据必须与“自然 LSPB 时长”一致
        natural = lspb_profile(0.0, D, vmax, amax)["T"]
        assert plan["feasible"] == (natural <= T + 1e-9), \
            f"可行性判据与自然时长不一致: D={D}, T={T}, 自然时长={natural:.4f}"
    # 梯形剖面的自然时长就是 D/vmax + vmax/amax（若够长跑得到 vmax）
    assert np.isclose(lspb_profile(0.0, 0.8, vmax, amax)["T"], 0.8 / vmax + vmax / amax, atol=1e-9)

    # 具体结论：0.8 rad 可行；1.2 rad 在 2s 内不可行（需要 >= 2.3s）
    assert plan_with_fixed_duration(0.0, 0.8, 2.0, vmax, amax)["feasible"]
    tight = plan_with_fixed_duration(0.0, 1.2, 2.0, vmax, amax)
    assert not tight["feasible"]
    assert np.isclose(lspb_profile(0.0, 1.2, vmax, amax)["T"], 2.3, atol=1e-6)
    assert not plan_with_fixed_duration(0.0, 1.2, 0.5, vmax, amax)["feasible"]


def check_cartesian_vs_joint() -> None:
    """笛卡尔直线插值走直线；同样的两个端点若在关节空间线性插值，
    末端会走一条弧线（起终点距离相同，但路径更长）。"""
    p_start = np.array([0.5, 0.0, 0.3])
    p_end = np.array([0.0, 0.5, 0.3])
    straight = cartesian_line_path_length(p_start, p_end)
    assert np.isclose(straight, np.linalg.norm(p_end - p_start), atol=1e-9)

    # 用“关节空间线性插值 + 正运动学”造出一条弧线作对比
    def fk2(q1: float, q2: float) -> np.ndarray:
        return np.array([0.6 * np.cos(q1) + 0.4 * np.cos(q1 + q2),
                         0.6 * np.sin(q1) + 0.4 * np.sin(q1 + q2), 0.3])

    qa = np.array([0.0, np.pi / 2])
    qb = np.array([np.pi / 2, -np.pi / 2])
    arc = np.array([fk2(*(1 - t) * qa + t * qb) for t in np.linspace(0, 1, 400)])
    arc_len = np.sum(np.linalg.norm(np.diff(arc, axis=0), axis=1))
    chord = np.linalg.norm(arc[-1] - arc[0])
    assert arc_len > chord * 1.01, f"关节空间插值末端应走弧线：{arc_len:.4f} vs 直线 {chord:.4f}"


# ------------------------------------------------------ 主流程 ----
def main() -> None:
    check_polynomial_boundaries()
    check_lspb_limits()
    check_cubic_violates_limits()
    check_fixed_duration()
    check_cartesian_vs_joint()

    q0, q1 = 0.0, 1.0
    T_cubic, vmax, amax = 1.0, 0.6, 2.0

    # ---- 三次多项式 vs LSPB：峰值对比 ----
    cubic = cubic_coeffs(q0, q1, T_cubic)
    t_cubic = np.linspace(0, T_cubic, 1001)
    q_c, dq_c, ddq_c = poly_eval(cubic, t_cubic)
    prof = lspb_profile(q0, q1, vmax, amax)
    t_lspb = np.linspace(0, prof["T"], 1001)
    q_l, dq_l, ddq_l = lspb_eval(prof, t_lspb)

    print(f"三次多项式 T={T_cubic}s: 峰值速度 {np.max(np.abs(dq_c)):.3f} rad/s "
          f"(限 {vmax}), 峰值加速度 {np.max(np.abs(ddq_c)):.3f} rad/s^2 (限 {amax})")
    print(f"LSPB       T={prof['T']:.3f}s: 峰值速度 {np.max(np.abs(dq_l)):.3f} rad/s, "
          f"峰值加速度 {np.max(np.abs(ddq_l)):.3f} rad/s^2  [{prof['kind']}]")
    assert np.max(np.abs(dq_l)) <= vmax + 1e-9 and np.max(np.abs(ddq_l)) <= amax + 1e-9

    # ---- 固定时长 2s 的规划 ----
    vmax, amax = 0.6, 2.0
    print(f"\n速度上限 {vmax} rad/s、加速度上限 {amax} rad/s^2，要求在 2.0s 内完成：")
    for D in (0.8, 1.2):
        p = plan_with_fixed_duration(q0, q0 + D, 2.0, vmax, amax)
        natural = lspb_profile(q0, q0 + D, vmax, amax)["T"]
        print(f"  行程 {D:.2f} rad: 自然时长 {natural:.3f}s -> "
              f"v_peak={p['v_peak']:.3f} ({p['v_peak'] / vmax:.2f}x vmax), "
              f"a_peak={p['a_peak']:.3f} ({p['a_peak'] / amax:.2f}x amax), 可行={p['feasible']}")

    # ---- 笛卡尔轨迹 ----
    p_start, p_end = np.array([0.5, 0.0, 0.3]), np.array([0.0, 0.5, 0.3])
    R_start = euler_zyx_to_matrix([0.0, np.pi / 2, 0.0])
    R_end = euler_zyx_to_matrix([0.4, 0.0, np.pi / 2])
    positions, rotations = cartesian_line(p_start, p_end, R_start, R_end, 200)
    print(f"\n笛卡尔直线: 路径长度 {cartesian_line_path_length(p_start, p_end):.6f} m, "
          f"直线距离 {np.linalg.norm(p_end - p_start):.6f} m")

    # ---- 导出数据：LSPB 轨迹，就位后保持静止 ----
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    T_req = 2.0
    t = np.linspace(0, T_req, 401)
    reprof = lspb_profile(q0, q0 + 0.8, vmax, amax)
    q_lspb, dq_lspb, ddq_lspb = lspb_eval(reprof, np.minimum(t, reprof["T"]))
    settled = t > reprof["T"]
    dq_lspb[settled] = 0.0
    ddq_lspb[settled] = 0.0
    csv_path = RESULT_DIR / "trajectory.csv"
    np.savetxt(csv_path, np.column_stack([t, q_lspb, dq_lspb, ddq_lspb]),
               delimiter=",", header="t,q,dq,ddq", comments="", fmt="%.6f")
    print(f"\nLSPB 行程 0.8 rad 在 {reprof['T']:.3f}s 就位，剩余时间保持静止")
    print(f"[csv] {csv_path}")

    # ---- 绘图 ----
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))

    ax = axes[0]
    ax.plot(t_cubic, q_c, label=f"三次多项式 T={T_cubic}s")
    ax.plot(t_lspb, q_l, "--", label=f"LSPB T={prof['T']:.2f}s")
    quint = quintic_coeffs(q0, q1, T_cubic, 0.0, 0.0, 0.0, 0.0)
    q_q, _, _ = poly_eval(quint, t_cubic)
    ax.plot(t_cubic, q_q, ":", label=f"五次多项式 T={T_cubic}s")
    ax.grid(alpha=0.3)
    ax.set_title("(a) 关节位置曲线")
    ax.set_xlabel("时间 [s]")
    ax.set_ylabel("位置 [rad]")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(t_cubic, dq_c, label="三次多项式 速度")
    ax.plot(t_lspb, dq_l, "--", label="LSPB 速度")
    ax.plot(t_cubic, ddq_c, label="三次多项式 加速度")
    ax.plot(t_lspb, ddq_l, "--", label="LSPB 加速度")
    ax.axhline(vmax, color="r", lw=1, ls=":", label=f"$v_{{max}}$={vmax}")
    ax.axhline(-vmax, color="r", lw=1, ls=":")
    ax.axhline(amax, color="g", lw=1, ls=":", label=f"$a_{{max}}$={amax}")
    ax.axhline(-amax, color="g", lw=1, ls=":")
    ax.grid(alpha=0.3)
    ax.set_title("(b) 速度/加速度与限制线")
    ax.set_xlabel("时间 [s]")
    ax.legend(fontsize=6, ncol=2)

    ax = axes[2]
    ts = np.linspace(0, 1, 200)

    def fk2(q1: float, q2: float) -> np.ndarray:
        return np.array([0.6 * np.cos(q1) + 0.4 * np.cos(q1 + q2),
                         0.6 * np.sin(q1) + 0.4 * np.sin(q1 + q2)])

    qa, qb = np.array([0.0, np.pi / 2]), np.array([np.pi / 2, -np.pi / 2])
    arc_pts = np.array([fk2(*((1 - t) * qa + t * qb)) for t in ts])
    line_pts = np.array([(1 - t) * arc_pts[0] + t * arc_pts[-1] for t in ts])
    ax.plot(arc_pts[:, 0], arc_pts[:, 1], label="关节空间线性插值")
    ax.plot(line_pts[:, 0], line_pts[:, 1], "--", label="笛卡尔直线插值")
    ax.plot([0.0], [0.0], "k+", ms=12, label="基座")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title("(c) 末端路径：关节空间 vs 笛卡尔空间")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_trajectory.png", dpi=140)
    plt.close(fig)
    print(f"[fig] {FIG_DIR / 'fig_trajectory.png'}")
    print("03_trajectory_planning: all assertions passed")


if __name__ == "__main__":
    main()
