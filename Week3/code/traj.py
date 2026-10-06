"""轨迹时间参数化：梯形/三角形速度剖面（LSPB, Linear Segments with Parabolic Blends）。

被 03_trajectory_planning.py 和 04_rrt_planner.py 复用。

工业控制器最常用的点对点运动方式：先匀加速、再匀速、最后匀减速。
相比三次/五次多项式（只约束端点，中间可能超限），LSPB 对速度和加速度
都有硬上限，代价是加速度不连续（加加速度冲击）。
"""

from __future__ import annotations

import numpy as np


def lspb_profile(q0: float, q1: float, vmax: float, amax: float) -> dict:
    """按 (vmax, amax) 生成耗时最短的梯形/三角形速度剖面。

    能跑到 vmax 就是梯形，跑不到就是三角形（顶速由 sqrt(D*amax) 决定）。
    """
    assert vmax > 0 and amax > 0, "vmax / amax 必须为正"
    D = q1 - q0
    s = np.sign(D)
    dist = abs(D)
    if dist < 1e-15:
        return {"T": 0.0, "t_acc": 0.0, "t_flat": 0.0, "v_peak": 0.0,
                "kind": "zero", "q0": q0, "q1": q1}

    t_acc = vmax / amax
    if dist >= vmax * t_acc:  # 能到达 vmax -> 梯形
        t_flat = (dist - vmax * t_acc) / vmax
        v_peak, kind = vmax, "trapezoid"
    else:  # 到不了 vmax -> 三角形
        v_peak = np.sqrt(dist * amax)
        t_acc = v_peak / amax
        t_flat = 0.0
        kind = "triangle"
    return {"T": 2 * t_acc + t_flat, "t_acc": t_acc, "t_flat": t_flat,
            "v_peak": v_peak * s, "kind": kind, "q0": q0, "q1": q1}


def lspb_eval(profile: dict, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在给定时刻求 LSPB 剖面的位置、速度、加速度。"""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    q0 = profile["q0"]
    v_peak = profile["v_peak"]
    t_acc = profile["t_acc"]
    t_flat = profile["t_flat"]
    T = profile["T"]
    acc = v_peak / t_acc if t_acc > 0 else 0.0

    q = np.empty_like(t)
    dq = np.empty_like(t)
    ddq = np.empty_like(t)
    for i, ti in enumerate(t):
        if ti < 0 or ti > T + 1e-12:
            raise ValueError(f"时刻 {ti} 超出轨迹区间 [0, {T}]")
        if ti <= t_acc:  # 加速段
            q[i] = q0 + 0.5 * acc * ti**2
            dq[i], ddq[i] = acc * ti, acc
        elif ti <= t_acc + t_flat:  # 匀速段
            q[i] = q0 + 0.5 * acc * t_acc**2 + v_peak * (ti - t_acc)
            dq[i], ddq[i] = v_peak, 0.0
        else:  # 减速段
            tau = T - ti
            q[i] = profile["q1"] - 0.5 * acc * tau**2
            dq[i], ddq[i] = acc * tau, -acc
    return q, dq, ddq


def plan_with_fixed_duration(q0: float, q1: float, T: float, vmax: float, amax: float) -> dict:
    """给定总时长 T，找出最优梯形/三角形剖面的加速时间。

    记加速时间 t_a，则峰值速度 v = D/(T - t_a)、峰值加速度 a = v/t_a。
    在 (0, T/2] 上 v/vmax 单调递增、a/amax 单调递减，所以
    max(v/vmax, a/amax) 的最小值就在两曲线交点：t_a* = vmax/amax。
    交点落在区间外时取边界 T/2。此时两个限制中先超 1 的那个就是要放宽的瓶颈。

    可以证明“T 秒内可行” <=> T >= lspb_profile(q0, q1, vmax, amax)["T"]。
    """
    D = abs(q1 - q0)
    if D < 1e-15:
        return {"T": T, "t_acc": 0.0, "v_peak": 0.0, "a_peak": 0.0, "load": 0.0, "feasible": True}
    assert T > 0, "总时长必须为正"
    t_acc = min(vmax / amax, T / 2)
    v_peak = D / (T - t_acc)
    a_peak = v_peak / t_acc
    load = max(v_peak / vmax, a_peak / amax)
    return {"T": T, "t_acc": t_acc, "v_peak": v_peak, "a_peak": a_peak,
            "load": load, "feasible": load <= 1.0 + 1e-12}
