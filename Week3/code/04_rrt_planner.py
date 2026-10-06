"""复现 4：采样法路径规划（RRT / RRT*）+ 捷径平滑 + 时间参数化

对应周报第二部分提到的 RRT / PRM。完整走一遍机器人“从起点到终点”的链路：

    碰撞检测 -> RRT 采样 -> 捷径平滑 -> 按 (vmax, amax) 做时间参数化

运行：

    python 04_rrt_planner.py

输出：../figures/fig_rrt.png 与 ../results/rrt_path.csv
"""

from __future__ import annotations

import math
import operator
from pathlib import Path

import numpy as np
from matplotlib.patches import Rectangle
from mpl_style import setup
from traj import lspb_eval, lspb_profile

plt = setup()

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
RESULT_DIR = ROOT / "results"

# 工作空间：x, y 各在 [0, 1]
BOUNDS = np.array([[0.0, 1.0], [0.0, 1.0]])
START = np.array([0.08, 0.08])
GOAL = np.array([0.92, 0.92])

# 障碍物：圆形和矩形各来一些
OBSTACLES = [
    {"type": "circle", "center": np.array([0.35, 0.35]), "radius": 0.10},
    {"type": "circle", "center": np.array([0.65, 0.60]), "radius": 0.12},
    {"type": "circle", "center": np.array([0.30, 0.72]), "radius": 0.09},
    {"type": "rect", "min": np.array([0.50, 0.10]), "max": np.array([0.60, 0.42])},
]


# ---------------------------------------------------------------- 碰撞检测 ----
def collides(points: np.ndarray, obstacles: list[dict] = OBSTACLES) -> np.ndarray:
    """判断一批点是否落在障碍物内，返回布尔数组。"""
    pts = np.atleast_2d(points)
    bad = np.zeros(len(pts), dtype=bool)
    for ob in obstacles:
        if ob["type"] == "circle":
            bad |= np.linalg.norm(pts - ob["center"], axis=1) <= ob["radius"]
        else:
            inside = np.all((pts >= ob["min"]) & (pts <= ob["max"]), axis=1)
            bad |= inside
    return bad


def segment_free(a: np.ndarray, b: np.ndarray, obstacles: list[dict] = OBSTACLES,
                 step: float = 0.005) -> bool:
    """线段是否完全在自由空间。

    ponytail: 用密集采样而不是解析求交 —— 采样步长 5mm 远小于最小障碍尺寸
    (9cm)，不会穿透；若要处理薄壁障碍或换取速度再上解析求交。
    """
    length = math.dist(a, b)
    n = max(2, np.ceil(length / step).astype(np.int64) + 1)
    pts = a + np.outer(np.linspace(0.0, 1.0, n), b - a)
    return not np.any(collides(pts, obstacles))


# ---------------------------------------------------------------- RRT / RRT* ----
def rrt(start: np.ndarray, goal: np.ndarray, *, star: bool = False, step: float = 0.06,
        max_iter: int = 4000, goal_bias: float = 0.02, rewire_radius: float = 0.15,
        seed: int = 0, obstacles: list[dict] = OBSTACLES) -> dict:
    """RRT（star=False）或 RRT*（star=True）。

    - RRT  ：最近邻 -> 朝随机点走一步 -> 撞了丢弃。快，但路径很绕。
    - RRT* ：新点先从邻域里选一个“累计代价最小”的父节点（choose parent），
             再让邻域里其他点改认它当父节点（rewire）。渐进最优，但更慢。
    """
    rng = np.random.default_rng(seed)
    nodes: list[np.ndarray] = [np.asarray(start, dtype=float)]
    parents: list[int] = [-1]
    costs: list[float] = [0.0]
    goal_idx = -1

    for _ in range(max_iter):
        q_rand = goal.copy() if rng.random() < goal_bias else rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1])
        dists = np.linalg.norm(np.array(nodes) - q_rand, axis=1)
        i_near = operator.index(np.argmin(dists))
        q_near = nodes[i_near]

        direction = q_rand - q_near
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            continue
        q_new = q_near + direction / norm * min(step, norm)
        if not segment_free(q_near, q_new, obstacles):
            continue

        parent = i_near
        cost = costs[i_near] + math.dist(q_new, q_near)

        if star:  # 在邻域内挑代价最小的父节点
            for i, n in enumerate(nodes):
                if math.dist(n, q_new) <= rewire_radius and segment_free(n, q_new, obstacles):
                    cand = costs[i] + math.dist(q_new, n)
                    if cand < cost:
                        parent, cost = i, cand

        nodes.append(q_new)
        parents.append(parent)
        costs.append(cost)
        new_idx = len(nodes) - 1

        if star:  # 重连：让邻域点通过新点走更短的累计代价
            for i, n in enumerate(nodes[:-1]):
                if math.dist(n, q_new) > rewire_radius:
                    continue
                cand = costs[new_idx] + math.dist(n, q_new)
                if cand < costs[i] and segment_free(q_new, nodes[i], obstacles):
                    parents[i] = new_idx
                    costs[i] = cand

        if math.dist(q_new, goal) <= step and segment_free(q_new, goal, obstacles):
            nodes.append(goal.copy())
            parents.append(new_idx)
            costs.append(costs[new_idx] + math.dist(q_new, goal))
            goal_idx = len(nodes) - 1
            break

    path = None
    if goal_idx >= 0:
        path = []
        i = goal_idx
        while i >= 0:
            path.append(nodes[i])
            i = parents[i]
        path.reverse()
    return {"path": None if path is None else np.array(path),
            "nodes": np.array(nodes), "parents": np.array(parents),
            "costs": np.array(costs), "iterations": len(nodes)}


def path_length(path: np.ndarray) -> float:
    """折线路径的总长度（用 math.fsum 精确求和）。"""
    return math.fsum(math.dist(a, b) for a, b in zip(path[:-1], path[1:], strict=True))


def shortcut(path: np.ndarray, *, iterations: int = 800, seed: int = 0,
             obstacles: list[dict] = OBSTACLES) -> np.ndarray:
    """捷径平滑：随机取路径上两点，若直连无碰撞就用直线段替换中间部分。

    这是“先采样出可行解、再局部优化”这一思路的标准收尾步骤。
    """
    rng = np.random.default_rng(seed)
    pts = [p.copy() for p in path]
    for _ in range(iterations):
        if len(pts) <= 2:
            break
        i, j = sorted(rng.choice(len(pts), size=2, replace=False))
        if j - i < 1:
            continue
        if segment_free(pts[i], pts[j], obstacles):
            pts = pts[:i + 1] + pts[j:]
    return np.array(pts)


def time_parameterize(path: np.ndarray, vmax: float, amax: float,
                      n: int = 400) -> dict:
    """把几何路径变成时间轨迹：沿弧长做梯形速度剖面。

    这是标准的“路径-速度解耦”：路径只关心几何（走哪条线），
    时间参数化只关心动力学（多快走完）。
    """
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    assert s[-1] > 0, "路径长度必须为正"
    profile = lspb_profile(0.0, s[-1], vmax, amax)
    t = np.linspace(0, profile["T"], n)
    s_t, v_t, a_t = lspb_eval(profile, t)
    # 按弧长在折线上插值出位置与切向速度
    xy = np.column_stack([np.interp(s_t, s, path[:, k]) for k in range(path.shape[1])])
    tangents = np.gradient(xy, axis=0)
    norms = np.linalg.norm(tangents, axis=1)
    velocity = tangents / np.maximum(norms, 1e-12)[:, None] * v_t[:, None]
    return {"t": t, "s": s_t, "xy": xy, "speed": v_t, "accel": a_t,
            "velocity": velocity, "profile": profile, "arclength": s}


# ---------------------------------------------------------------- 自检 ----
def check_collision_detection() -> None:
    # 障碍物圆心一定在障碍内，外部点一定在障碍外
    assert collides(OBSTACLES[0]["center"])
    assert not collides(np.array([0.05, 0.95]))
    assert collides(np.array([0.55, 0.25]))          # 矩形内部
    assert not collides(np.array([0.55, 0.46]))      # 矩形上方、圆形障碍外的空场
    # 起终点必须不在障碍里
    assert not collides(START) and not collides(GOAL)
    # 穿过圆心正中间 -> 必然碰撞
    c = OBSTACLES[0]["center"]
    assert not segment_free(c + np.array([-0.3, 0.0]), c + np.array([0.3, 0.0]))
    # 贴着障碍上方走 -> 不碰
    assert segment_free(np.array([0.1, 0.95]), np.array([0.9, 0.95]))


def check_rrt_solution() -> None:
    """RRT 与 RRT* 都要给出无碰撞、连着起终点的路径。"""
    res_plain = rrt(START, GOAL, star=False, seed=0)
    res_star = rrt(START, GOAL, star=True, seed=0)
    for name, res in (("RRT", res_plain), ("RRT*", res_star)):
        path = res["path"]
        assert path is not None, f"{name} 未找到路径"
        assert np.allclose(path[0], START, atol=1e-12), f"{name} 起点错误"
        assert np.allclose(path[-1], GOAL, atol=1e-12), f"{name} 终点错误"
        # 每一段都不能碰障碍
        for a, b in zip(path[:-1], path[1:], strict=True):
            assert segment_free(a, b), f"{name} 路径段碰撞"
    # RRT* 的渐进最优性：同一随机种子下最终代价不劣于 RRT
    assert path_length(res_star["path"]) <= path_length(res_plain["path"]) + 1e-9, \
        f"RRT* 应不劣于 RRT: {path_length(res_star['path']):.4f} vs {path_length(res_plain['path']):.4f}"


def check_shortcut() -> None:
    res = rrt(START, GOAL, seed=0)
    before = path_length(res["path"])
    smoothed = shortcut(res["path"], seed=0)
    after = path_length(smoothed)
    assert after <= before + 1e-9, "平滑不应让路径变长"
    assert np.allclose(smoothed[0], START) and np.allclose(smoothed[-1], GOAL)
    for a, b in zip(smoothed[:-1], smoothed[1:], strict=True):
        assert segment_free(a, b), "平滑后出现碰撞段"


def check_time_parameterization() -> None:
    """时间参数化必须满足速度/加速度上限，且末端点落在终点上。"""
    res = rrt(START, GOAL, star=True, seed=0)
    smoothed = shortcut(res["path"], seed=0)
    vmax, amax = 0.5, 1.5
    traj = time_parameterize(smoothed, vmax, amax)
    assert np.allclose(traj["xy"][0], START, atol=1e-9)
    assert np.allclose(traj["xy"][-1], GOAL, atol=1e-9)
    assert np.max(np.abs(traj["speed"])) <= vmax + 1e-9, "速度超限"
    assert np.max(np.abs(traj["accel"])) <= amax + 1e-9, "加速度超限"
    # 弧长单调递增，且走到总长度
    assert np.all(np.diff(traj["s"]) >= -1e-12)
    assert np.isclose(traj["s"][-1], path_length(smoothed), atol=1e-9)
    # 路径上的每个时刻都不能在障碍里
    assert not np.any(collides(traj["xy"])), "时间参数化后的轨迹穿障"


def check_rrt_star_improves_with_iterations() -> None:
    """RRT* 的代价随迭代次数下降（渐进最优），RRT 不会。"""
    costs_star = [path_length(rrt(START, GOAL, star=True, seed=s)["path"]) for s in range(5)]
    costs_plain = [path_length(rrt(START, GOAL, star=False, seed=s)["path"]) for s in range(5)]
    assert np.mean(costs_star) < np.mean(costs_plain), \
        f"RRT* 平均代价应低于 RRT: {np.mean(costs_star):.4f} vs {np.mean(costs_plain):.4f}"


# ---------------------------------------------------------------- 主流程 ----
def main() -> None:
    check_collision_detection()
    check_rrt_solution()
    check_shortcut()
    check_time_parameterization()
    check_rrt_star_improves_with_iterations()

    res_plain = rrt(START, GOAL, star=False, seed=0)
    res_star = rrt(START, GOAL, star=True, seed=0)
    smoothed = shortcut(res_star["path"], seed=0)
    vmax, amax = 0.5, 1.5
    traj = time_parameterize(smoothed, vmax, amax)

    print(f"RRT   : 采样 {res_plain['iterations']:5d} 个节点, 路径长 {path_length(res_plain['path']):.4f} m")
    print(f"RRT*  : 采样 {res_star['iterations']:5d} 个节点, 路径长 {path_length(res_star['path']):.4f} m")
    print(f"平滑后: {len(smoothed)} 个途经点, 路径长 {path_length(smoothed):.4f} m")
    print(f"时间参数化 (vmax={vmax}, amax={amax}): 总时长 {traj['profile']['T']:.3f}s, "
          f"峰值速度 {np.max(np.abs(traj['speed'])):.3f} m/s, "
          f"峰值加速度 {np.max(np.abs(traj['accel'])):.3f} m/s^2  [{traj['profile']['kind']}]")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULT_DIR / "rrt_path.csv"
    np.savetxt(csv_path, np.column_stack([traj["t"], traj["xy"], traj["speed"]]),
               delimiter=",", header="t,x,y,speed", comments="", fmt="%.6f")
    print(f"[csv] {csv_path}")

    # ---- 绘图 ----
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    def draw_map(ax, title):
        theta = np.linspace(0, 2 * np.pi, 100)
        for ob in OBSTACLES:
            if ob["type"] == "circle":
                c, r = ob["center"], ob["radius"]
                ax.fill(c[0] + r * np.cos(theta), c[1] + r * np.sin(theta), color="0.75", zorder=1)
            else:
                ax.add_patch(Rectangle(ob["min"], *(ob["max"] - ob["min"]), color="0.75", zorder=1))
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

    for ax, res, name in ((axes[0], res_plain, "RRT"), (axes[1], res_star, "RRT*")):
        draw_map(ax, f"({'ab'[name == 'RRT*']}) {name} 树与路径")
        for i, par in enumerate(res["parents"]):
            if par >= 0:
                ax.plot(*np.array([res["nodes"][i], res["nodes"][par]]).T, "-", color="tab:blue",
                        lw=0.4, alpha=0.5, zorder=2)
        ax.plot(res["path"][:, 0], res["path"][:, 1], "-", color="tab:red", lw=2, zorder=3, label="路径")
        ax.plot(*START, "go", ms=7, zorder=4, label="起点")
        ax.plot(*GOAL, "r*", ms=13, zorder=4, label="终点")
        ax.legend(fontsize=7, loc="upper left")

    ax = axes[2]
    draw_map(ax, "(c) 平滑 + 时间参数化后的末端轨迹")
    ax.plot(res_star["path"][:, 0], res_star["path"][:, 1], ":", color="tab:red", lw=1.5, label="RRT* 原始路径")
    ax.plot(smoothed[:, 0], smoothed[:, 1], "o--", color="tab:orange", ms=4, lw=1.5, label="捷径平滑")
    speed = np.abs(traj["speed"])
    sc = ax.scatter(traj["xy"][:, 0], traj["xy"][:, 1], c=speed, cmap="viridis",
                    s=16, zorder=3, linewidths=0)
    fig.colorbar(sc, ax=ax, label="速度 [m/s]", fraction=0.046)
    ax.plot(*START, "go", ms=7, zorder=4)
    ax.plot(*GOAL, "r*", ms=13, zorder=4)
    ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_rrt.png", dpi=140)
    plt.close(fig)
    print(f"[fig] {FIG_DIR / 'fig_rrt.png'}")
    print("04_rrt_planner: all assertions passed")


if __name__ == "__main__":
    main()
