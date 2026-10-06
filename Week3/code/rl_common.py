"""强化学习复现脚本的公共部分（环境、离散化、评估、结果落盘）。

被 05/06/07/08 复用。统一约定：

- 环境用 Gymnasium 的 CartPole-v1，观测 4 维连续值，动作 2 个离散值。
- 评估一律用贪心策略跑若干 episode，返回回报统计。
- 每个脚本把自己的指标写进 ../results/<name>.json，供 08 汇总成对比表。
"""

from __future__ import annotations

import json
import operator
from collections.abc import Callable
from pathlib import Path

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"

ENV_ID = "CartPole-v1"

# CartPole 的观测在 x、x_dot、theta、theta_dot 四个维度上无界或半无界，
# 表格法必须先把它们裁到有限区间再分箱。
DISCRETE_BOUNDS = np.array([
    [-2.4, 2.4],        # 小车位置 x
    [-3.0, 3.0],        # 小车速度 x_dot
    [-0.2095, 0.2095],  # 杆角度 theta（±12°）
    [-3.0, 3.0],        # 杆角速度 theta_dot
])


def make_env(seed: int | None = None) -> gym.Env:
    """创建 CartPole 环境；seed 通过 reset(seed=...) 传入。"""
    del seed  # 播种统一在 reset 里做，避免两套随机源
    return gym.make(ENV_ID)


def n_states(n_bins: int) -> int:
    """给定每维分箱数，表格法一共多少个状态。"""
    return n_bins ** DISCRETE_BOUNDS.shape[0]


def discretize(obs: np.ndarray, n_bins: int) -> tuple[int, ...]:
    """把连续观测映射成离散格子下标。

    超出区间的值被裁到边界格（CartPole 出界就终止，所以影响很小）。
    """
    ratios = (np.asarray(obs, dtype=float) - DISCRETE_BOUNDS[:, 0]) / (
        DISCRETE_BOUNDS[:, 1] - DISCRETE_BOUNDS[:, 0])
    idx = np.clip((ratios * n_bins).astype(np.int64), 0, n_bins - 1)
    return tuple(idx.tolist())


def greedy_discrete(Q: np.ndarray, n_bins: int) -> Callable[[np.ndarray], int]:
    """由 Q 表构造贪心策略。"""
    return lambda obs: operator.index(np.argmax(Q[discretize(obs, n_bins)]))


def evaluate(policy: Callable[[np.ndarray], int], episodes: int = 50, seed: int = 9999) -> np.ndarray:
    """跑若干 episode 的贪心评估，返回每个 episode 的累计回报。"""
    env = make_env()
    returns = np.empty(episodes)
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        total = 0.0
        done = False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(policy(obs))
            total += reward
            done = terminated or truncated
        returns[i] = total
    env.close()
    return returns


def rollout_random_policy(episodes: int = 20, seed: int = 0) -> np.ndarray:
    """随机策略的基线回报，用来对照“学没学到东西”。

    注意要显式给 action_space 播种：env.reset(seed=...) 只保证初始状态可复现，
    env.action_space.sample() 用的是另一个随机源，不播它结果每次都不一样。
    """
    env = make_env()
    returns = np.empty(episodes)
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        env.action_space.seed(seed + i)
        total, done = 0.0, False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            total += reward
            done = terminated or truncated
        returns[i] = total
    env.close()
    return returns


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """滑动平均，长度与输入一致（前 window-1 个用已有数据平均）。"""
    if len(x) < window:
        return np.full(len(x), np.mean(x))
    kernel = np.ones(window) / window
    padded = np.concatenate([np.full(window - 1, x[0]), x])
    return np.convolve(padded, kernel, mode="valid")


def save_results(name: str, payload: dict) -> Path:
    """把指标写到 results/<name>.json（numpy 类型自动转成 Python 标量）。"""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    def default(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"无法序列化 {type(obj)}")

    path = RESULT_DIR / f"{name}.json"
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=default)
    except OSError as exc:
        raise SystemExit(f"无法写入 {path}: {exc}") from exc
    return path


def load_results(name: str) -> dict:
    """读回 results/<name>.json。"""
    path = RESULT_DIR / f"{name}.json"
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except OSError as exc:
        raise SystemExit(f"无法读取 {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} 不是合法 JSON: {exc}") from exc


def summarize(returns: np.ndarray) -> dict:
    """回报统计：均值、标准差、最好、最差。"""
    values = np.asarray(returns, dtype=float)
    return {"mean": values.mean().item(), "std": values.std().item(),
            "max": values.max().item(), "min": values.min().item()}
