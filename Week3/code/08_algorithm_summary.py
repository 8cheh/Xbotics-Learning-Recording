"""复现 8：算法横向对比 + A3C 并行采样的方差缩减验证

对应周报第三部分作业题「A3C 算法的主要优势是什么？」。

A3C 相比 A2C/PPO 的关键改动是**异步并行**：多个 worker 各自在环境副本里跑，
把梯度汇总到共享参数上。它带来三个好处，本脚本直接把第一条量化出来：

  1. 并行采样 = 把 B 条轨迹的梯度平均，估计方差大约降到 1/B
     （本脚本实测并拟合，见输出与图 (b)）；
  2. 样本天然去相关，不需要经验回放池（省内存）；
  3. 各 worker 异步推进，没有同步等待，吞吐更高。

其余数据直接从 05/06/07 的 results/*.json 读取，汇总成一张横向对比表。

运行（需先跑过 05、06、07）：

    python 08_algorithm_summary.py

输出：../figures/fig_rl_summary.png 与 ../results/rl_summary.json
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from mpl_style import setup
from rl_common import FIG_DIR, load_results, make_env, save_results

plt = setup()

GAMMA = 0.99
WORKER_COUNTS = (1, 2, 4, 8, 16, 32)
REPEATS = 24


class PolicyNet(nn.Module):
    """与 06 同构的策略网络，保证 A3C 实验和前面的结论可比。"""

    def __init__(self, obs_dim: int = 4, act_dim: int = 2, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def discounted_returns(rewards: list[float], gamma: float = GAMMA) -> np.ndarray:
    """逆序递推算折扣回报。"""
    out = np.zeros(len(rewards))
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def sample_trajectory(env, policy: PolicyNet, seed: int) -> dict:
    """用当前策略跑一个 episode，返回算梯度所需的 states/actions/returns。

    episode 的初始状态必须显式播种，否则每次跑出来的结果都不一样，
    实验就没法复现（这个坑在并行采样实验里特别隐蔽）。
    """
    obs, _ = env.reset(seed=seed)
    states, actions, rewards = [], [], []
    done = False
    with torch.no_grad():
        while not done:
            logits = policy(torch.as_tensor(obs, dtype=torch.float32))
            action = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            obs = next_obs
            done = terminated or truncated
    return {"states": torch.as_tensor(np.array(states, dtype=np.float32)),
            "actions": torch.as_tensor(actions),
            "returns": torch.as_tensor(discounted_returns(rewards), dtype=torch.float32)}


def mean_gradient(policy: PolicyNet, trajectories: list[dict]) -> torch.Tensor:
    """把若干条轨迹的策略梯度平均起来 —— 这正是 A3C 多个 worker 汇总的动作。"""
    policy.zero_grad()
    total_loss = torch.zeros(())
    for traj in trajectories:
        dist = torch.distributions.Categorical(logits=policy(traj["states"]))
        adv = traj["returns"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        total_loss = total_loss - (dist.log_prob(traj["actions"]) * adv.detach()).mean()
    (total_loss / len(trajectories)).backward()
    grad = torch.cat([p.grad.flatten() for p in policy.parameters() if p.grad is not None])
    return grad.detach().clone()


def measure_parallel_variance(policy: PolicyNet, workers: int, repeats: int = REPEATS,
                              seed: int = 0) -> float:
    """固定策略下，估计“平均 workers 条轨迹”的梯度标准差。

    每个 repeat 独立重采样一批轨迹，得到 repeats 个梯度估计，
    再统计这些估计的协方差迹的平方根。
    """
    torch.manual_seed(seed)
    env = make_env()
    estimates = []
    for r in range(repeats):
        trajectories = [sample_trajectory(env, policy, seed=seed * 1_000_000 + r * 1_000 + i)
                        for i in range(workers)]
        estimates.append(mean_gradient(policy, trajectories))
    env.close()
    stacked = torch.stack(estimates)
    return math.sqrt(stacked.var(dim=0).sum().item())


def check_variance_scales_inversely(repeats: int = REPEATS) -> dict:
    """并行 worker 越多，梯度估计的方差越小，且大致按 1/B 下降。"""
    torch.manual_seed(0)
    policy = PolicyNet()  # 未训练的策略：状态覆盖广，梯度信号不退化
    policy.eval()

    stds = {}
    for workers in WORKER_COUNTS:
        stds[workers] = measure_parallel_variance(policy, workers, repeats=repeats, seed=0)

    # 单调下降（留 5% 容差：repeats 有限，统计量本身有噪声）
    for a, b in zip(WORKER_COUNTS[:-1], WORKER_COUNTS[1:], strict=True):
        assert stds[b] <= stds[a] * 1.05, \
            f"worker 从 {a} 加到 {b} 反而变差: {stds[a]:.4f} -> {stds[b]:.4f}"
    # 1/B 缩放：B=32 时标准差应降到 B=1 的 1/2 以下（理想是 1/sqrt(32)≈0.18）
    assert stds[32] < stds[1] / 2.0, \
        f"并行 32 个 worker 的方差缩减不足: {stds[1]:.4f} -> {stds[32]:.4f}"

    # 用 log-log 斜率拟合幂律 std ~ B^(-alpha)，理想 alpha = 0.5
    log_b = np.log(np.array(WORKER_COUNTS, dtype=float))
    log_s = np.log(np.array([stds[b] for b in WORKER_COUNTS], dtype=float))
    alpha = -np.polyfit(log_b, log_s, 1)[0]
    assert 0.2 < alpha < 0.9, f"幂律指数应在 1/sqrt(B) (0.5) 附近，实际 {alpha:.3f}"
    return {"stds": {str(k): v for k, v in stds.items()}, "alpha": alpha}


def build_table() -> list[dict]:
    """从 05/06/07 的结果文件汇总横向对比表。"""
    q = load_results("rl_qlearning_dqn")
    pg = load_results("rl_policy_gradient")
    ppo = load_results("rl_ppo")

    rows = [
        {"method": "随机策略", "family": "基线", "return": q["random"]["mean"],
         "note": "CartPole 随机动作，任务难度下界"},
        {"method": "恒定动作", "family": "基线", "return": q["constant_action"]["mean"],
         "note": "始终推左，最差参考"},
        {"method": "表格 Q-learning (256 状态)", "family": "值迭代 / 表格", "return": q["coarse_q"]["mean"],
         "note": "离散化粗，状态覆盖好"},
        {"method": "表格 Q-learning (4096 状态)", "family": "值迭代 / 表格", "return": q["fine_q"]["mean"],
         "note": "格子更细但更稀疏，提升有限"},
        {"method": "DQN（最优快照）", "family": "值函数逼近", "return": q["dqn_best"]["mean"],
         "note": "连续观测 + 经验回放 + 目标网络"},
        {"method": "DQN（最后一个网络）", "family": "值函数逼近", "return": q["dqn_final"]["mean"],
         "note": "同一配置的末态；训练中会崩，靠快照选择兜底"},
        {"method": "REINFORCE", "family": "策略梯度", "return": pg["reinforce_eval"]["mean"],
         "note": "无基线，梯度方差最大"},
        {"method": "REINFORCE + 基线", "family": "策略梯度", "return": pg["baseline_eval"]["mean"],
         "note": "减去蒙特卡洛评论家 V(s)"},
        {"method": "A2C", "family": "演员-评论家", "return": pg["a2c_eval"]["mean"],
         "note": "单步自举优势，偏差大方差小"},
        {"method": "PPO (clip=0.2)", "family": "演员-评论家", "return": ppo["ppo_eval"]["mean"],
         "note": "裁剪 + 同批数据复用 10 个 epoch"},
    ]
    return rows


def main() -> None:
    rows = build_table()
    variance = check_variance_scales_inversely()

    print("=" * 100)
    print(f"{'方法':<30}{'类别':<16}{'贪心回报':>10}   关键机制")
    print("-" * 100)
    for row in rows:
        print(f"{row['method']:<30}{row['family']:<16}{row['return']:>10.1f}   {row['note']}")
    print("=" * 100)
    print("说明：CartPole 上多种方法都能打满 500 分，单看最终分数区分不出优劣；")
    print("      真正的差异在下面两处 —— 信用分配信号的方差（信号质量）和每次更新的 KL（步长稳定性）。")

    print("\nA3C 并行采样的方差缩减（固定策略，每个 worker 跑一条轨迹）：")
    for workers in WORKER_COUNTS:
        std = variance["stds"][str(workers)]
        print(f"  worker 数 {workers:2d}: 梯度估计标准差 {std:.4f}"
              f"   (相对单 worker {std / variance['stds']['1'] * 100:5.1f}%)")
    print(f"  幂律拟合 std ~ B^(-alpha)，alpha = {variance['alpha']:.3f}"
          f"（理论值 0.5，即标准差按 1/sqrt(B) 下降，方差按 1/B 下降）")

    save_results("rl_summary", {
        "table": rows,
        "parallel_variance_std": variance["stds"],
        "parallel_variance_alpha": variance["alpha"],
        "worker_counts": list(WORKER_COUNTS),
        "dqn_eval_trace": load_results("rl_qlearning_dqn")["dqn_eval_trace"],
        "ppo_kl_mean": load_results("rl_ppo")["ppo_kl_mean"],
        "ppo_kl_max": load_results("rl_ppo")["ppo_kl_max"],
        "unclipped_kl_mean": load_results("rl_ppo")["unclipped_kl_mean"],
        "unclipped_kl_max": load_results("rl_ppo")["unclipped_kl_max"],
        "signal_std_no_baseline": load_results("rl_policy_gradient")["signal_std_no_baseline"],
        "signal_std_with_baseline": load_results("rl_policy_gradient")["signal_std_with_baseline"],
    })

    # ---- 绘图 ----
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0))

    ax = axes[0]
    names = [r["method"] for r in rows]
    values = [r["return"] for r in rows]
    palette = {"基线": "0.6", "值迭代 / 表格": "tab:blue", "值函数逼近": "tab:orange",
               "策略梯度": "tab:red", "演员-评论家": "tab:green"}
    colors = [palette[r["family"]] for r in rows]
    bars = ax.barh(names[::-1], values[::-1], color=colors[::-1])
    ax.bar_label(bars, fmt="%.0f", fontsize=7, padding=2)
    ax.axvline(500, color="k", ls=":", lw=1)
    ax.set_xlim(0, 560)
    ax.grid(alpha=0.3, axis="x")
    ax.set_title("(a) 各算法最终贪心回报")
    ax.set_xlabel("回报")
    ax.tick_params(axis="y", labelsize=7)

    ax = axes[1]
    workers = np.array(WORKER_COUNTS, dtype=float)
    stds = np.array([variance["stds"][str(w)] for w in WORKER_COUNTS], dtype=float)
    ax.loglog(workers, stds, "o-", label="实测")
    ideal = stds[0] / np.sqrt(workers)
    ax.loglog(workers, ideal, "--", label=r"理想 $1/\sqrt{B}$")
    ax.grid(alpha=0.3, which="both")
    ax.set_title("(b) A3C 并行采样：梯度估计标准差")
    ax.set_xlabel("并行 worker 数 B")
    ax.set_ylabel("std of gradient")
    ax.legend(fontsize=8)

    ax = axes[2]
    labels = ["REINFORCE\n(无基线)", "REINFORCE\n+基线", "PPO\n(裁剪)", "PPO\n(不裁剪)"]
    kl_values = [np.nan, np.nan, load_results("rl_ppo")["ppo_kl_mean"],
                 load_results("rl_ppo")["unclipped_kl_mean"]]
    signal = [load_results("rl_policy_gradient")["signal_std_no_baseline"],
              load_results("rl_policy_gradient")["signal_std_with_baseline"], np.nan, np.nan]
    x = np.arange(4)
    ax.bar(x - 0.2, signal, width=0.4, color="tab:red", label="信用分配信号 std")
    left = ax.get_ylim()[1]
    ax.set_ylim(0, max([v for v in signal + kl_values if not np.isnan(v)]) * 1.35)
    ax.bar(x + 0.2, kl_values, width=0.4, color="tab:purple", label="每次更新近似 KL")

    def annotate_value(xs: float, value: float, text: str) -> None:
        if not np.isnan(value):
            ax.text(xs, value + left * 0.02, text, ha="center", fontsize=7)

    for xi, value in zip(x, signal, strict=True):
        annotate_value(xi - 0.2, value, f"{value:.2f}")
    for xi, value in zip(x, kl_values, strict=True):
        annotate_value(xi + 0.2, value, f"{value:.3f}")
    ax.set_xticks(x, labels, fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    ax.set_title("(c) 两个“稳定化改造”的量化效果")
    ax.set_ylabel("数值")
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_rl_summary.png", dpi=140)
    plt.close(fig)
    print(f"\n[fig] {FIG_DIR / 'fig_rl_summary.png'}")
    print(f"汇总表包含 {len(rows)} 个方法，已写入 results/rl_summary.json")
    print("08_algorithm_summary: all assertions passed")


if __name__ == "__main__":
    main()
