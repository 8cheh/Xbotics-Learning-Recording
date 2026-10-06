"""复现 5：Q-learning（表格）与 DQN 的对比

对应周报第三部分作业题「DQN 和 Q-learning 的核心区别是什么？」。
两者优化的是同一个目标（Bellman 最优方程），差别全在“怎么表示 Q 函数”：

    Q-learning : Q 存成一张表，一格一个状态，只能处理离散状态，
                 每步只更新被访问的那一格 —— 天然 off-policy，不需要回放。
    DQN        : Q 用一个神经网络拟合，可以直接吃连续观测（不用手工离散化），
                 但“函数逼近 + 自举 + off-policy”凑成 deadly triad，
                 必须靠经验回放（打散样本相关性）和目标网络（稳住自举目标）才能收敛。

本脚本在同一环境下训练三种策略做对照：
  - 粗离散表格 Q-learning（每维 4 格，共 256 个状态）
  - 细离散表格 Q-learning（每维 8 格，共 4096 个状态）
  - DQN（直接吃 4 维连续观测）

实测结论（见运行输出）：
  - 表格法约 150 分就封顶 —— 再细分格子也上不去，瓶颈是离散化误差；
  - DQN 能打满 500 分，但**最终网络**对随机种子敏感，训练中途会出现策略崩溃，
    所以脚本按固定间隔做贪心评估、保留最优快照（标准做法），并同时报告两者，
    这个差距本身就是 DQN 稳定性不如 PPO 的直接证据。

运行：

    python 05_qlearning_dqn.py

输出：../figures/fig_rl_qlearning_dqn.png 与 ../results/rl_qlearning_dqn.json
"""

from __future__ import annotations

import copy
import operator
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mpl_style import setup
from rl_common import (
    DISCRETE_BOUNDS,
    FIG_DIR,
    discretize,
    evaluate,
    greedy_discrete,
    make_env,
    moving_average,
    n_states,
    rollout_random_policy,
    save_results,
    summarize,
)

plt = setup()

GAMMA = 0.99

# 表格法超参：注意 eps 衰减期要拉长，太早收敛到贪心会让状态覆盖不足
TABULAR_STEPS = 100_000
TABULAR_ALPHA = 0.1
TABULAR_EPS_DECAY = 30_000

# DQN 超参：lr 是这里最关键的旋钮。实测 lr=1e-4 会欠拟合卡在 100 分附近，
# lr=5e-4 学得快但更容易在中途崩掉；2.5e-4 是较稳的折中。
DQN_STEPS = 60_000
DQN_LR = 2.5e-4
DQN_BATCH = 64
DQN_LEARNING_STARTS = 1_000
DQN_TARGET_UPDATE = 200
DQN_EPS = 0.1
DQN_EVAL_EVERY = 5_000


# ---------------------------------------------------------------- 表格 Q-learning ----
def train_q_learning(n_bins: int, steps: int = TABULAR_STEPS, alpha: float = TABULAR_ALPHA,
                     gamma: float = GAMMA, eps_start: float = 1.0, eps_end: float = 0.02,
                     eps_decay_steps: int = TABULAR_EPS_DECAY, seed: int = 0) -> dict:
    """表格 Q-learning：epsilon-greedy 采样，每步做一次 Bellman 最优更新。

    更新式：Q(s, a) <- Q(s, a) + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))
    注意 max 用的是行为策略之外的最优动作 —— 这就是 off-policy。
    """
    rng = np.random.default_rng(seed)
    env = make_env()
    # Q 用 4 维表格：Q[x, x_dot, theta, theta_dot, a]，正好用 discretize 返回的下标直接索引
    Q = np.zeros((n_bins,) * 4 + (2,))
    obs, _ = env.reset(seed=seed)

    episode_returns: list[float] = []
    episode_steps: list[int] = []
    current_return, current_len = np.float64(0.0), 0

    for step in range(steps):
        eps = eps_end + (eps_start - eps_end) * max(0.0, 1.0 - step / eps_decay_steps)
        s = discretize(obs, n_bins)
        a = operator.index(rng.integers(2)) if rng.random() < eps else operator.index(np.argmax(Q[s]))
        obs_next, reward, terminated, truncated, _ = env.step(a)
        s_next = discretize(obs_next, n_bins)
        # 终止状态不 bootstrap；截断（时间上限）仍然 bootstrap
        target = reward if terminated else reward + gamma * np.max(Q[s_next])
        Q[s + (a,)] += alpha * (target - Q[s + (a,)])

        current_return += reward
        current_len += 1
        obs = obs_next
        if terminated or truncated:
            episode_returns.append(current_return)
            episode_steps.append(current_len)
            current_return, current_len = np.float64(0.0), 0
            obs, _ = env.reset()
    env.close()
    return {"Q": Q, "n_bins": n_bins, "returns": np.array(episode_returns),
            "lengths": np.array(episode_steps)}


# ---------------------------------------------------------------- DQN ----
class QNet(nn.Module):
    """把观测映射到每个动作的 Q 值。"""

    def __init__(self, obs_dim: int = 4, act_dim: int = 2, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """定长经验回放池：打散相邻样本的时间相关性，让梯度假设成立。"""

    def __init__(self, capacity: int, obs_dim: int = 4) -> None:
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add(self, obs, action, reward, next_obs, done) -> None:
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.action[self.pos] = action
        self.reward[self.pos] = reward
        self.done[self.pos] = done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict:
        idx = rng.integers(0, self.size, size=batch_size)
        return {"obs": torch.as_tensor(self.obs[idx]), "action": torch.as_tensor(self.action[idx]),
                "reward": torch.as_tensor(self.reward[idx]), "next_obs": torch.as_tensor(self.next_obs[idx]),
                "done": torch.as_tensor(self.done[idx])}

    def __len__(self) -> int:
        return self.size


def net_policy(net: QNet):
    """把网络包成 (obs) -> action 的贪心策略。"""
    def policy(obs: np.ndarray) -> int:
        with torch.no_grad():
            return net(torch.as_tensor(obs, dtype=torch.float32)).argmax().item()
    return policy


def train_dqn(steps: int = DQN_STEPS, batch_size: int = DQN_BATCH, lr: float = DQN_LR,
              buffer_size: int = 100_000, learning_starts: int = DQN_LEARNING_STARTS,
              target_update: int = DQN_TARGET_UPDATE, eps: float = DQN_EPS,
              eval_every: int = DQN_EVAL_EVERY, seed: int = 0) -> dict:
    """DQN：经验回放 + 目标网络 + Double DQN + Huber 损失 + epsilon-greedy。

    几个 trick 各治一个病：
      - 经验回放   -> 样本强相关，SGD 的 i.i.d. 假设不成立
      - 目标网络   -> 自举目标 Q(s',a') 跟着在线网络一起动，容易发散/振荡
      - Double DQN -> max 算子系统性高估动作价值
      - Huber 损失 -> 时序差分误差尖峰把梯度带飞

    另外每 eval_every 步做一次贪心评估并保存最优快照：DQN 训练中途会崩，
    只交最后一个网络是不可靠的，这也是业界默认的模型选择做法。
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = make_env()
    obs, _ = env.reset(seed=seed)

    online = QNet()
    target = copy.deepcopy(online)
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_size)

    episode_returns: list[float] = []
    episode_steps: list[int] = []
    losses: list[float] = []
    eval_trace: list[tuple[int, float]] = []
    best_score, best_state = -np.inf, copy.deepcopy(online.state_dict())
    current_return, current_len = np.float64(0.0), 0

    for step in range(steps):
        if step < learning_starts or rng.random() < eps:
            action = operator.index(rng.integers(2))
        else:
            action = net_policy(online)(obs)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        # done 只记真正的终止；时间截断仍要 bootstrap，否则价值会被系统性低估
        buffer.add(obs, action, reward, next_obs, terminated)

        current_return += reward
        current_len += 1
        obs = next_obs
        if terminated or truncated:
            episode_returns.append(current_return)
            episode_steps.append(current_len)
            current_return, current_len = np.float64(0.0), 0
            obs, _ = env.reset()

        if len(buffer) < learning_starts:
            continue

        batch = buffer.sample(batch_size, rng)
        with torch.no_grad():
            # Double DQN：在线网络选动作、目标网络估值，抑制 max 的高估
            next_actions = online(batch["next_obs"]).argmax(dim=1, keepdim=True)
            next_q = target(batch["next_obs"]).gather(1, next_actions).squeeze(1)
            y = batch["reward"] + GAMMA * next_q * (1.0 - batch["done"])
        q = online(batch["obs"]).gather(1, batch["action"].unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q, y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(online.parameters(), 10.0)
        optimizer.step()
        losses.append(loss.item())

        if step % target_update == 0:
            target.load_state_dict(online.state_dict())

        if step > 0 and step % eval_every == 0:
            score = evaluate(net_policy(online), episodes=20, seed=5000).mean()
            eval_trace.append((step, score))
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(online.state_dict())

    best_net = QNet()
    best_net.load_state_dict(best_state)
    best_net.eval()
    env.close()
    return {"online": online, "best": best_net, "policy": net_policy(online),
            "best_policy": net_policy(best_net), "best_eval_score": best_score,
            "returns": np.array(episode_returns), "lengths": np.array(episode_steps),
            "losses": np.array(losses), "eval_trace": eval_trace}


# ---------------------------------------------------------------- 可视化用 ----
def q_surface_tabular(Q: np.ndarray, n_bins: int,
                      theta_dot_range: tuple[float, float] = (-2.0, 2.0)) -> tuple:
    """取 theta-theta_dot 切片，画出表格 Q 表的最优动作。"""
    thetas = np.linspace(DISCRETE_BOUNDS[2, 0], DISCRETE_BOUNDS[2, 1], 120)
    theta_dots = np.linspace(theta_dot_range[0], theta_dot_range[1], 120)
    actions = np.zeros((len(theta_dots), len(thetas)))
    for i, td in enumerate(theta_dots):
        for j, th in enumerate(thetas):
            s = discretize(np.array([0.0, 0.0, th, td]), n_bins)
            actions[i, j] = operator.index(np.argmax(Q[s]))
    return thetas, theta_dots, actions


def q_diff_surface_dqn(policy_net: QNet,
                       theta_dot_range: tuple[float, float] = (-2.0, 2.0)) -> tuple:
    """DQN 在同一 theta-theta_dot 切片上 Q(s,右推) - Q(s,左推)。

    差值为正表示该状态下推右更优。因为 DQN 吃的是连续观测，
    这张图是光滑曲面；表格法只能给出分块常数的阶梯。
    """
    thetas = np.linspace(DISCRETE_BOUNDS[2, 0], DISCRETE_BOUNDS[2, 1], 120)
    theta_dots = np.linspace(theta_dot_range[0], theta_dot_range[1], 120)
    grid = np.stack(np.meshgrid(thetas, theta_dots, indexing="ij"), axis=-1)
    obs = np.zeros((*grid.shape[:2], 4), dtype=np.float32)
    obs[..., 2] = grid[..., 0]
    obs[..., 3] = grid[..., 1]
    with torch.no_grad():
        q = policy_net(torch.as_tensor(obs.reshape(-1, 4))).numpy().reshape(*grid.shape[:2], 2)
    return thetas, theta_dots, (q[..., 1] - q[..., 0]).T


# ---------------------------------------------------------------- 自检 ----
def check_discretization() -> None:
    assert n_states(4) == 256 and n_states(8) == 4096
    # 区间内外的观测都必须落到合法格子下标
    rng = np.random.default_rng(0)
    for _ in range(200):
        obs = rng.normal(scale=2.0, size=4)
        idx = discretize(obs, 8)
        assert len(idx) == 4 and all(0 <= i < 8 for i in idx), f"非法格子 {idx} for {obs}"
    # 同格内的点必须映射到同一状态（选一个远离格边界的点）
    inside = np.array([0.1, 0.3, 0.01, 0.3])
    nudged = np.array([0.101, 0.301, 0.0105, 0.3005])
    assert discretize(inside, 8) == discretize(nudged, 8), "微小扰动不应跳出当前格子"
    # 相隔超过一个格宽的点必须换格（x 维格宽 = 4.8/8 = 0.6）
    assert discretize(inside, 8) != discretize(np.array([0.7, 0.3, 0.01, 0.3]), 8)


def check_replay_buffer() -> None:
    """回放池必须存得进、取得到，且丢掉最旧的一条。"""
    buf = ReplayBuffer(capacity=5)
    rng = np.random.default_rng(0)
    for i in range(7):
        buf.add(np.full(4, i, dtype=np.float32), i % 2, np.float32(i),
                np.full(4, i + 1, dtype=np.float32), i == 3)
    assert len(buf) == 5, "回放池应写满后保持容量上限"
    batch = buf.sample(4, rng)
    assert len(batch["obs"]) == 4 and batch["obs"].shape[1] == 4
    assert len(batch["action"]) == 4
    # 最旧的两条（i=0,1）已被覆盖
    assert buf.obs[:, 0].min() >= 2.0, "应丢弃最旧的经验"
    # 终止标记被如实保留
    assert buf.done.sum() == (1.0 if 3 in buf.obs[:, 0] else 0.0)


def check_tabular_learning(q_res: dict) -> np.ndarray:
    """表格 Q-learning 必须显著超过随机策略。"""
    baseline = rollout_random_policy(episodes=30, seed=123)
    trained = evaluate(greedy_discrete(q_res["Q"], q_res["n_bins"]), episodes=50, seed=9999)
    assert trained.mean() > baseline.mean() * 3, \
        f"训练后应远超随机策略: {trained.mean():.1f} vs {baseline.mean():.1f}"
    assert trained.mean() > 100, f"细离散表格 Q-learning 应能到 100+，实际 {trained.mean():.1f}"
    return trained


def check_dqn_learning(dqn_res: dict) -> tuple[np.ndarray, np.ndarray]:
    """DQN 必须能打满 CartPole；同时确认训练过程没发散。"""
    final = evaluate(dqn_res["policy"], episodes=50, seed=9999)
    best = evaluate(dqn_res["best_policy"], episodes=50, seed=9999)
    assert np.isfinite(dqn_res["losses"]).all(), "DQN 损失出现 NaN/Inf，训练发散"
    assert len(dqn_res["losses"]) > 10_000, "DQN 实际更新次数太少，检查 learning_starts 设置"
    assert best.mean() >= 400, f"最优快照应接近满分 500，实际 {best.mean():.1f}"
    return final, best


# ---------------------------------------------------------------- 主流程 ----
def main() -> None:
    check_discretization()
    check_replay_buffer()

    t0 = time.time()
    coarse = train_q_learning(n_bins=4, seed=0)
    fine = train_q_learning(n_bins=8, seed=0)
    print(f"[1/3] 表格 Q-learning ×2 用时 {time.time() - t0:.1f}s")
    dqn = train_dqn(seed=0)
    print(f"[2/3] DQN ({DQN_STEPS} 步) 用时 {time.time() - t0:.1f}s")

    eval_coarse = evaluate(greedy_discrete(coarse["Q"], 4), episodes=50)
    eval_fine = check_tabular_learning(fine)
    eval_final, eval_best = check_dqn_learning(dqn)
    baseline = evaluate(lambda obs: 0, episodes=50)  # 恒定左推，作下限参考
    random_baseline = rollout_random_policy(episodes=50, seed=123)
    print(f"[3/3] 评估完成 用时 {time.time() - t0:.1f}s")

    print("\n方法                          贪心回报(50 episodes)")
    print(f"  随机策略基线              {random_baseline.mean():8.1f} ± {random_baseline.std():.1f}")
    print(f"  恒定动作基线              {baseline.mean():8.1f} ± {baseline.std():.1f}")
    print(f"  粗离散表格 Q-learning     {eval_coarse.mean():8.1f} ± {eval_coarse.std():.1f}   (4^4 = 256 状态)")
    print(f"  细离散表格 Q-learning     {eval_fine.mean():8.1f} ± {eval_fine.std():.1f}   (8^4 = 4096 状态)")
    print(f"  DQN（最后一个网络）       {eval_final.mean():8.1f} ± {eval_final.std():.1f}   (连续观测)")
    print(f"  DQN（最优快照）           {eval_best.mean():8.1f} ± {eval_best.std():.1f}")
    print("\n  DQN 训练过程贪心评估: "
          + ", ".join(f"{s}->{v:.0f}" for s, v in dqn["eval_trace"]))
    print("  最后网络与最优快照的差距 = DQN 训练不稳定的直接证据。")

    save_results("rl_qlearning_dqn", {
        "coarse_q_n_bins": 4,
        "fine_q_n_bins": 8,
        "coarse_q": summarize(eval_coarse),
        "fine_q": summarize(eval_fine),
        "dqn_final": summarize(eval_final),
        "dqn_best": summarize(eval_best),
        "random": summarize(random_baseline),
        "constant_action": summarize(baseline),
        "coarse_curve": moving_average(coarse["returns"], 20),
        "fine_curve": moving_average(fine["returns"], 20),
        "dqn_curve": moving_average(dqn["returns"], 20),
        "dqn_eval_trace": dqn["eval_trace"],
        "dqn_loss": dqn["losses"][::50],
    })

    # ---- 绘图 ----
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))

    ax = axes[0]
    ax.plot(moving_average(coarse["returns"], 20), lw=1.3, label="粗离散 Q-learning (256 状态)")
    ax.plot(moving_average(fine["returns"], 20), lw=1.3, label="细离散 Q-learning (4096 状态)")
    ax.plot(moving_average(dqn["returns"], 20), lw=1.3, label="DQN (连续观测)")
    ax.axhline(random_baseline.mean(), color="gray", ls=":", lw=1, label="随机策略")
    ax.set_ylim(0, 520)
    ax.grid(alpha=0.3)
    ax.set_title("(a) 训练过程中的 episode 回报")
    ax.set_xlabel("episode")
    ax.set_ylabel("回报")
    ax.legend(fontsize=6.5, loc="lower right")

    ax = axes[1]
    thetas, theta_dots, act_coarse = q_surface_tabular(coarse["Q"], 4)
    ax.pcolormesh(np.rad2deg(thetas), theta_dots, act_coarse, cmap="coolwarm", shading="auto")
    ax.set_title("(b) 表格 Q 的策略：分块常数（粗离散）")
    ax.set_xlabel("杆角度 θ [deg]")
    ax.set_ylabel(r"杆角速度 $\dot{\theta}$ [rad/s]")

    ax = axes[2]
    _, _, diff = q_diff_surface_dqn(dqn["best"])
    im = ax.pcolormesh(np.rad2deg(thetas), theta_dots, diff, cmap="coolwarm", shading="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, label="$Q(s,\\to) - Q(s,\\leftarrow)$")
    ax.set_title("(c) DQN 的 Q 值差：光滑曲面（最优快照）")
    ax.set_xlabel("杆角度 θ [deg]")
    ax.set_ylabel(r"杆角速度 $\dot{\theta}$ [rad/s]")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_rl_qlearning_dqn.png", dpi=140)
    plt.close(fig)
    print(f"\n[fig] {FIG_DIR / 'fig_rl_qlearning_dqn.png'}")
    print("05_qlearning_dqn: all assertions passed")


if __name__ == "__main__":
    main()
