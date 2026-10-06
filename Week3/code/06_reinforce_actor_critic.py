"""复现 6：策略梯度方法 —— REINFORCE 与 Actor-Critic

对应周报第三部分「策略梯度方法：REINFORCE、Actor-Critic 架构」，同时为
「为什么 PPO 比 REINFORCE 更稳定」提供可量化的依据。

三种变体（网络结构、随机种子、episode 预算完全相同，只改优势函数）：

    1. REINFORCE        A_t = G_t（整条轨迹的折扣回报），不带基线
                        -> 梯度方差最大，学得最慢、最抖。
    2. REINFORCE+基线    A_t = G_t - V(s_t)，V 用蒙特卡洛回报拟合
                        -> 减掉与动作无关的状态价值，方差立刻降下来。
    3. A2C(演员-评论家)  A_t = r + gamma * V(s_{t+1}) - V(s_t)（自举）
                        -> 评论家单步自举，偏差大一点但方差更小、收敛更快。

脚本会实测并打印：
  * 最终贪心回报
  * 策略梯度范数的标准差（方差对比的直接证据）
  * 策略熵（探索是否过早塌缩）

运行：

    python 06_reinforce_actor_critic.py

输出：../figures/fig_rl_policy_gradient.png 与 ../results/rl_policy_gradient.json
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mpl_style import setup
from rl_common import (
    FIG_DIR,
    evaluate,
    make_env,
    moving_average,
    rollout_random_policy,
    save_results,
    summarize,
)

plt = setup()

GAMMA = 0.99
# 这三种方法的方差差距很大，单次运行的波动很明显（这正是下面要量的东西），
# 所以预算给到 1000 episode；再多结果更好看但脚本要跑好几分钟。
EPISODES = 1000
# lr=1e-2 时纯 REINFORCE 偶尔会过早塌缩成确定性动作（曾实测到 9 分）；
# 1e-3 收敛慢一点但稳得多，三种方法也才能在同一预算下公平对比。
LR_POLICY = 1e-3
LR_VALUE = 5e-3
HIDDEN = 64
METHODS = ("reinforce", "baseline", "a2c")


# ---------------------------------------------------------------- 网络 ----
class PolicyNet(nn.Module):
    """策略网络：输出两个动作的 logits。"""

    def __init__(self, obs_dim: int = 4, act_dim: int = 2, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ValueNet(nn.Module):
    """评论家：输出状态价值 V(s)。"""

    def __init__(self, obs_dim: int = 4, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------- 工具 ----
def discounted_returns(rewards: list[float], gamma: float = GAMMA) -> np.ndarray:
    """从后往前算折扣回报 G_t = r_t + gamma * G_{t+1}。

    注意是 O(n) 的逆序递推，不是每步重新求和的 O(n^2)。
    """
    out = np.zeros(len(rewards))
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def net_policy(net: PolicyNet):
    """把随机策略网络包成确定性的贪心策略，用于评估。"""
    def policy(obs: np.ndarray) -> int:
        with torch.no_grad():
            return net(torch.as_tensor(obs, dtype=torch.float32)).argmax().item()
    return policy


def evaluate_policy(net: PolicyNet, episodes: int = 50, seed: int = 9999) -> np.ndarray:
    """评估贪心策略的平均回报。"""
    return evaluate(net_policy(net), episodes=episodes, seed=seed)


# ---------------------------------------------------------------- 采样 ----
def collect_episode(env, policy: PolicyNet, seed: int | None = None) -> dict:
    """跑一个 episode，收集策略梯度需要的一切。

    `seed` 不为 None 时用它初始化 episode；传 None 则沿用环境当前随机流
    （训练循环里就靠这个，既保证初始状态多样，又保证整体可复现）。

    采样期间不开梯度、不建计算图：只记录 states/actions/rewards，
    结束后对整条轨迹做一次批量前向算 log 概率与熵。
    逐帧建图会把 500 步的计算图全留着，慢很多。

    额外记下 next_states 和 terminated：A2C 要正确地对最后一步做自举 ——
    真正终止时不能自举，被时间上限截断时则必须自举。
    """
    obs, _ = env.reset() if seed is None else env.reset(seed=seed)
    states, next_states, actions, rewards, terminated_flags = [], [], [], [], []
    done = False
    with torch.no_grad():
        while not done:
            probs = torch.softmax(policy(torch.as_tensor(obs, dtype=torch.float32)), dim=-1)
            action = torch.multinomial(probs, 1).item()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            states.append(obs)
            next_states.append(next_obs)
            actions.append(action)
            rewards.append(reward)
            terminated_flags.append(terminated)
            obs = next_obs
            done = terminated or truncated

    states_t = torch.as_tensor(np.array(states, dtype=np.float32))
    dist = torch.distributions.Categorical(logits=policy(states_t))
    actions_t = torch.as_tensor(actions)
    return {"states": states_t,
            "next_states": torch.as_tensor(np.array(next_states, dtype=np.float32)),
            "rewards": np.array(rewards, dtype=np.float32),
            "actions": actions_t,
            "log_probs": dist.log_prob(actions_t),
            "entropies": dist.entropy(),
            "terminated": np.array(terminated_flags, dtype=np.float32)}


def normalize(advantage: torch.Tensor) -> torch.Tensor:
    """优势归一化：让不同量级的回报不把步长带飞（REINFORCE 必须做）。"""
    return (advantage - advantage.mean()) / (advantage.std() + 1e-8)


# ---------------------------------------------------------------- 训练 ----
def train_policy_gradient(method: str = "reinforce", episodes: int = EPISODES,
                          gamma: float = GAMMA, lr_policy: float = LR_POLICY,
                          lr_value: float = LR_VALUE, entropy_coef: float = 0.01,
                          seed: int = 0) -> dict:
    """三种策略梯度变体共用一个训练循环，只换优势函数。

    method:
      "reinforce" -> 直接用折扣回报，无基线
      "baseline"  -> 折扣回报减去评论家的蒙特卡洛估计
      "a2c"       -> 单步自举的 TD 优势，评论家用 TD 目标训练
    """
    if method not in METHODS:
        raise ValueError(f"未知方法 {method}，可选 {METHODS}")

    torch.manual_seed(seed)
    env = make_env()
    # 播种一次即可：后续每次 env.reset() 都从这条已播种的随机流继续推进，
    # 所以 episode 的初始状态是随机的、但整个训练过程可复现。
    env.reset(seed=seed)

    policy = PolicyNet()
    opt_policy = torch.optim.Adam(policy.parameters(), lr=lr_policy)
    # 三种方法都建评论家：REINFORCE 用不到，但保持结构一致、分支更少
    value = ValueNet()
    opt_value = torch.optim.Adam(value.parameters(), lr=lr_value)

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    grad_norms: list[float] = []
    entropies: list[float] = []
    value_losses: list[float] = []
    signal_stds: list[float] = []

    for _ in range(episodes):
        rollout = collect_episode(env, policy)
        rewards = rollout["rewards"]
        states_t = rollout["states"]
        next_states_t = rollout["next_states"]
        logp = rollout["log_probs"]
        ent = rollout["entropies"].mean()

        returns = discounted_returns(rewards.tolist(), gamma)
        returns_t = torch.as_tensor(returns, dtype=torch.float32)
        episode_returns.append(math.fsum(rewards.tolist()))
        episode_lengths.append(len(rewards))
        entropies.append(ent.item())

        if method == "reinforce":
            # 无基线：直接拿折扣回报当优势，归一化只是为了不让步长发散
            raw_advantage = returns_t
            advantage = normalize(raw_advantage)
            gradient_term = (logp * advantage.detach()).mean()
        elif method == "baseline":
            with torch.no_grad():
                baseline = value(states_t)
                raw_advantage = returns_t - baseline
            advantage = normalize(raw_advantage)
            v_loss = F.mse_loss(value(states_t), returns_t)
            opt_value.zero_grad()
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(value.parameters(), 1.0)
            opt_value.step()
            value_losses.append(v_loss.item())
            gradient_term = (logp * advantage.detach()).mean()
        else:  # a2c：单步自举，终止状态不自举
            with torch.no_grad():
                values = value(states_t)
                next_values = value(next_states_t)
                targets = torch.as_tensor(rewards) + gamma * next_values * (
                    1.0 - torch.as_tensor(rollout["terminated"]))
                raw_advantage = targets - values
            advantage = normalize(raw_advantage)
            v_loss = F.mse_loss(value(states_t), targets)
            opt_value.zero_grad()
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(value.parameters(), 1.0)
            opt_value.step()
            value_losses.append(v_loss.item())
            gradient_term = (logp * advantage.detach()).mean()

        # 记录信用分配信号的离散程度（归一化前），仅作参考下
        signal_stds.append(raw_advantage.std().item())

        # 熵奖励鼓励探索，防止策略过早塌缩成确定性动作
        policy_loss = -gradient_term - entropy_coef * ent
        opt_policy.zero_grad()
        policy_loss.backward()
        # 记录策略梯度的总范数，作为方差对比的辅助证据
        grad_norms.append(math.sqrt(sum(p.grad.norm().item() ** 2
                                        for p in policy.parameters() if p.grad is not None)))
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt_policy.step()

    env.close()
    return {"policy": policy, "value": value, "returns": np.array(episode_returns),
            "lengths": np.array(episode_lengths),
            "grad_norms": np.array(grad_norms), "entropies": np.array(entropies),
            "value_losses": np.array(value_losses), "signal_stds": np.array(signal_stds),
            "method": method}


def measure_advantage_variance(policy: PolicyNet, episodes: int = 60, fit_steps: int = 400,
                               seed: int = 0) -> dict:
    """固定策略下，比较信用分配信号的方差：A_t = G_t  vs  A_t = G_t - V(s_t)。

    直接拿训练过程中的回报离散度去比会混进混杂因素（REINFORCE 收敛差、episode
    短、回报本来就集中），所以这里固定策略、用同一批轨迹，只换优势函数。

    理论依据：
      E[V(s) * ∇log π(a|s)] = 0   —— V 只依赖状态，∇log π 对动作积分为零，
    所以减去 V(s) 不引入偏差（无偏），却能降低估计方差。方差降多少取决于
    V 对回报的解释力：V 拟合得越好，Var[G - V(s)] 越小。
    """
    torch.manual_seed(seed)
    env = make_env()
    rollouts = [collect_episode(env, policy, seed=seed * 100_000 + i) for i in range(episodes)]
    env.close()

    all_states = torch.cat([r["states"] for r in rollouts])
    all_returns = torch.cat([torch.as_tensor(discounted_returns(r["rewards"].tolist()), dtype=torch.float32)
                             for r in rollouts])

    # 用同一批轨迹拟合一个评论家 V（就是 Actor-Critic 里的 critic）
    value = ValueNet()
    opt = torch.optim.Adam(value.parameters(), lr=1e-2)
    for _ in range(fit_steps):
        loss = F.mse_loss(value(all_states), all_returns)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        advantage_with = all_returns - value(all_states)
    var_explained = 1.0 - F.mse_loss(value(all_states), all_returns).item() / all_returns.var().item()

    return {
        "episodes": episodes,
        "var_explained": var_explained,
        "signal_std_no_baseline": all_returns.std().item(),
        "signal_std_with_baseline": advantage_with.std().item(),
        "signal_mean_no_baseline": all_returns.mean().item(),
    }


# ---------------------------------------------------------------- 自检 ----
def check_discounted_returns() -> None:
    """逆序递推必须等于按定义逐步求和。"""
    for rewards in ([1.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0]):
        got = discounted_returns(rewards, gamma=0.9)
        expected = [sum(0.9**k * rewards[t + k] for k in range(len(rewards) - t))
                    for t in range(len(rewards))]
        assert np.allclose(got, expected, atol=1e-12), f"折扣回报不符 {rewards}"
    # 全是 1 且 gamma=1 时，G_t = 剩余步数
    assert np.allclose(discounted_returns([1.0] * 5, gamma=1.0), [5, 4, 3, 2, 1])


def check_policy_gradient_direction() -> None:
    """梯度的符号要正确：提高“好动作”的概率，降低“差动作”的概率。

    用一个人为构造的单步问题验证：只有一个状态、两个动作，
    正优势应该把动作 1 的概率推上去。
    """
    torch.manual_seed(0)
    policy = PolicyNet(obs_dim=2, act_dim=2, hidden=8)
    opt = torch.optim.SGD(policy.parameters(), lr=0.5)
    obs = torch.zeros(2)

    def prob_action1() -> float:
        with torch.no_grad():
            return torch.softmax(policy(obs), dim=-1)[1].item()

    before = prob_action1()
    for _ in range(50):
        dist = torch.distributions.Categorical(logits=policy(obs))
        loss = -(dist.log_prob(torch.tensor(1)) * torch.tensor(1.0))  # 正优势
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = prob_action1()
    assert after > before, f"正优势应提升该动作概率: {before:.4f} -> {after:.4f}"
    assert after > 0.9, f"反复施加正优势应收敛到该动作，实际 {after:.4f}"

    # 反号：负优势应压低概率
    for _ in range(50):
        dist = torch.distributions.Categorical(logits=policy(obs))
        loss = -(dist.log_prob(torch.tensor(1)) * torch.tensor(-1.0))  # 负优势
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert prob_action1() < after, "负优势应压低该动作概率"


def check_a2c_bootstrap() -> None:
    """A2C 的自举必须遵守：终止不自举、截断自举。"""
    terminated = torch.tensor([0.0, 1.0])
    next_values = torch.tensor([5.0, 5.0])
    rewards = torch.tensor([1.0, 1.0])
    targets = rewards + GAMMA * next_values * (1.0 - terminated)
    # 第一步未终止 -> 自举；第二步终止 -> 只留即时奖励
    assert np.isclose(targets[0].item(), 1.0 + GAMMA * 5.0, atol=1e-12)
    assert np.isclose(targets[1].item(), 1.0, atol=1e-12)


def check_advantage_variance() -> dict:
    """实测：减去基线后信用分配信号的方差必须下降。

    在固定策略、同一批轨迹上比较，排除“收敛好坏导致 episode 长度不同”的干扰。
    """
    torch.manual_seed(0)
    # 用一个未训练的策略：状态覆盖广、动作接近均匀，能看出基线的真实效果
    policy = PolicyNet()
    measurement = measure_advantage_variance(policy, episodes=60, seed=0)
    assert measurement["var_explained"] > 0.2, \
        f"评论家应解释掉一部分回报方差，实际 {measurement['var_explained']:.3f}"
    assert measurement["signal_std_with_baseline"] < measurement["signal_std_no_baseline"], \
        (f"减基线后信号方差应下降: {measurement['signal_std_with_baseline']:.4f} "
         f"vs {measurement['signal_std_no_baseline']:.4f}")
    return measurement


def check_all_learned(results: dict) -> None:
    """三种方法都必须明显超过随机策略，且至少有一种能接近满分。

    不要求每种都打满：策略梯度本来就方差大，单次跑出次优解很正常，
    这恰恰是要展示的现象。但必须确保它们确实在学（远高于随机），
    而不是代码写错了导致什么都没学到。
    """
    baseline_return = rollout_random_policy(episodes=50, seed=123).mean()
    scores = {}
    for name, res in results.items():
        scores[name] = evaluate_policy(res["policy"]).mean()
        assert scores[name] > baseline_return * 2, \
            f"{name} 未学到有效策略: {scores[name]:.1f} vs 随机 {baseline_return:.1f}"
    assert max(scores.values()) > 300, \
        f"至少一种方法应接近满分，实际最好只有 {max(scores.values()):.1f}"


# ---------------------------------------------------------------- 主流程 ----
def main() -> None:
    check_discounted_returns()
    check_policy_gradient_direction()
    check_a2c_bootstrap()

    results = {}
    evals = {}
    t0 = time.time()
    for method in METHODS:
        results[method] = train_policy_gradient(method=method, seed=0)
        evals[method] = evaluate_policy(results[method]["policy"])
        print(f"[{method:9s}] {EPISODES} episodes, 用时 {time.time() - t0:.1f}s")

    # 先把结果打出来，再跑断言 —— 即使某个阈值没过，也能看到真实数据
    random_baseline = rollout_random_policy(episodes=50, seed=123)
    variance = measure_advantage_variance(PolicyNet(), episodes=60, seed=0)

    print("\n方法                  贪心回报      平均 episode 长度   平均熵")
    print(f"  {'随机策略':16s} {random_baseline.mean():6.1f} ± {random_baseline.std():5.1f}")
    for method in METHODS:
        res = results[method]
        print(f"  {method:16s} {evals[method].mean():6.1f} ± {evals[method].std():5.1f}   "
              f"{res['returns'].mean():13.1f}   {res['entropies'].mean():6.3f}")
    print(f"\n  固定策略下的信用分配信号标准差（同一批 {variance['episodes']} 条轨迹）：")
    print(f"    A = G_t        : {variance['signal_std_no_baseline']:8.3f}")
    print(f"    A = G_t - V(s) : {variance['signal_std_with_baseline']:8.3f}"
          f"   （评论家解释了 {variance['var_explained'] * 100:.1f}% 的回报方差）")
    print("  理论依据：E[V(s)·∇log π(a|s)] = 0，基线只依赖状态，对动作积分为零，")
    print("  所以减掉它不引入偏差，却能明确压低估计方差 —— 这就是 Actor-Critic 好训的原因。")
    print("  PPO 在此之上再加重要性采样裁剪，直接限住每次更新的策略变化量（见 07_ppo.py）。")

    check_advantage_variance()
    check_all_learned(results)

    payload = {"episodes": EPISODES, "gamma": GAMMA, "lr_policy": LR_POLICY,
               "random_baseline": summarize(random_baseline),
               "signal_std_no_baseline": variance["signal_std_no_baseline"],
               "signal_std_with_baseline": variance["signal_std_with_baseline"],
               "critic_var_explained": variance["var_explained"]}
    for method in METHODS:
        res = results[method]
        payload[f"{method}_eval"] = summarize(evals[method])
        payload[f"{method}_signal_std"] = res["signal_stds"].mean().item()
        payload[f"{method}_grad_std"] = res["grad_norms"].std().item()
        payload[f"{method}_entropy_mean"] = res["entropies"].mean().item()
        payload[f"{method}_curve"] = moving_average(res["returns"], 20)
        payload[f"{method}_lengths"] = res["lengths"]
        payload[f"{method}_returns_raw"] = res["returns"]
    save_results("rl_policy_gradient", payload)

    # ---- 绘图 ----
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    labels = {"reinforce": "REINFORCE（无基线）", "baseline": "REINFORCE + 基线",
              "a2c": "A2C（自举评论家）"}

    ax = axes[0]
    for method in METHODS:
        ax.plot(moving_average(results[method]["returns"], 20), lw=1.3, label=labels[method])
    ax.set_ylim(0, 520)
    ax.grid(alpha=0.3)
    ax.set_title("(a) episode 回报（滑动平均）")
    ax.set_xlabel("episode")
    ax.set_ylabel("回报")
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[1]
    bars = ax.bar(["A = G\n(REINFORCE)", "A = G - V(s)\n(Actor-Critic)"],
                  [variance["signal_std_no_baseline"], variance["signal_std_with_baseline"]],
                  color=["tab:red", "tab:green"])
    ax.bar_label(bars, fmt="%.2f", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.set_title("(b) 信用分配信号的标准差（固定策略）")
    ax.set_ylabel("std of advantage")
    ax.tick_params(axis="x", labelsize=8)

    ax = axes[2]
    for method in METHODS:
        ax.plot(moving_average(results[method]["entropies"], 20), lw=1.3, label=labels[method])
    ax.grid(alpha=0.3)
    ax.set_title("(c) 策略熵（探索程度）")
    ax.set_xlabel("episode")
    ax.set_ylabel("熵 [nats]")
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_rl_policy_gradient.png", dpi=140)
    plt.close(fig)
    print(f"\n[fig] {FIG_DIR / 'fig_rl_policy_gradient.png'}")
    print("06_reinforce_actor_critic: all assertions passed")


if __name__ == "__main__":
    main()
