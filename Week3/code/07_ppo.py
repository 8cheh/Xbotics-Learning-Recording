"""复现 7：PPO（近端策略优化）+ 裁剪消融实验

对应周报第三部分作业题「为什么 PPO 在实际应用中比 REINFORCE 更稳定？」
和「A3C 的主要优势是什么？」（A3C 的异步并行采样在 08 的对比表里说明）。

PPO 在 REINFORCE / Actor-Critic 之上做了两件事：

  1. 重要性采样 + 裁剪（clipped surrogate）
     一次采样可以重复训练多个 epoch（反复利用数据，样本效率高几个量级），
     同时把 ratio = π_new/π_old 裁到 [1-ε, 1+ε]：
         L = -min( ratio * A,  clip(ratio, 1-ε, 1+ε) * A )
     ratio 跑出范围时梯度直接归零，单次更新的策略变化量被硬性限住 ——
     这就是“稳定”两个字的来源。
  2. GAE(λ) 优势估计
     在“单步自举（偏差小但方差大）”和“蒙特卡洛回报（无偏但方差大）”之间
     用 λ 插值，兼顾两者。

本脚本跑三组：
  A. PPO（clip=0.2，10 个 epoch）
  B. 消融：不裁剪（clip=∞，其余完全相同）—— 去掉 PPO 的关键机制
  C. 消融：只训练 1 个 epoch（退化成普通的、每次只用一遍数据的策略梯度）

并把 06 里 REINFORCE 的学习曲线调出来，用“环境步数”作横轴对比样本效率。

运行（需要先跑过 06，以便读取 REINFORCE 的曲线）：

    python 07_ppo.py

输出：../figures/fig_rl_ppo.png 与 ../results/rl_ppo.json
"""

from __future__ import annotations

import math
import operator
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mpl_style import setup
from rl_common import (
    FIG_DIR,
    evaluate,
    load_results,
    make_env,
    moving_average,
    rollout_random_policy,
    save_results,
    summarize,
)

plt = setup()

GAMMA = 0.99
LAM = 0.95
CLIP = 0.2
TOTAL_STEPS = 100_000
ROLLOUT_STEPS = 2_048
EPOCHS = 10
MINIBATCH = 64
LR = 3e-4
ENTROPY_COEF = 0.01
VF_COEF = 0.5
HIDDEN = 64


class ActorCritic(nn.Module):
    """共享躯干 + 策略头 + 价值头（CartPole 上共享足够，也更省）。"""

    def __init__(self, obs_dim: int = 4, act_dim: int = 2, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, act_dim)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)


def net_policy(model: ActorCritic):
    """贪心策略，用于评估。"""
    def policy(obs: np.ndarray) -> int:
        with torch.no_grad():
            logits, _ = model(torch.as_tensor(obs, dtype=torch.float32))
            return logits.argmax().item()
    return policy


def compute_gae(rewards: np.ndarray, values: np.ndarray, next_value: float,
                terminated: np.ndarray, gamma: float = GAMMA, lam: float = LAM) -> tuple[np.ndarray, np.ndarray]:
    """广义优势估计 GAE(λ)。

        delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        A_t     = delta_t + gamma * lam * (1 - done_t) * A_{t+1}

    λ=0 退化成单步 TD（方差小、偏差大）；λ=1 退化成蒙特卡洛回报（无偏、方差大）。
    必须逆序递推，而且要在真正终止处截断。
    """
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float64)
    running = 0.0
    for t in reversed(range(n)):
        next_v = next_value if t == n - 1 else values[t + 1]
        non_terminal = 1.0 - terminated[t]
        delta = rewards[t] + gamma * next_v * non_terminal - values[t]
        running = delta + gamma * lam * non_terminal * running
        advantages[t] = running
    return advantages, advantages + values


def collect_rollout(env, model: ActorCritic, steps: int, obs) -> tuple[dict, object, float]:
    """用当前策略采样固定步数，返回训练 PPO 所需的一整批数据。"""
    obs_buf = np.zeros((steps, 4), dtype=np.float32)
    act_buf = np.zeros(steps, dtype=np.int64)
    logp_buf = np.zeros(steps, dtype=np.float32)
    rew_buf = np.zeros(steps, dtype=np.float32)
    val_buf = np.zeros(steps, dtype=np.float32)
    term_buf = np.zeros(steps, dtype=np.float32)

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    current_return, current_len = 0.0, 0

    with torch.no_grad():
        for t in range(steps):
            logits, value = model(torch.as_tensor(obs, dtype=torch.float32))
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action.item())

            obs_buf[t] = obs
            act_buf[t] = action.item()
            logp_buf[t] = dist.log_prob(action).item()
            val_buf[t] = value.item()
            rew_buf[t] = reward
            term_buf[t] = 1.0 if terminated else 0.0

            current_return += reward
            current_len += 1
            obs = next_obs
            if terminated or truncated:
                episode_returns.append(current_return)
                episode_lengths.append(current_len)
                current_return, current_len = 0.0, 0
                obs, _ = env.reset()

    with torch.no_grad():
        _, next_value = model(torch.as_tensor(obs, dtype=torch.float32))
    return {"obs": obs_buf, "actions": act_buf, "log_probs": logp_buf, "rewards": rew_buf,
            "values": val_buf, "terminated": term_buf,
            "returns": np.array(episode_returns), "lengths": np.array(episode_lengths)}, obs, next_value.item()


def train_ppo(total_steps: int = TOTAL_STEPS, rollout_steps: int = ROLLOUT_STEPS,
              epochs: int = EPOCHS, minibatch: int = MINIBATCH, lr: float = LR,
              clip: float = CLIP, entropy_coef: float = ENTROPY_COEF, vf_coef: float = VF_COEF,
              lam: float = LAM, seed: int = 0) -> dict:
    """PPO 主循环。clip=math.inf 即“不裁剪”的消融对照组。

    这里只保留“裁剪”作为限住策略变化的机制（原论文的做法），
    再叠加学习率线性退火。实现里常见的另一道保险是“近似 KL 超过
    1.5 倍目标值就提前结束本轮 epoch”；本脚本不默认开它，
    否则它会先于裁剪生效，消融实验就看不出裁剪本身的作用了。
    """
    torch.manual_seed(seed)
    env = make_env()
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    base_lr = lr

    obs, _ = env.reset(seed=seed)
    all_returns: list[float] = []
    all_lengths: list[int] = []
    update_kls: list[float] = []
    clip_fracs: list[float] = []
    entropies: list[float] = []
    eval_trace: list[tuple[int, float]] = []

    n_updates = total_steps // rollout_steps
    for update in range(n_updates):
        # 学习率线性退火
        for group in optimizer.param_groups:
            group["lr"] = base_lr * (1.0 - update / n_updates)

        batch, obs, next_value = collect_rollout(env, model, rollout_steps, obs)
        all_returns.extend(batch["returns"].tolist())
        all_lengths.extend(batch["lengths"].tolist())

        advantages, returns = compute_gae(batch["rewards"], batch["values"], next_value,
                                          batch["terminated"], GAMMA, lam)
        # 优势归一化：让 actor 和 critic 的损失量级匹配
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t = torch.as_tensor(batch["obs"])
        act_t = torch.as_tensor(batch["actions"])
        old_logp_t = torch.as_tensor(batch["log_probs"])
        adv_t = torch.as_tensor(advantages, dtype=torch.float32)
        ret_t = torch.as_tensor(returns, dtype=torch.float32)

        idx = np.arange(rollout_steps)
        update_kl = 0.0
        update_clip_frac = 0.0
        n_minibatches = 0
        for epoch in range(epochs):
            np.random.default_rng(update * 1000 + epoch).shuffle(idx)
            for start in range(0, rollout_steps, minibatch):
                mb = idx[start:start + minibatch]
                logits, value = model(obs_t[mb])
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(act_t[mb])
                entropy = dist.entropy().mean()

                # 新旧策略的概率比。取 exp 前先夹一下，避免极端值把 exp 撑爆
                log_ratio = torch.clamp(new_logp - old_logp_t[mb], -20.0, 20.0)
                ratio = torch.exp(log_ratio)
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv_t[mb]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value, ret_t[mb])

                loss = policy_loss + vf_coef * value_loss - entropy_coef * entropy
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                with torch.no_grad():
                    approx_kl = (-log_ratio).mean().item()
                    update_kl += approx_kl
                    # 裁剪真正生效的比例：ratio 落在 [1-ε, 1+ε] 之外
                    update_clip_frac += (torch.abs(ratio - 1.0) > clip).float().mean().item()
                    entropies.append(entropy.item())
                n_minibatches += 1

        update_kls.append(update_kl / max(n_minibatches, 1))
        clip_fracs.append(update_clip_frac / max(n_minibatches, 1))

        if (update + 1) % 10 == 0:
            score = evaluate(net_policy(model), episodes=20, seed=5000).mean()
            eval_trace.append(((update + 1) * rollout_steps, score))

    env.close()
    return {"model": model, "policy": net_policy(model), "returns": np.array(all_returns),
            "lengths": np.array(all_lengths), "kls": np.array(update_kls),
            "clip_fracs": np.array(clip_fracs), "entropies": np.array(entropies),
            "eval_trace": eval_trace, "clip": clip, "epochs": epochs}


# ---------------------------------------------------------------- 自检 ----
def check_gae() -> None:
    """GAE 的两端必须分别退化成单步 TD 和蒙特卡洛回报。"""
    rewards = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    values = np.array([2.0, 2.0, 2.0], dtype=np.float64)
    terminated = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    next_value = 2.0

    # lam=0 -> A_t = r_t + gamma*V(s_{t+1}) - V(s_t)
    adv0, _ = compute_gae(rewards, values, next_value, terminated, gamma=0.9, lam=0.0)
    expected0 = np.array([1.0 + 0.9 * 2.0 - 2.0, 1.0 + 0.9 * 2.0 - 2.0, 1.0 - 2.0])
    assert np.allclose(adv0, expected0, atol=1e-12), f"lam=0 应退化成 TD 误差: {adv0}"

    # lam=1 -> A_t = 折扣回报 - V(s_t)
    adv1, ret1 = compute_gae(rewards, values, next_value, terminated, gamma=0.9, lam=1.0)
    mc = np.array([1.0 + 0.9 * 1.0 + 0.9**2 * 1.0, 1.0 + 0.9 * 1.0, 1.0])
    assert np.allclose(adv1, mc - values, atol=1e-12), f"lam=1 应退化成蒙特卡洛优势: {adv1}"
    assert np.allclose(ret1, mc, atol=1e-12), "returns 应为 A + V"


def check_gae_truncation() -> None:
    """终止步必须切断自举，否则价值会被系统性高估。"""
    rewards = np.array([1.0], dtype=np.float64)
    values = np.array([0.0], dtype=np.float64)
    terminated = np.array([1.0], dtype=np.float64)
    adv, _ = compute_gae(rewards, values, next_value=100.0, terminated=terminated, gamma=0.99, lam=0.95)
    assert np.isclose(adv[0], 1.0, atol=1e-12), f"终止处不应自举，实际 {adv[0]}"
    adv_open, _ = compute_gae(rewards, values, next_value=100.0,
                              terminated=np.array([0.0]), gamma=0.99, lam=0.95)
    assert adv_open[0] > 1.0, "未终止处应自举"


def check_clip_objective() -> None:
    """裁剪目标的三个关键性质：
    1. ratio 在范围内时 min 不生效（梯度正常）；
    2. 正优势且 ratio 过大时，裁剪把梯度掐掉；
    3. 裁剪让 surrogate 对 ratio 的导数在越界处为零。
    """
    clip = 0.2
    adv = torch.tensor([1.0])

    def surrogate(ratio_value: float) -> float:
        ratio = torch.tensor([ratio_value], requires_grad=True)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
        out = torch.min(surr1, surr2)
        out.backward()
        grad = ratio.grad
        assert grad is not None, "surrogate 必须对 ratio 可导"
        return grad.item()

    # 范围内：梯度 = adv
    assert np.isclose(surrogate(1.0), 1.0, atol=1e-12)
    assert np.isclose(surrogate(1.1), 1.0, atol=1e-12)
    # 正优势 + ratio 超过上界：min 选中被裁剪的那支，梯度为 0
    assert np.isclose(surrogate(1.3), 0.0, atol=1e-12), "正优势越界时梯度应被掐掉"
    # 正优势 + ratio 低于下界：目标是 ratio*adv，梯度仍为 adv（鼓励往回走）
    assert np.isclose(surrogate(0.5), 1.0, atol=1e-12)

    # 负优势对称地相反：ratio 过小时梯度被掐掉
    adv = torch.tensor([-1.0])
    assert np.isclose(surrogate(0.5), 0.0, atol=1e-12), "负优势越界时梯度应被掐掉"
    assert np.isclose(surrogate(1.3), -1.0, atol=1e-12)

    # ratio 恒为 1 时，PPO 目标就等于普通策略梯度目标（第一轮 epoch 必然如此）
    ratio1 = torch.tensor([1.0], requires_grad=True)
    surr1 = ratio1 * adv
    surr2 = torch.clamp(ratio1, 1 - clip, 1 + clip) * adv
    assert torch.isclose(torch.min(surr1, surr2), surr1)


def check_ppo_clip_bounds_kl() -> tuple[dict, dict]:
    """核心结论：裁剪把每次更新的近似 KL 压在很小的范围内。"""
    t0 = time.time()
    clipped = train_ppo(total_steps=40_960, seed=0, clip=CLIP)
    unclipped = train_ppo(total_steps=40_960, seed=0, clip=math.inf)
    print(f"    裁剪消融对比用时 {time.time() - t0:.1f}s")

    kl_clip = clipped["kls"].mean()
    kl_free = unclipped["kls"].mean()
    print(f"    PPO  平均KL={kl_clip:.5f} 最大KL={clipped['kls'].max():.5f} "
          f"裁剪比例={clipped['clip_fracs'].mean():.3f}")
    print(f"    不裁剪 平均KL={kl_free:.5f} 最大KL={unclipped['kls'].max():.5f}")
    assert kl_clip < kl_free, \
        f"裁剪应把 KL 压得更小: {kl_clip:.5f} vs 不裁剪 {kl_free:.5f}"
    assert clipped["clip_fracs"].mean() > 0.0, "裁剪从未生效，说明不算 PPO"
    # 裁剪生效的样本占比应该是个可观的比例，否则这项机制形同虚设
    assert clipped["clip_fracs"].mean() > 0.01, \
        f"裁剪触发比例过低（{clipped['clip_fracs'].mean():.4f}），机制没起作用"
    return clipped, unclipped


def check_ppo_solves() -> tuple[dict, float, int]:
    """PPO 必须解掉 CartPole，而且比 06 的 REINFORCE 更省样本。"""
    result = train_ppo(total_steps=TOTAL_STEPS, seed=0)
    score = evaluate(result["policy"], episodes=50, seed=9999).mean()
    assert score >= 400, f"PPO 应接近满分，实际 {score:.1f}"

    # 用环境步数比较样本效率：PPO 到达 400 分所需的步数
    steps_to_solve = next((s for s, v in result["eval_trace"] if v >= 400), TOTAL_STEPS)
    print(f"    PPO 达到 400 分用了 {steps_to_solve} 环境步")
    return result, score, steps_to_solve


# ---------------------------------------------------------------- 主流程 ----
def main() -> None:
    check_gae()
    check_gae_truncation()
    check_clip_objective()

    print("[1/3] 裁剪消融（PPO vs 不裁剪）")
    clipped, unclipped = check_ppo_clip_bounds_kl()

    print("[2/3] 完整 PPO 训练")
    result, score, steps_to_solve = check_ppo_solves()
    print(f"    PPO 最终贪心回报 {score:.1f}")

    print("[3/3] 与 06 的 REINFORCE 对比样本效率")
    pg = load_results("rl_policy_gradient")
    reinforce_steps = np.cumsum(pg["reinforce_lengths"])
    # 06 已经存好了滑动平均后的曲线，直接复用
    reinforce_curve = np.asarray(pg["reinforce_curve"])
    ppo_steps = np.cumsum(result["lengths"])
    ppo_curve = moving_average(result["returns"], 20)

    # REINFORCE 达到 400 分 / 500 分的环境步数
    def steps_to(curve, threshold: float, steps) -> int:
        return next((operator.index(steps[i]) for i, v in enumerate(curve) if v >= threshold),
                    -1)

    reinforce_400 = steps_to(reinforce_curve, 400, reinforce_steps)
    reinforce_500 = steps_to(reinforce_curve, 500, reinforce_steps)
    ppo_500 = steps_to(ppo_curve, 500, ppo_steps)
    print(f"    REINFORCE 首次到 400 分：{reinforce_400} 步；到 500 分："
          f"{'未达到' if reinforce_500 < 0 else f'{reinforce_500} 步'}")
    print(f"    PPO       首次到 400 分：{steps_to_solve} 步；到 500 分："
          f"{'未达到' if ppo_500 < 0 else f'{ppo_500} 步'}")
    print(f"    总预算：REINFORCE {operator.index(reinforce_steps[-1])} 环境步，"
          f"PPO {TOTAL_STEPS} 环境步；最终回报 "
          f"{np.mean(pg['reinforce_returns_raw'][-50:]):.1f} vs {score:.1f}")
    print("    注意：两条训练回报曲线都是“随机策略”跑出来的，PPO 训练时采样用的是")
    print("    随机策略，所以训练回报会低于其贪心评估（绿虚线，很早就到 500）；")
    print("    REINFORCE 每 episode 只更新一次，PPO 则把同一批数据反复训 10 个 epoch。")

    print("\n对比项                          PPO          不裁剪(消融)   REINFORCE(06)")
    print(f"  平均每次更新 KL               {clipped['kls'].mean():.5f}      "
          f"{unclipped['kls'].mean():.5f}        -")
    print(f"  最大 KL                       {clipped['kls'].max():.5f}      "
          f"{unclipped['kls'].max():.5f}        -")
    print(f"  裁剪生效比例                  {clipped['clip_fracs'].mean():.3f}        "
          f"{0.0:.3f}          -")
    print(f"  环境步数                      {TOTAL_STEPS:6d}       {40_960:6d}      "
          f"{operator.index(reinforce_steps[-1]):6d}")
    print(f"  贪心回报                      {score:6.1f}            -      "
          f"{np.mean(pg['reinforce_returns_raw'][-50:]):6.1f}")

    payload = {"total_steps": TOTAL_STEPS, "rollout_steps": ROLLOUT_STEPS,
               "epochs": EPOCHS, "clip": CLIP, "lam": LAM,
               "ppo_eval": {"mean": score},
               "ppo_kl_mean": clipped["kls"].mean().item(),
               "ppo_kl_max": clipped["kls"].max().item(),
               "ppo_clip_frac": clipped["clip_fracs"].mean().item(),
               "ppo_eval_trace": [[s, v] for s, v in result["eval_trace"]],
               "ppo_curve": ppo_curve, "ppo_steps": ppo_steps,
               "unclipped_kl_mean": unclipped["kls"].mean().item(),
               "unclipped_kl_max": unclipped["kls"].max().item(),
               "unclipped_curve": moving_average(unclipped["returns"], 20),
               "unclipped_clip_frac": unclipped["clip_fracs"].mean().item(),
               "reinforce_curve": reinforce_curve,
               "reinforce_steps": reinforce_steps,
               "reinforce_total_steps": operator.index(reinforce_steps[-1]),
               "reinforce_final_return": np.mean(pg["reinforce_returns_raw"][-50:]).item(),
               "reinforce_steps_to_400": reinforce_400,
               "ppo_steps_to_400": steps_to_solve,
               "ppo_steps_to_500": ppo_500,
               "random_baseline": summarize(rollout_random_policy(episodes=50, seed=123))}
    save_results("rl_ppo", payload)

    # ---- 绘图 ----
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))

    ax = axes[0]
    ax.plot(reinforce_steps, reinforce_curve, lw=1.2, color="tab:blue",
            label="REINFORCE 训练回报（06）")
    ax.plot(ppo_steps, ppo_curve, lw=1.2, color="tab:orange", label="PPO 训练回报")
    if result["eval_trace"]:
        eval_steps = [s for s, _ in result["eval_trace"]]
        eval_scores = [v for _, v in result["eval_trace"]]
        ax.plot(eval_steps, eval_scores, "--", lw=1.6, marker="o", ms=3, color="tab:green",
                label="PPO 贪心评估")
    ax.set_xscale("log")
    ax.set_ylim(0, 540)
    ax.grid(alpha=0.3, which="both")
    ax.set_title("(a) 回报 vs 环境步数")
    ax.set_xlabel("环境步数（对数轴）")
    ax.set_ylabel("回报")
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[1]
    ax.plot(clipped["kls"], lw=1.2, marker="o", ms=3, label=f"PPO 裁剪 (ε={CLIP})")
    ax.plot(unclipped["kls"], lw=1.2, marker="s", ms=3, label="不裁剪（消融）")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.set_title("(b) 每次更新的近似 KL")
    ax.set_xlabel("更新次数")
    ax.set_ylabel("KL(π_old ‖ π_new)")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(moving_average(clipped["clip_fracs"], 3), lw=1.4, color="tab:green",
            label="裁剪生效的样本比例")
    ax.set_ylim(0, max(0.05, clipped["clip_fracs"].max() * 1.2))
    ax.grid(alpha=0.3)
    ax.set_title("(c) 裁剪触发比例")
    ax.set_xlabel("更新次数")
    ax.set_ylabel("被裁剪的样本占比")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_rl_ppo.png", dpi=140)
    plt.close(fig)
    print(f"\n[fig] {FIG_DIR / 'fig_rl_ppo.png'}")
    print("07_ppo: all assertions passed")


if __name__ == "__main__":
    main()
