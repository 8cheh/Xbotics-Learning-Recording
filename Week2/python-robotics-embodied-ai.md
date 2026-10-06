# Python 机器人 / 具身智能常用库与函数速查

面向"从零搭一个能跑的机器人具身智能项目"的场景：仿真 → 建模 → 感知 → 策略学习 → 真机部署。

约定：

- 所有 `import` 均为官方推荐写法；`pip install` 命令按需改 `pip3` / `uv pip`。
- 代码片段是最小可运行骨架，去掉了一个真实项目里的错误处理。
- 库的 API 变动很快（尤其 LeRobot / Isaac Lab），片段用于理解"该调哪个函数"，具体签名以官方文档为准。

---

## 0. 全局地图

| 层次 | 干什么 | 首选库 | 备注 |
|---|---|---|---|
| 数学 / 几何 | 数组、旋转、SE(3)、插值 | `numpy` `scipy.spatial.transform` `pytransform3d` | 一定要先统一四元数序 |
| 运动学 / 动力学 | FK、IK、雅可比、RNEA、CRBA | `pinocchio` `roboticstoolbox-python` `ikpy` `drake` | 真机 + MPC 用 pinocchio |
| 物理仿真 | 刚体、接触、相机 | `pybullet` `mujoco` `sapien` `Isaac Lab` | 快速验证 pybullet，GPU 并行 mujoco/Isaac |
| 环境接口 | 统一 `reset/step` | `gymnasium` | 所有 RL 代码的事实标准 |
| 操作 Benchmark | 标准任务 + 演示数据 | `robosuite` `ManiSkill` `LIBERO` `RLBench` | 复现论文必用 |
| 策略学习 | 模仿学习 / RL / VLA | `LeRobot` `robomimic` `diffusers` `transformers` `torch` | ACT、Diffusion Policy 都在 LeRobot |
| 感知 | 点云、检测、分割、标定 | `open3d` `opencv-python` `ultralytics` `segment-anything` `pytorch3d` | 手眼标定是最大坑 |
| 中间件 | 话题、服务、实时控制 | `rclpy` (ROS 2) `ros2_control` | 部署环节绕不过 |
| 真机 SDK | 电机 / 机械臂驱动 | `dynamixel-sdk` `feetech-servo-sdk` `xarm-python-sdk` `pymycobot` | 先做限位和急停 |
| 加速推理 | 部署时延 | `onnxruntime` `tensorrt` `torch.compile` | 控制周期 <50ms 才上真机 |

---

## 1. 数学与几何基础

### 1.1 numpy —— 机器人的通用语言

```bash
pip install numpy
```

```python
import numpy as np

q = np.zeros(7)                      # 关节角，float64
dq = np.array([0.1] * 7)             # 关节速度
np.clip(q, -2.9, 2.9)                # 关节限位，写控制前必做
np.linalg.norm(dq)                   # 速度范数（限制末端速度常用）
np.linalg.pinv(J)                    # 阻尼最小二乘的简化版
np.linalg.solve(A, b)                # 解 Ax=b，别用 inv(A) @ b
np.linalg.svd(J, full_matrices=False) # 奇异值，判断奇异位形
np.deg2rad(q); np.rad2deg(q)         # 角度换算
np.interp(t, ts, ys)                 # 轨迹重采样
np.cumsum(dq) * dt                   # 速度积分成位置
rng = np.random.default_rng(0)       # 固定种子，复现实验
```

要点：

- 机器人代码里几乎全是 float64；进网络前再转 `torch.float32`。
- 关节顺序（URDF 里 joint 的声明顺序）必须和控制器一致，写错不会报错，只会动错。
- `np.random.seed()` 是全局状态；新代码用 `default_rng`。

### 1.2 scipy —— 旋转 / 插值 / 优化

```bash
pip install scipy
```

```python
from scipy.spatial.transform import Rotation as R, Slerp

# 欧拉角 -> 旋转；degrees=True 别忘
r = R.from_euler("xyz", [0, 0, 90], degrees=True)
r.as_matrix()                      # 3x3 旋转矩阵
r.as_quat()                        # [x, y, z, w] —— scipy 是 xyzw！
r.as_rotvec()                      # 轴角
R.from_quat([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])   # 输入也是 xyzw

# 旋转矩阵 -> 欧拉角（解决万向锁时要注意顺序）
R.from_matrix(np.eye(3)).as_euler("zyx", degrees=True)

# 姿态复合：先转 A 再转 B（注意顺序，矩阵左乘）
(R.from_euler("z", 90, degrees=True) * R.from_euler("x", 30, degrees=True)).as_matrix()

# 逆、作用到向量
r.inv().apply([1, 0, 0])
r.apply(np.array([[1,0,0],[0,1,0]]))    # 支持批量 (N,3)

# 姿态插值（笛卡尔轨迹常用）
# 注意：单轴 from_euler 要求 angles 形状为 (N, 1)，写 [0, 90] 会被当成"一个旋转含两个轴"
rots = R.from_euler("z", [[0], [90]], degrees=True)
slerp = Slerp([0, 1], rots)
slerp(0.5).as_euler("zyx", degrees=True)[0]   # 45.0
slerp(0.25).as_euler("zyx", degrees=True)[0] # 22.5

# 数值积分 / 优化 / 滤波
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, least_squares
from scipy.signal import butter, sosfilt
solve_ivp(f, [0, 1], y0, t_eval=np.linspace(0, 1, 101))
least_squares(residual, x0)          # 手眼标定 / 参数辨识
```

> **最容易踩的坑**：四元数序。
> `scipy` = `[x, y, z, w]`（xyzw）；ROS / `geometry_msgs` / 很多 URDF 工具 = `[x, y, z, w]`；
> pybullet `getQuaternionFromEuler` 也是 xyzw；但 **Isaac Sim / USD / 部分内部实现用 wxyz**。
> 只在边界处转换一次，中间层永远用一种。

### 1.3 SE(3) 一体封装（可选，但省事）

```bash
pip install pytransform3d        # 或 spatialmath-python / manif
```

```python
import pytransform3d.transformations as pt
T = pt.transform_from_pq([0.5, 0, 0.3, 0, 0, 0, 1])   # p + quat(wxyz)
pt.invert_transform(T)
pt.concat(A2B, B2C)              # 链式变换，比手写 @ 更不容易搞错方向
pt.transform(A2B, p)             # 变换一个点
from pytransform3d import rotations as pr
pr.matrix_from_compact_axis_angle([0, 0, np.pi/2])
```

用途：正运动学链、相机外参、多坐标系叠加，手写容易出错。

---

## 2. 物理仿真

### 2.1 PyBullet —— 最快上手的验证环境

```bash
pip install pybullet
```

```python
import pybullet as p
import pybullet_data
import numpy as np

cid = p.connect(p.GUI)                          # GUI / DIRECT（无头批量）
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(1 / 240)                          # 注意：物理步长 ≠ 控制步长

plane = p.loadURDF("plane.urdf")
robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
n_joints = p.getNumJoints(robot)
joint_ids = [j for j in range(n_joints)
             if p.getJointInfo(robot, j)[2] != p.JOINT_FIXED]

# 关节状态
p.resetJointState(robot, joint_ids[0], 0.5)
p.getJointState(robot, joint_ids[0])            # (pos, vel, reaction, torque)

# 控制：三种模式
p.setJointMotorControl2(robot, joint_ids[0], p.POSITION_CONTROL,
                        targetPosition=0.5, force=500)
p.setJointMotorControlArray(robot, joint_ids, p.POSITION_CONTROL,
                            targetPositions=[0.0]*len(joint_ids))
p.setJointMotorControl2(robot, joint_ids[-1], p.TORQUE_CONTROL, force=1.0)

# 末端位姿 / IK
state = p.getLinkState(robot, n_joints - 1, computeForwardKinematics=True)
pos, orn = state[4], state[5]                   # 世界系位置 + 四元数(xyzw)
q_ik = p.calculateInverseKinematics(robot, n_joints - 1, [0.5, 0, 0.5])

# 相机（用于视觉策略）
view = p.computeViewMatrixFromYawPitchRoll([0.6,0,0.6], 1.5, 90, -30, 0, 2)
proj = p.computeProjectionMatrixFOV(60, 1.0, 0.01, 10)
_, _, rgb, depth, seg = p.getCameraImage(320, 320, view, proj)
depth_m = 1.0 / (depth * (1/0.01 - 1/10) + 1/10)     # 归一化深度 -> 米

# 接触信息（抓取判定）
for c in p.getContactPoints(bodyA=robot, bodyB=obj):
    c[8]                                     # 法向接触力

# 步进
for _ in range(10):
    p.stepSimulation()
```

常用函数：`getBasePositionAndOrientation` `resetBasePositionAndOrientation` `applyExternalForce` `changeDynamics`（摩擦/质量）`getAABB` `rayTest` `resetSimulation` `setRealTimeSimulation`。

优点：pip 就装，URDF 直接吃，`DIRECT` 模式可开几十个进程并行采数据。
缺点：接触解算不精确，不适合精细插拔任务。

### 2.2 MuJoCo —— 接触更准、GPU 并行

```bash
pip install mujoco        # 3.x，含 viewer；Linux 无头需 EGL/OSMesa
```

```python
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("panda_scene.xml")
data = mujoco.MjData(model)

# 模型查询（按名字拿 id，不要写死下标）
bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7")
sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint1")
print(model.jnt_qposadr[jid], model.jnt_dofadr[jid])   # qpos/qvel 起始下标

# 状态与控制
data.qpos[:] = model.key_qpos[0]      # 或 model.qpos0
data.ctrl[:] = 0.0                    # 执行器目标（位置/力矩由 actuator 类型决定）
mujoco.mj_forward(model, data)        # 只做运动学 + 传感器，不积分
mujoco.mj_step(model, data)           # 积一步（nsubsteps 由 model.opt.timestep 决定）

# 从状态直接算：速度、雅可比、动力学
mujoco.mj_kinematics(model, data)
mujoco.mj_comPos(model, data)
data.xpos[bid], data.site_xpos[sid]
jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
mujoco.mj_jacSite(model, data, jacp, jacr, sid)       # 末端位置/姿态雅可比
mujoco.mj_inverse(model, data)                        # 逆动力学 -> data.qfrc_inverse

# 传感器与接触
data.sensordata      # 对应 model.sensor_* 定义
for i in range(data.ncon):
    c = data.contact[i]
    c.dist, c.pos, c.geom1, c.geom2

# 可视化
import mujoco.viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        import time; time.sleep(model.opt.timestep)
```

进阶：

- `mujoco.mjx` —— MuJoCo 的 JAX 后端，能在 GPU 上并行几千个环境，做大规模 RL 采样。
- `dm_control` —— DeepMind 的环境封装（`suite.load`、`Environment`），很多经典任务基于它。
- MJCF 里 `option.timestep` 是物理步长，控制频率靠 `frame_skip` / 外层循环控制。

### 2.3 SAPIEN / Isaac Lab —— GPU 仿真

```bash
pip install sapien                  # 需要 Vulkan
# Isaac Lab 不走 pip，按官方脚本装 Isaac Sim + isaaclab 扩展
```

```python
import sapien
scene = sapien.Scene()
scene.add_ground(0)
loader = scene.create_urdf_loader()
robot = loader.load("panda.urdf")
robot.set_qpos([0.0] * 7)
for _ in range(100):
    scene.step()
```

Isaac Lab 走"脚本 + 配置类"风格，核心是 `RLEnv` / `ManagerBasedRLEnv`：

```python
# python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Lift-Franka-v0 --headless
from isaaclab.envs import ManagerBasedRLEnv
env = ManagerBasedRLEnv(cfg=MyEnvCfg())
obs, _ = env.reset()
obs, rew, terminated, truncated, info = env.step(actions)   # actions 形状 (num_envs, act_dim)
```

### 2.4 Gymnasium —— 统一环境接口

```bash
pip install gymnasium
```

```python
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit

env = gym.make("Reacher-v5", render_mode="rgb_array")
obs, info = env.reset(seed=0)          # 新 API 返回 (obs, info)，旧版只返回 obs
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
done = terminated or truncated          # 这两个必须分开处理，别合成一个 done

env.action_space, env.observation_space
env.observation_space.sample()
env = RecordEpisodeStatistics(env)      # info["episode"]["r"] 记录回报

# 向量化环境（并行采样，训练提速的关键）
venv = gym.make_vec("Reacher-v5", num_envs=8, vectorization_mode="async")
obs, info = venv.reset()
obs, rew, term, trunc, info = venv.step(venv.action_space.sample())

# 自定义环境：继承 gym.Env，实现 step / reset / render
class MyEnv(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (10,), np.float32)
    action_space = gym.spaces.Box(-1, 1, (7,), np.float32)
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(10, np.float32), {}
    def step(self, action):
        return np.zeros(10, np.float32), 0.0, False, False, {}
```

---

## 3. 运动学与动力学

### 3.1 Pinocchio —— 真机动力学首选

```bash
pip install pin          # 注意包名：pin，或 conda install -c conda-forge pinocchio
```

```python
import pinocchio as pin
import numpy as np

model = pin.buildModelFromUrdf("panda.urdf")
data = model.createData()
q = pin.neutral(model)
v = np.zeros(model.nv)

# 正运动学
pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)
fid = model.getFrameId("ee_link")
data.oMf[fid].translation          # 末端位置（世界系）
data.oMf[fid].rotation

# 雅可比
pin.computeJointJacobians(model, data, q)
J = pin.getFrameJacobian(model, data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

# 逆动力学 tau = M(q)qdd + C(q,qd)qd + g(q)
tau = pin.rnea(model, data, q, v, np.zeros(model.nv))
M = pin.crba(model, data, q)                  # 惯性矩阵
pin.computeGeneralizedGravity(model, data, q) # 只算重力项（前馈补偿常用）
pin.nonLinearEffects(model, data, q, v)       # C*v + g

# 逆运动学（数值解，带约束）
q_sol = pin.ik.ik(model, data, q_init, pin.SE3(np.eye(3), np.array([0.5, 0, 0.5])),
                  frame_id=fid, max_iter=100, eps=1e-4)

# 数值积分（仿真真机模型）
q_next = pin.integrate(model, q, v * dt)
v_err = pin.difference(model, q, q_next)      # 流形上的差，配置空间不能直接相减
```

要点：`qpos` 维度 `nq` 与速度维度 `nv` 在球关节/自由关节下**不相等**，永远用 `model.nq` / `model.nv`。

### 3.2 Robotics Toolbox / ikpy / Drake

```bash
pip install roboticstoolbox-python ikpy
```

```python
import roboticstoolbox as rtb
panda = rtb.models.Panda()
T = panda.fkine(panda.qz)                       # SE3 位姿
sol = panda.ikine_LM(T)                         # 数值 IK，返回 .q / .success
panda.plot(panda.qz)                            # Swift 可视化
panda.jacob0(panda.qz)
panda.jacob_dot(...)                            # 雅可比导数（奇异分析）
panda.accel(...); panda.rne(...)                # 加速度、逆动力学

import numpy as np
from ikpy.chain import Chain
chain = Chain.from_urdf_file("robot.urdf", base_elements=["base_link"],
                             active_links_mask=[False]+[True]*6+[False])
q = chain.inverse_kinematics([0.3, 0.0, 0.4])    # 只给位置
q2 = chain.inverse_kinematics(np.array([0.3, 0, 0.4]), orientation_mode="Z")
chain.forward_kinematics(q)
```

- `ikpy`：几行就能跑，但没考虑动力学，做实时控制不够。
- `drake`：`pydrake` 里有数学规划 IK（`InverseKinematics`）、轨迹优化、`MultibodyPlant`。学术上最完整，安装最重。
- 需要 GPU 批量 FK（比如学习模型里做可微运动学）：`pytorch-kinematics` 或自己写 `torch` 版齐次矩阵乘法。

---

## 4. 操作任务 Benchmark

### 4.1 robosuite

```bash
pip install robosuite
```

```python
import numpy as np
import robosuite as suite

env = suite.make(
    "Lift", robots="Panda",
    has_renderer=True, use_camera_obs=False,
    control_freq=20,                     # 控制频率 ≠ 仿真频率
    horizon=500,
)
obs = env.reset()                        # dict：robot0_eef_pos、robot0_gripper_qpos...
low, high = env.action_spec              # 逐维动作上下限
obs, reward, done, info = env.step(np.zeros(env.action_dim))
env.render()
env.close()
```

`env.step` 的动作通常是末端 6 维增量 + 夹爪 1 维，顺序写错是新手最常见的 bug。

### 4.2 ManiSkill / LIBERO / RLBench

```python
# ManiSkill 3：Gymnasium 接口，GPU 并行
import gymnasium as gym, mani_skill.envs
env = gym.make("PickCube-v1", num_envs=64, obs_mode="rgbd",
               control_mode="pd_joint_delta_pos", render_mode="rgb_array")
obs, _ = env.reset()

# LIBERO：语言条件下的长程操作，模仿学习评测标准
from libero.libero import benchmark
suite = benchmark.get_benchmark_dict()["libero_object"]()
task = suite.get_task(0)
```

| 库 | 任务类型 | 数据 | 典型用途 |
|---|---|---|---|
| robosuite | 单臂/双臂操作 | 需自己采 | 论文基线、demo 生成 |
| ManiSkill | 大规模操作 | 有示范 | GPU RL、点云策略 |
| LIBERO | 语言指令长程 | 有 | VLA 微调评测 |
| RLBench | 100+ 任务 | 有 | 少样本模仿学习 |

---

## 5. 策略学习：模仿学习 / RL / VLA

### 5.1 PyTorch 训练骨架

```python
import torch, torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
net = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU(),
                    nn.Linear(256, 256), nn.ReLU(),
                    nn.Linear(256, act_dim)).to(device)
opt = torch.optim.AdamW(net.parameters(), lr=3e-4)

for epoch in range(epochs):
    for obs, act in loader:
        pred = net(obs.to(device))
        loss = nn.functional.mse_loss(pred, act.to(device))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)   # 策略网络必备
        opt.step()

# 部署时一定要关梯度、切 eval
net.eval()
with torch.no_grad():
    action = net(obs)
```

要点：

- 动作归一化（用数据集统计量做 `mean/std`）几乎决定 BC 能不能收敛。
- 观测里混了不同量纲（关节角 + 图像 + 力）时，分头编码再 concat，别直接拼。
- `torch.compile(net)` 在控制回路里能省可观时延，但首次编译慢，别放在控制线程里。

### 5.2 LeRobot —— 模仿学习 / VLA 全家桶

```bash
pip install lerobot
```

```python
# 1) 数据集：统一格式存 episode（观测 + 动作 + 时间戳）
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("lerobot/aloha_mobile_cabinet")      # 或本地 repo_id
sample = ds[0]                                            # dict[str, Tensor]
sample.keys()
ds.meta.fps, ds.num_episodes, ds.num_frames

# 2) 写自己的数据集（采集真机数据）
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset.create(repo_id="user/my_dataset", fps=30,
                           features=features, root="./data")
ds.add_frame({"observation.state": q, "action": a, "task": "pick cube"})
ds.save_episode()
ds.finalize()

# 3) 训练（CLI 比 API 稳定，策略名 ACT / diffusion / pi0 / smolvla）
# lerobot-train --policy.type=act --dataset.repo_id=user/my_dataset --output_dir=./out

# 4) 推理：策略是一个 nn.Module，输入当前观测，输出动作块
from lerobot.policies.act.modeling_act import ACTPolicy
policy = ACTPolicy.from_pretrained("user/act_pick_cube")
policy.eval()
action = policy.select_action(batch)          # 内部维护动作队列，按控制频率调用
policy.reset()                                # 每个 episode 开始必须 reset
```

概念对应关系（看懂这四条，LeRobot 就通了）：

- `observation.*`：策略输入，键名固定，图像用 `observation.images.<cam>`。
- `action`：策略输出，和 `observation.state` 同维度（关节空间）。
- **action chunk**：一次预测未来 N 步动作，缓解推理时延（ACT / Diffusion Policy 的核心）。
- `select_action` 内部缓存 chunk，不用自己切片。

### 5.3 Diffusion Policy / ACT / VLA

```python
# 扩散策略：用 diffusers 的调度器做动作去噪
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
noise_scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2")
noise = torch.randn_like(action)
t = torch.randint(0, 100, (B,))
noisy = noise_scheduler.add_noise(action, noise, t)
loss = nn.functional.mse_loss(model(noisy, t, obs_cond), noise)

# VLA：把"图像 + 语言 + 本体状态"喂进多模态模型，输出动作 token
from transformers import AutoProcessor, AutoModelForVision2Seq
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b",
                                               torch_dtype=torch.bfloat16,
                                               trust_remote_code=True).to("cuda")
inputs = processor("pick up the red cube", image).to("cuda", dtype=torch.bfloat16)
action = model.predict_action(**inputs, unnorm_key="bridge_orig")   # (7,) 连续动作
```

VLA 现实提醒：7B 模型在单卡消费级 GPU 上推理 100~300ms，直接做闭环控制不够快，通常配 action chunk 或蒸馏成小模型（`smolvla` 之类）再上真机。

### 5.4 RL 算法库

```bash
pip install stable-baselines3 sb3-contrib
```

```python
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env

venv = make_vec_env("PandaReach-v3", n_envs=8)
model = PPO("MultiInputPolicy", venv, n_steps=256, batch_size=256,
            tensorboard_log="./tb/", seed=0)
model.learn(total_timesteps=1_000_000)
model.save("ppo_reach")
obs, _ = venv.reset()
action, _ = model.predict(obs, deterministic=True)    # 部署用 deterministic=True
```

其他：`cleanrl`（单文件可读实现）、`tianshou`、JAX 系 `brax` / `jaxrl`（配合 mjx 上千并行环境）。

---

## 6. 感知

### 6.1 OpenCV —— 标定与基础视觉

```bash
pip install opencv-python
```

```python
import cv2, numpy as np

img = cv2.imread("a.png")                        # BGR！
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)       # 喂给网络前必须转
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 相机标定
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
undist = cv2.undistort(img, K, dist)

# PnP：已知 3D 点 + 像素点 -> 位姿（抓取位姿估计常用）
ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist, flags=cv2.SOLVEPNP_IPPE)
R, _ = cv2.Rodrigues(rvec)                       # 罗德里格斯 -> 旋转矩阵

# 手眼标定（eye-in-hand）：解 AX = XB
R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
    R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, method=cv2.CALIB_HAND_EYE_TSAI)

# 简单目标定位（颜色阈值 + 轮廓）
mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), (0,100,100), (10,255,255))
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
cx, cy = x + w/2, y + h/2
```

### 6.2 Open3D —— 点云

```bash
pip install open3d
```

```python
import open3d as o3d, numpy as np

pcd = o3d.io.read_point_cloud("scene.pcd")
pcd = pcd.voxel_down_sample(0.005)                # 降采样（先做，后面都便宜）
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(0.01, 30))

# 平面分割（抠桌面）
plane, inliers = pcd.segment_plane(distance_threshold=0.008, ransac_n=3, num_iterations=1000)
table = pcd.select_by_index(inliers)
objs = pcd.select_by_index(inliers, invert=True)

# 聚类找物体
labels = np.array(objs.cluster_dbscan(eps=0.02, min_points=10))

# 从相机内外参造点云（深度图 -> 世界系点云）
o3d.geometry.PointCloud.create_from_depth_image(
    o3d.geometry.Image(depth_mm), intrinsics, extrinsic=np.linalg.inv(T_world_cam))

# 配准：粗（FPFH+RANSAC）+ 精（ICP）
result = o3d.pipelines.registration.registration_icp(
    src, dst, 0.02, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPlane())
result.transformation
```

### 6.3 检测 / 分割

```python
# YOLO：检测抓取目标
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
res = model("scene.jpg")[0]
res.boxes.xyxy, res.boxes.conf, res.boxes.cls
res.masks                                # 分割模型才有

# SAM / SAM2：给点或框，出掩码（做通用抓取分割）
from segment_anything import sam_model_registry, SamPredictor
predictor = SamPredictor(sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth"))
predictor.set_image(rgb)
masks, scores, _ = predictor.predict(point_coords=np.array([[cx, cy]]),
                                     point_labels=np.array([1]), multimask_output=True)
```

`pytorch3d`：可微渲染、点云/网格算子，用于 6D 位姿和神经重建（安装麻烦，注意 CUDA 版本）。

---

## 7. ROS 2（部署中间件）

### 7.1 节点骨架

```bash
# 不是 pip 装，走系统源：apt install ros-jazzy-desktop
```

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray

class Controller(Node):
    def __init__(self):
        super().__init__("controller")
        self.declare_parameter("control_hz", 100.0)
        hz = self.get_parameter("control_hz").value

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)  # 传感器用 BEST_EFFORT
        self.pub = self.create_publisher(Float64MultiArray, "/arm/cmd", 10)
        self.sub = self.create_subscription(JointState, "/joint_states", self.on_state, qos)
        self.timer = self.create_timer(1.0 / hz, self.on_timer)
        self.state = None

    def on_state(self, msg):
        self.state = msg                                   # 只存不处理，回调要短

    def on_timer(self):
        if self.state is None:
            return
        cmd = Float64MultiArray()
        cmd.data = [0.0] * len(self.state.position)
        self.pub.publish(cmd)

def main():
    rclpy.init()
    node = Controller()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

# launch：ros2 launch my_pkg bringup.launch.py
# 调试：ros2 topic hz /joint_states ; ros2 topic echo /arm/cmd
```

### 7.2 关键概念 ↔ 机器人

| ROS 2 概念 | 机器人里的用途 |
|---|---|
| `JointState` | 关节角/速度/力矩，控制回路的主输入 |
| `PoseStamped` / `TF2` | 坐标系变换，`tf2_ros.Buffer.lookup_transform("base", "ee", rclpy.time.Time())` |
| QoS `BEST_EFFORT` | 高频传感器/图像，丢帧无所谓，别堵 |
| QoS `RELIABLE` | 指令类话题，不能丢 |
| `ros2_control` | `SystemInterface` 子类实现 `read()`/`write()`，把 SDK 接进标准控制环 |
| `ros2 bag` | 录数据回放，采模仿学习数据常用 |

`ros2_control` 硬件接口（真机部署的标准姿势）：

```python
from hardware_interface import SystemInterface, return_type, CallbackReturn
class MyArm(SystemInterface):
    def on_init(self): ...        # 打开串口/CAN
    def export_state_interfaces(self): ...
    def read(self, time, period): # 从 SDK 读关节状态写入接口
    def write(self, time, period):# 把命令下发给 SDK
```

---

## 8. 真机 SDK

```python
# Dynamixel 总线舵机
from dynamixel_sdk import PortHandler, PacketHandler
port = PortHandler("COM3"); port.openPort(); port.setBaudRate(1_000_000)
pkt = PacketHandler(2.0)
pkt.write4ByteTxRx(port, dxl_id, ADDR_GOAL_POSITION, ticks)   # ticks = rad -> 0~4095 标定
pkt.read4ByteTxRx(port, dxl_id, ADDR_PRESENT_POSITION)

# 瑞尔/宇树/飞特总线舵机（SO-ARM100 之类）
from feetech_servo_sdk import PortHandler, PacketHandler, GroupSyncWrite

# xArm / UFACTORY
from xarm.wrapper import XArmAPI
arm = XArmAPI("192.168.1.100")
arm.motion_enable(True); arm.set_mode(0); arm.set_state(0)
arm.set_position(x=300, y=0, z=200, roll=180, pitch=0, yaw=0, speed=100, wait=True)
arm.get_position()                     # (code, [x,y,z,roll,pitch,yaw])
arm.set_servo_angle(angle=[0]*6, speed=0.5, wait=True)
arm.emergency_stop()

# myCobot / myArm（M5Stack）
from pymycobot.mycobot import MyCobot
mc = MyCobot("COM3", 115200)
mc.send_angles([0, 0, 0, 0, 0, 0], 50)  # 速度 0~100
mc.get_coords(); mc.get_angles()
```

真机铁律（这些不是代码洁癖，是安全问题）：

1. 任何命令下发前做**关节限位裁剪**，舵机堵转烧的是钱。
2. 位置模式下限制单步最大增量（`np.clip(q_cmd, q_prev - max_step, q_prev + max_step)`）。
3. 必须有物理急停或独立于软件的断电开关。
4. 先空跑（不装夹爪、抬高 Z）验证轨迹，再上物体。
5. 每个舵机的零点和方向做一次标定表，别信 `rad → tick` 是简单线性且同向。

---

## 9. 加速与部署

```python
# ONNX 导出 + 推理（跨平台部署最省事）
torch.onnx.export(policy, dummy_obs, "policy.onnx", opset_version=17,
                  input_names=["obs"], output_names=["action"], dynamic_axes={"obs": {0: "batch"}})

import onnxruntime as ort
sess = ort.InferenceSession("policy.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
action = sess.run(None, {"obs": obs_np.astype(np.float32)})[0]

# 半精度 + 不建图
with torch.inference_mode():
    action = policy(obs.half().cuda())

# TensorRT（NVIDIA 上最快，但和模型结构耦合，改网络要重编）
# trtexec --onnx=policy.onnx --saveEngine=policy.plan --fp16
```

时延预算参考（真机闭环 30Hz 意味着整条链路 <33ms）：

| 环节 | 典型耗时 |
|---|---|
| 相机采图 + 传输 | 5~30 ms |
| 图像预处理 + resize | 2~5 ms |
| 小策略网络（<10M） | 1~5 ms |
| VLA 7B 单卡 | 100~300 ms → 必须配 action chunk |
| 串口/CAN 下发 | 1~5 ms |

控制回路与推理回路**分线程**：推理线程算 action chunk，控制线程按固定周期从队列取，避免抖动。

---

## 10. 常见坑速查

| 症状 | 原因 |
|---|---|
| 机械臂动作方向怪、绕圈子 | 四元数 xyzw / wxyz 搞反 |
| 仿真里正常，真机反了 | 关节零点 / 转向符号 / 单位（rad vs deg）不一致 |
| IK 有解但会瞬移 | 没做关节限位和相邻解连续性检查（`wrapToPi`） |
| 策略训练 loss 降但真机不动 | 动作没做归一化/反归一化，或训练用归一化动作、推理忘了反归一化 |
| `env.step` 后 episode 不结束 | `terminated` 和 `truncated` 只看了一个 |
| 仿真比实时快/慢，控制不跟随 | 物理步长、控制频率、`frame_skip` 三者没对齐 |
| MuJoCo 拿到错的 qpos | 用了关节声明顺序下标，没用 `model.jnt_qposadr` |
| 点云和图像对不上 | 手眼外参没标定，或内参 resize 后没同步缩放 |
| 多进程采数据随机性不可复现 | 每个子进程要单独 `seed`（`env.reset(seed=base+i)`） |
| 真机跑久了漂移 | 电机零点温漂 → 定期回零 / 用 ROS 里程计校准 |

---

## 11. 一条最小可跑路线

想一周内从零到一个能动的 demo：

1. `gymnasium` + `pybullet` 起一个 `Panda` 环境，脚本里手写 IK 让它到固定点。
2. 加 `RecordEpisodeStatistics`，用键盘/脚本采集 30 条演示轨迹存成 `.npz`。
3. 用 `torch` 写一个最小 BC（MLP → 关节增量），训练到能复现轨迹。
4. 换成 `mujoco` 验证接触更准的场景；或换 `robosuite` 复现论文基线。
5. 数据规模上来后切 `LeRobot` 的 `LeRobotDataset` + ACT，享受现成的 action chunk。
6. 部署：`onnxruntime` 导出 + 串口/CAN 下发，先空跑再上物体。

---

参考自检脚本：`check_math_snippets.py`（只依赖 numpy/scipy，验证本文档第 1 节的旋转/插值/流形写法）。
