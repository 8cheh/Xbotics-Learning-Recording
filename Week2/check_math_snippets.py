"""Self-check for the math/geometry snippets in python-robotics-embodied-ai.md §1.

Only depends on numpy + scipy. Run: python check_math_snippets.py
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

# --- scipy Rotation conventions -------------------------------------------------
r = R.from_euler("xyz", [0, 0, 90], degrees=True)
assert np.allclose(r.as_matrix(), [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)
assert np.allclose(r.apply([1, 0, 0]), [0, 1, 0], atol=1e-12)  # 绕 z 转 90°: x -> y

q = r.as_quat()
assert len(q) == 4  # scipy quat is [x, y, z, w]
assert np.isclose(q[3], np.cos(np.pi / 4))  # w = cos(theta/2) for a 90 deg rotation
assert np.allclose(R.from_quat(q).as_matrix(), r.as_matrix(), atol=1e-12)

# round trip through every representation
assert np.allclose(R.from_matrix(r.as_matrix()).as_quat(), q, atol=1e-12)
assert np.allclose(R.from_rotvec(r.as_rotvec()).as_matrix(), r.as_matrix(), atol=1e-12)

# composition order: rotate z90 then x30 (left-multiply), NOT the other way round
a = R.from_euler("z", 90, degrees=True)
b = R.from_euler("x", 30, degrees=True)
assert np.allclose((a * b).as_matrix(), a.as_matrix() @ b.as_matrix(), atol=1e-12)
assert not np.allclose((a * b).as_matrix(), (b * a).as_matrix(), atol=1e-12)

# inverse undoes the forward rotation
assert np.allclose(r.inv().apply(r.apply([0.3, -0.2, 0.5])), [0.3, -0.2, 0.5], atol=1e-12)

# batch apply keeps leading dims
assert r.apply(np.zeros((5, 3))).shape == (5, 3)  # batch (N,3) in -> (N,3) out

# --- SLERP ----------------------------------------------------------------------
# single-axis from_euler needs angles shaped (N, 1)
rots = R.from_euler("z", [[0], [90]], degrees=True)
assert rots.as_euler("zyx", degrees=True)[1][0] == 90.0
slerp = Slerp([0, 1], rots)
assert np.isclose(slerp(0.5).as_euler("zyx", degrees=True)[0], 45.0, atol=1e-9)
assert np.isclose(slerp(0.0).as_euler("zyx", degrees=True)[0], 0.0, atol=1e-9)
assert np.isclose(slerp(1.0).as_euler("zyx", degrees=True)[0], 90.0, atol=1e-9)
assert np.isclose(slerp(0.25).as_euler("zyx", degrees=True)[0], 22.5, atol=1e-9)

# --- numpy control-path primitives ---------------------------------------------
q_lim = np.clip(np.array([5.0, -5.0, 0.5]), -2.9, 2.9)
assert np.allclose(q_lim, [2.9, -2.9, 0.5])

J = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
assert np.allclose(np.linalg.pinv(J) @ np.array([1.0, 2.0, 3.0]), [1.0, 2.0])

# pinocchio-style integration on a manifold: q_next = q + v*dt is only valid for
# tiny dt / small angles; here we just check the linearization matches at dt -> 0
q0 = R.from_euler("z", 0.0)
dq = np.array([0.0, 0.0, 1.0])  # angular velocity about z
dt = 1e-6
assert np.isclose(R.from_rotvec(dq * dt).as_euler("zyx")[0], dq[2] * dt, atol=1e-12)

# radians/degrees and resampling
assert np.isclose(np.rad2deg(np.deg2rad(37.0)), 37.0)
assert np.isclose(np.interp(0.5, [0, 1], [0, 10]), 5.0)

# seeded RNG is reproducible
assert np.allclose(np.random.default_rng(0).normal(size=3),
                   np.random.default_rng(0).normal(size=3))

print("check_math_snippets: all assertions passed")
