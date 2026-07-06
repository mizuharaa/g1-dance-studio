import sys
sys.path.insert(0, "/workspace/notebook-data/cloud")
import torch
import mjlab.tasks  # noqa
import sim2real_task_v3 as v3
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg

MOTION = "/workspace/notebook-data/motions/thriller_deploy.npz"

# --- cfg-level checks -----------------------------------------------------
stock = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1", play=False)
a = load_env_cfg(v3.TASK_V3A, play=False)
b = load_env_cfg(v3.TASK_V3B, play=False)
c = load_env_cfg(v3.TASK_V3C, play=False)
ge = load_env_cfg(v3.TASK_V3B_GAPEVAL, play=False)
bp = load_env_cfg(v3.TASK_V3B, play=True)

def arm_stiff(cfg):
    out = {}
    for act in cfg.scene.entities["robot"].articulation.actuators:
        names = tuple(getattr(act, "target_names_expr", ()) or ())
        if names and all(any(p in n for p in ("shoulder","elbow","wrist")) for n in names):
            out[names[0]] = (float(act.stiffness), float(act.damping))
    return out

s_stock, s_a, s_b, s_ge, s_bp = map(arm_stiff, (stock, a, b, ge, bp))
print("stock arm gains:", s_stock)
print("v3a   arm gains:", s_a)
print("v3b   arm gains:", s_b)
print("gapev arm gains:", s_ge)
print("v3b-play gains :", s_bp)
for k in s_stock:
    assert abs(s_a[k][0] - s_stock[k][0]) < 1e-9, "v3a must NOT scale arm gains"
    assert abs(s_b[k][0] - 2.5*s_stock[k][0]) < 1e-6, "v3b stiffness x2.5"
    assert abs(s_b[k][1] - 2.5*s_stock[k][1]) < 1e-6, "v3b damping x2.5"
    assert abs(s_ge[k][0] - 2.5*s_stock[k][0]) < 1e-6, "gapeval stiffness x2.5"
    assert abs(s_bp[k][0] - 2.5*s_stock[k][0]) < 1e-6, "v3b play stiffness x2.5"
print("ARM GAIN CHECKS OK (stock untouched, v3b/gapeval/play x2.5)")

for cfg, name in ((a,"v3a"),(b,"v3b"),(c,"v3c")):
    assert cfg.rewards["action_rate_l2"].weight == -0.1, name
    assert cfg.rewards["motion_body_pos"].weight == 1.5, name
    assert cfg.rewards["motion_body_ori"].weight == 1.5, name
    assert cfg.rewards["ankle_torque_l2"].weight == -4e-4, name
    assert cfg.rewards["joint_torques_l2"].weight == -2e-5, name
print("REWARD CHECKS OK (action_rate -0.1, body_pos/ori 1.5, torque penalties kept)")

# v2 recipe DR still present on v3a/b/c; friction split only on v3b
for cfg, name in ((a,"v3a"),(b,"v3b"),(c,"v3c")):
    for ev in ("dr_pd_gains","dr_effort_limits","dr_joint_armature","dr_ankle_zero_offset",
               "dr_torso_mass","dr_hand_payload"):
        assert ev in cfg.events, (name, ev)
assert "dr_joint_friction" in a.events and "dr_joint_friction" in c.events
assert "dr_joint_friction" not in b.events
assert b.events["dr_joint_friction_arms"].params["ranges"] == (0.1, 0.6)
assert b.events["dr_joint_friction_body"].params["ranges"] == (0.0, 0.4)
# disjointness of the two friction terms' name patterns
arm_exprs = b.events["dr_joint_friction_arms"].params["asset_cfg"].joint_names
body_exprs = b.events["dr_joint_friction_body"].params["asset_cfg"].joint_names
print("friction arms:", arm_exprs, " body:", body_exprs)
print("DR CHECKS OK")

# stock rewards untouched (no cross-contamination)
assert stock.rewards["action_rate_l2"].weight == -0.1
assert stock.rewards["motion_body_pos"].weight == 1.0
assert "ankle_torque_l2" not in stock.rewards
print("STOCK CONTAMINATION CHECK OK")

# --- live 30-step env runs ------------------------------------------------
for task in (v3.TASK_V3A, v3.TASK_V3B, v3.TASK_V3B_GAPEVAL):
    cfg = load_env_cfg(task, play=False)
    cfg.scene.num_envs = 8
    cfg.commands["motion"].motion_file = MOTION
    agent = load_rl_cfg(task)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    env = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    obs = env.get_observations()
    tens = obs if torch.is_tensor(obs) else obs["actor"]
    shape = tuple(tens.shape)
    assert shape[-1] == 160, f"{task}: obs dim {shape}"
    for i in range(30):
        act = torch.zeros(8, 29, device="cuda:0")
        obs, _, _, _ = env.step(act)
    env.close()
    print(f"{task}: obs {shape} + 30 steps OK")
print("SMOKE ALL OK")
