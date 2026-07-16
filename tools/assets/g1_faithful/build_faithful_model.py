"""Build the faithful mjlab-aligned G1 preview model (Agent E).

Source geometry/inertia: official Unitree G1 MJCF (g1_29dof.xml) — the same model
pipeline/g1_limits.py treats as hardware truth (correct inertias + LAFAN1 joint order
+ foot contact spheres). We then patch the per-joint ARMATURE to the mjlab/BeyondMimic
values (from g1_limits.ARMATURE) so the joint-space inertia matches the TRAINING model,
and zero the XML joint damping / frictionloss (mjlab impedance-actuator convention: the
deploy PD kd supplies damping; joint friction lives in the actuator model, not the XML).
A floor + light + skybox are added so it is a self-contained standable scene.

Output: tools/assets/g1_faithful/g1_mjlab_faithful.xml (meshdir -> absolute meshes dir).
"""
import os
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco

ROOT = Path("/home/alois/g1-dance")
import sys
sys.path.insert(0, str(ROOT))
from pipeline import g1_limits as L

SRC = ROOT / "third_party/unitree_mujoco/unitree_robots/g1/g1_29dof.xml"
MESHDIR = (ROOT / "third_party/unitree_mujoco/unitree_robots/g1/meshes").resolve()
OUTDIR = ROOT / "tools/assets/g1_faithful"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "g1_mjlab_faithful.xml"

spec = mujoco.MjSpec.from_file(str(SRC))
spec.meshdir = str(MESHDIR)          # absolute so the saved XML resolves meshes anywhere
spec.modelname = "g1_29dof_mjlab_faithful"

# --- patch per-joint armature to mjlab values; zero XML damping/frictionloss ---
arm_by_name = dict(zip(L.JOINT_ORDER, L.ARMATURE))
patched = 0
for j in spec.joints:
    if j.name in arm_by_name:
        j.armature = float(arm_by_name[j.name])
        j.damping = [0.0, 0.0, 0.0]
        j.frictionloss = 0.0
        patched += 1
print(f"patched {patched}/29 joints (armature->mjlab, damping/frictionloss->0)")

# --- add floor + lighting + skybox so it is a self-contained standable scene ---
spec.add_texture(name="skybox", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                 builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                 rgb1=[0.3, 0.5, 0.7], rgb2=[0, 0, 0], width=512, height=3072)
spec.add_texture(name="groundplane", type=mujoco.mjtTexture.mjTEXTURE_2D,
                 builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, mark=mujoco.mjtMark.mjMARK_EDGE,
                 rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.2, 0.3], markrgb=[0.8, 0.8, 0.8],
                 width=300, height=300)
mat = spec.add_material(name="groundplane", texrepeat=[5, 5], texuniform=True, reflectance=0.2)
mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "groundplane"

light = spec.worldbody.add_light()
light.pos = [0, 0, 1.5]; light.dir = [0, 0, -1]
light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
floor = spec.worldbody.add_geom()
floor.name = "floor"; floor.type = mujoco.mjtGeom.mjGEOM_PLANE
floor.size = [0, 0, 0.05]; floor.material = "groundplane"

model = spec.compile()
print("compiled OK: nq", model.nq, "nv", model.nv, "nu", model.nu, "njnt", model.njnt)
OUT.write_text(spec.to_xml())
print("wrote", OUT)

# sanity: verify patched armatures survive a reload
m2 = mujoco.MjModel.from_xml_path(str(OUT))
for name in ("left_ankle_pitch_joint", "left_knee_joint", "left_shoulder_pitch_joint"):
    jid = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_JOINT, name)
    dof = m2.jnt_dofadr[jid]
    print(f"  {name}: armature={m2.dof_armature[dof]:.6f} damping={m2.dof_damping[dof]:.4f}")
