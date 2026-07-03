"""Gantry-scope authorization + deploy-path integration-seam fixes (robot-day).

Covers the production-audit deploy blockers and the gantry-only path:
  - mjlab_verify records BOTH motion_sha256 (deployable csv) and motion_npz_sha256.
  - gen_config consumes a real heldout verdict (glob + no duration_s KeyError).
  - --gantry authorizes a signed sub-99% (fail) policy; ground/show still requires pass.
"""
from pathlib import Path

import pipeline.mjlab_verify as mv
from deploy import gen_config as gc


def _eval(n_success, n=128):
    rate = n_success / n
    cond = lambda: {"num_episodes": n, "n_success": n_success, "success_rate": rate,
                    "mpkpe_m": 0.17, "ee_pos_error_m": 0.1, "seed": 7}
    return {"dance": "t", "conditions": {"nominal": cond(), "push": cond()}}


def _artifacts(tmp_path, n_success):
    pol = tmp_path / "policy.onnx"; pol.write_bytes(b"onnx-bytes")
    csv = tmp_path / "motion_deploy.csv"; csv.write_text("h\n" + "0,1,2\n" * 60)  # 60 rows
    npz = tmp_path / "motion.npz"; npz.write_bytes(b"npz-bytes-differ")
    v = mv.build_verdict(_eval(n_success), pol, csv, eval_motion_path=npz)
    return pol, csv, npz, v


def test_records_both_digests(tmp_path):
    pol, csv, npz, v = _artifacts(tmp_path, 128)
    assert v["motion_sha256"] == gc.full_sha256(csv)          # deployable identity
    assert v["motion_npz_sha256"] == gc.full_sha256(npz)      # provenance
    assert v["motion_sha256"] != v["motion_npz_sha256"]       # genuinely different files


def test_gantry_accepts_signed_sub99(tmp_path):
    pol, csv, npz, v = _artifacts(tmp_path, 126)              # 98.4% -> fail
    (tmp_path / "heldout_verdict.json").write_text(__import__("json").dumps(v))
    p_sha, m_sha = gc.full_sha256(pol), gc.full_sha256(csv)
    # gantry: a signed, bound, sub-99% verdict is sufficient
    assert gc.find_gantry_verdict(p_sha, m_sha, tmp_path) is not None
    # ground/show: the SAME verdict must NOT authorize (not >=99%)
    assert gc.find_passing_exam(p_sha, m_sha, tmp_path) is None


def test_full_accepts_pass_only(tmp_path):
    pol, csv, npz, v = _artifacts(tmp_path, 128)              # 100% -> pass
    (tmp_path / "heldout_verdict.json").write_text(__import__("json").dumps(v))
    p_sha, m_sha = gc.full_sha256(pol), gc.full_sha256(csv)
    assert gc.find_passing_exam(p_sha, m_sha, tmp_path) is not None
    assert gc.find_gantry_verdict(p_sha, m_sha, tmp_path) is not None


def test_gantry_rejects_wrong_sha_and_unsigned(tmp_path):
    pol, csv, npz, v = _artifacts(tmp_path, 126)
    (tmp_path / "heldout_verdict.json").write_text(__import__("json").dumps(v))
    p_sha, m_sha = gc.full_sha256(pol), gc.full_sha256(csv)
    assert gc.find_gantry_verdict("0" * 64, m_sha, tmp_path) is None   # wrong policy
    assert gc.find_gantry_verdict(p_sha, "0" * 64, tmp_path) is None   # wrong motion
    tampered = dict(v); tampered["signature"] = "deadbeef"
    (tmp_path / "bad_verdict.json").write_text(__import__("json").dumps(tampered))
    (tmp_path / "heldout_verdict.json").unlink()
    assert gc.find_gantry_verdict(p_sha, m_sha, tmp_path) is None      # bad signature


def test_duration_from_motion_not_verdict(tmp_path):
    csv = tmp_path / "m.csv"; csv.write_text("h\n" + "x\n" * 30)   # 30 rows @30fps = 1.0s
    assert gc.motion_duration_s(csv) == 1.0
