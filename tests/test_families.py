"""Every model family of the journal pipeline builds, runs and trains its intended parts."""
import copy, sys
from pathlib import Path

import pytest, torch, yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "pipeline"))
from storm_pipeline.data import SW_COLS, HSTEPS, SCALED  # noqa: E402
from storm_pipeline.families import SPEC, configure, build  # noqa: E402
from src.training.trainer import Trainer  # noqa: E402
from src.training.physics_loss import PhysicsInformedLoss, LossWeights  # noqa: E402

CONFERENCE = ["lstm", "transformer_matched", "tf_ffn_res", "tf_flux_only", "storm_bz", "storm_no_delay", "storm_no_gate",
              "storm_fixed_ma", "tf_res_fixed_ma"]
JOURNAL = ["transformer", "tf_ffn256", "mlp", "cnn", "storm_no_physics", "storm_cathode", "storm_cathode_spec",
           "storm_radiotrophic", "storm_lstm_backbone", "storm_magnetopause",
           "lstm_res", "storm_lstm_no_gate", "storm_lstm_no_delay", "storm_lstm_no_physics",
           "storm_physfix", "storm_lstm_physfix"]


def make(fam, tmp_path, scaler=None):
    cfg = yaml.safe_load(open(ROOT / "configs/config.yaml"))
    cfg["training"]["checkpoint_dir"] = str(tmp_path / "ck"); cfg["training"]["log_dir"] = str(tmp_path / "tb")
    tr = Trainer(configure(copy.deepcopy(cfg), fam)); m = build(fam, tr, len(SW_COLS), "cpu", scaler=scaler); tr.writer.close()
    return m


def test_registry_lists_exactly_the_planned_families():
    assert sorted(SPEC) == sorted(CONFERENCE + JOURNAL)


@pytest.mark.parametrize("fam", CONFERENCE + JOURNAL)
def test_family_forward_shape(fam, tmp_path):
    torch.manual_seed(0)
    m = make(fam, tmp_path).eval()
    B, H = 3, len(HSTEPS)
    with torch.no_grad():
        o = m(x_sw=torch.randn(B, 72, len(SW_COLS)), x_flux=torch.randn(B, 72, 1), y_persist=torch.zeros(B, H))
    assert tuple(o["flux_pred"].shape) == (B, H), f"{fam}: got {tuple(o['flux_pred'].shape)}"
    assert torch.isfinite(o["flux_pred"]).all()


def test_conference_families_keep_their_parameter_names(tmp_path):
    """The registry must build the same modules as the AGERS 2026 code, so conference checkpoints still load."""
    cfg = yaml.safe_load(open(ROOT / "configs/config.yaml"))
    cfg["training"]["checkpoint_dir"] = str(tmp_path / "ck"); cfg["training"]["log_dir"] = str(tmp_path / "tb")
    for fam, mt, match in (("storm_bz", "storm_physnet", False), ("transformer_matched", "transformer", True)):
        c = copy.deepcopy(cfg); c["model_type"], c["ablation"], c["match_storm_capacity"] = mt, "none", match
        tr = Trainer(c); old = tr.build_model(len(SW_COLS)); tr.writer.close()
        new = make(fam, tmp_path)
        assert list(old.state_dict()) == list(new.state_dict()), fam
        assert [v.shape for v in old.state_dict().values()] == [v.shape for v in new.state_dict().values()], fam


def test_variants_differ_from_storm_bz_where_intended(tmp_path):
    base = make("storm_bz", tmp_path)
    assert type(make("storm_cathode", tmp_path).bz_gate).__name__ == "CathodeAnodeGate"
    assert type(make("storm_radiotrophic", tmp_path).bz_gate).__name__ == "RadiotrophicGate"
    assert make("storm_cathode_spec", tmp_path).spectral_head is not None and base.spectral_head is None
    assert make("storm_lstm_backbone", tmp_path).backbone == "lstm"
    assert make("storm_magnetopause", tmp_path).mag_geo is not None and base.mag_geo is None


def test_magnetopause_uses_physical_units(tmp_path):
    """With the block scaler, Shue (1998) r0 drops from ~10.4 Re at quiet Pdyn to ~8 Re at 10 nPa."""
    mu = torch.zeros(len(SCALED)); sd = torch.ones(len(SCALED))
    i_v, i_bz, i_p = SW_COLS.index("vsw"), SW_COLS.index("bz"), SW_COLS.index("pdyn")
    mu[i_v], sd[i_v] = 420.0, 94.0; mu[i_p], sd[i_p] = 1.9, 1.5; mu[i_bz], sd[i_bz] = 0.0, 3.0
    m = make("storm_magnetopause", tmp_path, scaler=(mu.numpy(), sd.numpy()))
    x = torch.zeros(2, 72, len(SW_COLS)); x[1, -1, i_p] = (10.0 - 1.9) / 1.5
    r0 = m.mag_geo(x)["r0"]
    assert r0[0].item() == pytest.approx(10.35, abs=0.05) and r0[1].item() == pytest.approx(8.03, abs=0.05)
    # the scaler is saved with the checkpoint
    sd_ = m.state_dict(); assert sd_["mag_geo.in_mu"][i_v].item() == 420.0


def test_spectral_head_is_trained_by_the_loss(tmp_path):
    m = make("storm_cathode_spec", tmp_path).train()
    fn = PhysicsInformedLoss(LossWeights())
    B, H = 8, len(HSTEPS)
    x = torch.randn(B, 72, len(SW_COLS))
    out = m(x_sw=x, x_flux=torch.randn(B, 72, 1), y_persist=torch.zeros(B, H))
    tg = {"y_flux": torch.randn(B, H), "y_dst": torch.zeros(B, H), "y_kp": torch.zeros(B, H), "y_storm": torch.zeros(B, 1),
          "storm_flag": torch.zeros(B, 1), "y_persist": torch.zeros(B, H)}
    total, comps = fn(out, tg, x)
    assert comps["spectral"] > 0
    total.backward()
    g = m.spectral_head.spectral_net[-1].weight.grad
    assert g is not None and g.abs().sum() > 0, "spectral head received no gradient"
    out2 = make("storm_bz", tmp_path)(x_sw=x, x_flux=torch.randn(B, 72, 1), y_persist=torch.zeros(B, H))
    assert fn(out2, tg, x)[1]["spectral"] == 0.0


@pytest.mark.parametrize("fam", ["storm_bz", "storm_lstm_backbone", "storm_magnetopause"])
def test_freeze_encoder_leaves_heads_trainable(fam, tmp_path):
    m = make(fam, tmp_path); m.freeze_encoder()
    frozen = {n for n, p in m.named_parameters() if not p.requires_grad}
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert any(n.startswith("heads.") for n in trainable)
    assert any(n.startswith(("transformer.", "lstm.", "ssm.", "itransformer.")) for n in frozen), fam
    assert not any(n.startswith(("transformer.", "lstm.", "ssm.", "itransformer.", "prop_delay.")) for n in trainable), fam


def test_lstm_attribution_variants(tmp_path):
    """Each LSTM-encoder ablation keeps the LSTM encoder and removes exactly one component."""
    full = make("storm_lstm_backbone", tmp_path)
    ng, nd = make("storm_lstm_no_gate", tmp_path), make("storm_lstm_no_delay", tmp_path)
    assert full.backbone == ng.backbone == nd.backbone == "lstm"
    assert (full.ablation, ng.ablation, nd.ablation) == ("none", "no_bz_gate", "no_delay")
    assert make("storm_lstm_no_physics", tmp_path).ablation == "no_physics"
    # the gate is bypassed in forward(): its parameters get no gradient
    x = torch.randn(4, 72, len(SW_COLS))
    ng(x_sw=x, x_flux=torch.randn(4, 72, 1), y_persist=torch.zeros(4, len(HSTEPS)))["flux_pred"].sum().backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in ng.bz_gate.parameters())
    r = make("lstm_res", tmp_path)
    assert type(r).__name__ == "ResLSTM" and r.residual
    o = r(x_sw=torch.zeros(2, 72, len(SW_COLS)), x_flux=torch.zeros(2, 72, 1), y_persist=torch.full((2, len(HSTEPS)), 3.0))
    assert o["flux_pred"].shape == (2, len(HSTEPS))


def test_physfix_physics_loss_sees_physical_units(tmp_path):
    """With the block scaler the Bz-response term switches on for Bz < -10 nT over > 6 h; without it (the
    original runs) it never does, because the standardised Bz never falls below -10."""
    import numpy as np
    from storm_pipeline.families import configure
    i_bz, i_dur, i_v = SW_COLS.index("bz"), SW_COLS.index("bz_south_duration"), SW_COLS.index("vsw")
    mu = np.zeros(len(SCALED)); sd = np.ones(len(SCALED))
    mu[i_v], sd[i_v] = 420.0, 93.0; sd[i_bz] = 2.75; mu[i_dur], sd[i_dur] = 0.33, 1.74
    cfg = yaml.safe_load(open(ROOT / "configs/config.yaml"))
    c = configure(copy.deepcopy(cfg), "storm_physfix", scaler=(mu, sd))
    assert c["loss"]["input_scaler"][1][i_bz] == 2.75
    assert "input_scaler" not in configure(copy.deepcopy(cfg), "storm_bz", scaler=(mu, sd)).get("loss", {})
    B, H = 4, len(HSTEPS)
    x = torch.zeros(B, 72, len(SW_COLS))
    x[:, -1, i_bz] = (-15.0 - 0.0) / 2.75          # Bz = -15 nT
    x[:, -1, i_dur] = (8.0 - 0.33) / 1.74          # southward for 8 h
    out = {"flux_pred": torch.zeros(B, H), "flux_residual": torch.zeros(B, H)}
    tg = {"y_flux": torch.ones(B, H), "y_dst": torch.zeros(B, H), "y_kp": torch.zeros(B, H), "y_storm": torch.zeros(B, 1),
          "storm_flag": torch.zeros(B, 1), "y_persist": torch.zeros(B, H)}
    old = PhysicsInformedLoss(LossWeights(horizon_hours=[float(h) for h in HSTEPS]))
    new = PhysicsInformedLoss(LossWeights(horizon_hours=[float(h) for h in HSTEPS])); new.set_input_scaler(mu, sd)
    assert old(out, tg, x)[1]["bz_resp"] == 0.0
    assert new(out, tg, x)[1]["bz_resp"] > 0.0
    xp = new.physical_inputs(x)
    assert xp[0, -1, i_bz].item() == pytest.approx(-15.0, abs=1e-4) and xp[0, -1, -1].item() == x[0, -1, -1].item()
