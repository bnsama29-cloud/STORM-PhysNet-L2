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
           "storm_physfix", "storm_lstm_physfix", "lstm_res_w64", "lstm_res_w256",
           "storm_physfix_e1", "storm_lstm_physfix_e1",
           "lstm_res_tp", "storm_lstm_backbone_tp", "tf_ffn_res_tp", "storm_bz_tp",
           "lstm_res_L24", "lstm_res_L168", "tf_ffn_res_tp_L24", "tf_ffn_res_tp_L168"]


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


def test_physfix_auxiliary_targets_in_physical_units():
    """Dst and Kp targets arrive standardised; with the scaler and indices the loss compares in nT and Kp units."""
    import numpy as np
    i_dst, i_kp = SCALED.index("dst"), SCALED.index("kp")
    mu = np.zeros(len(SCALED)); sd = np.ones(len(SCALED)); mu[i_dst], sd[i_dst] = -12.0, 18.0; mu[i_kp], sd[i_kp] = 1.9, 1.2
    B, H = 3, len(HSTEPS)
    y_dst_nt = torch.full((B, H), -100.0); y_kp = torch.full((B, H), 6.0)
    out = {"flux_pred": torch.zeros(B, H), "dst_pred": y_dst_nt.clone(), "kp_pred": y_kp.clone()}
    tg = {"y_flux": torch.zeros(B, H), "y_dst": (y_dst_nt - mu[i_dst]) / sd[i_dst], "y_kp": (y_kp - mu[i_kp]) / sd[i_kp],
          "y_storm": torch.zeros(B, 1), "storm_flag": torch.zeros(B, 1), "y_persist": torch.zeros(B, H)}
    fn = PhysicsInformedLoss(LossWeights(horizon_hours=[float(h) for h in HSTEPS]))
    fn.set_input_scaler(mu, sd, dst_idx=i_dst, kp_idx=i_kp)
    comps = fn(out, tg, torch.zeros(B, 72, len(SW_COLS)))[1]
    assert comps["dst"] == pytest.approx(0.0, abs=1e-8) and comps["kp"] == pytest.approx(0.0, abs=1e-8)


def test_physics_ramp_default_and_full_weight_from_first_epoch():
    from src.training.trainer import physics_scale
    # default: off during the 5 warm-up epochs, full weight from epoch 15
    assert [physics_scale(e, 5, 15) for e in (1, 5, 6, 10, 15, 16)] == [0.0, 0.0, 0.1, 0.5, 1.0, 1.0]
    # "_e1" families: full weight from the first epoch
    assert physics_scale(1, 0, 1) == 1.0


def test_e1_families_configure_ramp_and_physical_units():
    import copy
    import numpy as np
    from storm_pipeline.families import configure
    cfg = {"model": {}, "loss": {}, "training": {"warmup_epochs": 5}}
    mu, sd = np.zeros(16, np.float32), np.ones(16, np.float32)
    c = configure(copy.deepcopy(cfg), "storm_lstm_physfix_e1", scaler=(mu, sd))
    assert c["training"]["physics_ramp_start"] == 0 and c["training"]["physics_ramp_len"] == 1
    assert c["loss"]["input_scaler"] is not None and c["model"]["backbone"] == "lstm"
    c2 = configure(copy.deepcopy(cfg), "storm_lstm_physfix", scaler=(mu, sd))
    assert "physics_ramp_start" not in c2["training"]


def test_selection_mse_last_and_mean():
    import pytest
    import torch
    from src.training.trainer import selection_mse
    pred = torch.tensor([[0.0, 0.0], [0.0, 0.0]]); target = torch.tensor([[1.0, 3.0], [1.0, 3.0]])
    assert selection_mse(pred, target) == 9.0                  # 24 h (last) only
    assert selection_mse(pred, target, "mean") == 5.0           # mean of 1 and 9
    with pytest.raises(ValueError, match="select_on"):
        selection_mse(pred, target, "first")


def test_protocol_families_copy_base_and_override_training():
    import copy
    import numpy as np
    from storm_pipeline.families import configure, SPEC, PROTOCOL
    for base in ("lstm_res", "storm_lstm_backbone", "tf_ffn_res", "storm_bz"):
        assert SPEC[base + "_tp"]["model_type"] == SPEC[base]["model_type"]
    cfg = {"model": {}, "loss": {}, "training": {"warmup_epochs": 5, "learning_rate": 3e-4, "epochs": 40}}
    mu, sd = np.zeros(16, np.float32), np.ones(16, np.float32)
    c = configure(copy.deepcopy(cfg), "storm_lstm_backbone_tp", scaler=(mu, sd))
    assert c["training"]["learning_rate"] == 1e-4 and c["training"]["select_on"] == "mean"
    assert c["training"]["epochs"] == 60 and c["training"]["warmup_epochs"] == 1 and c["model"]["backbone"] == "lstm"
    assert configure(copy.deepcopy(cfg), "storm_lstm_backbone", scaler=(mu, sd))["training"]["learning_rate"] == 3e-4


def test_protocol_families_build_same_architecture_as_base():
    import copy
    from storm_pipeline.families import build, configure
    from src.training.trainer import Trainer
    cfg0 = __import__("yaml").safe_load(open("configs/config.yaml"))
    for base in ("lstm_res", "tf_ffn_res"):
        m_tp = build(base + "_tp", Trainer(configure(copy.deepcopy(cfg0), base + "_tp")), 16, "cpu")
        m = build(base, Trainer(configure(copy.deepcopy(cfg0), base)), 16, "cpu")
        assert type(m_tp) is type(m)
        assert sum(p.numel() for p in m_tp.parameters()) == sum(p.numel() for p in m.parameters())


def test_lookback_families_carry_window_and_base_settings():
    from storm_pipeline.families import SPEC, base_family
    for base in ("lstm_res", "tf_ffn_res_tp"):
        for L in (24, 168):
            s = SPEC[f"{base}_L{L}"]
            assert s["lookback"] == L
            assert {k: v for k, v in s.items() if k != "lookback"} == SPEC[base]
    assert base_family("tf_ffn_res_tp_L168") == "tf_ffn_res" and base_family("lstm_res_L24") == "lstm_res"
    assert base_family("storm_bz_tp") == "storm_bz" and base_family("lstm_res_w64") == "lstm_res_w64"
    assert "lookback" not in SPEC["lstm_res"]


@pytest.mark.parametrize("L", [24, 168])
def test_lookback_families_run_on_their_window_length(L, tmp_path):
    for fam in (f"lstm_res_L{L}", f"tf_ffn_res_tp_L{L}"):
        m = make(fam, tmp_path).eval()
        x = torch.randn(2, L, len(SW_COLS)); f = torch.randn(2, L, 1)
        with torch.no_grad():
            out = m(x, f)
        assert out["flux_pred"].shape == (2, len(HSTEPS)) and torch.isfinite(out["flux_pred"]).all()


def test_kaggle_lookback_of_accounts():
    from make_journal_kaggle import lookback_of, CAMPAIGN7
    assert lookback_of("lstm_res,tf_ffn_res") == 72
    assert [lookback_of(f) for f, _ in CAMPAIGN7.values()] == [24, 168, 168]
    with pytest.raises(AssertionError):
        lookback_of("lstm_res_L24,lstm_res_L168")
