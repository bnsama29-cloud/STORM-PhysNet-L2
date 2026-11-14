"""Model families of the block pipeline: one registry shared by train_block.py and transfer_journal.py.

SPEC[family] = model_type, ablation, match (Transformer sized to STORM-PhysNet) and optional "model" overrides of
cfg["model"]. build() constructs a family; families not covered by Trainer.build_model are built here.
The default families reproduce the AGERS 2026 conference runs; the "journal additions" come from the original
STORM-PhysNet code base (alternative gates, spectral head, encoder backbones, magnetopause features, baselines).
"""
from .data import SW_COLS, SCALED, HSTEPS
from .models_extra import FFNResTransformer, FluxOnly, FixedTwoPointAverage, SmoothedInput, ResLSTM

SPEC = {
    "lstm": dict(model_type="lstm", ablation="none", match=False),
    "transformer_matched": dict(model_type="transformer", ablation="none", match=True),
    "tf_ffn_res": dict(model_type="transformer", ablation="none", match=True),     # model built separately
    "tf_flux_only": dict(model_type="transformer", ablation="none", match=True),   # model built separately
    "storm_bz": dict(model_type="storm_physnet", ablation="none", match=False),
    "storm_no_delay": dict(model_type="storm_physnet", ablation="no_delay", match=False),
    "storm_no_gate": dict(model_type="storm_physnet", ablation="no_bz_gate", match=False),
    # reviewer controls (AGERS 2026): the learned delay replaced by a fixed, non-learned two-point average
    "storm_fixed_ma": dict(model_type="storm_physnet", ablation="none", match=False),
    "tf_res_fixed_ma": dict(model_type="transformer", ablation="none", match=True),  # model built separately
    # journal additions (variants from the original STORM-PhysNet code base); "model" overrides cfg["model"]
    "transformer": dict(model_type="transformer", ablation="none", match=False),       # default-size Transformer
    "tf_ffn256": dict(model_type="transformer", ablation="none", match=True),          # FFN-matched, no residual; built separately
    "mlp": dict(model_type="mlp", ablation="none", match=False),
    "cnn": dict(model_type="cnn", ablation="none", match=False),
    "storm_no_physics": dict(model_type="storm_physnet", ablation="no_physics", match=False),
    "storm_cathode": dict(model_type="storm_physnet", ablation="none", match=False, model=dict(gate_type="cathode_anode")),
    "storm_cathode_spec": dict(model_type="storm_physnet", ablation="none", match=False,
                               model=dict(gate_type="cathode_anode", use_spectral_head=True)),
    "storm_radiotrophic": dict(model_type="storm_physnet", ablation="none", match=False, model=dict(gate_type="radiotrophic")),
    "storm_lstm_backbone": dict(model_type="storm_physnet", ablation="none", match=False, model=dict(backbone="lstm")),
    # attribution of the LSTM-encoder variant (second journal campaign)
    "lstm_res": dict(model_type="lstm", ablation="none", match=False),                 # built separately (ResLSTM)
    # width sensitivity of the residual LSTM (final campaign); built separately
    "lstm_res_w64": dict(model_type="lstm", ablation="none", match=False),
    "lstm_res_w256": dict(model_type="lstm", ablation="none", match=False),
    "storm_lstm_no_gate": dict(model_type="storm_physnet", ablation="no_bz_gate", match=False, model=dict(backbone="lstm")),
    "storm_lstm_no_delay": dict(model_type="storm_physnet", ablation="no_delay", match=False, model=dict(backbone="lstm")),
    "storm_lstm_no_physics": dict(model_type="storm_physnet", ablation="no_physics", match=False, model=dict(backbone="lstm")),
    # loss as intended: physics terms and Dst/Kp auxiliary targets in physical units (unit bug found while writing the
    # journal paper; the earlier families keep the original behaviour so their runs reproduce); otherwise identical
    "storm_physfix": dict(model_type="storm_physnet", ablation="none", match=False, physical_loss=True),
    "storm_lstm_physfix": dict(model_type="storm_physnet", ablation="none", match=False, physical_loss=True,
                               model=dict(backbone="lstm")),
    # physics terms in physical units and at full weight from the first epoch (no ramp), so that the weights chosen
    # by early stopping (usually epochs 2-5) were trained with the constraints active
    "storm_physfix_e1": dict(model_type="storm_physnet", ablation="none", match=False, physical_loss=True,
                             training=dict(physics_ramp_start=0, physics_ramp_len=1)),
    "storm_lstm_physfix_e1": dict(model_type="storm_physnet", ablation="none", match=False, physical_loss=True,
                                  model=dict(backbone="lstm"), training=dict(physics_ramp_start=0, physics_ramp_len=1)),
    "storm_magnetopause": dict(model_type="storm_physnet", ablation="none", match=False,
                               model=dict(use_magnetopause=True, vsw_idx=SW_COLS.index("vsw"),
                                          density_idx=SW_COLS.index("density"), pdyn_idx=SW_COLS.index("pdyn"))),
}



# Training-protocol check: the same models trained with a lower learning rate (1e-4), a one-epoch warm-up, up to 60
# epochs, and model selection on the mean validation MSE over all lead times instead of the 24 h one.
PROTOCOL = dict(learning_rate=1e-4, warmup_epochs=1, epochs=60, select_on="mean")
for _base in ("lstm_res", "storm_lstm_backbone", "tf_ffn_res", "storm_bz"):
    SPEC[_base + "_tp"] = dict(SPEC[_base], training={**SPEC[_base].get("training", {}), **PROTOCOL})

# Lookback sensitivity (campaign 7): the residual LSTM and the protocol-B residual Transformer with 24 h and 168 h of
# input history instead of 72 h. train_block.py must be started with the matching --lookback.
for _base in ("lstm_res", "tf_ffn_res_tp"):
    for _L in (24, 168):
        SPEC[f"{_base}_L{_L}"] = dict(SPEC[_base], lookback=_L)


def base_family(family):
    """The family whose architecture a variant uses: strips the _L<h> lookback and _tp protocol suffixes."""
    import re
    family = re.sub(r"_L\d+$", "", family)
    return family[:-3] if family.endswith("_tp") else family


def configure(cfg, family, scaler=None):
    """Apply a family's settings to a deep copy of the base config (in place) and return it.
    scaler: (mu, sd) of the block; train_block.py always passes it, so the "physfix" families train their physics
    loss in physical units. Without it (transfer.py, which only builds architectures to load weights) it is not set."""
    s = SPEC[family]
    cfg["model_type"], cfg["ablation"], cfg["match_storm_capacity"] = s["model_type"], s["ablation"], s["match"]
    cfg["model"].update(s.get("model", {}))
    cfg["training"].update(s.get("training", {}))
    if s.get("physical_loss") and scaler is not None:   # training needs it; building a model to load weights does not
        mu, sd = scaler
        assert list(SW_COLS[: len(mu)]) == list(SCALED), "scaler columns must be the leading input channels"
        cfg.setdefault("loss", {})["input_scaler"] = ([float(v) for v in mu], [float(v) for v in sd])
        cfg["loss"]["dst_idx"], cfg["loss"]["kp_idx"] = SCALED.index("dst"), SCALED.index("kp")
    return cfg


def build(family, trainer, n_sw, device, scaler=None):
    """trainer: a src.training.trainer.Trainer made from configure(cfg, family). scaler: (mu, sd) of the block,
    needed by storm_magnetopause for new models (a loaded checkpoint carries its own)."""
    family = base_family(family)   # protocol-B and lookback variants build exactly like their base family
    n_h = len(HSTEPS)
    if family == "tf_ffn_res":
        return FFNResTransformer(n_sw, n_h=n_h).to(device)
    if family == "tf_flux_only":
        return FluxOnly(trainer.build_model(n_sw)).to(device)
    if family == "tf_res_fixed_ma":
        return SmoothedInput(FFNResTransformer(n_sw, n_h=n_h)).to(device)
    if family == "storm_fixed_ma":
        m = trainer.build_model(n_sw); m.prop_delay = FixedTwoPointAverage().to(device); return m
    if family == "lstm_res":
        return ResLSTM(n_sw, n_h=n_h).to(device)
    if family in ("lstm_res_w64", "lstm_res_w256"):
        return ResLSTM(n_sw, d_model=int(family.rsplit("_w", 1)[1]), n_h=n_h).to(device)
    if family == "tf_ffn256":
        return FFNResTransformer(n_sw, n_h=n_h, residual=False).to(device)
    m = trainer.build_model(n_sw).to(device)
    if family == "storm_magnetopause" and scaler is not None:   # Shue (1998) needs physical units
        mu, sd = scaler
        assert list(SW_COLS[: len(mu)]) == list(SCALED), "scaler columns must be the leading input channels"
        m.mag_geo.set_input_scaler(mu, sd)
    return m
