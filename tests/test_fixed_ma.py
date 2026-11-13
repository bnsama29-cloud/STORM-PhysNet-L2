"""FixedTwoPointAverage must equal the learned delay stage at tau = 1.5 h, and STORM must train with it."""
import sys, pathlib, torch
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "pipeline")]
from src.model.propagation_delay import AdaptivePropagationDelay
from storm_pipeline.models_extra import FixedTwoPointAverage, SmoothedInput, FFNResTransformer


def test_matches_learned_stage_at_saturation():
    torch.manual_seed(0); x = torch.randn(4, 72, 17)
    learned = AdaptivePropagationDelay(n_sw_features=17)
    with torch.no_grad(): learned.tau_logit_bias.fill_(8.0)        # clamp limit -> sigmoid(8) ~ 0.99966, tau ~ 1.4997
    learned.compute_tau = lambda x: torch.full((x.shape[0], 1), 1.5)  # exact saturation
    a, _ = learned(x); b, tau = FixedTwoPointAverage()(x)
    assert torch.allclose(a, b, atol=1e-6) and torch.all(tau == 1.5)


def test_average_definition_and_causality():
    x = torch.arange(5.).view(1, 5, 1)
    y, _ = FixedTwoPointAverage()(x)
    assert y.flatten().tolist() == [0.5, 0.5, 1.5, 2.5, 3.5]          # hour t -> mean(t-1, t); hour 0 -> mean(0, 1)
    x2 = x.clone(); x2[0, 4, 0] = 100.                                # changing the last hour changes only the last output
    y2, _ = FixedTwoPointAverage()(x2)
    assert torch.equal(y[0, :4], y2[0, :4])


def test_no_parameters_and_wrapper_runs():
    assert sum(p.numel() for p in FixedTwoPointAverage().parameters()) == 0
    m = SmoothedInput(FFNResTransformer(17))
    out = m(torch.randn(2, 72, 17), torch.randn(2, 72, 1), y_persist=torch.zeros(2, 3))
    assert out["flux_pred"].shape == (2, 3)


def test_storm_trains_with_fixed_stage():
    import yaml
    from src.training.trainer import Trainer
    cfg = yaml.safe_load(open(ROOT / "configs/config.yaml"))
    cfg["model_type"], cfg["ablation"], cfg["match_storm_capacity"] = "storm_physnet", "none", False
    t = Trainer(cfg); m = t.build_model(17); m.prop_delay = FixedTwoPointAverage()
    out = m(x_sw=torch.randn(2, 72, 17), x_flux=torch.randn(2, 72, 1), y_persist=torch.zeros(2, 3))
    assert out["flux_pred"].shape == (2, 3) and torch.all(out["tau"] == 1.5)
    groups = [{"params": [p for n, p in m.named_parameters() if "prop_delay" not in n]},
              {"params": [p for n, p in m.named_parameters() if "prop_delay" in n]}]
    torch.optim.AdamW(groups, lr=1e-3)                                 # the trainer builds exactly these two groups
