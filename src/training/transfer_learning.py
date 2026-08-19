"""
GOES → GRASP Transfer Learning.
Cross-satellite transfer learning for GEO electron flux at Indian longitude (GSAT-19 GRASP).

Strategy:
  Phase 1: Pre-train full STORM-PhysNet on ~10 years of GOES data
  Phase 2: Freeze encoder (propagation delay, iTransformer, SSM,
           cross-modal attention, Bz gate)
  Phase 3: Fine-tune forecasting heads only on 1-2 years of GRASP data
  Phase 4: Evaluate GOES vs GRASP predictions on held-out test period

Why freeze the encoder:
  The solar wind → flux physics is the same at all longitudes.
  Only the magnitude/calibration of flux differs (different radiation
  belt sampling geometry). The heads learn this calibration shift.

Novel contribution for paper:
  First demonstration of domain adaptation for GEO electron flux
  across satellites/longitudes. Opens door to cross-satellite nowcasting.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from src.model.storm_physnet import STORMPhysNet, STORMPhysNetEnsemble
from src.training.physics_loss import PhysicsInformedLoss
from src.evaluation.metrics import StormEvaluator


class GRASPTransferLearner:
    """
    Fine-tunes a pre-trained STORM-PhysNet on GRASP data.
    Implements GOES→GRASP transfer learning via frozen encoder + fine-tuned heads.
    """

    def __init__(
        self,
        config:   dict,
        device:   torch.device = None,
    ):
        self.cfg    = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def load_pretrained(
        self,
        checkpoint_path: str,
        n_sw_features:   int,
    ) -> STORMPhysNet:
        """Load pre-trained GOES model.

        Fixed: this used to construct STORMPhysNet with only the plain
        default backbone/gate_type/use_spectral_head, regardless of what
        config.yaml actually specifies — so loading a checkpoint trained
        with e.g. gate_type='cathode_anode' would crash with a strict
        key-mismatch RuntimeError (bz_gate.voltage/emission_net keys don't
        exist in a plain-bz-constructed model). Not currently called
        anywhere in the pipeline (dead code as of this fix), but fixed so
        it's safe if/when something does call it — including a clear error
        instead of an opaque PyTorch key-mismatch trace if the checkpoint
        still doesn't match despite reading the current config correctly.
        """
        m = self.cfg["model"]
        model = STORMPhysNet(
            n_sw_features        = n_sw_features,
            seq_len              = self.cfg["data"]["sequence_length"],
            d_model              = m["d_model"],
            n_heads              = m["transformer"]["n_heads"],
            n_transformer_layers = m["transformer"]["n_layers"],
            n_ssm_layers         = m["ssm"]["n_layers"],
            d_state              = m["ssm"]["d_state"],
            d_ff                 = m["transformer"]["d_ff"],
            hidden_dim           = m["heads"]["hidden_dim"],
            n_horizons           = 3,
            dropout              = m["transformer"]["dropout"],
            backbone             = m.get("backbone", "transformer"),
            gate_type            = m.get("gate_type", "bz"),
            use_spectral_head    = m.get("use_spectral_head", False),
        ).to(self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        try:
            model.load_state_dict(ckpt)
        except RuntimeError as e:
            raise RuntimeError(
                f"[Transfer] Failed to load {checkpoint_path} into a model built with "
                f"backbone={m.get('backbone', 'transformer')}, gate_type={m.get('gate_type', 'bz')}, "
                f"use_spectral_head={m.get('use_spectral_head', False)}. This checkpoint was likely "
                f"trained with different settings — check config.yaml's model: section matches what "
                f"produced this checkpoint. Original error: {e}"
            ) from e
        print(f"[Transfer] Loaded pre-trained model from {checkpoint_path}")
        return model

    def fine_tune(
        self,
        model:           STORMPhysNet,
        grasp_loader:    DataLoader,
        grasp_val_loader: DataLoader,
        epochs:          int = 30,
        lr:              float = 1e-4,
    ) -> STORMPhysNet:
        """
        Fine-tune on GRASP data with frozen encoder.

        Only trains the forecasting heads (flux, Dst, Kp, storm heads).
        This is the key transfer learning step.
        """
        # Freeze encoder
        model.freeze_encoder()

        # Only optimize unfrozen parameters (heads)
        trainable = [p for p in model.parameters() if p.requires_grad]
        print(f"[Transfer] Fine-tuning {sum(p.numel() for p in trainable):,} "
              f"parameters (heads only)")

        optimizer = torch.optim.Adam(trainable, lr=lr)
        loss_fn   = PhysicsInformedLoss().to(self.device)

        best_val_loss = float("inf")
        ckpt_dir = Path(self.cfg["transfer"]["grasp_checkpoint_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs + 1):
            # ── Fine-tune ────────────────────────────────────────────────────
            model.train()
            train_losses = []
            for batch in grasp_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = model(batch["x_sw"], batch["x_flux"],
                                batch["y_persist"])
                loss, _ = loss_fn(outputs, batch, batch["x_sw"])
                loss.backward()
                nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            # ── Validate ─────────────────────────────────────────────────────
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in grasp_val_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = model(batch["x_sw"], batch["x_flux"],
                                    batch["y_persist"])
                    loss, _ = loss_fn(outputs, batch, batch["x_sw"])
                    val_losses.append(loss.item())

            val_loss = np.mean(val_losses)
            print(f"  Epoch {epoch:2d} | "
                  f"Train: {np.mean(train_losses):.4f} | Val: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(),
                           ckpt_dir / "grasp_best.pt")

        # Unfreeze for final evaluation
        model.unfreeze_all()
        print("[Transfer] Fine-tuning complete.")
        return model

    def evaluate_domain_gap(
        self,
        goes_model:   STORMPhysNet,
        grasp_model:  STORMPhysNet,
        grasp_loader: DataLoader,
    ) -> dict:
        """
        Quantify domain gap before/after transfer learning.
        This is the key paper result for GRASP section.

        Returns dict with metrics for:
            - GOES model on GRASP data (no adaptation)
            - GRASP fine-tuned model on GRASP data
        """
        evaluator = StormEvaluator()

        def get_preds(model):
            model.eval()
            preds, trues, kps = [], [], []
            with torch.no_grad():
                for batch in grasp_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    out   = model(batch["x_sw"], batch["x_flux"],
                                  batch["y_persist"])
                    preds.append(out["flux_pred"].cpu().numpy())
                    trues.append(batch["y_flux"].cpu().numpy())
                    kps.append(batch["y_kp"].cpu().numpy()[:, 0])
            return (np.concatenate(preds, 0),
                    np.concatenate(trues, 0),
                    np.concatenate(kps,  0))

        goes_pred,  y_true, kp = get_preds(goes_model)
        grasp_pred, _,      _  = get_preds(grasp_model)

        goes_metrics  = evaluator.evaluate_all(y_true, goes_pred,  kp, None)
        grasp_metrics = evaluator.evaluate_all(y_true, grasp_pred, kp, None)

        print("\n[Transfer Learning Results — GRASP Indian Longitude]")
        print("─" * 70)
        print(f"{'Model':<30} {'Horizon':<6} {'PE':>6} {'RMSE':>6} {'R²':>6}")
        print("─" * 70)
        for df, name in [(goes_metrics, "GOES-only (no adapt)"),
                          (grasp_metrics, "GRASP fine-tuned (ours)")]:
            for _, row in df.iterrows():
                if row["period"] == "all":
                    print(f"  {name:<28} {row['horizon']:<6} "
                          f"{row['pe']:>6.3f} {row['rmse']:>6.3f} "
                          f"{row['r2']:>6.3f}")
        print("─" * 70)

        return {"goes": goes_metrics, "grasp": grasp_metrics}
