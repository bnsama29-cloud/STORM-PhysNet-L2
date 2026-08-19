"""
Training loop for STORM-PhysNet.
Supports: Deep Ensemble training, early stopping, cosine LR schedule,
gradient clipping, TensorBoard logging, checkpoint saving.
"""

import os
import time
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.model.storm_physnet import STORMPhysNet, STORMPhysNetEnsemble
from src.model.baselines import StandardLSTM, VanillaTransformer, StandardMLP, StandardCNN
from src.training.physics_loss import PhysicsInformedLoss, LossWeights

from src.data.dataloader import make_dataloaders


class Trainer:
    """
    Training orchestrator for STORM-PhysNet (single or ensemble).

    Features
    --------
    - Storm-biased batch sampling (default 12× storm oversampling, see config.yaml)
    - Physics-informed loss with asymmetric storm penalty
    - Cosine annealing LR schedule with warmup
    - Early stopping on validation MSE
    - TensorBoard logging of all loss components
    - Automatic checkpoint saving (best val loss)
    """

    def __init__(self, config: dict):
        self.cfg     = config
        self.device  = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Trainer] Device: {self.device}")

        self.ckpt_dir = Path(config["training"]["checkpoint_dir"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir  = Path(config["training"]["log_dir"])
        self.writer   = SummaryWriter(log_dir=str(self.log_dir))

    # ──────────────────────────────────────────────────────────────────────────
    # Model construction
    # ──────────────────────────────────────────────────────────────────────────

    def build_model(self, n_sw_features: int) -> nn.Module:
        model_type = self.cfg.get("model_type", "storm_physnet")
        ablation = self.cfg.get("ablation", "none")
        m = self.cfg["model"]
        
        if model_type == "lstm":
            model = StandardLSTM(n_sw_features=n_sw_features, seq_len=self.cfg["data"]["sequence_length"], n_horizons=len(self.cfg["data"]["forecast_horizons"])).to(self.device)
        elif model_type == "mlp":
            model = StandardMLP(n_sw_features=n_sw_features, seq_len=self.cfg["data"]["sequence_length"], n_horizons=len(self.cfg["data"]["forecast_horizons"])).to(self.device)
        elif model_type == "cnn":
            model = StandardCNN(n_sw_features=n_sw_features, seq_len=self.cfg["data"]["sequence_length"], n_horizons=len(self.cfg["data"]["forecast_horizons"])).to(self.device)
        elif model_type == "transformer":
            _tf_kw = dict(
                n_sw_features=n_sw_features,
                seq_len=self.cfg["data"]["sequence_length"],
                n_horizons=len(self.cfg["data"]["forecast_horizons"]),
            )
            if self.cfg.get("match_storm_capacity", False):
                _tf_kw["d_model"] = int(m.get("d_model", 128))
                _tf_kw["nhead"] = int(m.get("transformer", {}).get("n_heads", 4))
                _tf_kw["num_layers"] = int(m.get("transformer", {}).get("n_layers", 2))
            model = VanillaTransformer(**_tf_kw).to(self.device)
        else:
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
                n_horizons           = len(self.cfg["data"]["forecast_horizons"]),
                dropout              = m["transformer"]["dropout"],
                ablation             = ablation,
                backbone             = m.get("backbone", "transformer"),
                gate_type            = m.get("gate_type", "bz"),
                use_spectral_head    = m.get("use_spectral_head", False),
            ).to(self.device)
        print(f"[Model] Built {model_type.upper()} | Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        return model

    def build_ensemble(self, n_sw_features: int) -> nn.Module:
        model_type = self.cfg.get("model_type", "storm_physnet")
        ablation = self.cfg.get("ablation", "none")
        if model_type != "storm_physnet":
            print(f"[Ensemble] Warning: {model_type} does not support STORMPhysNetEnsemble natively. Building a single model wrapper.")
            # For simplicity, baselines are usually evaluated as single models in standard literature unless specifically ensembling them.
            # We'll just return a single model wrapped in a list-like structure for the trainer loop
            class DummyEnsemble(nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.members = nn.ModuleList([m])
                    self.n_members = 1
            return DummyEnsemble(self.build_model(n_sw_features))
            
        m = self.cfg["model"]
        ensemble = STORMPhysNetEnsemble(
            n_members            = int(self.cfg.get("ensemble", {}).get("n_members", 5)),
            n_sw_features        = n_sw_features,
            seq_len              = self.cfg["data"]["sequence_length"],
            d_model              = m["d_model"],
            n_heads              = m["transformer"]["n_heads"],
            n_transformer_layers = m["transformer"]["n_layers"],
            n_ssm_layers         = m["ssm"]["n_layers"],
            d_state              = m["ssm"]["d_state"],
            d_ff                 = m["transformer"]["d_ff"],
            hidden_dim           = m["heads"]["hidden_dim"],
            n_horizons           = len(self.cfg["data"]["forecast_horizons"]),
            dropout              = m["transformer"]["dropout"],
            ablation             = ablation,
            backbone             = m.get("backbone", "transformer"),
            gate_type            = m.get("gate_type", "bz"),
            use_spectral_head    = m.get("use_spectral_head", False),
        ).to(self.device)
        return ensemble

    # ──────────────────────────────────────────────────────────────────────────
    # Train single model
    # ──────────────────────────────────────────────────────────────────────────

    def train_single(
        self,
        model:        STORMPhysNet,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        model_name:   str = "goes",
    ) -> STORMPhysNet:
        """Train one model instance."""
        tr     = self.cfg["training"]
        lw     = self.cfg["loss"]
        # Horizon weights from config
        hw = tr.get("horizon_weights", [1.0, 0.7, 0.5])
        phs = lw.get("physics_horizon_scale", [1.0, 0.0, 0.0])
        # Base loss weights — physics terms start at 0 and ramp up (PINN warmup)
        ablation = self.cfg.get("ablation", "none")
        if ablation == "no_physics" or self.cfg.get("model_type", "storm_physnet") != "storm_physnet":
            print("[Trainer] Physics loss disabled for this run")
            base_weights = LossWeights(
                horizon_weights=hw,
                physics_horizon_scale=phs,
                lambda_dst=0, lambda_kp=0, lambda_storm_cls=0,
                lambda_mono=0, lambda_bz=0, lambda_smooth=0, lambda_delay=0,
                lambda_var=0, lambda_epsilon=0
            )
        else:
            base_weights = LossWeights(
                horizon_weights=hw,
                physics_horizon_scale=phs,
                lambda_dst       = 0.05,                        # Dst auxiliary task
                lambda_kp        = 0.05,                        # Kp auxiliary task
                lambda_storm_cls = 0.10,                        # storm classification
                lambda_mono      = lw["lambda_monotonicity"],   # energy transfer bound
                lambda_bz        = lw["lambda_bz_response"],    # Bz gate response
                lambda_smooth    = lw["lambda_smooth"],         # temporal smoothness
                lambda_delay     = lw["lambda_delay"],          # delay regularization
                lambda_var       = 0.02,                        # uncertainty calibration
                lambda_epsilon   = lw.get("lambda_epsilon", 0.08),  # Akasofu epsilon coupling
            )
        print(f"[Trainer] horizon_weights={hw} physics_horizon_scale={phs}")
        physics_loss_fn = PhysicsInformedLoss(base_weights).to(self.device)
        physics_ramp_start = tr["warmup_epochs"]            # epoch when physics turns on
        physics_ramp_end   = tr["warmup_epochs"] + 10      # full weight after 10 more epochs

        # Store original physics lambda values for proper warmup scaling
        original_physics_lambdas = {
            "lambda_mono": physics_loss_fn.w.lambda_mono,
            "lambda_bz": physics_loss_fn.w.lambda_bz,
            "lambda_smooth": physics_loss_fn.w.lambda_smooth,
            "lambda_delay": physics_loss_fn.w.lambda_delay,
            "lambda_var": physics_loss_fn.w.lambda_var,
            "lambda_epsilon": physics_loss_fn.w.lambda_epsilon,
        }

        # Separate parameter groups
        delay_params = [p for n, p in model.named_parameters() if "prop_delay" in n or "tau_logit" in n]
        other_params = [p for n, p in model.named_parameters() if "prop_delay" not in n and "tau_logit" not in n]

        optimizer = torch.optim.AdamW([
            {"params": other_params, "lr": tr["learning_rate"]},
            {"params": delay_params, "lr": tr["learning_rate"] * 5.0}  # 5× higher LR for delay
        ], weight_decay=tr["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tr["epochs"] - tr["warmup_epochs"],
        )

        best_val_mse  = float("inf")
        patience_ctr  = 0
        best_ckpt     = self.ckpt_dir / f"{model_name}_best.pt"

        for epoch in range(1, tr["epochs"] + 1):
            # ── LR Warmup ───────────────────────────────────────────────────
            if epoch <= tr["warmup_epochs"]:
                lr_scale = epoch / tr["warmup_epochs"]
                for pg in optimizer.param_groups:
                    pg["lr"] = tr["learning_rate"] * lr_scale

            # ── Physics Loss Warmup (PINN standard: ramp from 0 → target) ──
            # Epochs 1..warmup: physics weight = 0 (learn data pattern first)
            # Epochs warmup+1..warmup+10: linearly ramp to full weight
            # Epoch warmup+10+: full physics weight
            if epoch <= physics_ramp_start:
                phys_scale = 0.0
            elif epoch <= physics_ramp_end:
                phys_scale = (epoch - physics_ramp_start) / (physics_ramp_end - physics_ramp_start)
            else:
                phys_scale = 1.0
            # Scale physics lambda weights during warmup (SET, not multiply)
            for attr, orig_val in original_physics_lambdas.items():
                if hasattr(physics_loss_fn.w, attr):
                    setattr(physics_loss_fn.w, attr, orig_val * phys_scale)

            # ── Train ────────────────────────────────────────────────────────
            model.train()
            train_losses = []
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                optimizer.zero_grad()
                outputs = model(
                    x_sw      = batch["x_sw"],
                    x_flux    = batch["x_flux"],
                    y_persist = batch["y_persist"],
                )
                loss, comps = physics_loss_fn(outputs, batch, batch["x_sw"])
                # Skip batches that produce NaN/Inf loss to prevent weight corruption
                if torch.isnan(loss) or torch.isinf(loss):
                    print("NaN/Inf loss detected — skipping batch")
                    optimizer.zero_grad()
                    continue
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), tr["gradient_clip"])
                optimizer.step()
                train_losses.append(loss.item())

            if epoch > tr["warmup_epochs"]:
                scheduler.step()

            # ── Validate ─────────────────────────────────────────────────────
            val_loss, val_mse = self._validate(model, val_loader, physics_loss_fn)
            train_avg = np.mean(train_losses)

            # ── Log ──────────────────────────────────────────────────────────
            self.writer.add_scalars(
                "Loss", {"train": train_avg, "val": val_loss}, epoch)
            self.writer.add_scalar(
                "Val_MSE", val_mse, epoch)
            self.writer.add_scalar(
                "LR", optimizer.param_groups[0]["lr"], epoch)
            delay_val = getattr(model, "prop_delay", None)
            delay_h = delay_val.tau.item() if delay_val is not None else 0.0
            
            self.writer.add_scalar(
                "Delay_hours",
                delay_h, epoch)

            print(f"Epoch {epoch:3d} | "
                  f"Train: {train_avg:.4f} | "
                  f"Val: {val_loss:.4f} | "
                  f"Val MSE: {val_mse:.4f} | "
                  f"delay: {delay_h:.3f}h | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

            # ── Checkpoint ───────────────────────────────────────────────────
            # Track pure MSE for early stopping! NLL loss fluctuates due to physics 
            # constraint ramp-up and variance calibration. MSE measures pure accuracy.
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                patience_ctr  = 0
                torch.save(model.state_dict(), best_ckpt)
                print(f"  + Saved best model (val_mse={val_mse:.4f})")
            else:
                patience_ctr += 1
                if patience_ctr >= tr["early_stopping_patience"]:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        # Load best weights (only if checkpoint was saved)
        if best_ckpt.exists():
            model.load_state_dict(torch.load(best_ckpt, map_location=self.device,
                                             weights_only=True))
        else:
            print("  [Warning] No checkpoint saved — all epochs produced NaN loss.")
        return model

    def _validate(
        self,
        model:   STORMPhysNet,
        loader:  DataLoader,
        loss_fn: PhysicsInformedLoss,
    ) -> tuple[float, float]:
        model.eval()
        losses, sq_errs = [], []
        with torch.no_grad():
            for batch in loader:
                batch   = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(
                    x_sw=batch["x_sw"], 
                    x_flux=batch["x_flux"],
                    y_persist=batch["y_persist"]
                )
                loss, _ = loss_fn(outputs, batch, batch["x_sw"])
                # Skip NaN/Inf batches (same guard as training loop)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                losses.append(loss.item())
                sq_errs.append(
                    ((outputs["flux_pred"][:, -1] -
                      batch["y_flux"][:, -1]).pow(2).mean().item()))
        if not losses:   # all batches were NaN — should never happen after input guard
            return float("nan"), float("nan")
        return np.mean(losses), np.mean(sq_errs)

    # ──────────────────────────────────────────────────────────────────────────
    # Train Deep Ensemble
    # ──────────────────────────────────────────────────────────────────────────

    def train_ensemble(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        n_sw_features: int,
    ) -> STORMPhysNetEnsemble:
        """Train N independent ensemble members with different seeds."""
        ensemble = self.build_ensemble(n_sw_features)

        for i, member in enumerate(ensemble.members):
            print(f"\n{'='*50}")
            print(f"Training ensemble member {i+1}/{ensemble.n_members}")
            print(f"{'='*50}")

            # Different seed per member
            torch.manual_seed(42 + i * 100)
            np.random.seed(42 + i * 100)

            self.train_single(
                model        = member,
                train_loader = train_loader,
                val_loader   = val_loader,
                model_name   = f"ensemble_member_{i}",
            )

        # Save full ensemble
        ens_path = self.ckpt_dir / "ensemble_goes_best.pt"
        torch.save(
            {f"member_{i}": m.state_dict()
             for i, m in enumerate(ensemble.members)},
            ens_path,
        )
        print(f"\n[Ensemble] All members trained. Saved to {ens_path}")
        return ensemble

    # ──────────────────────────────────────────────────────────────────────────
    # Quick single-model training (for development/ablation)
    # ──────────────────────────────────────────────────────────────────────────

    def fit(
        self,
        train_loader:  DataLoader,
        val_loader:    DataLoader,
        n_sw_features: int,
        use_ensemble:  bool = True,
    ):
        """Main entry point: train model(s)."""
        if use_ensemble:
            return self.train_ensemble(train_loader, val_loader, n_sw_features)
        else:
            model = self.build_model(n_sw_features)
            model_type = self.cfg.get("model_type", "storm_physnet")
            ablation = self.cfg.get("ablation", "none")
            name = model_type if ablation == "none" else f"{model_type}_{ablation}"
            return self.train_single(model, train_loader, val_loader, model_name=name)
