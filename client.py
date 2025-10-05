# client.py
import torch
from tqdm import tqdm
import sys
import logging
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import calculate_metrics, save_visual_results, CharbonnierLoss
import config
from models import VQVAE2
import lpips
import piq
import torch.nn.functional as F
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import torch.nn as nn

class Client:
    def __init__(self, client_id, train_loader, test_loader, val_loader, results_logger):
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.results_logger = results_logger
        self.logger = logging.getLogger()

        # --- Model & Optimizer ---
        self.model = VQVAE2().to(config.DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)

        # --- Loss functions ---
        self.recon_loss_fn = CharbonnierLoss(eps=config.LOSS_CONFIG["charbonnier_eps"]).to(config.DEVICE)
        self.perceptual_loss_fn = lpips.LPIPS(net='vgg').to(config.DEVICE)
        self.structural_loss_fn = piq.MultiScaleSSIMLoss(data_range=1.0).to(config.DEVICE)

        # --- Metrics ---
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(config.DEVICE)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(config.DEVICE)
        self.lpips_metric = lpips.LPIPS(net='vgg').to(config.DEVICE)
        self.mse_metric = nn.MSELoss()

        # --- Scheduler ---
        total_steps = config.GLOBAL_ROUNDS * config.LOCAL_EPOCHS * len(self.train_loader)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=1e-6)

    def get_shared_params(self):
        return self.model.get_shared_state_dict()

    def set_shared_params(self, global_params):
        self.model.load_shared_state_dict(global_params)

    def _to_lpips(self, x):
        """Convert 1-channel images to 3-channel [-1,1] for LPIPS."""
        x = torch.clamp(x, -1, 1)
        if x.shape[1] == 1:         # if grayscale, repeat to RGB
            x = x.repeat(1, 3, 1, 1)
        return x

    def train(self, current_round, global_shared_params):
        self.model.train()
        is_interactive = sys.stdout.isatty()

        for epoch in range(config.LOCAL_EPOCHS):
            running_loss = 0.0
            progress_bar = tqdm(self.train_loader, desc=f"Client {self.client_id} Epoch {epoch+1}/{config.LOCAL_EPOCHS}", disable=not is_interactive)

            for batch in progress_bar:
                lr_images = batch['lr'].to(config.DEVICE)
                hr_images = batch['hr'].to(config.DEVICE)
                self.optimizer.zero_grad()

                reconstructed_images, vq_loss = self.model(lr_images)

                # --- Loss Calculation ---
                recon_loss = self.recon_loss_fn(reconstructed_images, hr_images)

                x_lpips = self._to_lpips(reconstructed_images)
                y_lpips = self._to_lpips(hr_images)
                perceptual_loss = self.perceptual_loss_fn(x_lpips, y_lpips).mean()

                reconstructed_images_0_1 = torch.clamp((reconstructed_images + 1) / 2, 0.0, 1.0)
                hr_images_0_1 = torch.clamp((hr_images + 1) / 2, 0.0, 1.0)
                structural_loss = self.structural_loss_fn(reconstructed_images_0_1, hr_images_0_1)
                
                total_loss = (
                        config.LOSS_CONFIG["reconstruction_loss_weight"] * recon_loss
                        + config.LOSS_CONFIG["perceptual_loss_weight"] * perceptual_loss
                        + config.LOSS_CONFIG["structural_loss_weight"] * structural_loss
                        + config.LOSS_CONFIG["vq_loss_weight"] * vq_loss
                    )

                total_loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                running_loss += total_loss.item()
                if is_interactive:
                    current_lr = self.scheduler.get_last_lr()[0]
                    progress_bar.set_postfix({"Loss": total_loss.item(), "LR": f"{current_lr:.1e}"})
            
            avg_epoch_loss = running_loss / len(self.train_loader)
            self.logger.info(f"Client {self.client_id} > Epoch {epoch+1}/{config.LOCAL_EPOCHS} | Average Loss: {avg_epoch_loss:.4f}")

    def validate(self):
        """Validation with PSNR, SSIM, LPIPS, MSE, and loss."""
        self.model.eval()
        all_psnr, all_ssim, all_lpips, all_mse, all_loss = [], [], [], [], []

        with torch.no_grad():
            for batch in self.val_loader:
                lr_images = batch['lr'].to(config.DEVICE)
                hr_images = batch['hr'].to(config.DEVICE)

                reconstructed_images, vq_loss = self.model(lr_images)

                reconstructed_images_0_1 = torch.clamp((reconstructed_images + 1) / 2, 0.0, 1.0)
                hr_images_0_1 = torch.clamp((hr_images + 1) / 2, 0.0, 1.0)

                # --- Losses ---
                recon_loss = self.recon_loss_fn(reconstructed_images, hr_images)
                x_lpips = self._to_lpips(reconstructed_images)
                y_lpips = self._to_lpips(hr_images)
                perceptual_loss = self.perceptual_loss_fn(x_lpips, y_lpips).mean()
                structural_loss = self.structural_loss_fn(reconstructed_images_0_1, hr_images_0_1)

                loss = (
                    config.LOSS_CONFIG["reconstruction_loss_weight"] * recon_loss +
                    config.LOSS_CONFIG["perceptual_loss_weight"] * perceptual_loss +
                    config.LOSS_CONFIG["structural_loss_weight"] * structural_loss +
                    config.LOSS_CONFIG["vq_loss_weight"] * vq_loss
                )

                # --- Metrics ---
                psnr_val = self.psnr_metric(reconstructed_images_0_1, hr_images_0_1)
                ssim_val = self.ssim_metric(reconstructed_images_0_1, hr_images_0_1)
                lpips_val = self.lpips_metric(reconstructed_images, hr_images)  # [-1,1]
                mse_val = self.mse_metric(reconstructed_images_0_1, hr_images_0_1)

                all_psnr.append(psnr_val.item())
                all_ssim.append(ssim_val.item())
                all_lpips.append(lpips_val.mean().item())
                all_mse.append(mse_val.item())
                all_loss.append(loss.item())

        return {
            "psnr": np.mean(all_psnr),
            "ssim": np.mean(all_ssim),
            "lpips": np.mean(all_lpips),
            "mse": np.mean(all_mse),
            "loss": np.mean(all_loss)
        }

    def evaluate(self, round_num, results_path):
        """Evaluation on test set with metrics + visual save."""
        self.model.eval()
        all_psnr, all_ssim, all_lpips, all_mse, all_loss = [], [], [], [], []

        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                lr_images = batch['lr'].to(config.DEVICE)
                hr_images = batch['hr'].to(config.DEVICE)

                reconstructed_images, vq_loss = self.model(lr_images)

                reconstructed_images_0_1 = torch.clamp((reconstructed_images + 1) / 2, 0.0, 1.0)
                hr_images_0_1 = torch.clamp((hr_images + 1) / 2, 0.0, 1.0)

                # Losses
                recon_loss = self.recon_loss_fn(reconstructed_images, hr_images)
                x_lpips = self._to_lpips(reconstructed_images)
                y_lpips = self._to_lpips(hr_images)
                perceptual_loss = self.perceptual_loss_fn(x_lpips, y_lpips).mean()
                structural_loss = self.structural_loss_fn(reconstructed_images_0_1, hr_images_0_1)

                loss = (
                    config.LOSS_CONFIG["reconstruction_loss_weight"] * recon_loss +
                    config.LOSS_CONFIG["perceptual_loss_weight"] * perceptual_loss +
                    config.LOSS_CONFIG["structural_loss_weight"] * structural_loss +
                    config.LOSS_CONFIG["vq_loss_weight"] * vq_loss
                )

                # Metrics
                psnr_val = self.psnr_metric(reconstructed_images_0_1, hr_images_0_1)
                ssim_val = self.ssim_metric(reconstructed_images_0_1, hr_images_0_1)
                lpips_val = self.lpips_metric(reconstructed_images, hr_images)
                mse_val = self.mse_metric(reconstructed_images_0_1, hr_images_0_1)

                all_psnr.append(psnr_val.item())
                all_ssim.append(ssim_val.item())
                all_lpips.append(lpips_val.mean().item())
                all_mse.append(mse_val.item())
                all_loss.append(loss.item())

                # Save first visual example
                if i == 0:
                    save_visual_results(
                        lr_images[0], hr_images[0], reconstructed_images[0],
                        results_path, round_num, self.client_id, sample_idx=0
                    )

        avg_psnr = np.mean(all_psnr)
        avg_ssim = np.mean(all_ssim)
        avg_lpips = np.mean(all_lpips)
        avg_mse = np.mean(all_mse)
        avg_loss = np.mean(all_loss)

        self.results_logger.add_round_results(
            round_num, self.client_id, avg_psnr, avg_ssim, avg_loss, is_global=False
        )

        return {
            "psnr": avg_psnr,
            "ssim": avg_ssim,
            "lpips": avg_lpips,
            "mse": avg_mse,
            "loss": avg_loss
        }