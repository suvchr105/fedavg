# server.py
import torch, os, logging, matplotlib.pyplot as plt
from collections import OrderedDict
import config
from models import VQVAE2 

class Server:
    def __init__(self, clients, results_logger):
        self.clients, self.results_logger = clients, results_logger
        self.logger = logging.getLogger()
        self.global_model = VQVAE2()
        self.global_shared_params = self.global_model.get_shared_state_dict()

        # Track best model by validation loss
        self.best_val_loss = float("inf")
        self.best_val_psnr = 0.0
        self.best_val_ssim = 0.0
        self.patience_counter = 0
        
        # Track metrics for plotting
        self.round_metrics = {
            'psnr': [],
            'ssim': [],
            'loss': [],
            'lpips': [],   
            'mse': []     
        }

    def aggregate_updates(self, client_updates, client_sizes):
        """Weighted FedAvg aggregation"""
        self.logger.info("Using FedAvg aggregation (weighted)")
        total = float(sum(client_sizes))
        aggregated_params = OrderedDict()

        for key in self.global_shared_params.keys():
            agg = torch.zeros_like(self.global_shared_params[key])
            for upd, n in zip(client_updates, client_sizes):
                agg += upd[key] * (n / total)
            aggregated_params[key] = agg

        self.global_shared_params = aggregated_params


    def save_metrics_graphs(self, round_results_dir):
        """Save PSNR, SSIM, Loss, LPIPS, and MSE graphs for current round"""
        if not self.round_metrics['psnr']:
            return

        rounds = list(range(1, len(self.round_metrics['psnr']) + 1))
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))

        metrics = ['psnr', 'ssim', 'loss', 'lpips', 'mse']
        colors  = ['b', 'g', 'r', 'm', 'c']
        titles  = ['PSNR (dB)', 'SSIM', 'Loss', 'LPIPS', 'MSE']

        for i, (m, c, t) in enumerate(zip(metrics, colors, titles)):
            axes[i].plot(rounds, self.round_metrics[m], f"{c}-o", linewidth=2, markersize=6)
            axes[i].set_xlabel("Round")
            axes[i].set_ylabel(t)
            axes[i].set_title(f"{t} vs Rounds")
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xticks(rounds)

        plt.tight_layout()
        plot_path = os.path.join(round_results_dir, "metrics_plot.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        self.logger.info(f"Metrics plot saved to: {plot_path}")

    def run(self):
        for round_num in range(1, config.GLOBAL_ROUNDS + 1):
            self.logger.info(f"========== Starting Round {round_num}/{config.GLOBAL_ROUNDS} ==========")
            round_results_dir = os.path.join(config.RESULTS_DIR, config.AGGREGATOR, f"round_{round_num}")
            os.makedirs(round_results_dir, exist_ok=True)
            client_updates = []
            client_sizes = [] 

            # ---- Train clients ----
            for client in self.clients:
                self.logger.info(f"--- Training Client {client.client_id} ---")
                client.set_shared_params(self.global_shared_params)
                client.train(round_num, self.global_shared_params)
                client_updates.append(client.get_shared_params())
                client_sizes.append(len(client.train_loader.dataset))

            # ---- Aggregate updates ----
            self.logger.info("--- Server Aggregating Client Updates ---")
            self.aggregate_updates(client_updates, client_sizes)  


            # ---- Validate ----
            self.logger.info("--- Validating Models Post-Aggregation ---")
            total_val_psnr, total_val_ssim, total_val_loss = 0.0, 0.0, 0.0

            for client in self.clients:
                client.set_shared_params(self.global_shared_params)
                val_metrics = client.validate()
                self.logger.info(
                    f"Client {client.client_id} Validation: "
                    f"PSNR={val_metrics['psnr']:.4f}, "
                    f"SSIM={val_metrics['ssim']:.4f}, "
                    f"LPIPS={val_metrics['lpips']:.4f}, "
                    f"MSE={val_metrics['mse']:.4f}, "
                    f"Loss={val_metrics['loss']:.4f}"
                )
                total_val_psnr += val_metrics['psnr']
                total_val_ssim += val_metrics['ssim']
                total_val_loss += val_metrics['loss']

            avg_val_psnr = total_val_psnr / len(self.clients)
            avg_val_ssim = total_val_ssim / len(self.clients)
            avg_val_loss = total_val_loss / len(self.clients)

            self.logger.info(
                f"Global Average Validation: PSNR={avg_val_psnr:.4f}, "
                f"SSIM={avg_val_ssim:.4f}, Loss={avg_val_loss:.4f}"
            )

            # ---- Early stopping based on validation loss ----
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.best_val_psnr = avg_val_psnr
                self.best_val_ssim = avg_val_ssim
                self.patience_counter = 0
                self.logger.info(
                    f"New best validation loss: {self.best_val_loss:.6f} "
                    f"(PSNR={self.best_val_psnr:.4f}, SSIM={self.best_val_ssim:.4f}). Saving model state."
                )

                # Save best model checkpoint
                torch.save(
                    self.global_shared_params,
                    os.path.join(config.RESULTS_DIR, config.AGGREGATOR, "best_model.pth")
                )

            else:
                self.patience_counter += 1
                self.logger.info(
                    f"Validation loss did not improve. Patience: {self.patience_counter}/{config.EARLY_STOPPING_PATIENCE}"
                )

            if self.patience_counter >= config.EARLY_STOPPING_PATIENCE:
                self.logger.warning("Early stopping triggered (based on validation loss)!")
                break

            # ---- Evaluate ----
            self.logger.info("--- Evaluating Models Post-Aggregation ---")
            total_global_psnr, total_global_ssim, total_global_loss = 0, 0, 0
            total_global_lpips, total_global_mse = 0, 0

            for client in self.clients:
                client.set_shared_params(self.global_shared_params)
                metrics = client.evaluate(round_num, round_results_dir)
                self.logger.info(
                    f"Client {client.client_id} Metrics: "
                    f"PSNR={metrics['psnr']:.4f}, "
                    f"SSIM={metrics['ssim']:.4f}, "
                    f"LPIPS={metrics['lpips']:.4f}, "
                    f"MSE={metrics['mse']:.4f}, "
                    f"Loss={metrics['loss']:.4f}"
                )
                total_global_psnr += metrics['psnr']
                total_global_ssim += metrics['ssim']
                total_global_lpips += metrics['lpips']
                total_global_mse += metrics['mse']
                total_global_loss += metrics['loss']

            avg_global_psnr = total_global_psnr / len(self.clients)
            avg_global_ssim = total_global_ssim / len(self.clients)
            avg_global_lpips = total_global_lpips / len(self.clients)
            avg_global_mse = total_global_mse / len(self.clients)
            avg_global_loss = total_global_loss / len(self.clients)

            # Store metrics for plotting
            self.round_metrics['psnr'].append(avg_global_psnr)
            self.round_metrics['ssim'].append(avg_global_ssim)
            self.round_metrics['lpips'].append(avg_global_lpips)
            self.round_metrics['mse'].append(avg_global_mse)
            self.round_metrics['loss'].append(avg_global_loss)

            self.results_logger.add_round_results(
                round_num, 'N/A',
                avg_global_psnr,
                avg_global_ssim,
                avg_global_loss,
                lpips_val=avg_global_lpips,
                mse_val=avg_global_mse,
                is_global=True
            )
            self.logger.info(
                f"Global Average Metrics: "
                f"PSNR={avg_global_psnr:.4f}, SSIM={avg_global_ssim:.4f}, "
                f"LPIPS={avg_global_lpips:.4f}, MSE={avg_global_mse:.4f}, "
                f"Loss={avg_global_loss:.4f}"
            )

            # Save metrics graphs + model
            self.save_metrics_graphs(round_results_dir)
            torch.save(self.global_shared_params, os.path.join(round_results_dir, 'global_model.pth'))
            self.results_logger.save()

        self.logger.info("Federated Learning process completed.")