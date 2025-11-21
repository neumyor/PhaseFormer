import config.base_config as config_module
from src.dataset.data_factory import data_provider
from src.dataset.data_info import DATASET_INFO
from src.models.PhaseFormer import PhaseFormer
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import Callback
import os
import csv
import torch
from datetime import datetime


class EpochTestCallback(Callback):
    """在每个epoch结束后运行测试的自定义回调"""

    def __init__(self, test_loader):
        super().__init__()
        self.test_loader = test_loader

    def on_fit_start(self, trainer, pl_module):
        print("🚀 每epoch测试回调已激活")

    def on_train_epoch_end(self, trainer, pl_module):
        was_training = pl_module.training
        pl_module.eval()
        try:
            total_mae = 0.0
            total_mse = 0.0
            num_batches = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(self.test_loader):
                    device = next(pl_module.parameters()).device
                    if isinstance(batch, (list, tuple)):
                        batch = [b.to(device) if torch.is_tensor(b) else b for b in batch]
                    elif torch.is_tensor(batch):
                        batch = batch.to(device)
                    test_metrics = pl_module.test_step(batch, batch_idx)
                    if isinstance(test_metrics, dict):
                        mae = test_metrics.get("mae", test_metrics.get("test_mae", 0.0))
                        mse = test_metrics.get("mse", test_metrics.get("test_mse", 0.0))
                        total_mae += float(mae)
                        total_mse += float(mse)
                    num_batches += 1

            if num_batches > 0:
                avg_mae = total_mae / num_batches
                avg_mse = total_mse / num_batches
                print(f"Epoch {trainer.current_epoch + 1} 测试结果: MAE={avg_mae:.6f}, MSE={avg_mse:.6f}")
                if trainer.logger:
                    trainer.logger.log_metrics(
                        {"epoch_test_mae": avg_mae, "epoch_test_mse": avg_mse},
                        step=trainer.current_epoch
                    )
        except Exception as e:
            print(f"❌ 测试过程中出现错误: {e}")
        finally:
            if was_training:
                pl_module.train()


DEFAULT_NORM_HYPERS = dict(
    revin_affine=False, revin_eps=1e-5,
)


def get_best_config_for_horizon(horizon):
    """根据预测长度返回ETTm1的最佳配置"""
    if horizon in [96, 192, 720]:
        return {
            'layers': 2,
            'latent_dim': 8,
            'phase_encoder_hidden': 32,
            'predictor_hidden': 64,
            'phase_num_routers': 8,
            'learning_rate': 0.001,
            'phase_attn_heads': 1
        }
    elif horizon == 336:
        return {
            'layers': 1,
            'latent_dim': 8,
            'phase_encoder_hidden': 32,
            'predictor_hidden': 64,
            'phase_num_routers': 8,
            'learning_rate': 0.001,
            'phase_attn_heads': 1
        }


def main():
    # 设置随机种子确保可复现性
    pl.seed_everything(2021, workers=True)
    torch.set_float32_matmul_precision("medium")
    
    exp_args = config_module.config
    dataset_name = "ETTm1"

    # 基础配置
    exp_args.model_args.model = "PhaseFormer"
    exp_args.model_args.input_len = exp_args.dataset_args.seq_len = 720

    # 训练参数
    exp_args.training_args.itr = 1
    exp_args.training_args.patience = 8
    exp_args.training_args.ema = False
    exp_args.training_args.train_epochs = 30
    exp_args.training_args.lr_schedule_config.type = "type3"
    exp_args.training_args.loss_func = "mse"

    # Huber loss 配置
    exp_args.training_args.use_huber_loss = True
    exp_args.training_args.huber_delta = 1.0

    # 数据集配置
    exp_args.dataset_args.percent = 100
    exp_args.dataset_args.data = DATASET_INFO[dataset_name]["data"]
    exp_args.dataset_args.root_path = DATASET_INFO[dataset_name]["root_path"]
    exp_args.dataset_args.data_path = DATASET_INFO[dataset_name]["data_path"]
    exp_args.training_args.batch_size = 16
    exp_args.dataset_args.batch_size = 16

    exp_args.dataset_args.var_needed = exp_args.model_args.num_variants = int(
        DATASET_INFO[dataset_name]["num_variants"]
    )
    
    # 实验配置
    lookback = 720
    horizons = [96, 192, 336, 720]  # 四个预测窗口
    
    # 收集每次实验的结果
    summary_records = []

    for horizon in horizons:
        print(f"\n{'='*50}")
        print(f"开始ETTm1实验: {lookback}->{horizon}")
        print(f"{'='*50}")

        # 获取该预测长度的最佳配置
        best_config = get_best_config_for_horizon(horizon)
        
        exp_args.dataset_args.seq_len = lookback
        exp_args.dataset_args.pred_len = horizon
        exp_args.dataset_args.noisy_ratio = 0.0
        exp_args.training_args.learning_rate = best_config['learning_rate']

        class PhaseFormerConfig:
            def __init__(self):
                # 基础配置
                self.seq_len = lookback
                self.pred_len = horizon
                self.enc_in = exp_args.model_args.num_variants
                self.period_len = 24

                # DefaultPL 需要
                self.target_var_index = -1
                self.training_args = exp_args.training_args
                self.dataset_args = exp_args.dataset_args

                # PhaseFormer 最佳超参
                self.latent_dim = best_config['latent_dim']
                self.phase_encoder_hidden = best_config['phase_encoder_hidden']
                self.predictor_hidden = best_config['predictor_hidden']
                self.phase_layers = best_config['layers']

                self.phase_attn_heads = best_config['phase_attn_heads']
                self.phase_attn_dropout = 0.1
                self.phase_attn_use_relpos = True
                self.phase_attn_window = None
                self.phase_attention_dim = None
                self.phase_num_routers = best_config['phase_num_routers']
                self.phase_use_pos_embed = True
                self.phase_pos_dropout = 0.0

                # RevIN
                self.use_revin = True
                self.revin_affine = DEFAULT_NORM_HYPERS["revin_affine"]
                self.revin_eps = DEFAULT_NORM_HYPERS["revin_eps"]

                # Loss 超参
                self.use_huber_loss = exp_args.training_args.use_huber_loss
                self.huber_delta = exp_args.training_args.huber_delta

            def get(self, key, default=None):
                return getattr(self, key, default)

        print(f"使用配置: layers={best_config['layers']}, latent_dim={best_config['latent_dim']}, "
              f"routers={best_config['phase_num_routers']}, lr={best_config['learning_rate']}")

        model_config = PhaseFormerConfig()

        # 数据加载
        _, train_loader = data_provider(exp_args.dataset_args, "train")
        _, vali_loader = data_provider(exp_args.dataset_args, "val")
        _, test_loader = data_provider(exp_args.dataset_args, "test")

        # 实例化模型
        model = PhaseFormer(model_config)

        # 日志配置
        loss_suffix = f"-losshuber-d{model_config.huber_delta}"
        logger_version = (
            f"{dataset_name}-{lookback}-{horizon}-PhaseFormer-p{model_config.period_len}"
            f"-layers{model_config.phase_layers}-lat{model_config.latent_dim}"
            f"-hid{model_config.predictor_hidden}-revin{model_config.use_revin}"
            f"-dimred{model_config.phase_num_routers}{loss_suffix}"
        )
        
        log_dir = f"./log/training_results/PhaseFormer/{logger_version}"
        if os.path.exists(log_dir):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger_version = f"{logger_version}_new_{timestamp}"
            print(f"📁 日志目录已存在，修改为: {logger_version}")

        logger = CSVLogger(
            save_dir="./log/training_results",
            name="PhaseFormer",
            version=logger_version,
        )

        # 回调配置
        epoch_test_callback = EpochTestCallback(test_loader)
        callbacks_list = [
            pl.callbacks.EarlyStopping(
                monitor="val_loss", patience=exp_args.training_args.patience
            ),
            epoch_test_callback,
        ]

        # 训练器配置
        trainer = pl.Trainer(
            max_epochs=exp_args.training_args.train_epochs,
            logger=logger,
            enable_checkpointing=True,
            callbacks=callbacks_list,
            devices=[0],  # 使用GPU 0
            enable_progress_bar=True,
            log_every_n_steps=1,
            deterministic=True,  # 确保可复现性
        )

        # 训练和测试
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=vali_loader)
        test_results = trainer.test(model, dataloaders=test_loader)

        # 提取测试结果
        last_test_mae = None
        last_test_mse = None
        if isinstance(test_results, list) and len(test_results) > 0:
            try:
                test_dict = dict(test_results[0])
                if "test_mae" in test_dict:
                    last_test_mae = float(test_dict["test_mae"])
                else:
                    for k, v in test_dict.items():
                        if isinstance(k, str) and "mae" in k.lower():
                            try:
                                last_test_mae = float(v)
                                break
                            except Exception:
                                pass
                if "test_mse" in test_dict:
                    last_test_mse = float(test_dict["test_mse"])
                else:
                    for k, v in test_dict.items():
                        if isinstance(k, str) and "mse" in k.lower():
                            try:
                                last_test_mse = float(v)
                                break
                            except Exception:
                                pass
            except Exception as e:
                print(f"⚠️ 解析测试结果出错: {e}")

        summary_records.append({
            "dataset": dataset_name,
            "lookback": lookback,
            "horizon": horizon,
            "layers": best_config['layers'],
            "latent_dim": best_config['latent_dim'],
            "routers": best_config['phase_num_routers'],
            "learning_rate": best_config['learning_rate'],
            "test_mae": last_test_mae,
            "test_mse": last_test_mse,
            "log_dir": logger.log_dir,
        })

    # 汇总和导出结果
    if summary_records:
        summary_records = sorted(summary_records, key=lambda x: x.get("horizon", 0))

        # 导出CSV
        os.makedirs("./log/training_results/PhaseFormer", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_csv = os.path.join("./log/training_results/PhaseFormer", f"summary_ettm1_{ts}.csv")
        headers = [
            "dataset", "lookback", "horizon", "layers", "latent_dim", "routers", 
            "learning_rate", "test_mae", "test_mse", "log_dir",
        ]
        try:
            with open(summary_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                for r in summary_records:
                    writer.writerow([
                        r.get("dataset"), r.get("lookback"), r.get("horizon"), 
                        r.get("layers"), r.get("latent_dim"), r.get("routers"),
                        r.get("learning_rate"), r.get("test_mae"), r.get("test_mse"), 
                        r.get("log_dir"),
                    ])
            print(f"✅ ETTm1实验汇总已导出到: {summary_csv}")
        except Exception as e:
            print(f"❌ 导出CSV失败: {e}")

        # 打印结果表格
        def _fmt(v):
            if v is None:
                return "-"
            if isinstance(v, float):
                return f"{v:.6f}"
            return str(v)

        print("\nETTm1实验结果汇总:")
        print("| Horizon | Layers | Latent Dim | Routers | Learning Rate | Test MAE | Test MSE |")
        print("|---------|--------|------------|---------|---------------|----------|----------|")
        for r in summary_records:
            print(
                "| "
                + " | ".join([
                    _fmt(r.get("horizon")),
                    _fmt(r.get("layers")),
                    _fmt(r.get("latent_dim")),
                    _fmt(r.get("routers")),
                    _fmt(r.get("learning_rate")),
                    _fmt(r.get("test_mae")),
                    _fmt(r.get("test_mse")),
                ])
                + " |"
            )


if __name__ == "__main__":
    main()
