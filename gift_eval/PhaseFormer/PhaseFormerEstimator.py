from typing import Optional, Iterable, Dict, Any

import torch
import lightning.pytorch as pl

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import as_stacked_batches
from gluonts.itertools import Cyclic
from gluonts.transform import (
    Transformation,
    AddObservedValuesIndicator,
    InstanceSampler,
    InstanceSplitter,
    ValidationSplitSampler,
    TestSplitSampler,
    ExpectedNumInstanceSampler,
    SelectFields,
)

from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.distributions import Output, StudentTOutput
from gluonts.model.forecast_generator import SampleForecastGenerator

from .PhaseFormerLightningModule import PhaseFormerLightningModule


PREDICTION_INPUT_NAMES = ["past_target", "past_observed_values"]
TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target",
    "future_observed_values",
]


class PhaseFormerEstimator(PyTorchLightningEstimator):
    """
    用来训练你贴的 PhaseFormer 模型的 GluonTS Estimator（Torch 版）。

    用法示意：
        estimator = PhaseFormerEstimator(
            prediction_length=24,
            context_length=96,
            enc_in=1,
            seq_len=96,
            pred_len=24,
            period_len=24,
            phase_layers=2,
            trainer_kwargs={'max_epochs': 50, 'accelerator': 'gpu'},
        )
        predictor = estimator.train(training_data=train_ds)
    """

    @validated()
    def __init__(
        self,
        prediction_length: int,
        context_length: Optional[int],
        distr_output,
        # ---- 优化相关 ----
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
        # ---- 下面这些直接对应你 Model(configs) 里的 configs 字段 ----
        seq_len: Optional[int] = None,
        enc_in: int = 1,
        period_len: int = 24,
        latent_dim: int = 8,
        phase_encoder_hidden: int = 32,
        predictor_hidden: int = 64,
        phase_attn_heads: int = 4,
        phase_attn_dropout: float = 0.0,
        phase_attn_use_relpos: bool = True,
        phase_attn_window=None,
        phase_attention_dim=None,
        phase_num_routers: int = 8,
        phase_use_pos_embed: bool = False,
        phase_pos_dropout: float = 0.0,
        use_revin: bool = True,
        revin_affine: bool = False,
        revin_eps: float = 1e-5,
        task_name: str = "long_term_forecast",
        phase_layers: int = 1,
        phase_encoder_use_mlp: bool = False,
        phase_encoder_dropout: float = 0.0,
        predictor_use_mlp: bool = False,
        predictor_dropout: float = 0.0,
    ) -> None:
        # 默认 Trainer 参数
        default_trainer_kwargs = {
            "max_epochs": 100,
            "gradient_clip_val": 10.0,
        }
        if trainer_kwargs is not None:
            default_trainer_kwargs.update(trainer_kwargs)

        super().__init__(trainer_kwargs=default_trainer_kwargs)

        # ---- Estimator 自己要记住的一些东西 ----
        self.prediction_length = prediction_length
        self.context_length = context_length or prediction_length

        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.distr_output = distr_output

        self.train_sampler = train_sampler or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=prediction_length
        )
        self.validation_sampler = validation_sampler or ValidationSplitSampler(
            min_future=prediction_length
        )

        # ---- 把要传给 Model(configs) 的配置收集成一个 dict ----
        self.model_cfg = dict(
            seq_len=seq_len or self.context_length,
            pred_len=self.prediction_length,
            enc_in=enc_in,
            period_len=period_len,
            latent_dim=latent_dim,
            phase_encoder_hidden=phase_encoder_hidden,
            predictor_hidden=predictor_hidden,
            phase_attn_heads=phase_attn_heads,
            phase_attn_dropout=phase_attn_dropout,
            phase_attn_use_relpos=phase_attn_use_relpos,
            phase_attn_window=phase_attn_window,
            phase_attention_dim=phase_attention_dim,
            phase_num_routers=phase_num_routers,
            phase_use_pos_embed=phase_use_pos_embed,
            phase_pos_dropout=phase_pos_dropout,
            use_revin=use_revin,
            revin_affine=revin_affine,
            revin_eps=revin_eps,
            task_name=task_name,
            phase_layers=phase_layers,
            phase_encoder_use_mlp=phase_encoder_use_mlp,
            phase_encoder_dropout=phase_encoder_dropout,
            predictor_use_mlp=predictor_use_mlp,
            predictor_dropout=predictor_dropout,
            distr_output=distr_output,
        )

    # -------- GluonTS Estimator 接口部分 --------

    def create_transformation(self) -> Transformation:
        """
        定义从 raw Dataset entry 到训练样本之间的 entry-wise transform。
        这里仅使用 target + observed_values，结构和 PatchTST/DeepAR 类似。
        """
        return (
            SelectFields(
                [
                    FieldName.ITEM_ID,
                    FieldName.INFO,
                    FieldName.START,
                    FieldName.TARGET,
                ],
                allow_missing=True,
            )
            + AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            )
        )

    def _create_instance_splitter(self, module: PhaseFormerLightningModule, mode: str):
        """
        把长时间序列切成 (past, future) 窗口。
        """
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            time_series_fields=[FieldName.OBSERVED_VALUES],
            dummy_value=getattr(self, "distr_output", None).value_in_support
            if getattr(self, "distr_output", None) is not None
            else 0.0,
        )

    def create_lightning_module(self) -> pl.LightningModule:
        """
        真正创建训练用的 LightningModule（里面 new 你的 Model）。
        """
        return PhaseFormerLightningModule(
            model_cfg=self.model_cfg,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def create_training_data_loader(
        self,
        data: Dataset,
        module: PhaseFormerLightningModule,
        shuffle_buffer_length: Optional[int] = None,
        **kwargs,
    ) -> Iterable:
        """
        把 Dataset -> (transform -> split -> batchify) 得到一个 Iterable[batch]。
        """
        data_stream = Cyclic(data).stream()
        instances = self._create_instance_splitter(module, "training").apply(
            data_stream, is_train=True
        )
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            shuffle_buffer_length=shuffle_buffer_length,
            field_names=TRAINING_INPUT_NAMES,
            output_type=torch.tensor,
            num_batches_per_epoch=self.num_batches_per_epoch,
        )

    def create_validation_data_loader(
        self,
        data: Dataset,
        module: PhaseFormerLightningModule,
        **kwargs,
    ) -> Iterable:
        instances = self._create_instance_splitter(module, "validation").apply(
            data, is_train=True
        )
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            field_names=TRAINING_INPUT_NAMES,
            output_type=torch.tensor,
        )

    def create_predictor(
        self, transformation: Transformation, module: PhaseFormerLightningModule
    ) -> PyTorchPredictor:
        """
        训练完以后，Estimator.train(...) 会调用这里来构造 Predictor。
        这里我们用 SampleForecastGenerator，把 deterministic 预测当作单样本输出。
        """
        prediction_splitter = self._create_instance_splitter(module, "test")

        forecast_generator = (
            getattr(self, "distr_output", None).forecast_generator
            if getattr(self, "distr_output", None) is not None
            else SampleForecastGenerator()
        )

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=PREDICTION_INPUT_NAMES,
            prediction_net=module,
            forecast_generator=forecast_generator,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            device="auto",
        )