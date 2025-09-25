from easydict import EasyDict

config = EasyDict()

# Model
model_config = EasyDict()

# Dataset
dataset_config = EasyDict(
    features="M",
    target="OT",
    embed="timeF",
    percent=100,
    max_len=-1,
    freq="t",
    num_workers=6,
    label_len=0,  # TODO will not use for cal loss actually, then why it's here?
)


# Training
training_config = EasyDict(
    decay_fac=0.75,
    num_workers=6,
    patience=5,
    ema=True,
    loss_func="mse",
    loss_type="multi-variate",
    checkpoints=r"./log/gpt2_traffic_ckpts",
    lr_schedule_config=EasyDict(
        type="cos",
        tmax=16,
    ),
)
# training_config.lr_schedule_config = EasyDict(
#     type="type1",
#     decay_fac=0.75
# )

config.dataset_args = dataset_config
config.model_args = model_config
config.training_args = training_config
