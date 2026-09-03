from .data_loader import *
from .input_component_ablation import InputComponentConfig, InputComponentDataset
from .input_candidate_discovery import (
    CandidateDataset,
    GaussianNotchBank,
    TailZeroBank,
    TrajectoryComponentBank,
)
from torch.utils.data import DataLoader

data_dict = {
    "custom": Dataset_Custom_Multi,
    "ett_h": Dataset_ETT_hour_Multi,
    "ett_m": Dataset_ETT_minute_Multi,
    "custom_uni": Dataset_Custom,
    "ett_h_uni": Dataset_ETT_hour,
    "ett_m_uni": Dataset_ETT_minute,
    "ett_all": ConcatDataset,
    "pems": Dataset_PEMS,
}


def data_provider(args, flag, drop_last_test=False, train_all=False):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != "timeF" else 1
    percent = args.percent
    if "scale" in args:
        will_scale = args.scale
    else:
        will_scale = True
    max_len = args.max_len
    var_needed = args.var_needed
    noisy_ratio = args.noisy_ratio

    if flag == "test":
        shuffle_flag = False
        drop_last = drop_last_test
        batch_size = args.batch_size
        freq = args.freq
        # var_needed = None # for the test set, we use all the variables, as [:None] is equal to :
        # shuffle_flag = True
        noisy_ratio = 0.0

    elif flag == "val":
        # Validation order must be stable for auditable sample indices and bad cases.
        shuffle_flag = False
        drop_last = drop_last_test
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    if args.data == "ett_all":
        if flag in ("val", "test"):
            data_infos = args.multiple_dataset_info["test"]
        else:
            data_infos = args.multiple_dataset_info["train"]

        data_sets = [
            data_dict[data_info.data](
                root_path=data_info.root_path,
                data_path=data_info.data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                timeenc=timeenc,
                freq=freq,
                percent=percent,
                max_len=max_len,
                train_all=train_all,
                var_needed=var_needed,
            )
            for data_info in data_infos
        ]

        data_set = ConcatDataset(datasets=data_sets)
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            scale=will_scale,
            max_len=max_len,
            train_all=train_all,
            var_needed=var_needed,
            noisy_ratio=noisy_ratio,
        )

    hypothesis = getattr(args, "input_hypothesis", "none")
    variant = getattr(args, "input_variant", "full")
    if hypothesis == "d1":
        if variant != "remove_full":
            raise ValueError("D1 retraining requires input_variant=remove_full")
        period = float(getattr(args, "input_d1_period", 0.0))
        if period <= 2:
            raise ValueError("D1 requires --input-d1-period > 2")
        sigma = float(getattr(args, "input_d1_sigma", 0.0))
        data_set = CandidateDataset(
            data_set,
            GaussianNotchBank(int(args.seq_len), period, None if sigma == 0.0 else sigma),
        )
    elif hypothesis == "d2":
        if variant != "remove_full":
            raise ValueError("D2 retraining requires input_variant=remove_full")
        recent_length = int(getattr(args, "input_d2_recent_length", 0))
        data_set = CandidateDataset(data_set, TailZeroBank(int(args.seq_len), recent_length))
    elif hypothesis == "d3":
        if variant != "remove_full":
            raise ValueError("D3 retraining requires input_variant=remove_full")
        component = str(getattr(args, "input_d3_component", ""))
        data_set = CandidateDataset(
            data_set,
            TrajectoryComponentBank(
                int(args.seq_len),
                component,
                period_len=int(getattr(args, "input_period_len", 24)),
            ),
        )
    elif hypothesis != "none":
        component_config = InputComponentConfig(
            hypothesis=hypothesis,
            variant=variant,
            period_len=int(getattr(args, "input_period_len", 24)),
            ema_window=int(getattr(args, "input_ema_window", 96)),
            intervention_seed=int(getattr(args, "intervention_seed", 9102)),
            max_phase_shift=int(getattr(args, "input_max_phase_shift", 6)),
            mad_epsilon=float(getattr(args, "input_mad_epsilon", 1e-6)),
            minimum_phase_correlation=float(
                getattr(args, "input_minimum_phase_correlation", 0.15)
            ),
        )
        component_config.validate(int(args.seq_len))
        namespace = f"{getattr(args, 'data_path', args.data)}|{flag}"
        data_set = InputComponentDataset(data_set, component_config, namespace)

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )
    return data_set, data_loader
