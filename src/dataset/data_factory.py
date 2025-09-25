from .data_loader import *
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
        shuffle_flag = True
        drop_last = drop_last_test
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    if args.data == "ett_all":
        if flag == "val" or "test":
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

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )
    return data_set, data_loader
