import math
import random
import sys
from contextlib import contextmanager
from typing import Iterable
import numpy as np
import torch
import utils
from utils import adjust_learning_config, SmoothedValue, MetricLogger
import torch_clustering


@contextmanager
def preserve_random_state():
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def train_one_epoch(
    model: torch.nn.Module,
    data_loader_train: Iterable,
    data_loader_test: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    state_logger=None,
    args=None,
):
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    print_freq = 50

    if args.print_this_epoch:
        data_loader = enumerate(metric_logger.log_every(data_loader_train, print_freq, header))
    else:
        data_loader = enumerate(data_loader_train)

    model.train(True)
    optimizer.zero_grad()

    for data_iter_step, (ids, samples, mask, label1, label2, id1, id2) in data_loader:
        smooth_epoch = epoch + (data_iter_step + 1) / len(data_loader_train)
        lr = adjust_learning_config(optimizer, smooth_epoch, args)
        mmt = args.momentum

        for i in range(args.n_views):
            samples[i] = samples[i].to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        with torch.autocast('cuda', enabled=False):
            loss = model(samples, mask, mmt, epoch < args.start_rectify_epoch,
                         idx=ids, epoch=epoch, singular_thresh=args.singular_thresh)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if args.print_this_epoch:
            metric_logger.update(lr=lr)
            metric_logger.update(loss=loss_value)

            debug_state = model.get_debug_state() if hasattr(model, 'get_debug_state') else {}
            if debug_state:
                debug_state = {k: v for k, v in debug_state.items() if v is not None}
                if debug_state:
                    metric_logger.update(**debug_state)

    if args.print_this_epoch:
        print("Averaged stats:", metric_logger)
        with preserve_random_state():
            eval_result = evaluate(model, data_loader_test, device, epoch, args)
    else:
        eval_result = None

    return eval_result


def evaluate(
    model: torch.nn.Module,
    data_loader_test: Iterable,
    device: torch.device,
    epoch: int,
    args=None
):
    model.eval()

    with torch.no_grad():
        features_0to1 = []
        labels_0to1 = []
        features_1to0 = []
        labels_1to0 = []
        features_realign = []
        labels_realign = []
        shuffle_noisy_view = getattr(args, "shuffle_noisy_view", "view1")
        realign_anchor_arg = getattr(args, "realign_anchor_view", "clean")
        if realign_anchor_arg == "clean":
            realign_anchor_view = 1 if shuffle_noisy_view == "view1" else 0
        elif realign_anchor_arg == "view1":
            realign_anchor_view = 0
        elif realign_anchor_arg == "view2":
            realign_anchor_view = 1
        else:
            raise ValueError(f"Unknown realign_anchor_view: {realign_anchor_arg}")
        realign_label_name = f"label{realign_anchor_view + 1}"

        for indexs, samples, mask, label1, label2, id1, id2 in data_loader_test:
            for i in range(args.n_views):
                samples[i] = samples[i].to(device, non_blocking=True)

            mask = mask.to(device, non_blocking=True)

            feat_0to1, valid_0_src, valid_0_tgt = model.extract_directional_feature(samples, mask, source_view=0)
            if feat_0to1 is not None:
                features_0to1.append(feat_0to1)
                labels_parts = []
                if valid_0_src is not None and valid_0_src.any():
                    labels_parts.append(label1[valid_0_src.cpu()])
                if valid_0_tgt is not None and valid_0_tgt.any():
                    labels_parts.append(label2[valid_0_tgt.cpu()])
                labels_0to1.append(torch.cat(labels_parts, dim=0).to(device, non_blocking=True))
        features_0to1 = torch.cat(features_0to1, dim=0)
        labels_0to1 = torch.cat(labels_0to1, dim=0)

        features_0to1 = torch.nn.functional.normalize(features_0to1, dim=-1)
        kmeans_0to1 = run_k_means_pytorch(features_0to1, args).cpu().numpy()


    nmi_0to1, ari_0to1, f_0to1, acc_0to1 = utils.evaluate(np.asarray(labels_0to1.cpu()), kmeans_0to1)


    view1_to_view2 = {'nmi': nmi_0to1, 'ari': ari_0to1, 'f': f_0to1, 'acc': acc_0to1}

    result = {
        'view1_to_view2': view1_to_view2,
        'realign_label_name': realign_label_name,
    }
    return result


def run_k_means_pytorch(feature, args, random_state=0, return_centroids=False, verbose=False):
    kwargs = {
        'metric': 'cosine',
        'distributed': False,
        'random_state': random_state,
        'n_clusters': args.n_classes,
        'verbose': verbose
    }

    clustering_model = torch_clustering.PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)
    pseudo_labels = clustering_model.fit_predict(feature)
    if return_centroids:
        centroids = clustering_model.cluster_centers_
        return pseudo_labels, centroids
    else:
        return pseudo_labels
