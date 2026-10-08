import argparse
import datetime
import os
import time
import warnings
from pathlib import Path
import numpy as np
import torch
import yaml
from model import Ours
import utils
from engine_train import train_one_epoch
from dataset_loader import load_dataset, IncompleteDatasetSampler, MixedCompleteIncompleteBatchSampler
warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument('--config_file', type=str, default=None)
    parser.add_argument('--encoder_dim', type=list, nargs='+', default=[])
    parser.add_argument('--embed_dim', type=int, default=0)

    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--start_rectify_epoch', type=int, default=20)
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--drop_rate', type=float, default=0.2)
    parser.add_argument('--n_views', type=int, default=2, help='number of views')
    parser.add_argument('--n_classes', type=int, default=10, help='number of classes')

    parser.add_argument('--batch_size', type=int, default=256, help='batch size per GPU')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--warmup_epochs', type=int, default=20, help='epochs to warmup learning rate')
    parser.add_argument('--data_norm', type=str, default='standard', choices=['standard', 'min-max', 'l2-norm'])
    parser.add_argument('--train_time', type=int, default=5)

    parser.add_argument('--weight_decay', type=float, default=0, help='Initial value of the weight decay. (default: 0)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')

    parser.add_argument('--dataset', type=str, default='LandUse21', choices=['LandUse21', 'Scene15'])
    parser.add_argument('--missing_rate', type=float, default=0.0)
    parser.add_argument('--data_path', type=str, default='./', help='path to your folder of dataset')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--output_dir', type=str, default='./', help='path where to save, empty for no saving')
    parser.add_argument('--print_freq', default=10)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--train_type', type=str, default='None', choices=['NC', 'Incomplete', 'Mixture', 'None'])
    parser.add_argument('--test_type', type=str, default='None', choices=['NC', 'Incomplete', 'Mixture', 'None'])
    parser.add_argument('--fp_ratio', type=float, default=0)
    parser.add_argument('--shuffle_noisy_view', type=str, default='view2', choices=['view1', 'view2'])
    parser.add_argument('--realign_anchor_view', type=str, default='view1', choices=['clean', 'view1', 'view2'])
    parser.add_argument('--which_model', type=str, default='CANDY', choices=['DIVIDE', 'CANDY', 'Ours'])
    parser.add_argument('--singular_thresh', type=float, default=0.2)
    parser.add_argument('--proto_update_freq', type=int, default=5, help='frequency to update prototypes')
    parser.add_argument('--incomplete_ratio', type=float, default=0.25, help='ratio of incomplete samples in each training batch for Mixture setting')
    parser.add_argument('--missing_loss_weight', type=float, default=1.0, help='weight for missing-view supervision loss')
    parser.add_argument('--sparse_topk', type=int, default=3, help='number of transported prototypes kept for sparse pseudo-view construction')
    parser.add_argument('--sparse_temp', type=float, default=0.25, help='temperature for sharpening transported prototype distributions')
    parser.add_argument('--missing_start_epoch', type=int, default=100, help='epoch to start joint missing-view supervision and mixed batching')
    parser.add_argument('--min_missing_conf', type=float, default=0.2, help='minimum confidence required to use a missing-view pseudo target')
    parser.add_argument('--loss_distance', type=float, default=0.8, help='minimum 1D-kmeans center distance to split clean/noisy correspondence losses')
    return parser


def build_train_loader_for_epoch(dataset_train, args, epoch):
    if epoch < args.missing_start_epoch:
        sampler_train = IncompleteDatasetSampler(dataset_train, seed=args.seed, drop_last=True)
        if hasattr(sampler_train, 'set_epoch'):
            sampler_train.set_epoch(epoch)
        effective_batch_size = min(args.batch_size, max(1, len(sampler_train)))
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            sampler=sampler_train,
            batch_size=effective_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        sampler_mode = 'complete_only'
    else:
        sampler_train = MixedCompleteIncompleteBatchSampler(
            dataset_train,
            batch_size=args.batch_size,
            incomplete_ratio=args.incomplete_ratio,
            seed=args.seed,
            drop_last=True,
        )
        if hasattr(sampler_train, 'set_epoch'):
            sampler_train.set_epoch(epoch)
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_sampler=sampler_train,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )
        sampler_mode = 'mixed_complete_incomplete'
    return data_loader_train, sampler_mode


def train_one_time(args, state_logger):
    utils.fix_random_seeds(args.seed)
    device = torch.device(args.device)

    if args.train_type == 'Mixture' and args.test_type == 'Mixture':
        print('Loading Mixture dataset (shared for train & test)...')
        dataset = load_dataset(args, noise_type='Mixture')
        dataset_train, dataset_test = dataset, dataset
        sampler_test = torch.utils.data.RandomSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    if args.which_model == 'Ours':
        model = Ours(
            n_views=args.n_views,
            layer_dims=args.encoder_dim,
            temperature=args.temperature,
            n_classes=args.n_classes,
            drop_rate=args.drop_rate,
            missing_loss_weight=args.missing_loss_weight,
            sparse_topk=args.sparse_topk,
            sparse_temp=args.sparse_temp,
            missing_start_epoch=args.missing_start_epoch,
            min_missing_conf=args.min_missing_conf,
            loss_distance=args.loss_distance,
        )
    else:
        raise NotImplementedError('This curriculum script currently supports Ours only.')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

    if args.train_id == 0:
        print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
        state_logger.write('Batch size: {}'.format(args.batch_size))
        state_logger.write('Start time: {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M')))
        state_logger.write('Train parameters: {}'.format(args).replace(', ', ',\n'))
        state_logger.write(model.__repr__())
        state_logger.write(optimizer.__repr__())
        print('Data loaded: there are {:} samples.'.format(len(dataset_train)))

    state_logger.write('\n>> Start training {}-th initial, seed: {},'.format(args.train_id + 1, args.seed))

    last_sampler_mode = None
    for epoch in range(args.start_epoch, args.epochs):
        data_loader_train, sampler_mode = build_train_loader_for_epoch(dataset_train, args, epoch)
        if sampler_mode != last_sampler_mode:
            state_logger.write(f'Epoch {epoch}: training sampler -> {sampler_mode}')
            last_sampler_mode = sampler_mode

        args.print_this_epoch = (epoch + 1) % args.print_freq == 0 or epoch + 1 == args.epochs or epoch == args.start_rectify_epoch - 1

        is_rectify_epoch = (epoch >= args.start_rectify_epoch) and (epoch < args.missing_start_epoch) and ((epoch - args.start_rectify_epoch) % args.proto_update_freq == 0)
        if args.which_model == 'Ours' and is_rectify_epoch:
            sampler_for_prototypes = torch.utils.data.SequentialSampler(dataset_train)
            data_loader_for_prototypes = torch.utils.data.DataLoader(
                dataset_train,
                sampler=sampler_for_prototypes,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False,
            )
            model.compute_and_store_prototypes(
                data_loader_for_prototypes,
                device,
                epoch=epoch,
                seed=args.seed,
                fp_ratio=args.fp_ratio,
                msrt=args.missing_rate,
                data_set_name=args.dataset,
            )

        train_state = train_one_epoch(
            model,
            data_loader_train,
            data_loader_test,
            optimizer,
            device,
            epoch,
            state_logger,
            args,
        )

        if args.print_this_epoch:
            view1_to_view2 = train_state['view1_to_view2']
            state_logger.write(
                'Epoch {} K-means View1->View2(label1): ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} F = {:.4f}'.format(
                    epoch,
                    view1_to_view2['acc'],
                    view1_to_view2['nmi'],
                    view1_to_view2['ari'],
                    view1_to_view2['f'],
                )
            )
    return train_state


def main(args):
    start_time = time.time()

    view1_to_view2_avr = {'nmi': [], 'ari': [], 'f': [], 'acc': []}
    batch_scale = args.batch_size / 256
    if args.lr is None:
        args.lr = args.blr * batch_scale

    state_logger = utils.FileLogger(os.path.join(args.output_dir, 'log_train.txt'))

    for t in range(args.train_time):
        args.train_id = t
        train_state = train_one_time(args, state_logger)
        args.seed = args.seed + 1
        for k in view1_to_view2_avr:
            view1_to_view2_avr[k].append(train_state['view1_to_view2'][k])

    def summarize_metrics(metric_dict):
        summary = {}
        for k, v in metric_dict.items():
            x = np.asarray(v) * 100
            summary[k] = [x.mean(), x.std()]
        return summary

    view1_to_view2_avr = summarize_metrics(view1_to_view2_avr)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    state_logger.write('\nTraining time {}\n'.format(total_time_str))
    state_logger.write(
        'Average K-means Result: ACC = {:.2f}({:.2f}) NMI = {:.2f}({:.2f}) ARI = {:.2f}({:.2f})'.format(
            *view1_to_view2_avr['acc'], *view1_to_view2_avr['nmi'], *view1_to_view2_avr['ari']
        )
    )


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    if args.config_file is not None:
        with open(args.config_file) as f:
            if hasattr(yaml, 'FullLoader'):
                configs = yaml.load(f.read(), Loader=yaml.FullLoader)
            else:
                configs = yaml.load(f.read())
        args = vars(args)
        args.update(configs)
        args = argparse.Namespace(**args)

    folder_name = '_'.join([
        args.which_model, args.dataset,
        'train', str(args.train_type), 'test', str(args.test_type),
        'NCrt', str(args.fp_ratio), 'msrt', str(args.missing_rate),
        'tau', str(args.temperature), 'bs', str(args.batch_size), 'blr', str(args.blr)
    ])

    args.embed_dim = args.encoder_dim[0][-1]
    args.output_dir = os.path.join(args.output_dir, folder_name)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, 'visualize')).mkdir(parents=True, exist_ok=True)

    main(args)
