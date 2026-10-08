import os.path
import torch
from torch.utils.data import Dataset
import scipy.io as sio
from scipy import sparse
import sklearn.preprocessing as skp
import numpy as np
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder


def _shuffle_sources(indices, rng):
    indices = np.asarray(indices).copy()
    if indices.size <= 1:
        return indices, indices
    source_indices = indices.copy()
    rng.shuffle(source_indices)
    return indices, source_indices


def _apply_view_shuffle(data_X, label_y_view1, label_y_view2, id1, id2, view, indices, rng):
    target_idx, source_idx = _shuffle_sources(indices, rng)
    if target_idx.size <= 1:
        return

    data_X[view][target_idx] = data_X[view][source_idx]
    if view == 0:
        label_y_view1[target_idx] = label_y_view1[source_idx]
        id1[target_idx] = id1[source_idx]
    elif view == 1:
        label_y_view2[target_idx] = label_y_view2[source_idx]
        id2[target_idx] = id2[source_idx]
    else:
        raise ValueError("Noisy correspondence currently supports two views only.")


def _apply_noisy_correspondence(data_X, label_y_view1, label_y_view2, id1, id2, candidate_idx, fp_ratio, rng, mode="view1"):
    m = int(fp_ratio * len(candidate_idx))
    if m <= 0:
        return

    noisy_idx = np.asarray(candidate_idx).copy()
    rng.shuffle(noisy_idx)
    noisy_idx = noisy_idx[:m]

    if mode == "view1":
        _apply_view_shuffle(data_X, label_y_view1, label_y_view2, id1, id2, 0, noisy_idx, rng)
    elif mode == "view2":
        _apply_view_shuffle(data_X, label_y_view1, label_y_view2, id1, id2, 1, noisy_idx, rng)
    else:
        raise ValueError(f"Unknown shuffle_noisy_view mode: {mode}")


def load_mat(args,noise_type='None'):
    data_X = []
    label_y = None

    if args.dataset == 'Scene15':
        mat = sio.loadmat(os.path.join(args.data_path, 'Scene_15.mat'))
        X = mat['X'][0]
        data_X.append(X[0].astype('float32'))
        data_X.append(X[1].astype('float32'))
        label_y = np.squeeze(mat['Y'])

    elif args.dataset == 'LandUse21':
        mat = sio.loadmat(os.path.join(args.data_path, 'LandUse_21.mat'))
        data_X.append(sparse.csr_matrix(mat['X'][0, 1]).toarray())
        data_X.append(sparse.csr_matrix(mat['X'][0, 2]).toarray())
        label_y = np.squeeze(mat['Y']).astype('int')

    elif args.dataset == 'Reuters':
        mat = sio.loadmat(os.path.join(args.data_path, 'Reuters_dim10.mat'))
        data_X = []  # 18758 samples
        data_X.append(np.vstack((mat['x_train'][0], mat['x_test'][0])))
        data_X.append(np.vstack((mat['x_train'][1], mat['x_test'][1])))
        label_y = np.squeeze(np.hstack((mat['y_train'], mat['y_test'])))

    elif args.dataset == "CCV":
        mat = sio.loadmat(os.path.join(args.data_path, "CCV.mat"))
        X = np.ravel(mat["X"], order="F")
        label_y = np.squeeze(mat["Y"]).astype("int")

        # data_X.append(X[0].astype("float32"))
        data_X.append(X[1].astype("float32"))
        data_X.append(X[2].astype("float32"))

    elif args.dataset == 'HandWritten':
        mat = sio.loadmat(
            os.path.join(args.data_path, 'HandWritten.mat')
        )
        X = mat['X'][0]
        data_X.append(X[0].astype('float32'))
        data_X.append(X[1].astype('float32'))
        label_y = np.squeeze(mat['Y'])

    else:
        raise KeyError(f"Unknown Dataset {args.dataset}")

    if args.data_norm == 'standard':
        for i in range(args.n_views):
            data_X[i] = skp.scale(data_X[i])
    elif args.data_norm == 'l2-norm':
        for i in range(args.n_views):
            data_X[i] = skp.normalize(data_X[i])
    elif args.data_norm == 'min-max':
        for i in range(args.n_views):
            data_X[i] = skp.minmax_scale(data_X[i])

    args.n_sample = data_X[0].shape[0]

    if noise_type == 'NC':
        rng = np.random.RandomState(1234)

        label_y_view1 = label_y.copy()
        label_y_view2 = label_y.copy()
        id1 = np.arange(data_X[0].shape[0])
        id2 = id1.copy()
        if args.fp_ratio > 0:
            candidate_idx = np.arange(data_X[0].shape[0])
            _apply_noisy_correspondence(
                data_X,
                label_y_view1,
                label_y_view2,
                id1,
                id2,
                candidate_idx,
                args.fp_ratio,
                rng,
                getattr(args, "shuffle_noisy_view", "view1"),
            )
        return data_X, label_y_view1, label_y_view2, id1, id2
    
    return data_X, label_y


def load_mat_mixture(args):
    # 1) return all samples without both NC and missing views
    data_X, label_y = load_mat(args, noise_type="None")
    N = data_X[0].shape[0]
    args.n_sample = N

    # 2) generate mask matrix of missing view
    missing_mask = IncompleteMultiviewDataset._get_mask(args.n_views, N, args.missing_rate)
    missing_mask = torch.from_numpy(missing_mask).bool()
    # index of complete samples with all views present
    complete_idx = torch.where(missing_mask.sum(dim=1) == args.n_views)[0].cpu().numpy()

    # 3) NC
    rng = np.random.RandomState(1234)
    label_y_view1 = label_y.copy()
    label_y_view2 = label_y.copy()
    id1 = np.arange(N)
    id2 = id1.copy()

    if args.fp_ratio > 0 and complete_idx.size > 0:
        _apply_noisy_correspondence(
            data_X,
            label_y_view1,
            label_y_view2,
            id1,
            id2,
            complete_idx,
            args.fp_ratio,
            rng,
            getattr(args, "shuffle_noisy_view", "view1"),
        )

    return data_X, label_y_view1, label_y_view2, id1, id2, missing_mask


def load_dataset(args, noise_type):
    if noise_type == 'NC':
        print('Loading dataset with Noisy Correspondence setting...')
        data, label1, label2, id1, id2 = load_mat(args,noise_type)
        dataset = NC_MultiviewDataset(args.n_views, data, label1, label2, id1, id2)
        return dataset

    elif noise_type == 'Mixture':
        print('Loading dataset with Mixture (Missing + NC) setting...')
        data_X, label_y_view1, label_y_view2, id1, id2, missing_mask = load_mat_mixture(args)
        dataset = MixtureMultiviewDataset(args.n_views, data_X, label_y_view1, label_y_view2, id1, id2, missing_mask)
        return dataset

    else:
        # for Incomplete setting like DIVIDE
        print('Loading dataset...')
        data_x, label_y = load_mat(args,noise_type)
        dataset = IncompleteMultiviewDataset(args.n_views, data_x, label_y, args.missing_rate)
        return dataset


class NC_MultiviewDataset(torch.utils.data.Dataset):
    def __init__(self, n_views, data_X, label1, label2, id1, id2):
        super(NC_MultiviewDataset, self).__init__()
        self.n_views = n_views
        self.data = data_X
        self.label1 = label1 - np.min(label1)
        self.label2 = label2 - np.min(label2)
        self.id1 = id1
        self.id2 = id2

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        data = []
        for i in range(self.n_views):
            data.append(torch.tensor(self.data[i][idx].astype("float32")))
        label1 = torch.tensor(self.label1[idx], dtype=torch.long)
        label2 = torch.tensor(self.label2[idx], dtype=torch.long)
        id1 = torch.tensor(self.id1[idx], dtype=torch.long)
        id2 = torch.tensor(self.id2[idx], dtype=torch.long)
        return idx, data, label1, label2, id1, id2


class IncompleteMultiviewDataset(torch.utils.data.Dataset):

    def __init__(self, n_views, data_X, label_y, missing_rate):
        super(IncompleteMultiviewDataset, self).__init__()
        self.n_views = n_views
        self.data = data_X
        self.targets = label_y - np.min(label_y)
        self.missing_mask = torch.from_numpy(self._get_mask(n_views, self.data[0].shape[0], missing_rate)).bool()


    def __len__(self):
        return self.data[0].shape[0]


    def __getitem__(self, idx):
        data = []
        for i in range(self.n_views):
            data.append(torch.tensor(self.data[i][idx].astype('float32')))
        label = torch.tensor(self.targets[idx], dtype=torch.long)

        mask = self.missing_mask[idx]

        return idx, data, mask, label


    @staticmethod
    def _get_mask(view_num, alldata_len, missing_rate):
        full_matrix = np.ones((int(alldata_len * (1 - missing_rate)), view_num))
        alldata_len = alldata_len - int(alldata_len * (1 - missing_rate)) 
        missing_rate = 0.5 # indicates

        if alldata_len != 0:
            one_rate = 1.0 - missing_rate

            if one_rate <= (1 / view_num):
                enc = OneHotEncoder()
                view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
                full_matrix = np.concatenate([view_preserve, full_matrix], axis=0)
                choice = np.random.choice(full_matrix.shape[0], size=full_matrix.shape[0], replace=False)
                matrix = full_matrix[choice]
                return matrix

            error = 1

            if one_rate == 1:
                matrix = randint(1, 2, size=(alldata_len, view_num))
                full_matrix = np.concatenate([matrix, full_matrix], axis=0)
                choice = np.random.choice(full_matrix.shape[0], size=full_matrix.shape[0], replace=False)
                matrix = full_matrix[choice]
                return matrix

            while error >= 0.005:
                enc = OneHotEncoder()
                view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
                one_num = view_num * alldata_len * one_rate - alldata_len
                ratio = one_num / (view_num * alldata_len)
                matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int)
                a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int))
                one_num_iter = one_num / (1 - a / one_num)
                ratio = one_num_iter / (view_num * alldata_len)
                matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int)
                matrix = ((matrix_iter + view_preserve) > 0).astype(np.int)
                ratio = np.sum(matrix) / (view_num * alldata_len)
                error = abs(one_rate - ratio)

            full_matrix = np.concatenate([matrix, full_matrix], axis=0)

        choice = np.random.choice(full_matrix.shape[0], size=full_matrix.shape[0], replace=False)
        matrix = full_matrix[choice]

        return matrix


class MixtureMultiviewDataset(torch.utils.data.Dataset):
    def __init__(self, n_views, data_X, label_y_view1, label_y_view2, id1, id2, missing_mask):
        super().__init__()
        self.n_views = n_views
        self.data = data_X
        # Kept for compatibility with code paths that expect dataset.targets.
        self.targets = label_y_view2 - np.min(label_y_view2)
        # view-specific labels track the actual content after NC injection.
        self.label_y_view1 = label_y_view1 - np.min(label_y_view1)
        self.label_y_view2 = label_y_view2 - np.min(label_y_view2)
        self.id1 = id1
        self.id2 = id2
        self.missing_mask = missing_mask.bool()

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        data = []
        mask = self.missing_mask[idx]
        for v in range(self.n_views):
            x = self.data[v][idx].astype("float32")
            if not mask[v].item():
                x = np.zeros_like(x, dtype=np.float32)
            data.append(torch.tensor(x))
        label1 = torch.tensor(self.label_y_view1[idx], dtype=torch.long)
        label2 = torch.tensor(self.label_y_view2[idx], dtype=torch.long)
        id1 = torch.tensor(self.id1[idx], dtype=torch.long)
        id2 = torch.tensor(self.id2[idx], dtype=torch.long)
        return idx, data, mask, label1, label2, id1, id2


class IncompleteDatasetSampler:
    '''
    only sample complete samples whose views are all present to train the model
    '''

    def __init__(self, dataset: Dataset, seed: int = 0, drop_last: bool = False) -> None:
        self.dataset = dataset
        self.epoch = 0
        self.drop_last = drop_last
        self.seed = seed
        self.compelte_idx = torch.where(self.dataset.missing_mask.sum(dim=1) == self.dataset.n_views)[0]
        # index of complete samples whose views are all present
        self.num_samples = self.compelte_idx.shape[0] # number of complete samples

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = torch.randperm(self.num_samples, generator=g).tolist()
        indices = self.compelte_idx[indices].tolist()
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        self.epoch = epoch

class MixedCompleteIncompleteBatchSampler:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        incomplete_ratio: float = 0.1,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.incomplete_ratio = incomplete_ratio
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        self.complete_idx = torch.where(self.dataset.missing_mask.sum(dim=1) == self.dataset.n_views)[0]
        self.incomplete_idx = torch.where(self.dataset.missing_mask.sum(dim=1) < self.dataset.n_views)[0]

        if len(self.complete_idx) == 0:
            raise ValueError('Mixed sampler requires at least one complete sample.')

        if len(self.incomplete_idx) == 0 or incomplete_ratio <= 0:
            self.incomplete_batch_size = 0
        else:
            self.incomplete_batch_size = min(
                batch_size - 1,
                max(1, int(round(batch_size * incomplete_ratio)))
            )
        self.complete_batch_size = batch_size - self.incomplete_batch_size
        if self.complete_batch_size <= 0:
            raise ValueError('batch_size is too small for the chosen incomplete_ratio.')

        if self.drop_last:
            self.num_batches = max(1, len(self.dataset) // self.batch_size)
        else:
            self.num_batches = int(np.ceil(len(self.dataset) / self.batch_size))

    def __len__(self):
        return self.num_batches

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _draw_from_pool(self, pool: torch.Tensor, ptr: int, need: int, g: torch.Generator):
        pool_len = len(pool)
        if pool_len == 0 or need == 0:
            return [], ptr, pool

        out = []
        while need > 0:
            remain = pool_len - ptr
            take = min(remain, need)
            out.extend(pool[ptr:ptr+take].tolist())
            ptr += take
            need -= take
            if ptr >= pool_len and need > 0:
                pool = pool[torch.randperm(pool_len, generator=g)]
                ptr = 0
        return out, ptr, pool

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        complete_pool = self.complete_idx[torch.randperm(len(self.complete_idx), generator=g)]
        if len(self.incomplete_idx) > 0:
            incomplete_pool = self.incomplete_idx[torch.randperm(len(self.incomplete_idx), generator=g)]
        else:
            incomplete_pool = self.incomplete_idx

        ptr_c, ptr_i = 0, 0

        for _ in range(self.num_batches):
            batch = []

            draw_c = self.complete_batch_size
            draw_i = self.incomplete_batch_size

            c_part, ptr_c, complete_pool = self._draw_from_pool(complete_pool, ptr_c, draw_c, g)
            batch.extend(c_part)

            if draw_i > 0:
                i_part, ptr_i, incomplete_pool = self._draw_from_pool(incomplete_pool, ptr_i, draw_i, g)
                batch.extend(i_part)

            if len(batch) < self.batch_size:
                need = self.batch_size - len(batch)
                extra_c, ptr_c, complete_pool = self._draw_from_pool(complete_pool, ptr_c, need, g)
                batch.extend(extra_c)

            perm = torch.randperm(len(batch), generator=g).tolist()
            batch = [batch[i] for i in perm]
            yield batch
