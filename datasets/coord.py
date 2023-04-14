import numpy as np
import torch

from torch.utils.data import Dataset, Subset, DataLoader, Sampler
from pytorch_lightning import LightningDataModule
from typing import Union, Tuple, List, Sequence, Optional


class CoordDataset(Dataset):
    """
    Given a Tensor of shape (dim1, dim2, ..., dimN), treat a subset of dimensions as index, and the rest as data.
    For example, 2D + time Tensor with shape (T, H, W):
    (1). Spatial. Each sample is of shape (H * W, 3) and there are T samples. Note "3" is dimension of the data Tensor. 
    Iteration over all samples: 
        t = 0: [[0, 0, 0],
                [0, 0, 1],
                ...
                [0, H - 1, W - 1]]
        ...
        t = T - 1: [[T - 1, 0, 0],
                    [T - 1, 0, 1],
                    ...
                    [T - 1, H - 1, W - 1]], of shape (H * W, 3)
    (2). Temporal. Each sample is of shape (T, 3) and there are H * W samples
    Iteration over all samples: 
        h = 0, w = 0: [[0, 0, 0],
                       [1, 0, 0],
                       ...
                       [T - 1, 0, 0]],
        ...
        h = H - 1, w = W - 1: [[0, H - 1, W - 1],
                               [1, H - 1, W - 1],
                               ...
                               [T - 1, H - 1, W - 1]], of shape (T, 3)
    """
    def __init__(self, data_shape: Sequence[int], data_dims: Sequence[int], if_normalize=True):
        """
        data_shape: e.g. (T, H, W)
        data_dims: e.g. spatial: (1, 2); temporal: (0,)
        Comments follow spatial case.
        """
        super().__init__()
        self.data_shape = data_shape
        self.data_dims = list(data_dims)  # [1, 2]
        self.index_dims = []  # [0] -> 0, initialized in the for-loop below
        self.data_dims_range, self.index_dims_range = [], []  # [H, W], [T]
        self.if_normalize = if_normalize
        for dim_iter in range(len(self.data_shape)):
            if dim_iter in self.data_dims:
                self.data_dims_range.append(self.data_shape[dim_iter])
            else:
                self.index_dims_range.append(self.data_shape[dim_iter])
                self.index_dims.append(dim_iter)
        
        # slicing: e.g. X[:, [0]] is invalid; should be X[:, 0]
        if len(self.data_dims) == 1:
            self.data_dims = self.data_dims[0]
        if len(self.index_dims) == 1:
            self.index_dims = self.index_dims[0]

        self.sample_len = np.prod(self.data_dims_range)  # H * W
        self.ds_len = np.prod(self.index_dims_range)  # T
        if not self.if_normalize:
            mgrid = torch.meshgrid([torch.arange(dim_range_iter) for dim_range_iter in self.data_dims_range], indexing="ij")  # ((H, W)...)
        else:
            mgrid = torch.meshgrid([(torch.arange(dim_range_iter) / (dim_range_iter  - 1)) * 2 - 1 for dim_range_iter in self.data_dims_range], indexing="ij")
        self.mgrid = torch.stack(mgrid, dim=-1).float()  # (H, W, 2)
        self.mgrid = self.mgrid.reshape(-1, self.mgrid.shape[-1]).squeeze()  # (H * W, 2); squeezing for e.g. temporal: (T, 1) -> (T,)
    
    def __len__(self):
        return self.ds_len

    def __getitem__(self, idx):
        sample = torch.empty((self.sample_len, len(self.data_shape)), dtype=torch.float32)  # (H * W, 3)
        sub = self.idx2sub(idx)
        sub = torch.FloatTensor(sub)
        if self.if_normalize:
            sub  = (sub / (torch.FloatTensor(self.index_dims_range) - 1)) * 2 - 1
        sample[:, self.index_dims] = sub  # fill in index info into the sample; spatial: fill in "t"; temporal: fill in "h", "w"
        sample[:, self.data_dims] = self.mgrid.clone()

        return sample

    # Note: .idx2sub(.) and .sub2idx(.) should use the same order: default to "C" (row-major), e.g. unravel_index(6, (2, 4)) -> (1, 2) (counting row first)
    # When we sample e.g. spatial coords (h, w) we need to convert it to idx. The specific ordering doesn't matter as long as .idx2sub(.) 
    # and .sub2idx(.) match
    def idx2sub(self, idx: int) -> Tuple[int]:
        if isinstance(self.index_dims, int):  # e.g. idx = 6, .index_dims = 8 (this is T): returns (6,)
            return (idx,)
        return np.unravel_index(idx, self.index_dims_range)

    def sub2idx(self, sub: Tuple[int]) -> int:
        if isinstance(self.index_dims, int):  # e.g. sub = (6,), .index_dims = 8: returns 6
            return sub[0]
        return np.ravel_multi_index(sub, self.index_dims_range)  


class MetaCoordDataset(Dataset):
    """
    A container Dataset used for pl.LightningModule.
    Takes in multiple CoordDataset(s), and iterates over samples in a zip(.)-styled way.
    Note zip(.) stops when the shortest iterable is through, here we pad shorter CoordDataset(s)
    with torch.empty(.). We also use a bool tensor to specify whether the current sample is a padding or not.
    For example: 2D + time: (T, H, W); spatial: T samples of shape (H * W, 3) each; and temporal: H * W samples of shape (T, 3) each.
    One sample here: ((H * W, 3), (T, 3), [True, False]) for T <= idx < H * W
    """
    def __init__(self, datasets: List[CoordDataset], num_samples: List[int], seed=None):
        """
        num_samples: Number of samples to create Subset for each CoordDataset
        """
        super().__init__()
        self.datasets = datasets
        self.num_samples = num_samples
        self.seed = seed
        self.ds_len = max(self.num_samples)
        self.subsets = None
        self.subsets_indices = None  # for debugging
        self.update_subsets()

    def __len__(self):
        return self.ds_len

    def __getitem__(self, idx):
        sample = []
        is_valid = []
        for ds_iter in self.subsets:
            if idx >= len(ds_iter):
                sample_iter = torch.zeros_like(ds_iter[0])
                is_valid_iter = False
            else:
                sample_iter = ds_iter[idx]
                is_valid_iter = True
            sample.append(sample_iter)
            is_valid.append(is_valid_iter)

        sample.append(torch.tensor(is_valid))

        return sample

    def create_one_subset(self, ds: Dataset, num_samples: int) -> Subset:
        if self.seed is not None:
            torch.manual_seed(self.seed)
        indices = torch.randperm(len(ds))[:num_samples]
        sub_ds = Subset(ds, indices)

        return sub_ds, indices

    def update_subsets(self) -> None:
        self.subsets = []
        self.subsets_indices = []
        for ds_iter, num_samples_iter in  zip(self.datasets, self.num_samples):
            if len(ds_iter) == num_samples_iter:
                subset_iter = ds_iter
                indices_iter = torch.arange(num_samples_iter)
            else:
                subset_iter, indices_iter = self.create_one_subset(ds_iter, num_samples_iter)
            self.subsets.append(subset_iter)
            self.subsets_indices.append(indices_iter)


class MetaCoordDM(LightningDataModule):
    """
    Note we are solving an optimization problem for one image, thus we only need the training data. Instead of validation data, we'll
    use a callback with on_train_epoch_end hook to monitor metrics (e.g. NRMSE, SSIM) comparing the reconstructed image with the original. 
    """
    def __init__(self, params,  datasets: List[CoordDataset], num_samples: List[int], pred_ds: CoordDataset, seed=None):
        """
        params: batch_size, num_workers
        pred_ds: for example, spatial for 2D + time; used to obtain the whole reconstruction
        """
        super().__init__()
        self.params = params
        self.datasets = datasets
        self.num_samples = num_samples
        self.meta_ds = None
        self.pred_ds = pred_ds

    def prepare_data(self):
        return super().prepare_data()
    
    def setup(self, stage: Union[str, None] = None):
        self.meta_ds = MetaCoordDataset(self.datasets, self.num_samples)

    def train_dataloader(self):
        loader = DataLoader(self.meta_ds, batch_size=self.params["batch_size"], num_workers=self.params["num_workers"], shuffle=True)

        return loader  

    def predict_dataloader(self):
        # Prediction only needs spatial coord
        loader = DataLoader(self.pred_ds, batch_size=self.params["batch_size"], num_workers=self.params["num_workers"])

        return loader   

    def reload_dataloader(self) -> None:
        """
        Used in a callback to update (resampling) training DataLoader.
        """
        self.meta_ds.update_subsets()


class AligningSampler(Sampler):
    """
    E.g, iterate 25 samples in 11 steps. 25 = 10 * 2 + 5 where 2 = 25 % (11 - 1). Thus the first batch_size is 5,
    and the rest are 2. 
    """
    def __init__(self, num_samples: int, num_steps: int):
        assert num_steps <= num_samples, f"num_steps = {num_steps}, num_samples = {num_samples}"
        self.num_samples = num_samples
        self.num_steps = num_steps
        self.first_batch_size = self.num_samples % (self.num_steps - 1)
        self.rest_batch_size = self.num_samples // self.num_steps
        if self.first_batch_size == 0:
            self.first_batch_size = self.rest_batch_size
        self.current_idx = 0

    def __len__(self):
        return self.num_steps
    
    def __iter__(self):
        yield torch.arange(self.first_batch_size)
        self.current_idx += self.first_batch_size
        while self.current_idx < self.num_samples:
            yield torch.arange(self.current_idx, self.current_idx + self.rest_batch_size)
            self.current_idx += self.rest_batch_size


class WrapperDM(LightningDataModule):
    """
    Equivalent as zip(data_loader...)
    """
    def __init__(self, ds_collection: Sequence[Dataset], batch_size_collection: Sequence[int], num_samples_collection: Sequence[int], pred_ds: Dataset, pred_batch_size: int, num_workers: int = 0):
        """
        batch_size_collections: [None..., batch_size, None...]; only one batch_size is specified, and the rest DataLoaders are controlled by a Sampler
        """
        self.ds_collection = ds_collection
        self.batch_size_collection = batch_size_collection
        self.num_samples_collection = num_samples_collection
        for ds, batch_size in zip(self.ds_collection, self.batch_size_collection):
            if batch_size is not None:
                self.num_steps = np.ceil(len(ds) / batch_size).astype(int)
                break
        assert self.num_steps is not None
        self.pred_ds = pred_ds
        self.pred_batch_size = pred_batch_size
        self.num_workers = num_workers
        self.ds_resampled = None
        self.resample()  # set .ds_resampled here
    
    def prepare_data(self) -> None:
        return super().prepare_data()
    
    def setup(self, stage: Optional[str] = None) -> None:
        return super().setup(stage)
    
    def train_dataloader(self):
        data_loaders = []
        for ds, batch_size in zip(self.ds_resampled, self.batch_size_collection):
            # shuffle: by .resample()
            if batch_size is not None:
                data_loaders.append(DataLoader(ds, batch_size, num_workers=self.num_workers, pin_memory=True))
            else:
                sampler = AligningSampler(len(ds), self.num_steps)
                data_loaders.append(DataLoader(ds, batch_sampler=sampler, pin_memory=True))

        return data_loaders

    def predict_dataloader(self):
        pred_loader = DataLoader(self.pred_ds, self.pred_batch_size, num_workers=self.num_workers, pin_memory=True)

        return pred_loader

    def resample(self):
        self.ds_resampled = []
        for ds, num_samples in zip(self.ds_collection, self.num_samples_collection):
            indices = torch.randperm(len(ds))[:num_samples]
            self.ds_resampled.append(Subset(ds, indices))
