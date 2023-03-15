import numpy as np
import torch

from torch.utils.data import Dataset, Subset, DataLoader
from pytorch_lightning import LightningDataModule
from typing import Union, Tuple, List


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
    def __init__(self, data_shape: Tuple[int], data_dims: Tuple[int]):
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
        mgrid = torch.meshgrid([torch.arange(dim_range_iter) for dim_range_iter in self.data_dims_range], indexing="ij")  # ((H, W)...)
        self.mgrid = torch.stack(mgrid, dim=-1).float()  # (H, W, 2)
        self.mgrid = self.mgrid.reshape(-1, self.mgrid.shape[-1]).squeeze()  # (H * W, 2); squeezing for e.g. temporal: (T, 1) -> (T,)
    
    def __len__(self):
        return self.ds_len

    def __getitem__(self, idx):
        sample = torch.empty((self.sample_len, len(self.data_shape)), dtype=torch.float32)  # (H * W, 3)
        sub = self.idx2sub(idx)
        sample[:, self.index_dims] = torch.FloatTensor(sub)  # fill in index info into the sample; spatial: fill in "t"; temporal: fill in "h", "w"
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
                sample_iter = torch.empty_like(ds_iter[0])
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
    def __init__(self, params,  datasets: List[CoordDataset], num_samples: List[int], seed=None):
        """
        params: batch_size, num_workers
        """
        super().__init__()
        self.params = params
        self.datasets = datasets
        self.num_samples = num_samples
        self.meta_ds = None

    def prepare_data(self):
        return super().prepare_data()
    
    def setup(self, stage: Union[str, None] = None):
        self.meta_ds = MetaCoordDataset(self.datasets, self.num_samples)

    def train_dataloader(self):
        loader = DataLoader(self.meta_ds, batch_size=self.params["batch_size"], num_workers=self.params["num_workers"], shuffle=True)

        return loader       

    def reload_dataloader(self) -> None:
        """
        Used in a callback to update (resampling) training DataLoader.
        """
        self.meta_ds.update_subsets()
