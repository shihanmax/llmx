# ref: mmengine/dataset/sampler.py, xtuner/parallel/sequence/sampler.py
import math
import torch

from typing import Optional, Sized, Iterator

import numpy as np
from torch import distributed as torch_dist
from torch.utils.data.sampler import Sampler

from .setup_dist import get_data_parallel_rank, get_data_parallel_world_size
from .utils import (
    get_world_size, get_rank, get_default_group, get_comm_device, ProcessGroup,
)


def sync_random_seed(group: Optional[ProcessGroup] = None) -> int:
    """Synchronize a random seed to all processes.

    In distributed sampling, different ranks should sample non-overlapped
    data in the dataset. Therefore, this function is used to make sure that
    each rank shuffles the data indices in the same order based
    on the same seed. Then different ranks could use different indices
    to select non-overlapped data from the same data list.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        int: Random seed.

    Examples:
        >>> import torch
        >>> import mmengine.dist as dist

        >>> # non-distributed environment
        >>> seed = dist.sync_random_seed()
        >>> seed  # which a random number
        587791752

        >>> distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> seed = dist.sync_random_seed()
        >>> seed
        587791752  # Rank 0
        587791752  # Rank 1
    """
    seed = np.random.randint(2**31)
    if get_world_size(group) == 1:
        return seed

    if group is None:
        group = get_default_group()

    backend_device = get_comm_device(group)

    if get_rank(group) == 0:
        random_num = torch.tensor(seed, dtype=torch.int32).to(backend_device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32).to(backend_device)

    torch_dist.broadcast(random_num, src=0, group=group)

    return random_num.item()


class SequenceParallelSampler(Sampler):
    """The default data sampler for both distributed and non-distributed
    environment.

    It has several differences from the PyTorch ``DistributedSampler`` as
    below:

    1. This sampler supports non-distributed environment.

    2. The round up behaviors are a little different.

       - If ``round_up=True``, this sampler will add extra samples to make the
         number of samples is evenly divisible by the world size. And
         this behavior is the same as the ``DistributedSampler`` with
         ``drop_last=False``.
       - If ``round_up=False``, this sampler won't remove or add any samples
         while the ``DistributedSampler`` with ``drop_last=True`` will remove
         tail samples.

    Args:
        dataset (Sized): The dataset.
        shuffle (bool): Whether shuffle the dataset or not. Defaults to True.
        seed (int, optional): Random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Defaults to None.
        round_up (bool): Whether to add extra samples to make the number of
            samples evenly divisible by the world size. Defaults to True.
    """
    def __init__(
        self, dataset: Sized, shuffle: bool = True, seed: Optional[int] = None,
        round_up: bool = True,
    ) -> None:

        rank = get_data_parallel_rank()
        world_size = get_data_parallel_world_size()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.shuffle = shuffle
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0
        self.round_up = round_up

        if self.round_up:
            self.num_samples = math.ceil(len(self.dataset) / world_size)
            self.total_size = self.num_samples * self.world_size
        else:
            self.num_samples = math.ceil(
                (len(self.dataset) - rank) / world_size)
            self.total_size = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        # deterministically shuffle based on epoch and seed
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        if self.round_up:
            indices = (
                indices *
                int(self.total_size / len(indices) + 1))[:self.total_size]

        # subsample
        indices = indices[self.rank:self.total_size:self.world_size]

        return iter(indices)

    def __len__(self) -> int:
        """The number of samples in this rank."""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas use a different
        random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
