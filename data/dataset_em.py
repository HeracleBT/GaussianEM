import numpy as np
import torch
from typing import Optional, Tuple, Union
from relion import ImageSource
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler
from eval_em import spherical_window_mask


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        mrcfile,
        lazy=True,
        invert_data=False,
        ind=None,
        window=True,
        datadir=None,
        window_r=0.85,
        fourier_mask=None,
        max_threads=16,
        device: Union[str, torch.device] = "cpu",
    ):
        datadir = datadir or ""
        self.ind = ind
        self.src = ImageSource.from_file(
            mrcfile,
            lazy=lazy,
            datadir=datadir,
            indices=ind,
            max_threads=max_threads,
        )
        ny = self.src.D
        assert ny % 2 == 0, "Image size must be even."
        self.N = self.src.n
        self.D = ny
        self.invert_data = invert_data
        self.device = device
        self.lazy = lazy
        self.fourier_mask = fourier_mask

        if window:
            self.window = spherical_window_mask(D=ny, in_rad=window_r, out_rad=0.99).to(
                device
            )
        else:
            self.window = None

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        if isinstance(index, list):
            index = torch.Tensor(index).to(torch.long)

        particles = self.src.images(index).to(self.device)
        if self.window is not None:
            particles *= self.window

        if self.invert_data:
            particles *= -1

        # this is why it is tricky for index to be allowed to be a list!
        if len(particles.shape) == 2:
            particles = particles[np.newaxis, ...]

        return particles, index

    def get_slice(
        self, start: int, stop: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return (
            self.src.images(slice(start, stop), require_contiguous=True).numpy(),
            None,
        )


class DataShuffler:
    def __init__(
        self, dataset: ImageDataset, batch_size, buffer_size, dtype=np.float32
    ):
        if not all(dataset.src.indices == np.arange(dataset.N)):
            raise NotImplementedError(
                "Sorry dude, --ind is not supported for the data shuffler. "
                "The purpose of the shuffler is to load chunks contiguously during lazy loading on huge datasets, which doesn't work with --ind. "
                "If you really need this, maybe you should probably use `--ind` during preprocessing (e.g. cryodrgn downsample)."
            )
        self.dataset = dataset
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.dtype = dtype
        assert self.buffer_size % self.batch_size == 0, (
            self.buffer_size,
            self.batch_size,
        )  # FIXME
        self.batch_capacity = self.buffer_size // self.batch_size
        assert self.buffer_size <= len(self.dataset), (
            self.buffer_size,
            len(self.dataset),
        )
        self.ntilts = getattr(dataset, "ntilts", 1)  # FIXME

    def __iter__(self):
        return _DataShufflerIterator(self)


class _DataShufflerIterator:
    def __init__(self, shuffler: DataShuffler):
        self.dataset = shuffler.dataset
        self.buffer_size = shuffler.buffer_size
        self.batch_size = shuffler.batch_size
        self.batch_capacity = shuffler.batch_capacity
        self.dtype = shuffler.dtype
        self.ntilts = shuffler.ntilts

        self.buffer = np.empty(
            (self.buffer_size, self.ntilts, self.dataset.D - 1, self.dataset.D - 1),
            dtype=self.dtype,
        )
        self.index_buffer = np.full((self.buffer_size,), -1, dtype=np.int64)
        self.tilt_index_buffer = np.full(
            (self.buffer_size, self.ntilts), -1, dtype=np.int64
        )
        self.num_batches = (
            len(self.dataset) // self.batch_size
        )  # FIXME off-by-one? Nah, lets leave the last batch behind
        self.chunk_order = torch.randperm(self.num_batches)
        self.count = 0
        self.flush_remaining = -1  # at the end of the epoch, got to flush the buffer
        # pre-fill
        for i in range(self.batch_capacity):
            chunk, maybe_tilt_indices, chunk_indices = self._get_next_chunk()
            self.buffer[i * self.batch_size : (i + 1) * self.batch_size] = chunk
            self.index_buffer[
                i * self.batch_size : (i + 1) * self.batch_size
            ] = chunk_indices
            if maybe_tilt_indices is not None:
                self.tilt_index_buffer[
                    i * self.batch_size : (i + 1) * self.batch_size
                ] = maybe_tilt_indices

    def _get_next_chunk(self) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        chunk_idx = int(self.chunk_order[self.count])
        self.count += 1
        particles, maybe_tilt_indices = self.dataset.get_slice(
            chunk_idx * self.batch_size, (chunk_idx + 1) * self.batch_size
        )
        particle_indices = np.arange(
            chunk_idx * self.batch_size, (chunk_idx + 1) * self.batch_size
        )
        particles = particles.reshape(
            self.batch_size, self.ntilts, *particles.shape[1:]
        )
        if maybe_tilt_indices is not None:
            maybe_tilt_indices = maybe_tilt_indices.reshape(
                self.batch_size, self.ntilts
            )
        return particles, maybe_tilt_indices, particle_indices

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns a batch of images, and the indices of those images in the dataset.

        The buffer starts filled with `batch_capacity` random contiguous chunks.
        Each time a batch is requested, `batch_size` random images are selected from the buffer,
        and refilled with the next random contiguous chunk from disk.

        Once all the chunks have been fetched from disk, the buffer is randomly permuted and then
        flushed sequentially.
        """
        if self.count == self.num_batches and self.flush_remaining == -1:
            # since we're going to flush the buffer sequentially, we need to shuffle it first
            perm = np.random.permutation(self.buffer_size)
            self.buffer = self.buffer[perm]
            self.index_buffer = self.index_buffer[perm]
            self.flush_remaining = self.buffer_size

        if self.flush_remaining != -1:
            # we're in flush mode, just return chunks out of the buffer
            assert self.flush_remaining % self.batch_size == 0
            if self.flush_remaining == 0:
                raise StopIteration()
            particles = self.buffer[
                self.flush_remaining - self.batch_size : self.flush_remaining
            ]
            particle_indices = self.index_buffer[
                self.flush_remaining - self.batch_size : self.flush_remaining
            ]
            tilt_indices = self.tilt_index_buffer[
                self.flush_remaining - self.batch_size : self.flush_remaining
            ]
            self.flush_remaining -= self.batch_size
        else:
            indices = np.random.choice(
                self.buffer_size, size=self.batch_size, replace=False
            )
            particles = self.buffer[indices]
            particle_indices = self.index_buffer[indices]
            tilt_indices = self.tilt_index_buffer[indices]

            chunk, maybe_tilt_indices, chunk_indices = self._get_next_chunk()
            self.buffer[indices] = chunk
            self.index_buffer[indices] = chunk_indices
            if maybe_tilt_indices is not None:
                self.tilt_index_buffer[indices] = maybe_tilt_indices

        particles = torch.from_numpy(particles)
        particle_indices = torch.from_numpy(particle_indices)
        tilt_indices = torch.from_numpy(tilt_indices)

        # merge the batch and tilt dimension
        particles = particles.view(-1, *particles.shape[2:])
        tilt_indices = tilt_indices.view(-1, *tilt_indices.shape[2:])

        particles = self.dataset._process(particles.to(self.dataset.device))
        # print('ZZZ', particles.shape, tilt_indices.shape, particle_indices.shape)
        return particles, tilt_indices, particle_indices


def make_dataloader(
    data: ImageDataset,
    *,
    batch_size: int,
    num_workers: int = 0,
    shuffler_size: int = 0,
    shuffle=False,
):
    if shuffler_size > 0 and shuffle:
        assert data.lazy, "Only enable a data shuffler for lazy loading"
        return DataShuffler(data, batch_size=batch_size, buffer_size=shuffler_size)
    else:
        sampler_cls = RandomSampler if shuffle else SequentialSampler
        return DataLoader(
            data,
            num_workers=num_workers,
            sampler=BatchSampler(
                sampler_cls(data), batch_size=batch_size, drop_last=False
            ),
            batch_size=None,
            multiprocessing_context="spawn" if num_workers > 0 else None,
        )


if __name__ == "__main__":

    # source_dir = "dataset/10180"
    # source_dir = "/home/data_8T/EMPAIR/10005/gaussian_preprocess"
    # source_dir = "/home/data_8T/EMPAIR/cryobench/Ribosembly/gaussian_preprocess"
    # source_dir = "/home/data_8T/EMPAIR/cryobench/IgG-RL/gaussian_preprocess"
    # source_dir = "/home/data_8T/EMPAIR/cryobench/IgG-1D/gaussian_preprocess"
    # source_dir = "/home/data_8T/EMPAIR/cryobench/IgG-1D/gaussian_preprocess_total"
    # source_dir = "/home/data_8T/EMPAIR/10180/gaussian_preprocess"
    # source_dir = "/home/data_8T/EMPAIR/10180/gaussian_preprocess_256"
    # source_dir = "/home/data_8T/EMPAIR/10180/gaussian_preprocess_filtered"
    source_dir = "/home/data_8T/EMPAIR/10841/gaussian_preprocess"
    # source_dir = "/home/data_8T/EMPAIR/10059/gaussian_preprocess_128"
    # source_dir = "/home/data_8T/EMPAIR/10345/gaussian_preprocess_256"
    # source_dir = "/home/data_8T/EMPAIR/10345/gaussian_preprocess_128"
    # metafile = "%s/consensus_data.star" % source_dir
    ctf_dir = "%s/ctf.npy" % source_dir
    # metafile = "%s/img_stack_part.mrcs" % source_dir
    # metafile = "%s/particles.mrcs" % source_dir
    # metafile = "%s/particles.128.mrcs" % source_dir
    # metafile = "%s/particles.256.mrcs" % source_dir
    # metafile = "%s/particles.256.filtered.mrcs" % source_dir
    metafile = "%s/L17all.mrcs" % source_dir
    orientation_dir = "%s/poses.npy" % source_dir
    # ind = np.arange(50000)
    ind = None
    invert_flag = True
    # invert_flag = False
    window_flag = True
    data = ImageDataset(
        mrcfile=metafile,
        lazy=True,
        invert_data=invert_flag,
        ind=ind,
        window=window_flag,
        datadir=None,
    )
    data_generator = make_dataloader(
        data,
        batch_size=1,
        num_workers=1,
        shuffler_size=0,
    )

    # poses = []
    # df = np.load(orientation_dir)
    # rots = df[:, :9].reshape(-1, 3, 3)
    # trans = df[:, 9:] / (data.D / 2)
    # for i in range(data.N):
    #     poses.append([rots[i], trans[i]])

    # ctf = np.load(ctf_dir)
    # ctf = torch.from_numpy(ctf).float().cuda()
    D = data.D
    # print(np.mean(data.src), np.std(data.src))
    # mean = 0
    count = 0
    std = 0
    # raw_std = 0
    for i, minibatch in enumerate(data_generator):
        # print(minibatch[-1])
        # print(minibatch[0].shape)
        print(i)
        # orientation = poses[i]
        # # img = ht2_center(minibatch[0])
        img = minibatch[0].cuda()
        # structure_mask = circular_mask(data.D, data.D / 2 - orientation[1] * data.D / 2).cuda()
        # print((1 - structure_mask).shape)
        # output = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(img, dim=(-1, -2))), dim=(-1, -2))
        # # half_ctf = ctf[..., :D//2+1]
        # output = torch.multiply(output, torch.multiply(ctf, torch.sign(ctf)))
        # img = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(output, dim=(-1, -2))), dim=(-1, -2))
        # img = torch.real(img)[0]
        # structure_region = img[(1 - structure_mask).bool()]
        # img = ht2_center(img)
        # img = symmetrize_ht(img)
        std += torch.std(img)
        print(torch.std(img))
        # print(img.std())
        # mean += torch.mean(minibatch[0])
        # std += torch.sqrt(torch.sum(torch.square(minibatch[0] - 0.0078)) / (data.D * data.D - 1))
        # raw_std += torch.std(minibatch[0])
        count += 1
    # print("mean: ", mean / count)
    print("std: ", std / count)
    # print("raw_std: ", raw_std / count)
