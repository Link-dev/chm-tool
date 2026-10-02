"""Chip stacks on disk and PyTorch datasets.

A chip stack is one or more .npy files of shape [n, bands, 256, 256] (inputs, uint16) or [n, 256, 256]
(labels, float32, -999 = missing), concatenated along the first axis. Files are memory-mapped, so stacks
of any size can be used for training and prediction.
"""
import numpy as np

from . import channels


class Stack:
    """Consecutive .npy parts viewed as one array along axis 0 (memory-mapped)."""

    def __init__(self, files):
        if isinstance(files, (str, bytes)) or hasattr(files, "__fspath__"):
            files = [files]
        self.files = [str(f) for f in files]
        self.parts = [np.load(f, mmap_mode="r") for f in self.files]
        self.n = sum(p.shape[0] for p in self.parts)
        self.shape = (self.n,) + self.parts[0].shape[1:]
        self.dtype = self.parts[0].dtype
        for p in self.parts:
            assert p.shape[1:] == self.shape[1:], (self.files, p.shape)
        self._offsets = np.cumsum([0] + [p.shape[0] for p in self.parts])

    def __len__(self):
        return self.n

    # DataLoader workers started with 'spawn' (Windows, macOS) receive a pickled copy of the dataset; a pickled
    # memmap would carry the whole array, so only the file names travel and the worker maps them again.
    def __getstate__(self):
        return {"files": self.files}

    def __setstate__(self, state):
        self.__init__(state["files"])

    def chip(self, i):
        k = int(np.searchsorted(self._offsets, i, side="right") - 1)
        return self.parts[k][i - self._offsets[k]]

    def __getitem__(self, i):
        if isinstance(i, slice):
            start, stop, step = i.indices(self.n)
            assert step == 1
            out, j = [], start
            while j < stop:
                k = int(np.searchsorted(self._offsets, j, side="right") - 1)
                e = min(stop, self._offsets[k + 1])
                out.append(np.asarray(self.parts[k][j - self._offsets[k]:e - self._offsets[k]]))
                j = e
            return np.concatenate(out) if len(out) > 1 else out[0]
        return np.asarray(self.chip(i))


def open_stack(files):
    return None if not files else Stack(files)


def model_input(input_name, annual, seasonal, sl):
    """Assembled model input for chips `sl` (int or slice) of the annual / seasonal stacks."""
    a = annual[sl]
    s = seasonal[sl] if seasonal is not None else None
    if isinstance(sl, (int, np.integer)):
        return channels.assemble(input_name, a[None], None if s is None else s[None])[0]
    return channels.assemble(input_name, a, s)


def labelled_pixels(labels, zero_to_nodata=False, block=50):
    n = 0
    for i in range(0, len(labels), block):
        y = labels[i:min(i + block, len(labels))]
        v = y != -999
        if zero_to_nodata:
            v &= y != 0
        n += int(v.sum())
    return n
