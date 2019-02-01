import time
import argparse
import sys

import numpy as np
import numba
from numba import jit, njit, config, threading_layer


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="", help="Input file")
    parser.add_argument("--out", default="", help="Output file")
    parser.add_argument("--dtype", default="i8", help="Data type")

    return parser.parse_args()


@njit(fastmath=True)
def vol_min_max(in_file):
    n = len(in_file)
    vol_min = np.iinfo(in_file.dtype).max
    vol_max = -np.iinfo(in_file.dtype).min
    for i in numba.prange(n):
        x = in_file[i]
        if x < vol_min:
            vol_min = x
        if x > vol_max:
            vol_max = x

    return vol_min, vol_max


@njit(fastmath=True)
def dohist(in_file, vol_min, vol_max, num_bins):
    max_idx = numba.int64(num_bins - 1)
    bins = np.zeros(num_bins)

    n = len(in_file)
    for i in numba.prange(n):
        x = in_file[i]
        idx = numba.int64((x - vol_min) / (vol_max - vol_min) * numba.float64(max_idx) + 0.5)
        if idx > max_idx:
            idx = max_idx

        bins[idx] += 1

    # compute percentage of all values in each bin
    for i in numba.prange(len(bins)):
        bins[i] = bins[i] / n

    return bins


def main(cargs):
    fd = np.memmap(cargs.file, dtype=np.dtype(cargs.dtype), mode='r')

    # first traversal: find volume min and max
    vol_min, vol_max = vol_min_max(fd)

    # second traversal: compute histo bins
    num_bins = 1536
    bins = dohist(fd, vol_min, vol_max, num_bins)

    # compute the list of values corresponding to the bins
    bin_vals = np.linspace(0.0, 1.0, num_bins, float)

    with open(cargs.out, 'w') as f:
        for i in range(num_bins):
            a = bin_vals[i]
            b = bins[i]
            f.write(f"{a:.9f} {b:.9f}\n")


if __name__ == '__main__':
    cargs = parse_args(sys.argv[1:])
    main(cargs)
