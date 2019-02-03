import time
import argparse
import sys
import os
import numpy as np
import numba
from numba import jit, njit, config, threading_layer

import indexfile
import volume

#config.THREADING_LAYER = 'tbb'

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="", help="Input file")
    parser.add_argument("--out", default="", help="Output file")

    parser.add_argument("--bx", default=1, type=int, help="Blocks along x-dim")
    parser.add_argument("--by", default=1, type=int, help="Blocks along y-dim")
    parser.add_argument("--bz", default=1, type=int, help="Blocks along z-dim")

    parser.add_argument("--vx", default=1, type=int, help="Vol dims X")
    parser.add_argument("--vy", default=1, type=int, help="Vol dims Y")
    parser.add_argument("--vz", default=1, type=int, help="Vol dims Z")

    parser.add_argument("--tf", default='', type=str, help="Transfer function")

    return parser.parse_args(args)


@njit(fastmath=True)
def volume_analysis_jit(fd):
    mn = np.iinfo(fd.dtype).max
    mx = -np.iinfo(fd.dtype).min
    acc = numba.float64(0.0)
    n = len(fd)
    for i in numba.prange(n):
        x = fd[i]
        if x < mn:
            mn = x
        if x > mx:
            mx = x

        acc += x

    return mn, mx, acc

@njit(fastmath=True, parallel=True)
def block_analysis_jit(fd, xp, yp,
                     vmin: np.float64, vmax: np.float64,
                     vdims: np.ndarray, bdims: np.ndarray,
                     bcount: np.ndarray):
    diff = vmax - vmin
    num_vox = len(fd)

    blocks = np.zeros(np.prod(bcount))

    for i in numba.prange(num_vox):
        bI = numba.uint64((i % vdims[0]) / bdims[0])
        bJ = numba.uint64(((i / vdims[0]) % vdims[1]) / bdims[1])
        bK = numba.uint64(((i / vdims[0]) / vdims[1]) / bdims[2])

        if bI < bcount[0] and bJ < bcount[1] and bK < bcount[2]:
            x = numba.float64((fd[i] - vmin) / diff)
            # rel = interp(n, xp, yp)

            if len(xp) == 0 or len(yp) == 0:
                return 0.0

            if x <= xp[0]:
                return yp[0]

            max_idx = len(xp) - 1

            if x >= xp[-1]:
                return yp[-1]

            idx = int((x * max_idx) + 0.5)

            if idx > max_idx:
                k0 = int(max_idx - 1)
                k1 = int(max_idx)
            elif idx == 0:
                k0 = int(0)
                k1 = int(1)
            else:
                k0 = int(idx - 1)
                k1 = int(idx)

            d = (x - xp[k0]) / (xp[k1] - xp[k0])
            rel = numba.float64(yp[k0] * (1.0 - d) + yp[k1] * d)

            bIdx = bI + bcount[0] * (bJ + bK * bcount[1])
            blocks[bIdx] += rel

    return blocks

def run_volume(fd):
    start = time.time()
    vol_min, vol_max, vol_tot = volume_analysis_jit(fd)
    vol_end = time.time()
    vol_time = vol_end - start

    print(f"Min: {vol_min}, Max: {vol_max}")
    print(f"Total: {vol_tot}")
    print(f"Elapsed time: {vol_time}")
    return vol_min, vol_max, vol_tot

def run_block(fd,
        xp: np.ndarray,
        yp: np.ndarray,
        vmin: np.float64,
        vmax: np.float64,
        vdims: np.ndarray,
        bdims: np.ndarray,
        bcount: np.ndarray):
    """Run the block level analysis and return the relevancies as a list of np.float64
    """
    start = time.time()
    blocks = run_block_analysis(fd, xp, yp, vmin, vmax, vdims, bdims, bcount)
    volend = time.time()
    vol_time = volend - start
    print(f"Elapsed time: {vol_time}")

    block_vox_count = np.prod(args.block_dims)
    for i in range(len(blocks)):
        blocks[i] = blocks[i] / block_vox_count

    return blocks

def main():
    cargs = parse_args(sys.argv[1:])

    fd = np.memmap(cargs.raw, dtype=np.uint8, mode='r')
    tf_x = np.loadtxt(cargs.tf, dtype=np.float64, usecols=0, skiprows=1)
    tf_y = np.loadtxt(cargs.tf, dtype=np.float64, usecols=1, skiprows=1)
    vdims = np.array([cargs.vx, cargs.vy, cargs.vz], dtype=np.uint64)
    bcount = np.array([cargs.bx, cargs.by, cargs.bz], dtype=np.uint64)
    bdims = np.divide(vdims, bcount)
    blocks = np.zeros(cargs.bx * cargs.by * cargs.bz)

    print('Running volume analysis')
    vol_min, vol_max, vol_tot = run_volume(fd)

    print('Running relevance analysis')
    relevancies = run_block(fd, tf_x, tf_y, vol_min, vol_max, vdims, bdims, bcount)
    print(f"Elapsed time: {volTime}")

    print("Creating index file")
    vol_path, vol_name = os.path.split(cargs.raw)
    tr_path, tr_name = os.path.split(cargs.tf)
    vol_stats = volume.VolStats(min=vol_min, max=vol_max, avg=0.0, tot=vol_tot)
    volume = volume.Volume(vol_name, world_dims, world_origin, path=vol_path)
    world_dims = np.array((1.0, 1.0, 1.0), dtype=np.float64)
    world_origin = np.array((0.0, 0.0, 0.0), dtype=np.float64)
    blocks = create_blocks(relevancies)
    ifile = IndexFile({
        'world_dims': world_dims,
        'world_origin': world_origin,
        'vol_stats': vol_stats,
        'volume': volume,
        'tr_func': tf_name,
        'dtype': fd.dtype,
        'blocks': blocks,
        }

