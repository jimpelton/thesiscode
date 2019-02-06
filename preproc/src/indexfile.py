
from typing import List
import numpy as np
import json
from volume import Volume

class IndexFile:
    def __init__(self, from_file=None, **kwargs):
        """Keyword Arguments:
            - world_dims: the world dimensions of this volume
            - world_origin: the world origin of the volume (likely (0,0,0))
            - vol_min: min value of the volume
            - vol_max: max value of the volume
            - vol_avg: avg value of the volume
            - vol_total: total value of the voxel scalar values in the volume
            - vol_name: volume file name
            - vol_path: path to volume
            - tr_func: name of transfer function
            - dtype: data type of the volume data set
            - blocks: a list of dicts that are the blocks
        """
        ifile = {}
        if from_file is not None:
            ifile = self.open_from(from_file)
        elif kwargs is not None:
            vol = kwargs['volume']
            vol_stats = kwargs['vol_stats']

            ifile = {
                    'version': 1,
                    'vol_name': vol.name,
                    'vol_path': vol.path,
                    'tr_func': kwargs['tr_func'],
                    'dtype': kwargs['dtype'],
                    'volume' : vol.__dict__,
                    'vol_stats': vol_stats.__dict__,
                    'blocks': kwargs['blocks']
                    }

        self.index_file = ifile


    def write(self, file_name):
        with open(file_name, 'w') as f:
            f.write(json.dumps(self.index_file, indent="  "))

    def open_from(self, from_file):
        pass


class FileBlock:
    def __init__(self, **kwargs):
        self.dims = tuple(kwargs['dims'])
        self.origin = tuple(kwargs['origin'])
        self.vox_dims = tuple(kwargs['vox_dims'])
        self.index = kwargs['index']
        self.ijk = tuple(kwargs['ijk'])
        self.offset = int(kwargs['offset'])
        self.rel = float(kwargs['rel'])

def to1D(col, row, slab, maxCols, maxRows):
    return int(col + maxCols * (row + maxRows * slab))

def create_file_blocks(nblocks, dtype, vol: Volume, rels) -> List[FileBlock]:
    blk_dims_world = np.divide(vol.world_dims, nblocks)
    blk_dims_vox = np.array(np.divide(vol.vox_dims, nblocks), dtype=np.uint64)

    blocks = []

    for k in range(nblocks[2]):
        for j in range(nblocks[1]):
            for i in range(nblocks[0]):
                ijk = np.array([i,j,k], dtype=np.uint64)

                # this blocks location within the volume (world coords)
                world_loc = blk_dims_world * ijk - 0.5

                # block center in world coords
                origin = world_loc + (world_loc + blk_dims_world) * 0.5

                blkIdx = to1D(i, j, k, nblocks[0], nblocks[1]) #int(i + nblocks[0] * (j + nblocks[1] * k))

                # block start voxel
                start_vox = blk_dims_vox * ijk
                offset = dtype.itemsize * \
                        to1D(start_vox[0], start_vox[1], start_vox[2], vol.vox_dims[0],
                                vol.vox_dims[1])

                blk_args = {
                        'dims': blk_dims_world.tolist(),
                        'origin': origin.tolist(),
                        'vox_dims': blk_dims_vox.tolist(),
                        'index': blkIdx,
                        'ijk': ijk.tolist(),
                        'offset': offset,
                        'rel': float(rels[blkIdx])
                        }

                blocks.append(blk_args)

    return blocks

