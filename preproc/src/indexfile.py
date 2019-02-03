
from typing import List
import numpy as np
from block import Block


class IndexFile:
    def __init__(self, from_file=None, **kwargs):
    """
    Keyword Arguments:
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
            ifile = open_from(from_file)
        elif kwargs is not None:
            volume = {
                    'world_dims': kwargs['world_dims'],
                    'world_origin': kwargs['world_origin'],
                    }

            vol_stats = {
                    'min': kwargs['vol_min'],
                    'max': kwargs['vol_max'],
                    'averate': kwargs['vol_avg'],
                    'total': kwargs['vol_total'],
                    }

            ifile = {
                    'version': 1,
                    'vol_name': kwargs['vol_name'],
                    'vol_path': kwargs['vol_path'],
                    'tr_func': kwargs['tr_func'],
                    'dtype': kwargs['dtype'],
                    'volume' : volume,
                    'vol_stats': vol_stats,
                    'blocks': kwargs['blocks']
                    }

            self.index_file = ifile


    def writeOut(self, file_name):
        # write header
        
        #write blocks body

    def open_from(from_file):
        pass



class FileBlock(Block):
    def __init__(self, **kwargs):
        super(Block, self).__init__(kwargs['dims'], kwargs['origin'], kwargs['voxdims'])

        self.index = kwargs['index']
        self.ijk = kwargs['ijk']
        self.offset = kwargs['offset']

def create_file_blocks(nblocks, volume: Block) -> List[FileBlock]:
    blk_dims_world = np.divide(volume.world_dims, nblocks)
    blk_dims_vox = np.divide(volume.voxdims, nblocks)
    blk_dims_prod = np.prod(blk_dims_vox)
    blocks = []
    index = 0
    for k in range(nblocks[2]):
        for j in range(nblocks[1]):
            for i in range(nblocks[0]):
                world_loc = (volume.world_dims * [i, j, k]) - (volume.worl_dims * 0.5)
                origin = world_loc + (world_loc + blk_dims_world) * 0.5
                start_vox = [i, j, k] * blk_dims_prod

                blk_args = {
                        'dims': blk_dims_world,
                        'origin': origin,
                        'voxdims': blk_dims_vox,
                        'index': index,
                        'ijk': np.array([i,j,k]),
                        'offset': i + volume.voxdims[0] * (j + volume.voxdims[1] * k)
                        }

                blocks.append(FileBlock(blk_args))

                index += 1

    return blocks

