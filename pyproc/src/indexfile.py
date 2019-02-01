
from typing import List
import numpy as np
from block import Block


class IndexFile:
    def __init__(self, **kwargs):
        self.if_name = kwargs['fname']
        self.vol_name = kwargs['volname']
        self.vol_path =  kwargs['volpath']
        self.tf_name = kwargs['tfname']
        self.volume = kwargs['volume']
        self.volstats = kwargs['volstats']
        self.blocks = kwargs['blocks']

    def writeOut(self, file_name):
        # write header

        #write blocks body



class FileBlock(Block):
    def __init__(self, **kwargs):
        super(Block, self).__init__(kwargs['dims'], kwargs['origin'], kwargs['voxdims'])

        self.index = kwargs['index']
        self.ijk = kwargs['ijk']
        self.offset = kwargs['offset']

def CreateFileBlocks(nblocks, volume: Block) -> List[FileBlock]:
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

