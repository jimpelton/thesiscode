

import numpy as np

class Block:
    def __init__(self, dims, origin, voxdims):
        self.world_dims = dims
        self.origin = origin
        self.voxdims = voxdims
        self.rel = 0


def create_blocks(relevancies):

