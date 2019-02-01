

import numpy as np

class Block:
    __init__(self, dims, origin, voxdims):
        self.world_dims = dims
        self.origin = origin
        self.voxdims = voxdims
        self.rel = 0

