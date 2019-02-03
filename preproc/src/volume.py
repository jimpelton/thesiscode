

class VolStats:
    def __init__(self, min=0.0, max=1.0, avg=0.0, tot=0.0):
        self.min = min
        self.max = max
        self.avg = avg
        self.tot = tot

class Volume:
    def __init__(self, name, world_dims, world_origin, vox_dims, path=''):
        self.name = name
        self.path = ''
        self.world_dims = world_dims
        self.world_origin = world_origin
        self.vox_dims = vox_dims
