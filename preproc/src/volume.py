

class VolStats:
    def __init__(self, min=0.0, max=1.0, avg=0.0, tot=0.0):
        self.min = float(min)
        self.max = float(max)
        self.avg = float(avg)
        self.tot = float(tot)

class Volume:
    def __init__(self, world_dims, world_origin, vox_dims, rov_min, rov_max):
        self.world_dims = tuple(world_dims)
        self.world_origin = tuple(world_origin)
        self.vox_dims = tuple(vox_dims)
        self.rov_min = float(rov_min)
        self.rov_max = float(rov_max)

