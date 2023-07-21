import numpy as np
from base import base_layer

class pooling2d(base_layer):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_data, *args):
        mdd = input_data.shape[0]
        mxd = ((input_data.shape[1]-self.kernel_size)/2)+1
        myd = ((input_data.shape[2]-self.kernel_size)/2)+1
        mdd = int(mdd)
        mxd = int(mxd)
        myd = int(myd)
        map = np.zeros((mdd, mxd, myd))

        for d in range(mdd):
            for row in range(mxd):
                for column in range(myd):
                    target = input_data[d][row: row+self.kernel_size, column: column+self.kernel_size]
                    res = self.pooling(target)
                    map[d][row, column] = res
        map = np.array(map)
        # map = map.reshape(-1,1)
        return map

    def pooling(self):
        raise NotImplementedError("pooling is not implemented")
    
    def backward(self, *args):
        return args
    

class max_pooling2d(pooling2d):
    # TODO: add STRID and PAD
    # WARN: این فثط برای استرید ۲ کار می‌کنه
    def __init__(self, size=2) -> None:
        self.kernel_size = size
        # self.activator = activator
        # self.strid = strid
        # self.pad = pad
    
    def pooling(self, t):
        return np.max(t)