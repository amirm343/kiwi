import numpy as np
from base import base_layer

class linearizer(base_layer):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, data, *args):
        # print(data.shape)
        res = np.sum(data, 0)
        # print(res.shape)
        res = res.reshape(1, 1,-1)
        # print(res.shape)
        return res
    
    def backward(self, *args):
        return args