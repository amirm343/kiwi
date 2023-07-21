import numpy as np
from base import base_layer

class actv(base_layer):

    def __init__(self) -> None:
        super().__init__()

    def actv_func(self):
        raise NotImplementedError("activation function is not implemented")
    

def sf(x):
    return np.power(np.e, x)


class soft_max_1d(actv):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_data, *args):
        return self.actv_func(input_data)

    def actv_func(self, data):
        a = list(map(sf, data))
        s = sum(a)
        res = []
        for i in a:
            res.append(i/s)
        return res
    
    def backward(self, *args):
        return args
    

def sm(x):
    return (1/(1+np.power(np.e, -x)))

class sigmoid(actv):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_data, *args):
        return self.actv_func(input_data)

    def actv_func(self, data):
        return list(map(sm, data))
    
    def backward(self, *args):
        return args