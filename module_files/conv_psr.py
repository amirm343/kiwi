import numpy as np
from conv import qconv
from psr import psr_layer

class qconv_psr(qconv, psr_layer): # qconv_layer with parameter shift rule

    def __init__(self, kernel, kernel_size: int, weight, lr: float = 0.01) -> None:
        super().__init__(kernel, kernel_size, weight, lr)

        # TODO: باید گرادیان حساب کنیم و وزنارو آپدیت کنیم
    def backward(self, *args):
        # WARN: همچنان برای صرف همون فرمت کار میکنه
        self.gradient = np.zeros(self.weight.shape)

        for i, j in enumerate(self.weight):
            self.gradient[i] = self.grader(i, j)

        self.gradient = self.gradient
        self.weight -= self.gradient*self.lr

        return args
    
    def grader(self, index, value):
        differ = np.zeros(self.weight.shape)
        differ[index] = 0.001*value # -> UPSILON * THETA

        up_weights = self.weight+differ
        down_weights = self.weight-differ

        up_cost = []
        down_cost = []

        for i in self.last_label:
            up_cost.append(self.cost(
                            i, 
                            self.ron.res(self.forward(input_theta=up_weights))
                            ))
            down_cost.append(self.cost(
                            i, 
                            self.ron.res(self.forward(input_theta=down_weights))
                            ))
        
        up_cost = np.array(up_cost)
        down_cost = np.array(down_cost)

        res = (sum(up_cost)-sum(down_cost))/(2*0.001*len(self.last_label))
        
        return res
        # return (up_cost-down_cost)/(2*0.001)