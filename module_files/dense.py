import numpy as np
from base import base_layer

class dense(base_layer):
    
    # TODO: add the feature that make initial weights by the way
    def __init__(self, in_dim, out_dim, in_weights, lr = 0.01) -> None:
        self.lr = lr
        self.weight = in_weights
        self.in_dim = in_dim
        self.out_dim = out_dim

class classic_dense(dense):

    def __init__(self, in_dim, out_dim, in_weights, lr=0.01) -> None:
        super().__init__(in_dim, out_dim, in_weights, lr)

    def forward(self, input_data, *args):
        self.last_data = input_data
        output = np.matmul(input_data, self.weight)
        return output

    
    def backprop(self, grad_previous):
        self.grad = np.matmul((self.X.transpose()), grad_previous)/self.in_dim
        self.weight = self.weight - (self.lr*self.grad)
        output = np.matmul(grad_previous, self.Theta.transpose())
        return output
    
class classic_dense_psr(dense):

    def __init__(self, in_dim, out_dim, in_weights, lr=0.01) -> None:
        super().__init__(in_dim, out_dim, in_weights, lr)

    def forward(
            self, input_data=None,
            input_label=None,
            input_theta=None):

        
        # NOTE: اگه دیتای ورودی جدیدی نداشته باشیم لست اینپوتو استفاده می‌کنه
        if isinstance(input_data, type(None)):
            data = self.last_data
        # -NOTE: در غیر این‌صورت لست اینپوتو آپدیت می‌کنه و از ورودی جدید استفاده می‌کنه
        else:
            data = self.last_data = input_data

        if isinstance(input_label, type(None)):
            label = self.last_label
        else:
            label = self.last_label = input_label

        # NOTE: اگه تتای جدیدی نباشه از تتای درونی استفاده می‌کنه
        if isinstance(input_theta, type(None)):
            theta = self.weight
        # -NOTE: در غیر این‌صورت از تتاهای جدید استفاده می‌کنه و <<البته>> تتای درونی رو آپدیت نمیکنه
        # چون آپدیت تتا یه پروسه کلا جداست
        else:
            theta = input_theta

        output = []
        for d in data:
            output.append(np.matmul(d, self.weight))
        
        output = np.array(output)

        # print(output.shape)
        # print(label.shape)

        return np.sum(output, 0)
    

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
        # return (up_cost-down_cost)/(2*0.001) # > /2*upsilon