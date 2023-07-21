import numpy as np
from base import base_layer

class conv(base_layer):

    def __init__(self) -> None:
        super().__init__()

    def convolving(self):
        raise NotImplementedError("convolving is not implemented")
    

class qconv(conv):
    # TODO: add ACTIVATION, STRID and PAD
    # if you like DO: iylDO: add upsilon value
    def __init__(self, kernel, kernel_size: int, weight, lr: float = 0.01) -> None:
        self.kernel = kernel
        self.kernel_size = kernel_size
        self.weight = weight
        # self.activator = activator
        # self.strid = strid
        # self.pad = pad
        self.lr = lr

    # NOTE: این صرفا یه ورودی می‌گیره، دیتاست ر شبکه می‌گیره
    def forward(
            self, input_data=None, # صرفا موقع فوروارد جدید مقدار داره
            input_label=None, # صرفا موقع فوروارد جدید مقدار داره
            input_theta=None # صرفا موقع پارامتر شیقت رول مقدار داره
                ):
        
        # NOTE: اگه دیتای ورودی جدیدی نداشته باشیم لست اینپوتو استفاده می‌کنه
        if isinstance(input_data, type(None)):
        # if input_data  == None:
            data = self.last_data
        # -NOTE: در غیر این‌صورت لست اینپوتو آپدیت می‌کنه و از ورودی جدید استفاده می‌کنه
        else:
            data = self.last_data = input_data

        if isinstance(input_label, type(None)):
            label = self.last_label
        else:
            label = self.last_label = input_label

        # NOTE: اگه تتای جدیدی نباشه از تتای درونی استفاده می‌کنه
        # if input_theta == None:
        if isinstance(input_theta, type(None)):
            theta = self.weight
        # -NOTE: در غیر این‌صورت از تتاهای جدید استفاده می‌کنه و <<البته>> تتای درونی رو آپدیت نمیکنه
        # چون آپدیت تتا یه پروسه کلا جداست
        else:
            theta = input_theta
        
        # WARN: این با یکمدار بعنوان کرنل چند کرنل رو اجرا می‌کنه

        dd, xd, yd = data.shape

        mxd = xd-(self.kernel_size-1)
        myd = yd-(self.kernel_size-1)

        output = np.zeros((self.weight.shape[0], dd, mxd, myd))
        

        for d in range(dd):
            for i, j in enumerate(theta):
                output[i][d] = self.convolving(data[d], j)

        output = np.sum(output, 0)

        # return output.numpy()
        return output
    
    def convolving(self, data, param):

        xd, yd = data.shape

        # WARNING: این فقط برای یک قدم یک قدم بدون پدینگ جلو رفتنه
        mxd = xd-(self.kernel_size-1)
        myd = yd-(self.kernel_size-1)
        map = np.zeros((mxd, myd))

        for row in range(mxd):
            for column in range(myd):
                # NOTE: یه تارگت برمی‌داریم
                target = data[row: row+self.kernel_size, column: column+self.kernel_size]
                res = self.kernel(
                                data=(np.pi/(target.reshape(self.kernel_size*self.kernel_size)+1)), 
                                # phi=param
                                phi=[1,1,1,1,1,1,1,1,1]
                                ).numpy()
                map[row, column] = res

        # return map.numpy()
        return np.array(map)
    
    # TODO: باید گرادیان حساب کنیم و وزنارو آپدیت کنیم
    def backward(self, *args):
        raise NotImplementedError("backward is not implemented")