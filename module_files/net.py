import numpy as np

class Seq:

    def __init__(self, funcs):
        self.funcs = funcs

    def res(self, input):
        res = input
        for i in self.funcs:
            res = i.forward(res, res)
        return res
    

class PSR_Net:
# WARN: صرفا برای مدارهایی که تمامشون با پارامتر شیقت رول باشه( یا از اول تا یجایی جتما باشه)

    # WARN: اینجا م دیتاست نمی‌گیریم و یدوه دیتا میگیریم. دیتاست رو توی ترین میگیریم
    def __init__(self, network, cost) -> None :
        self.network = network
        self.cost = cost
        self.predict()

    # WARN: این قضیه بهینه نیست و اگه لایه ای از پارامنر شیفت استفاده نکنه کلی کاست و آراوان اضافه ایجاد می‌کنه
    def predict(self):
        for i, j in enumerate(self.network):
            j.cost = self.cost
            j.ron = Seq(self.network[i+1:])

    # TODO: صحت و کاست و تایم و اینارم برگردون
    def train(self, dataset, labelset, batchs):
        b = batchs
        d = dataset.shape[0]
        while d%b != 0:
            b -= 1

        for i, j in zip(np.split(dataset, b), np.split(labelset, b)):
            self.forward(i, j)
            self.backward()

    def forward(self, data, label):
        # print(label)
        res = data
        for i, j in enumerate(self.network):
            res = j.forward(res, label)
            print("step: ", i)

        return res
    
    def backward(self):
        back_g = 0
        for i, j in enumerate(reversed(self.network)):
            back_g = j.backward(back_g)
            print("step: ", i)