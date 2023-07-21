from base import base_layer

class psr_layer(base_layer):
    
    def __init__(self) -> None:
        super().__init__()

    def grader(self):
        raise NotImplementedError("grader is not implemented")
    
    # NOTE: کاست رو یطوری باید بنویسیم که نتیجه غایی رو بدیم بهش و کاست ر بده بهمون
    # کاست رو باید توی نتورک بیایم برای نود تعریف کنیم!
    def cost(self):
        raise NotImplementedError("cost is not implemented")
    
    # REST OF NET
    def ron(self):
        raise NotImplementedError("rest of net is not implemented")