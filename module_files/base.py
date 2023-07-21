class base_layer:

    def __init__(self) -> None:
        pass

    def forward(self, *args):
        raise NotImplementedError("forward is not implemented")
    
    def backward(self, *args):
        raise NotImplementedError("backward is not implemented")