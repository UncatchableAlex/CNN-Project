from model.layers.dense import Dense

class Sequential:
    def __init__(self):
        self.layers = []
    
    def add(self, layer):
        self.layers.append(layer)
    
    def forward(self, input, index=0):
        if index >= len(self.layers):
            return input
        output = self.layers[index].forward(input)
        return self.forward(output, index + 1)
    
    def backward(self, output, index=None):
        if index is None:
            index = len(self.layers) - 1
        if index < 0:
            return output
        grad = self.layers[index].backward(output)
        return self.backward(grad, index - 1)

    def compile(self,optimizer):
        for layer in self.layers:
            layer.compile(optimizer=optimizer)
            
            
    