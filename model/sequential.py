class Sequential:
    def __init__(self, layers):
        self.layers = layers
        self.optimizer = None
        self.loss = None
        self.metrics = None

    def compile(self, optimizer, loss):
        self.loss = loss
        self.optimizer = optimizer

    def _forward(input):
        pass

    def _backward(input):
        pass

    def fit(x_train, y_train, x_valid, epochs, batch_size):
        pass

    def predict(x_test):
        pass


