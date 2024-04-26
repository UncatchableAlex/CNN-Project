import numpy as np
# https://arxiv.org/pdf/1412.6980.pdf
class Adam:
    def __init__(self, beta1=0.9, beta2=0.999, alpha=0.1, epsilon=1e-8):
        # decay rate of the first moment
        self.beta1 = beta1
        # decay rate of the second moment
        self.beta2 = beta2
        # learning rate:
        self.alpha = alpha
        # small value:
        self.epsilon = epsilon
        # the time
        self.t = 0
        # the first moment
        self.m = 0
        # the second moment
        self.v = 0

    # blame for layer is a column vector with the gradients of each node in the layer w.r.t the final output
    def update(self, deriv_of_layer_wrt_final_output):
        self.t += 1
        g = deriv_of_layer_wrt_final_output
        self.m = self.beta1*self.m + (1 - self.beta1)*g
        self.v = self.beta2*self.v + (1 - self.beta2)*(g**2)
        m_hat = self.m/(1 - (self.beta1**self.t))
        v_hat = self.v/(1 - (self.beta2**self.t))
        return self.alpha*m_hat/(np.sqrt(v_hat) + self.epsilon)
        

        

        