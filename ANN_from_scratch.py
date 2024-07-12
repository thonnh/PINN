import numpy as np

class Dense():
    def __init__(self,input_size,output_size):
        self.weights = np.random.randn(output_size,input_size)
        self.bias    = np.random.randn(output_size,1)
    def forward(self,input):
        self.input   = input
        return np.dot(self.weights,self.input) + self.bias
    def backward(self,output_grad,lr):
        weights_grad  = np.dot(output_grad, self.input.T)
        self.weights -= lr*weights_grad
        self.bias    -= lr*output_grad
        return np.dot(self.weights.T,output_grad)
    
class Activation():
    def __init__(self, activation, activation_prime):
        self.activation       = activation
        self.activation_prime = activation_prime
    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    def backward(self, output_grad, lr):
        return np.multiply(output_grad,self.activation_prime(self.input))
    
class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(tanh,tanh_prime)
        
def mse(y_true,y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true,y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

X = np.reshape([[0,0],[0,1],[1,0],[1,1]],(4,2,1))
Y = np.reshape([[0],[1],[1],[0]],(4,1,1))

network = [
    Dense(2, 3),
    Tanh(),
    Dense(3,1),
    Tanh()
    ]

epochs = 10000
lr = 0.1

for e in range(epochs):
    error = 0
    for x,y in zip (X,Y):
        output = x
        for layer in network:
            output = layer.forward(output)
            
        error += mse(y,output)
        
        grad = mse_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, lr)
    
    error /= len(x)
    print('%d/%d,error=%f' % (e * 1, epochs, error))
    