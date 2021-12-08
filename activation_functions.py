import numpy as np

def tanh(x) :
    return np.tanh(x)

def tanh_prime(x) :
    return 1 - np.tanh(x)**2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return max(0, x)

def ident(x):
    return x

def ident_prime(x):
    return 1

def prime(activation_function):
    L = [
            (tanh, tanh_prime),
            (sigmoid, sigmoid_prime),
            (ident, ident_prime)
    ]
    for el in L:
        if el[0] == activation_function:
            return el[1]

def find_activation_function(activation_function):
    if activation_function == 'sigmoid':
        return sigmoid
    elif activation_function == 'tanh':
        return tanh
    elif activation_function == 'relu':
        return relu
    elif activation_function == 'ident':
        return ident
    else:
        raise InputError('in SensitiveLayer', f"{activation_function} unknown. The activation function must be 'sigmoid', 'relu', 'tanh', ident.")
