# special objects for fully connected networks
import numpy as np
from activation_functions import *
from losses import *
from Errors import InputError
import os

class Neurone:
    def __init__(self, activation_function):
        self.activation_function = find_activation_function(activation_function)
        self.weights = None
        self.bias = np.random.random() - 0.5
        self.input = None
        self.output = None
        self.delta = None

    def __repr__(self):
        string = 'weights: ' + str(self.weights)
        string += '\nbias: ' + str(self.bias) + '\n'
        return string

    def calculate_z(self):
        z = self.bias
        for k in range(len(self.input)):
            z += self.input[k] * self.weights[k]
        return z

    def activate(self):
        self.output = self.activation_function(self.calculate_z())
        return self.output

class SensorsLayer:
    def __init__(self, n_neurones, activation_function):
        self.input = None
        self.output = None
        self.next_layer = None
        self.neurones = []
        for k in range(n_neurones):
            neurone_k = Neurone(activation_function)
            neurone_k.weights = [np.random.random() - 0.5]
            self.neurones.append(neurone_k)

    def __repr__(self):
        string = "Sensors layer:\n"
        for k in range(len(self.neurones)):
            neurone = self.neurones[k]
            string += f'neurone {k}\n'
            string += neurone.__repr__() + '\n'
        return string

    def compute_forward(self):
        output = []
        for k in range(len(self.input)):
            self.neurones[k].input = [self.input[k]]
            output.append(self.neurones[k].activate())
        self.output = output
        return self.output


class FCHiddenLayer:
    def __init__(self, n_neurones, activation_function):
        self.input = None
        self.output = None
        self.neurones = [Neurone(activation_function) for _ in range(n_neurones)]
        self.previous_layer = None
        self.next_layer = None

    def __repr__(self):
        string = "FC hidden layer:\n"
        for k in range(len(self.neurones)):
            neurone = self.neurones[k]
            string += f'neurone {k}\n'
            string += neurone.__repr__() + '\n'
        return string

    def feed_neurones(self):
        for neurone in self.neurones:
            neurone.input = self.input

    def compute_forward(self):
        output = []
        for neurone in self.neurones:
            output.append(neurone.activate())
        self.output = output
        return self.output

class OutcomeLayer:
    def __init__(self, n_neurones, activation_function):
        self.input = None
        self.output = None
        self.neurones = [Neurone(activation_function) for _ in range(n_neurones)]
        self.previous_layer = None


    def __repr__(self):
        string = "Outcome layer:\n"
        for k in range(len(self.neurones)):
            neurone = self.neurones[k]
            string += f'neurone {k}\n'
            string += neurone.__repr__() + '\n'
        return string

    def feed_neurones(self):
        for neurone in self.neurones:
            neurone.input = self.input

    def compute_forward(self):
        output = []
        for neurone in self.neurones:
            output.append(neurone.activate())
        self.output = output
        return self.output


class Network:
    def __init__(self):
        self.layers = []
        self.classes = None

    def show(self):
        string = ""
        for layer in self.layers:
            string += "\n==========\n" + layer.__repr__()
        print(string)

    def add(self, layer):
        if self.layers != []:
            self.layers[-1].next_layer = layer
            layer.previous_layer = self.layers[-1]
            for neurone in layer.neurones:
                neurone.weights = [np.random.random() - 0.5 for _ in range(len(layer.previous_layer.neurones))]
        self.layers.append(layer)

    def feed_forward(self, input):
        self.layers[0].input = input
        input = self.layers[0].compute_forward()
        for layer in self.layers[1:]:
            layer.input = input
            layer.feed_neurones()
            input = layer.compute_forward()
        return input

    def back_propagation(self, target, alpha, loss_function):
        output = self.layers[-1].output
        loss_prime = find_loss_function(loss_function)[1]
        for k in range(len(output)):
            neurone_k = self.layers[-1].neurones[k]
            activation_function = neurone_k.activation_function
            tk = target[k]
            yk = output[k]
            delta_k = loss_prime(tk, yk) * prime(activation_function)(neurone_k.calculate_z())
            neurone_k.delta = delta_k
            neurone_k.bias += alpha * delta_k
            for j in range(len(neurone_k.weights)):
                zj = neurone_k.input[j]
                neurone_k.weights[j] += alpha * delta_k * zj

        for l in range(len(self.layers)-2, -1, -1):
            layer = self.layers[l]
            for j in range(len(layer.neurones)):
                neurone_j = layer.neurones[j]
                delta_inj = 0
                for k in range(len(layer.next_layer.neurones)):
                    neurone_k = layer.next_layer.neurones[k]
                    delta_inj += neurone_k.delta * neurone_k.weights[j]
                delta_j = delta_inj * prime(activation_function)(neurone_j.calculate_z())
                neurone_j.delta = delta_j
                neurone_j.bias += alpha * delta_j
                for i in range(len(neurone_j.weights)):
                    xi = neurone_j.input[i]
                    neurone_j.weights[i] += alpha * xi * delta_j

    def add_classes(self, target):
        # to code
        return 0

    def forward_backward(self, x_train, y_train, alpha, loss_function):
        for i in range(len(x_train)):
            y_i = self.feed_forward(x_train[i])
            target_i = y_train[i]
            self.back_propagation(target_i, alpha, loss_function)

    def evaluate(self, x_test, y_test, loss_function, visible=True):
        err = 0
        global_loss_function = find_loss_function(loss_function)[2]
        for i in range(len(x_test)):
            xi = x_test[i]
            target_i = y_test[i]
            output_i = self.feed_forward(xi)
            err += global_loss_function(target_i, output_i)
        if visible:
            print(f"{loss_function} of the model: {err}")
        return err

    def fit(self, x_train, y_train, x_test, y_test, n_epochs, loss_function="mse", alpha=0.3):
        for epoch in range(n_epochs):
            if (epoch+1) % (n_epochs/10) == 0:
                print(f"epoch {epoch+1}/{n_epochs}")
                self.evaluate(x_test, y_test, loss_function)
            self.forward_backward(x_train, y_train, alpha, loss_function)

    def fit_err(self, x_train, y_train, x_test, y_test, max_err, loss_function="mse", alpha=0.3):
        epoch = 0
        print("Training...")
        while self.evaluate(x_test, y_test, loss_function, visible=False) > max_err:
            epoch += 1
            self.forward_backward(x_train, y_train, alpha, loss_function)
        print(f'Training completed in {epoch} epoch.')
        self.evaluate(x_test, y_test, loss_function, visible=True)

    def predict(self, input):
        if len(self.layers[-1].neurones) == 1:
            output = self.feed_forward(input)[0]
            if abs(output) < abs(1 - output):
                return 0
            else:
                return 1
