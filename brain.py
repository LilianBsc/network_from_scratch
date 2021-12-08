from objects import *

data = [
        [1, 1, 1, 0, 1],
        [0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
        ]

x = [sample[:-1] for sample in data]
y = [[sample[-1]] for sample in data]

brain = Network()

brain.add(SensorsLayer(4, "sigmoid"))
brain.add(FCHiddenLayer(3, "sigmoid"))
brain.add(OutcomeLayer(1, "sigmoid"))

brain.fit_err(x, y, x, y, 0.01, 'mse')
print(brain.predict([1, 1, 1, 0]))
print(brain.predict([0, 1, 0, 0]))
print(brain.predict([1, 1, 1, 1]))
print(brain.predict([0, 0, 0, 1]))
print(brain.predict([0, 0, 0, 0]))
