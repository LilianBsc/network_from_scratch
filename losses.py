import numpy as np

def mse_global(target, output):
    #Mean Square Error on a vector
    target_vec = np.array(target)
    output_vec = np.array(output)
    return np.mean(np.power(target_vec - output_vec, 2))

def mse(tk, yk):
    return 0.5 * (tk - yk)**2

def mse_prime(tk, yk):
    return tk - yk

def find_loss_function(loss_function):
    if loss_function == 'mse':
        return mse, mse_prime, mse_global
    else:
        raise InputError('in back_propagation', f"{loss_function} unknown. The activation function must be 'mse'.")
