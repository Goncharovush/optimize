import numpy as np
from MySpice import MySpice as spice
from CirCreate import create_cir
from compare_ivc import compare_ivc
from matplotlib import pyplot as plt
import copy
xmin_c = 1e-12
xmax_c = 1e-6
xmin_r = 0e0
xmax_r = 1e6

# Objective function


def f(x):
    _x = copy.copy(x)
    _x = denorm(_x)
    _x[:5] = 1 / _x[:5]
    _x[5:] = 1 / (_x[5:] * _x[5:])
    _y = np.sum(_x) ** 2
    return _y


# Derivative
def f1(x):
    v_l = copy.copy(x)
    v_r = copy.copy(x)
    prod = []
    for i in range(len(x)):
        v_l[i] = v_l[i] - x[i] / 10 ** 3 if x[i] != 0 else 0e0
        v_r[i] = v_r[i] + x[i] / 10 ** 3 if x[i] != 0 else 1e-6
        prod.append((f(v_r) - f(v_l)) / (v_r[i] - v_l[i]))
        # print(f(v_r) - f(v_l))
        v_l[i] = x[i]
        v_r[i] = x[i]

    return np.array(prod)


def norm(x):
    global xmin_r, xmin_c, xmax_c, xmax_r
    _x = copy.copy(x)
    for i in range(len(x)):
        x_min = xmin_r if i >= 5 else xmin_c
        x_max = xmax_r if i >= 5 else xmax_c
        _x[i] = (x[i] - x_min)/(x_max - x_min)
    return _x


def denorm(x):
    global xmin_r, xmin_c, xmax_c, xmax_r
    _x = copy.copy(x)
    for i in range(len(x)):
        x_min = xmin_r if i >= 5 else xmin_c
        x_max = xmax_r if i >= 5 else xmax_c
        _x[i] = x_min + x[i] * (x_max - x_min)
    return _x


def bfgs_method(f, fprime, x0, epsi=1e-3):
    """
    Minimize a function func using the BFGS algorithm.

    Parameters
    ----------
    func : f(x)
        Function to minimise.
    x0 : ndarray
        Initial guess.
    fprime : fprime(x)
        The gradient of `func`.
    """
    # initial values
    k = 0
    xk = norm(x0)
    # xk = x0
    # print(f(xk))
    fprime(xk)
    _alfa = 0.1
    line = [f(xk)]
    while f(xk) > epsi:
        k += 1
        # print(_alfa * fprime(xk))
        xk = xk - _alfa * fprime(xk)
        # print(denorm(xk))
        if k > 50000:
            break
        line.append(f(xk))
        # print(denorm(xk))
        print(f(xk))
    figure1 = plt.figure(1, (10, 5))
    plt.clf()
    plt.plot(np.linspace(1, len(line), len(line)), line)
    plt.xlabel('Iter')
    plt.ylabel('Error')
    plt.savefig("error.png")
    plt.show()
    # print(line)
    return (denorm(xk), k)

result, k = bfgs_method(f, f1, np.array([1e-6, 1e-12, 1e-6, 2.22e-7, 2.22e-12, 1, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e6, 1e6]))

# print('Result of BFGS method:')
print('Final Result (best point): %s' % (result))
# print('Iterati