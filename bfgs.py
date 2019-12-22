import numpy as np
import numpy.linalg as ln
import scipy as sp
import scipy.optimize
import PySpice.Logging.Logging as Logging
from MySpice import MySpice as spice
import pyswarms as ps
import pandas as pd
from CirCreate import create_cir
from compare_ivc import compare_ivc
from matplotlib import pyplot as plt
import copy
circuit = spice.LoadFile('3_win.cir')
input_data = spice.Init_Data(1000, 0.3, SNR=10**6)
analysis = spice.CreateCVC(circuit, input_data, 100)
ivc = [analysis.input_dummy, analysis.VCurrent]
spice.SaveFile(analysis, "3_win.csv")


# Objective function
def f(x):
    global ivc
    _x = copy.copy(x)
    # for i in range(len(x)):
    #     _x[i] = x[i] * (1e-6 - 1e-12) + 1e-12 if i < 5 else x[i] * 10e6
    create_cir('test', _x)
    circuit = spice.LoadFile('test.cir')
    input_data = spice.Init_Data(1000, 0.3, SNR=10**6)
    analysis = spice.CreateCVC(circuit, input_data, 100)
    # spice.SaveFile(analysis, "test.csv")
    ivc_2 = [analysis.input_dummy, analysis.VCurrent]
    return compare_ivc(ivc, ivc_2)


# Derivative
def f1(x, k):
    v_l = copy.copy(x)
    v_r = copy.copy(x)
    prod = []
    for i in range(len(x)):
        if i == k:
            v_l[i] = v_l[i] - x[i] / 10 ** 3 if x[i] != 0 else 0e0
            if i < 5:
                v_r[i] = v_r[i] + x[i] / 10 ** 3 if x[i] != 0 else 10 ** -15
            else:
                v_r[i] = v_r[i] + x[i] / 10 ** 3 if x[i] != 0 else 10
            prod.append((f(v_r) - f(v_l)) / (v_r[i] - v_l[i]))
            print((f(v_r) - f(v_l)) / (v_r[i] - v_l[i]))
            v_l[i] = x[i]
            v_r[i] = x[i]
        else:
            prod.append(0.0)
    print(prod)
    return np.array(prod)


def bfgs_method(f, fprime, x0, maxiter=None, epsi=10e-3):
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

    if maxiter is None:
        maxiter = len(x0) * 200

    # initial values
    k = 0
    xk = x0
    alfa = 10 ** -11
    line = [f(xk)]
    while f(xk) > epsi:
        k += 1
        # alfa *= 0.1 if k % 20 == 0 else 1
        # alfa *= 0.1 if k % 8 == 0 else 1
        # alfa *= 0.1 if k == 50 else 1
        # print(xk)
        # for i in range(len(xk)):
        #     xk[i] = (xk[i] - 1e-12) / (1e-6 - 1e-12) if i < 5 else (xk[i]) / 10e6
        # print(xk)
        xk = xk - alfa * fprime(xk, 3)
        print(xk)
        if k > 200:
            break
        line.append(f(xk))
    # print(len(line))
    figure1 = plt.figure(1, (10, 5))
    plt.clf()
    plt.plot(np.linspace(1, len(line), len(line)), line)
    plt.xlabel('Iter')
    plt.ylabel('Error')
    plt.savefig("error.png")
    plt.show()
    print(line)
    return (xk, k)

# print(1e-6 + 1e-9)
result, k = bfgs_method(f, f1, np.array([1e-7, 1e-12, 1e-6, 2.22e-9, 2.22e-12, 1, 10e6, 10e6, 10e6, 10e6, 10e6, 10e6, 10e6]))

# print('Result of BFGS method:')
print('Final Result (best point): %s' % (result))
# print('Iteration Count: %s' % (k))