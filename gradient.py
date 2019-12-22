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
NUMBER = 7
circuit = spice.LoadFile('test_files/test_' + str(NUMBER) + '.cir')
input_data = spice.Init_Data(1000, 0.3, SNR=10**6)
analysis = spice.CreateCVC(circuit, input_data, 100)
ivc = [analysis.input_dummy, analysis.VCurrent]
spice.SaveFile(analysis, "test_files/test_" + str(NUMBER) + ".csv")


# Objective function
def f(x):
    global ivc
    _x = copy.copy(x)
    create_cir('test_files/test_res_' + str(NUMBER), _x)
    circuit = spice.LoadFile('test_files/test_res_' + str(NUMBER) + '.cir')
    input_data = spice.Init_Data(1000, 0.3, SNR=10**6)
    analysis = spice.CreateCVC(circuit, input_data, 100)
    ivc_2 = [analysis.input_dummy, analysis.VCurrent]
    return compare_ivc(ivc, ivc_2)


# Derivative
def f1(x):
    v_l = copy.copy(x)
    v_r = copy.copy(x)
    prod = []
    for i in range(len(x)):
        v_l[i] = v_l[i] - x[i] / 10 ** 7 if x[i] >= 1.e-12 else 1e-13
        v_r[i] = v_r[i] + x[i] / 10 ** 7 if x[i] >= 1.e-12 else 2e-12
        prod.append((f(v_r) - f(v_l)) / (2 * 10 ** -7))
        v_l[i] = x[i]
        v_r[i] = x[i]
    # print(prod)
    return np.array(prod)


def gradient_method(f, fprime, x0, epsi=1e-5):
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
    xk = x0
    # x = 1e-6
    # y = 1e3
    x = 1e-7
    y = 1e5
    line = [f(xk)]
    min = 1
    penalty = 0
    while f(xk) > epsi:
        k += 1
        _alfa = np.array([x, x, x, x, x, y, y, y, y, y, y, y, y])
        f_prev = f(xk)
        xk = xk - _alfa * fprime(xk)
        for i in range(len(xk)):
            if i < 5:
                xk[i] = 1e-12 if xk[i] < 1e-12 else xk[i]
                xk[i] = 1e-6 if xk[i] > 1e-6 else xk[i]
            else:
                xk[i] = 1e-12 if xk[i] < 0e-12 else xk[i]
                xk[i] = 1e6 if xk[i] > 1e6 else xk[i]

        # if (abs(f(xk) - f_prev) < 1e-3 * f(xk)) and f(xk) < 0.5:
        #    penalty += 1
        # else:
        #     penalty = 0
        if penalty > 10:
            y *= 0.5
            penalty = 0
            print("Penalty: y = %f" % y)
        if f(xk) < min:
            min = f(xk)
            x_min = xk
        if k > 4000:
            f(x_min)
            return x_min, min
        line.append(f(xk))
        print(k, f(xk))
    figure1 = plt.figure(1, (10, 5))
    plt.clf()
    plt.plot(np.linspace(1, len(line), len(line)), line)
    plt.xlabel('Iter')
    plt.ylabel('Error')
    plt.savefig("error.png")
    plt.show()
    return xk, f(xk)


# result, price = gradient_method(f, f1, np.array([1.e-06, 1.e-06, 1.e-06, 2.22e-12, 2.22e-12,
#                                                  1, 1e6, 1e6, 1, 1e6, 1, 1e6, 1e6]))
# result, price = gradient_method(f, f1, np.array([1.e-06, 1.e-06, 1.e-06, 2.22e-12, 2.22e-12,
#                                                  1, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6]))

result, price = gradient_method(f, f1, np.array([1.e-7, 1.e-6, 1.e-6, 2.22e-7, 2.22e-7,
                                                 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2]))
print('Result of Gradient method:')
print('Final Result (best point): %s' % result)
print('Final price: %f' % price)
