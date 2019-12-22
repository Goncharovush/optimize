# Import modules
import numpy as np
import PySpice.Logging.Logging as Logging
from MySpice import MySpice as spice
import pyswarms as ps
import pandas as pd
from CirCreate import create_cir
from compare_ivc import compare_ivc
from matplotlib import pyplot as plt
import itertools
import time
import pyswarms.utils.search

_x = np.ndarray(shape=(30, 5), dtype=float)
_y = np.ndarray(shape=(30, 8), dtype=int)
c = 0  # value for definition name picture corresponding number of iteration
time_func_0 = 0
time_func_1 = 0


def get_ivc(name):
    """
    This function read csv-file with iv-curve
    :param name: name of file
    :return: iv-curve
    """
    data = pd.read_csv(str(name)+".csv", sep=';', encoding='latin1')
    voltage = list(data)
    current = data.values.tolist()
    current = [float(n) for n in current[0]]
    voltage = [float(n) for n in voltage]
    return [voltage, current]


ivc_1 = get_ivc("3_win")


def func_x(x):
    """
    This function calculate scores for parameters of swarms and need curve
    :param x: population of swarms
    :return: array of score
    """
    global _x, _y, ivc_1
    _x = x
    # _y = y
    result = []
    for i in range(30):
        z = []
        z.extend(x[i])
        z.extend(_y[i])
        create_cir('test', z)
        # logger = Logging.setup_logging()
        circuit = spice.LoadFile('test.cir')
        input_data = spice.Init_Data(1000, 0.3)
        analysis = spice.CreateCVC(circuit, input_data, 100)
        ivc_2 = [analysis.input_dummy, analysis.VCurrent]
        result.append(compare_ivc(ivc_1, ivc_2))
    return result

def func_y(y):
    """
    This function calculate scores for parameters of swarms and need curve
    :param x: population of swarms
    :return: array of score
    """

    global _x, _y, ivc_1
    # _x = x
    _y = y
    result = []
    for i in range(30):
        z = []
        z.extend(_x[i])
        z.extend(y[i])
        create_cir('test', z)
        # logger = Logging.setup_logging()
        circuit = spice.LoadFile('test.cir')
        input_data = spice.Init_Data(1000, 0.3)
        analysis = spice.CreateCVC(circuit, input_data, 100)
        ivc_2 = [analysis.input_dummy, analysis.VCurrent]
        result.append(compare_ivc(ivc_1, ivc_2))
    return result


def plot(name, target_name):
    """
    Function plot and save picture of two iv-curves
    :param name: name of picture
    :return:
    """
    # curve1 = get_ivc("3_win")
    curve1 = get_ivc(str(target_name))
    curve2 = get_ivc("test")
    figure1 = plt.figure(1, (10, 5))
    plt.clf()
    plt.plot(curve1[0], curve1[1])
    plt.plot(curve2[0], curve2[1])
    plt.xlabel('Напряжение [В]')
    plt.ylabel('Сила тока [А]')
    plt.savefig("pic/" + str(name) + ".png")


def gen_init_pos(case, arr_of_res, arr_of_cap):
    init_pos = np.ndarray(shape=(case, 13), dtype=float)
    for i in range(case):
        array_of_resistor = np.random.permutation(arr_of_res)
        array_of_capacitor = np.random.permutation(arr_of_cap)
        init_pos[i][:5] = array_of_capacitor[-5:]
        # init_pos[i][:5] = init_pos[i][:5] * 10 ** -3 # if i % 2 is switch else init_pos[i][:5]
        init_pos[i][5:] = array_of_resistor[-8:]
        # init_pos[i][5:] = init_pos[i][5:] #* case[i]
    return init_pos


def gen_velocity_clamps(case=None):
    clamp = np.ndarray(shape=(2, 13), dtype=float)
    clamp[0][:5] = -1e-7
    clamp[1][:5] = 1e-7
    clamp[0][5:] = -1e5
    clamp[1][5:] = 1e5
    if case is not None:
        clamp[0][5:] = -clamp[0][5:] ** case
        clamp[1][5:] = clamp[0][5:] ** case
    # clamp[0][8:] = -1
    # clamp[1][8:] = 1
    return clamp


def gen_cases(first_level, second_level=None):
    f = [p for p in itertools.product(range(2), repeat=first_level)]
    cases = []
    del f[4]
    del f[5]
    if second_level is not None:
        for set in f:
            if set[0] == 0:
                f_1 = [p for p in itertools.product(range(2), repeat=second_level)]
                del f_1[3]
                del f_1[2]
            else:
                f_1 = [(1, 1)]
            if set[1] == 0:
                f_2 = [0, 1]
            else:
                f_2 = [1]
            if set[2] == 0:
                f_3 = [p for p in itertools.product(range(2), repeat=second_level)]
                del f_3[3]
                del f_3[2]
            else:
                f_3 = [(1, 1)]
            all_f = [p for p in itertools.product(f_1, f_2, f_3)]
            for _f in all_f:
                _set = list(set)
                _set.extend([_f[0][0], _f[1], _f[2][0], _f[0][1], _f[2][1]])
                cases.append(_set)
    return cases


time_opt_0 = time.clock()
clamps = []
costs = []
poses = []
array_of_resistor = np.linspace(0, 10 ** 6, 10 ** 6)
array_of_capacitor = np.linspace(0, 1e-6, 10 ** 6)
bounds_x = np.ndarray(shape=(2, 5), dtype=float)
bounds_y = np.ndarray(shape=(2, 8), dtype=int)
bounds_x[0] = 0.0000000
bounds_x[1] = 1e-6
bounds_y[0] = 0
bounds_y[1] = 10**6
cases = gen_cases(3, 2)
ivc_1 = get_ivc("3_win")
options = {'c1': 0.6, 'c2': 0.6, 'w': 0.4, 'k': 3, 'p': 1}
clamp = gen_velocity_clamps()
prev_cost = 1
none_error = 0
iter = 0

for i in range(30):
    _y[i] = np.random.randint(0, 10**5, 8)
    _x[i] = np.random.uniform(10**-12, 10**-6, 5)
while True:
    optimizer_x = ps.single.GlobalBestPSO(n_particles=30, dimensions=5, options=options,
                                          init_pos=_x, bounds=bounds_x)
    cost, pos_x = optimizer_x.optimize(func_x, iters=1)
    optimizer_y = pyswarms.discrete.binary.BinaryPSO(n_particles=30, dimensions=8, options=options,
                                          init_pos=_y)
                                                     # bounds=bounds_y)
    cost, pos_y = optimizer_y.optimize(func_y, iters=1)
    iter += 1
    print(cost, pos_x, pos_y)
    z = []
    z.extend(pos_x)
    z.extend(pos_y)
    create_cir('test', z)
    logger = Logging.setup_logging()
    circuit = spice.LoadFile('test.cir')
    input_data = spice.Init_Data(1000, 0.3)
    analysis = spice.CreateCVC(circuit, input_data, 100)
    spice.SaveFile(analysis, "test.csv")
    plot('result'+str(iter), '3_win')
    none_error += 1 if (cost-prev_cost) ** 2 < 10 ** -12 else 0
    if none_error > 5 or iter > 1000:
        break
# time_opt_1 = time.clock()
#
# print(time_func_1 - time_func_0)
# print("all optimization")
# print((time_opt_1 - time_opt_0)/60)
# y = []
# y.extend(init_pos[0][:5] / 10 ** (pos[:5] * 6))
# y.extend(init_pos[0][5:] * pos[5:])
# create_cir('test', y)
# logger = Logging.setup_logging()
# circuit = spice.LoadFile('test.cir')
# input_data = spice.Init_Data(1000, 0.3)
# analysis = spice.CreateCVC(circuit, input_data, 100)
# spice.SaveFile(analysis, "test.csv")
# plot('result', '3_win')
# #