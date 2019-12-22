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
from pyswarms.utils.plotters import plot_cost_history
import time
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


def func(x, ivc):
    """
    This function calculate scores for parameters of swarms and need curve
    :param x: population of swarms
    :return: array of score
    """
    # global c
    # c += 1
    global time_func_0, time_func_1
    # time_func_0 = time.clock()
    result = []
    for i in range(len(x)):
        # ivc_1 = get_ivc("3_win")
        time_func_0 = time.clock()
        create_cir('test', x[i])
        logger = Logging.setup_logging()
        circuit = spice.LoadFile('test.cir')
        input_data = spice.Init_Data(1000, 0.3)
        analysis = spice.CreateCVC(circuit, input_data, 100)
        # spice.SaveFile(analysis, "test.csv")
        # plot(i+c)
        ivc_2 = [analysis.input_dummy, analysis.VCurrent]
        result.append(compare_ivc(ivc, ivc_2))
        time_func_1 = time.clock()

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
    plt.savefig("pic/" + str(name) +".png")


def gen_init_pos(case, arr_of_res, arr_of_cap, num_sw):
    init_pos = np.ndarray(shape=(num_sw, 13), dtype=float)
    for i in range(num_sw):
        array_of_resistor = np.random.permutation(arr_of_res)
        array_of_capacitor = np.random.permutation(arr_of_cap)
        init_pos[i][:5] = array_of_capacitor[-5:]
        init_pos[i][:5] = init_pos[i][:5] * 10 ** -3 if i % 2 else init_pos[i][:5]
        init_pos[i][5:] = array_of_resistor[-8:]
        init_pos[i][5:] = init_pos[i][5:] * case
    return init_pos


def gen_velocity_clamps(case):
    clamp = np.ndarray(shape=(2, 13), dtype=float)
    clamp[0][:5] = -(10 ** -10)
    clamp[1][:5] = 10 ** -10
    clamp[0][5:] = -(10 ** 5)
    clamp[1][5:] = 10 ** 5
    clamp[0][5:] = -clamp[0][5:] ** case
    clamp[1][5:] = clamp[0][5:] ** case
    return clamp


def gen_cases(first_level, second_level):
    f = [p for p in itertools.product(range(2), repeat=first_level)]
    cases = []
    del f[4]
    del f[5]
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
array_of_resistor = np.linspace(0, 10**3, 10**3)
array_of_capacitor = np.linspace(10**-12, 10**-6, 10**3)
bounds = np.ndarray(shape=(2, 13), dtype=float)
bounds[0][3:] = 0.0
bounds[1][3:] = 10 ** 6
bounds[0][:3] = 0.0
bounds[1][:3] = 10 ** -6
cases = gen_cases(3, 2)
ivc_1 = get_ivc("3_win")
options = {'c1': 0.8, 'c2': 0.5, 'w': 0.5}
for case in cases:
    init_pos = gen_init_pos(case, array_of_resistor, array_of_capacitor, 20)
    clamp = gen_velocity_clamps(case)
    optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=13, bounds=bounds,
                                        options=options, velocity_clamp=clamp,
                                        init_pos=init_pos)
    cost, pos = optimizer.optimize(func, iters=28, ivc=ivc_1)
    costs.append(cost)
    poses.append(pos)
    clamps.append(clamp)

minpos = poses[np.argmin(costs)]
minclamp = clamps[np.argmin(costs)]
time_opt_1 = time.clock()
print(minpos, np.min(costs))
print(time_func_1 - time_func_0)
print("all optimization")
print((time_opt_1 - time_opt_0)/60)
create_cir('test', minpos)
logger = Logging.setup_logging()
circuit = spice.LoadFile('test.cir')
input_data = spice.Init_Data(1000, 0.3)
analysis = spice.CreateCVC(circuit, input_data, 100)
spice.SaveFile(analysis, "test.csv")
plot('result', '3_win')
