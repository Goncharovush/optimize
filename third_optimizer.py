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
    global time_func_0, time_func_1
    # time_func_0 = time.clock()
    result = []
    for i in range(len(x)):
        # time_func_0 = time.clock()
        create_cir('test', x[i])
        circuit = spice.LoadFile('test.cir')
        input_data = spice.Init_Data(1000, 0.3, SNR=10**6)
        analysis = spice.CreateCVC(circuit, input_data, 100)
        ivc_2 = [analysis.input_dummy, analysis.VCurrent]
        # y = [1 if y > 10**-12 else 0 for y in x[i]]
        result.append(compare_ivc(ivc, ivc_2))
        # time_func_1 = time.clock()

    return result


def plot(name, target_name):
    """
    Function plot and save picture of two iv-curves
    :param name: name of picture
    :return:
    """
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
    """
    This function generates initial positions for swarm
    :param case:
    :param arr_of_res: range of values of resistor
    :param arr_of_cap: range of values of capacitor
    :return:
    """

    init_pos = np.ndarray(shape=(len(case), 13), dtype=float)
    for i in range(len(case)):
        array_of_resistor = np.random.permutation(arr_of_res)
        array_of_capacitor = np.random.permutation(arr_of_cap)
        init_pos[i][:5] = array_of_capacitor[-5:]
        init_pos[i][5:] = array_of_resistor[-8:]
        init_pos[i][:] *= case[i]
        init_pos[i][:5] += 10**-12
    return init_pos


def gen_velocity_clamps():
    """
    This function set velocity clamps
    :return:
    """
    clamp = np.ndarray(shape=(2, 13), dtype=float)
    clamp[0][:5] = -(10 ** -6)   # min for capacitor and diod
    clamp[1][:5] = 10 ** -6    # max for capacitor and diod
    clamp[0][5:] = -(10 ** 6)  # min for resistor
    clamp[1][5:] = 10 ** 6   # max for resistor
    return clamp


def gen_cases(first_level, second_level):
    """
    This function generate all cases turn/off elements of electricak circuit
    :param first_level: number of resistors on high level
    :param second_level: number of resistors on low level
    :return:
    """
    f = [p for p in itertools.product(range(2), repeat=first_level)]
    cases = []
    del f[4]
    del f[5]
    # print(f)
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
        f_00 = [p for p in itertools.product(range(2), repeat=2)]
        del f_00[1]
        if set[0] == 0:
            _f_0 = f_00
        else:
            _f_0 = [(0, 0)]
        if set[1] == 0:
            _f_1 = [0, 1]
        else:
            _f_1 = [0]
        if set[2] == 0:
            _f_2 = f_00
        else:
            _f_2 = [(0, 0)]
        ff = [p for p in itertools.product(_f_0, _f_1, _f_2)]
        all_elem = [p for p in itertools.product(ff, all_f)]
        for _f in all_elem:
            _set = list(set)
            _f = list(_f)
            _f[0] = list(_f[0])
            _f[1] = list(_f[1])
            _set = [_f[0][0][0], _f[0][1], _f[0][2][0], _f[0][0][1], _f[0][2][1]] + _set
            _set.extend([_f[1][0][0], _f[1][1], _f[1][2][0], _f[1][0][1], _f[1][2][1]])
            cases.append(_set)
    return cases


time_opt_0 = time.clock()
clamps = []
costs = []
poses = []
array_of_resistor = np.linspace(0, 10**3, 10**3)
array_of_capacitor = np.linspace(10**-12, 10**-6, 10**3)
bounds = np.ndarray(shape=(2, 13), dtype=float)  # bounds for values of resistor/capacitor/diod
bounds[0][5:] = 0.00000000e+00  # bounds for values of resistor
bounds[1][5:] = 10 ** 6
bounds[0][:5] = 1.00000000e-12  # bounds for values of capacitor/diod
bounds[1][:5] = 1.00000000e-6
cases = gen_cases(3, 2)
ivc_1 = get_ivc("3_win")
options = {'c1': 0.6, 'c2': 0.6, 'w': 0.3}

clamp = gen_velocity_clamps()
logger = Logging.setup_logging()

init_pos = gen_init_pos(cases, array_of_resistor, array_of_capacitor)
for pos in init_pos:
    print(pos)
optimizer = ps.single.GlobalBestPSO(n_particles=len(cases), dimensions=13, bounds=bounds,
                                    options=options, velocity_clamp=clamp,
                                    init_pos=init_pos)
cost, pos = optimizer.optimize(func, iters=5, ivc=ivc_1)

# minclamp = clamps[np.argmin(costs)]
time_opt_1 = time.clock()
print(pos, cost)
# print(time_func_1 - time_func_0)
# print("all optimization")
# print((time_opt_1 - time_opt_0)/60)
create_cir('test', pos)
logger = Logging.setup_logging()
circuit = spice.LoadFile('test.cir')
input_data = spice.Init_Data(1000, 0.3, SNR=10**6)
analysis = spice.CreateCVC(circuit, input_data, 100)
spice.SaveFile(analysis, "test.csv")
plot('result', '3_win')
