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

nub = 0
init_pos = np.ndarray(shape=(215, 13), dtype=float)
def func(x, ivc):
    """
    This function calculate scores for parameters of swarms and need curve
    :param x: population of swarms
    :return: array of score
    """
    global time_func_0, time_func_1, nub, init_pos
    nub += 1
    init_pos = x
    # time_func_0 = time.clock()
    result = []
    for i in range(len(x)):
        time_func_0 = time.clock()
        create_cir('test', x[i])
        circuit = spice.LoadFile('test.cir')
        input_data = spice.Init_Data(1000, 0.3, SNR=10**6)
        analysis = spice.CreateCVC(circuit, input_data, 100)
        # spice.SaveFile(analysis, "test.csv")
        ivc_2 = [analysis.input_dummy, analysis.VCurrent]
        result.append(compare_ivc(ivc, ivc_2))
        time_func_1 = time.clock()
        # plot("test_"+str(nub)+"_" +str(i), "3_win")

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
    plt.show()


def gen_init_pos(case, arr_of_res, arr_of_cap, switch):

    init_pos = np.ndarray(shape=(2*len(case), 13), dtype=float)
    for i in range(len(case)):
        array_of_resistor = np.random.permutation(arr_of_res)
        array_of_capacitor = np.random.permutation(arr_of_cap)
        init_pos[i][:5] = array_of_capacitor[-5:]
        init_pos[i][:5] = init_pos[i][:5] * 10 ** -3 if i % 2 is switch else init_pos[i][:5]
        init_pos[i][5:] = array_of_resistor[-8:]
        init_pos[i][5:] = init_pos[i][5:] * case[i]
    for i in range(len(case), 2 * len(case)):
        array_of_resistor = np.random.permutation(arr_of_res)
        array_of_capacitor = np.random.permutation(arr_of_cap)
        init_pos[i][:5] = array_of_capacitor[-5:]
        init_pos[i][:5] = init_pos[i][:5] * 10 ** -3 if not(i % 2) else init_pos[i][:5]
        init_pos[i][5:] = array_of_resistor[-8:]
        init_pos[i][5:] = init_pos[i][5:] * case[i-len(case)]
    return init_pos


def gen_velocity_clamps(speed_c, speed_r):
    clamp = np.ndarray(shape=(2, 13), dtype=float)
    clamp[0][:5] = -(10 ** -speed_c)
    clamp[1][:5] = 10 ** -speed_c
    clamp[0][5:] = -(10 ** speed_r)
    clamp[1][5:] = 10 ** speed_r
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
        f_00 = [p for p in itertools.product((0, 1), repeat=2)]
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
array_of_resistor = np.linspace(0, 10**6, 10**6)
array_of_capacitor = np.linspace(10**-12, 10**-6, 10**6)
bounds = np.ndarray(shape=(2, 13), dtype=float)
bounds[0][5:] = -10 ** -17
bounds[1][5:] = 10 ** 6
bounds[0][:5] = -10 ** -17
bounds[1][:5] = 10 ** -6
cases = gen_cases(3, 2)
circuit = spice.LoadFile('3_win.cir')
input_data = spice.Init_Data(1000, 0.3, SNR=10**6)
analysis = spice.CreateCVC(circuit, input_data, 100)
ivc_1 = [analysis.input_dummy, analysis.VCurrent]
spice.SaveFile(analysis, "3_win.csv")
# ivc_1 = get_ivc("3_win")
# options = {'c1': 0.56, 'c2': 0.65, 'w': 0.2}
options = {'c1': 0.8, 'c2': 0.6, 'w': 0.6}

logger = Logging.setup_logging()

for i in range(215):
    array_of_resistor = np.random.randint(0, 10**6, 8)
    array_of_capacitor = np.random.randint(0, 6, 5) + 6
    init_pos[i][:5] = 1/(10 ** array_of_capacitor[:])
    init_pos[i][5:] = array_of_resistor[:] * cases[i][5:]
    for j in range(5):
        init_pos[i][j] = init_pos[i][j] if cases[i][j] == 1 else 10 ** -12
best_cost = 1
for i in range(5):
    clamp = gen_velocity_clamps(12-i, 0)
    print(clamp)
    optimizer = ps.single.GlobalBestPSO(n_particles=215, dimensions=13, bounds=bounds,
                                        options=options, velocity_clamp=clamp,
                                        init_pos=init_pos)
    cost, pos = optimizer.optimize(func, iters=12, ivc=ivc_1)
    if cost < best_cost:
        best_cost = cost
        best_pos = pos
print("\n", best_cost, best_pos)
create_cir('test', pos)
logger = Logging.setup_logging()
circuit = spice.LoadFile('test.cir')
input_data = spice.Init_Data(1000, 0.3, SNR=10**6)
analysis = spice.CreateCVC(circuit, input_data, 100)
spice.SaveFile(analysis, "test.csv")
plot('result', '3_win')
