import numpy as np
from collections import namedtuple
from scipy import interpolate
from MySpice import MySpice as spice
import matplotlib.pyplot as plt
import CirCreate as C

eq_k = 1


def dist2_pt_seg(p, a, b):
    v1 = np.subtract(b, a)
    v2 = np.subtract(p, a)
    seg_len2 = np.dot(v1, v1)
    proj = np.dot(v1, v2) / seg_len2
    if proj < 0:
        return np.dot(v2, v2)
    if proj > 1:
        return np.dot(np.subtract(p, b), np.subtract(p, b))

    return np.cross(v1, v2) ** 2 / seg_len2


v_dist2_pt_seg = np.vectorize(dist2_pt_seg)


def rescale_score(x):
    return 1 - np.exp(-8 * x)


def dist_curve_pts_slow(curve, pts):
    res = 0.0
    for pt in pts.T:
        min_v = np.inf
        for i in range(1, len(curve[0])):
            v = dist2_pt_seg(pt, curve[:, i - 1], curve[:, i])
            if v < min_v:
                min_v = v
                min_i = i
        res += min_v
        v = dist2(pt, curve[:, min_i])
        # if abs((v / min_v) - 1) > 0.1:
        #    print('seg, pts =', v, min_v)
    return res


def dist_curve_pts_rude(curve, pts):
    res = 0.0
    for pt in pts.T:
        minv = np.inf
        for i in range(0, len(curve[0])):
            v = dist2(pt, curve[:, i])
            # vdiff = pt[0] - curve[0, i]
            # cdiff = pt[1] - curve[1, i]
            # v = vdiff * vdiff + cdiff * cdiff
            if v < minv:
                minv = v
                min_i = i
        res += minv
    return res


def dist_curve_pts(curve, pts):
    # curve:                [[x1, x2, ...], y[y1, y2, ...]]
    # pts - set of points:  [[x1, x2, ...], y[y1, y2, ...]]
    # in fact pts - is an other curve

    res = 0.0

    # from matplotlib import pyplot as plt
    # plt.plot(*curve)
    # plt.scatter(*pts, c='red')

    p = []
    c1 = []
    c2 = []
    c3 = []

    for pt in pts.T:
        min_i = 0

        # Calculate distances for all points
        # d0 = a[0] - b[0]
        # d1 = a[1] - b[1]
        # d0 * d0 + d1 * d1
        v = (curve[0] - pt[0]) * (curve[0] - pt[0]) + (curve[1] - pt[1]) * (curve[1] - pt[1])
        min_i = v.argmin()

        """p.append(pt)


        if min_i != 0:
            c1.append(curve[:, min_i - 1])
        else:
            c1.append(curve[:, min_i + 1])

        c2.append(curve[:, min_i])

        if min_i != 99:
            c3.append(curve[:, min_i + 1])
        else:
            c3.append(curve[:, min_i - 1])
        """

        res += min(
            dist2_pt_seg(pt, curve[:, min_i - 1], curve[:, min_i]) if min_i > 0 else np.inf,
            dist2_pt_seg(pt, curve[:, min_i], curve[:, min_i + 1]) if min_i < len(
                curve[0]) - 1 else np.inf)

    """l = v_dist2_pt_seg(p, c1, c2)"""

    res /= len(pts.T)
    # plots = np.reshape((pt, curve[:, ordered[min_i]]), (2, 2)).T
    # plt.plot(*plots, c='green')
    # print('res %.4f' % res)
    # plt.show()
    return res


def remove_repeats_ivc(a, eps=1e-6):
    msk0 = np.append(np.abs(a[0][1:] - a[0][:-1]) > eps, True)
    msk1 = np.append(np.abs(a[1][1:] - a[1][:-1]) > eps, True)
    return a[0][msk0 | msk1], a[1][msk0 | msk1]


def compare_ivc(a, b=None, min_var_v=None, min_var_c=None):
    # a, b:  tuple(oscilloscope instance, curve)
    # curve: tuple(voltage_points, current_points)

    if a is None:
        return 0
    min_v = np.min(a[0])
    min_c = np.min(a[1])
    # Now a and b - curves - tuple(voltage_points, current_points)

    if min_var_v is None:
        min_var_v = min_v / 10000

    if min_var_c is None:
        min_var_c = min_c / 10000

    # The variance is the average of the squared deviations from the mean, i.e., var = mean(abs(x - x.mean())**2).
    var_v = max(np.var(a[0]) ** 0.5, np.var(b[0]) ** 0.5 if b is not None else 0, min_var_v)
    var_c = max(np.var(a[1]) ** 0.5, np.var(b[1]) ** 0.5 if b is not None else 0, min_var_c)

    # Rescale curves for standard size and center them
    an = np.subtract(a[0], np.mean(a[0])) / var_v, np.subtract(a[1], np.mean(a[1])) / var_c

    an = remove_repeats_ivc(an)

    tck, u = interpolate.splprep(an, s=0.00)
    eq1 = np.array(interpolate.splev(np.arange(0, 1, 1.0 / len(a[0]) / eq_k), tck))

    if b is None:
        return rescale_score(np.mean(eq1[1, :] ** 2))
        # return np.mean(eq1[1, :] ** 2))
    else:
        bn = np.subtract(b[0], np.mean(b[0])) / var_v, np.subtract(b[1], np.mean(b[1])) / var_c
        bn = remove_repeats_ivc(bn)
        tck, u = interpolate.splprep(bn, s=0.00)
        eq2 = np.array(interpolate.splev(np.arange(0, 1, 1.0 / len(b[0]) / eq_k), tck))

        return rescale_score((dist_curve_pts(eq1, eq2) + dist_curve_pts(eq2, eq1)) / 2.)
        # return (dist_curve_pts(eq1, eq2) + dist_curve_pts(eq2, eq1)) / 2.


if __name__ == "__main__":
    circuit = spice.LoadFile('test_files/test_8.cir')
    input_data = spice.Init_Data(1000, 0.3, SNR=10 ** 6)
    analysis = spice.CreateCVC(circuit, input_data, 100)
    ivc_1 = [analysis.input_dummy, analysis.VCurrent]
    _x = [1e-6, 1e-6, 1e-6, 2.22e-7, 2.22e-12, 1, 1, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6]
    # while True:
        # print("Input i and x:")
        # i = int(input())
        # x = input()
        # if x == "stop":
        #     break
        # _x[i] = x
        # C.create_cir('test', _x)
    circuit = spice.LoadFile('test_files/test_res_8.cir')
    input_data = spice.Init_Data(1000, 0.3, SNR=10 ** 6)
    analysis = spice.CreateCVC(circuit, input_data, 100)
    ivc_2 = [analysis.input_dummy, analysis.VCurrent]
    score = compare_ivc(ivc_1, ivc_2)
    print(score)
    figure1 = plt.figure(1, (10, 5))
    plt.clf()
    plt.plot(ivc_1[0], ivc_1[1])
    plt.plot(ivc_2[0], ivc_2[1])
    plt.xlabel('Напряжение [В]')
    plt.ylabel('Сила тока [А]')
    # if plt.ylim()[0] > -0.0002:
    #     plt.ylim(-0.0002, 0.0002)
    plt.show()

