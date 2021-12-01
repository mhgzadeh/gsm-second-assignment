import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    t = [0.000, 0.375, 0.475, 1.450, 2.050]
    s = [0.411, 0.330, 0.320, 0.300, 0.410]
    t0 = 0.621
    star = '*' * 15

    # ********* Q1 *********
    print('\n', '*' * 12, 'Q1', '*' * 12)
    start = -0.1
    stop = 2.2
    steps = 0.005
    tt = [np.round(step, 3) for step in np.arange(start, stop, steps)]
    print(f"t_min: {start}, dt: {steps}, t_max:{stop}")

    # ********* Q2 *********
    print('\n', '*' * 12, 'Q2', '*' * 12)
    a_mat = np.empty((0, 3))
    for i in t:
        a_mat = np.append(a_mat, np.array([[1, i, i ** 2]]), axis=0)
    x_cap = np.linalg.inv(np.transpose(a_mat).dot(a_mat)).dot(np.transpose(a_mat).dot(s))
    x_cap = np.round(x_cap, 6)
    print(f"x_cap: {x_cap}")
    ss_second_poly = [np.round(np.array([1, t, t ** 2]).dot(x_cap), 6) for t in tt]
    print('Plot 1')
    plt.gcf().number
    plt.figure(1)
    plt.plot(t, s, 'ro', tt, ss_second_poly, 'b-')
    plt.legend(['Points', '2nd Order Polynomial'], loc='best')
    plt.title('2nd Order Polynomial')
    plt.xlabel('t')
    plt.ylabel('s')

    # ********* Q3 *********
    print('\n', '*' * 12, 'Q3', '*' * 12)
    s0_second_poly = np.round(np.array([1, t0, t0 ** 2]).dot(x_cap), 6)
    print(f"s0 second order polynomial: {s0_second_poly}")

    # ********* Q4 *********
    print('\n', '*' * 12, 'Q4', '*' * 12)
    v = a_mat.dot(x_cap) - np.array(s)
    u = 3
    n = len(s)
    rms = np.sqrt((np.transpose(v).dot(v)) / (n - u))
    rms = np.round(rms, 6)
    print(f"rmse: {rms}")

    # ********* Q5 *********
    print('\n', '*' * 12, 'Q5', '*' * 12)
    di = np.empty((0, 1))
    for i in range(len(t)):
        di = np.append(di, 1 / (0.000001 + abs(t0 - t[i])))
    s0_idw = di.dot(s) / sum(di)
    s0_idw = np.round(s0_idw, 6)
    print(f"s0_idw: {s0_idw}")

    di = np.empty((0, 1))
    ss_idw = np.empty((0, 1))
    for ti in tt:
        for i in range(len(t)):
            di = np.append(di, 1 / (0.000001 + abs(ti - t[i])))
        ss_idw = np.append(ss_idw, np.round(di.dot(s) / sum(di), 6))
        di = np.empty((0, 1))
    print('Plot 2')
    plt.figure(2)
    plt.plot(t, s, 'ro', tt, ss_idw, 'g-')
    plt.legend(['Points', 'IDW'], loc='best')
    plt.title('IDW')
    plt.xlabel('t')
    plt.ylabel('s')

    # ********* Q6 *********
    print('\n', '*' * 12, 'Q6', '*' * 12)
    di = np.empty((0, 1))
    for i in range(len(t)):
        di = np.append(di, 1 / (0.000001 + abs(t0 - t[i])) ** 2)
    s0_isdw = di.dot(s) / sum(di)
    s0_isdw = np.round(s0_isdw, 6)
    print(f"s0_isdw: {s0_isdw}")

    di = np.empty((0, 1))
    ss_isdw = np.empty((0, 1))
    for ti in tt:
        for i in range(len(t)):
            di = np.append(di, 1 / (0.000001 + abs(ti - t[i])) ** 2)
        ss_isdw = np.append(ss_isdw, np.round(di.dot(s) / sum(di), 6))
        di = np.empty((0, 1))
    print('Plot 3')
    plt.figure(3)
    plt.plot(t, s, 'ro', tt, ss_isdw, 'y-')

    plt.legend(['Points', 'ISDW'], loc='best')
    plt.title('ISDW')
    plt.xlabel('t')
    plt.ylabel('s')

    print('Plot 4')
    plt.figure(4)
    plt.plot(t, s, 'ro', tt, ss_idw, 'g-', tt, ss_isdw, 'y-')
    plt.legend(['Points', 'IDW', 'ISDW'], loc='best')
    plt.title('IDW & ISDW')
    plt.xlabel('t')
    plt.ylabel('s')

    # ********* Q7 *********
    print('\n', '*' * 12, 'Q7', '*' * 12)
    d0 = np.round(np.mean(s), 6)
    print(f"d0: {d0}")
    a_mat = np.empty((0, 2))
    for ti in t:
        a_mat = np.append(a_mat, np.array([[1, ti]]), axis=0)
    ss_ll = np.empty((0, 1))
    for ti in tt:
        p = np.zeros([5, 5])
        for i in range(len(t)):
            d = np.abs(ti - t[i])
            p[i][i] = np.exp(-((d ** 2) / (2 * (d0 ** 2))))
        x_cap = np.linalg.inv(np.transpose(a_mat).dot(p).dot(a_mat)).dot(np.transpose(a_mat).dot(p).dot(s))
        a0_mat = np.empty((0, 2))
        a0_mat = np.append(a0_mat, np.array([[1, ti]]), axis=0)
        ss_ll = np.append(ss_ll, a0_mat.dot(x_cap))
    print('Plot 5')
    plt.figure(5)
    plt.plot(t, s, 'ro', tt, ss_ll, 'c-')
    plt.legend(['Points', 'LLP'], loc='best')
    plt.title('Local Linear Interpolation -- d0=0.3542')
    plt.xlabel('t')
    plt.ylabel('s')

    d0 = 0.1
    ss_ll_small = np.empty((0, 1))
    for ti in tt:
        p = np.zeros([5, 5])
        for i in range(len(t)):
            d = np.abs(ti - t[i])
            p[i][i] = np.exp(-((d ** 2) / (2 * (d0 ** 2))))
        x_cap = np.linalg.inv(np.transpose(a_mat).dot(p).dot(a_mat)).dot(np.transpose(a_mat).dot(p).dot(s))
        a0_mat = np.empty((0, 2))
        a0_mat = np.append(a0_mat, np.array([[1, ti]]), axis=0)
        ss_ll_small = np.append(ss_ll_small, a0_mat.dot(x_cap))
    print('Plot 6')
    plt.figure(6)
    plt.plot(t, s, 'ro', tt, ss_ll_small, 'c-')
    plt.legend(['Points', 'LLP'], loc='best')
    plt.title('Local Linear Interpolation - d0=0.1')
    plt.xlabel('t')
    plt.ylabel('s')

    d0 = 6
    ss_ll_big = np.empty((0, 1))
    for ti in tt:
        p = np.zeros([5, 5])
        for i in range(len(t)):
            d = np.abs(ti - t[i])
            p[i][i] = np.exp(-((d ** 2) / (2 * (d0 ** 2))))
        x_cap = np.linalg.inv(np.transpose(a_mat).dot(p).dot(a_mat)).dot(np.transpose(a_mat).dot(p).dot(s))
        a0_mat = np.empty((0, 2))
        a0_mat = np.append(a0_mat, np.array([[1, ti]]), axis=0)
        ss_ll_big = np.append(ss_ll_big, a0_mat.dot(x_cap))
    print('Plot 7')
    plt.figure(7)
    plt.plot(t, s, 'ro', tt, ss_ll_big, 'c-')
    plt.legend(['Points', 'LLP'], loc='best')
    plt.title('Local Linear Interpolation - d0=5')
    plt.xlabel('t')
    plt.ylabel('s')

    # ********* Q8 *********
    print('\n', '*' * 12, 'Q8', '*' * 12)
    nearest_num_t0 = np.abs(np.array(tt) - t0)
    t0_index = np.where(nearest_num_t0 == np.min(nearest_num_t0))
    s0_ll = np.round(ss_ll[t0_index][0], 6)
    print(f"s0_ll: {s0_ll}")

    plt.show()
