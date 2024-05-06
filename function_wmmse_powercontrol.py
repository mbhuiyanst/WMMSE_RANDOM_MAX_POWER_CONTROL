# ###############################################
# This file includes functions to perform the WMMSE algorithm [2].
# Codes have been tested successfully on Python 3.6.0 with Numpy 1.12.0 support.
#
# References: 
# [1] Haoran Sun, Xiangyi Chen, Qingjiang Shi, Mingyi Hong, Xiao Fu, and Nikos D. Sidiropoulos. 
# "Learning to optimize: Training deep neural networks for wireless resource management." 
# in proceedings of IEEE 18th International Workshop on Signal Processing Advances in Wireless Communications (SPAWC), 2017.
# 
# [2] Haoran Sun, Xiangyi Chen, Qingjiang Shi, Mingyi Hong, Xiao Fu, and Nikos D. Sidiropoulos, 
# "Learning to Optimize: Training Deep Neural Networks for Interference Management," 
# in IEEE Transactions on Signal Processing, vol. 66, no. 20, pp. 5438-5453, 15 Oct.15, 2018.
#
# [3] Qingjiang Shi, Meisam Razaviyayn, Zhi-Quan Luo, and Chen He.
# "An iteratively weighted MMSE approach to distributed sum-utility maximization for a MIMO interfering broadcast channel."
# IEEE Transactions on Signal Processing 59, no. 9 (2011): 4331-4340.
#
# version 1.0 -- February 2017. Written by Haoran Sun (hrsun AT iastate.edu)
# ###############################################

import numpy as np
import math
import time
import scipy.io as sio
import matplotlib.pyplot as plt


# Functions for objective (sum-rate) calculation


def obj_IA_sum_log_rate(H, p, var_noise, K):
    y = 0.0
    for i in range(K):
        s = var_noise
        for j in range(K):
            if j != i:
                s = s + H[i, j] ** 2 * p[j]
        y = y + math.log10(math.log2(1 + H[i, i] ** 2 * p[i] / s))
    return y


class ObjSumrate:
    def __init__(self, p_int, H, Pmax, var_noise):
        self.p_int = p_int
        self.H = H
        self.Pmax = Pmax
        self.var_noise = var_noise

    def WMMSE_sum_rate(self):
        K = np.size(self.p_int)
        vnew = 0
        b = np.sqrt(self.p_int)
        f = np.zeros(K)
        w = np.zeros(K)
        for i in range(K):
            f[i] = self.H[i, i] * b[i] / (np.square(self.H[i, :]) @ np.square(b) + self.var_noise)
            w[i] = 1 / (1 - f[i] * b[i] * self.H[i, i])
            vnew = vnew + math.log2(w[i])

        VV = np.zeros(10000)
        for iter_ in range(10000):
            vold = vnew
            for i in range(K):
                btmp = w[i] * f[i] * self.H[i, i] / sum(w * np.square(f) * np.square(self.H[:, i]))
                b[i] = min(btmp, np.sqrt(self.Pmax)) + max(btmp, 0) - btmp

            vnew = 0
            for i in range(K):
                f[i] = self.H[i, i] * b[i] / ((np.square(self.H[i, :])) @ (np.square(b)) + self.var_noise)
                w[i] = 1 / (1 - f[i] * b[i] * self.H[i, i])
                vnew = vnew + math.log2(w[i])

            VV[iter_] = vnew
            if vnew - vold <= 1e-5:
                break

        p_opt = np.square(b)
        return p_opt, iter_

    def obj_IA_sum_rate(self, H, p, var_noise, K):
        y = 0.0
        for i in range(K):
            s = var_noise
            for j in range(K):
                if j != i:
                    s = s + H[i, j] ** 2 * p[j]
            y = y + math.log2(1 + H[i, i] ** 2 * p[i] / s)
        return y

    def plot_cdf(self, data, label):
        sorted_data = np.sort(data)
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
        plt.plot(sorted_data, yvals, label=label)

    def perf_eval(self, Py_p, NN_p, K, var_noise=1):
        num_sample = self.H.shape[2]
        pyrate = np.zeros(num_sample)
        nnrate = np.zeros(num_sample)
        mprate = np.zeros(num_sample)
        rdrate = np.zeros(num_sample)
        for i in range(num_sample):
            pyrate[i] = self.obj_IA_sum_rate(self.H[:, :, i], Py_p[:, i], var_noise, K)
            nnrate[i] = self.obj_IA_sum_rate(self.H[:, :, i], NN_p[i, :], var_noise, K)
            mprate[i] = self.obj_IA_sum_rate(self.H[:, :, i], np.ones(K), var_noise, K)
            rdrate[i] = self.obj_IA_sum_rate(self.H[:, :, i], np.random.rand(K, 1), var_noise, K)
        print('Sum-rate: WMMSE: %0.3f, DNN: %0.3f, Max Power: %0.3f, Random Power: %0.3f' % (
            sum(pyrate) / num_sample, sum(nnrate) / num_sample, sum(mprate) / num_sample, sum(rdrate) / num_sample))
        print('Ratio: DNN: %0.3f%%, Max Power: %0.3f%%, Random Power: %0.3f%%\n' % (
            sum(nnrate) / sum(pyrate) * 100, sum(mprate) / sum(pyrate) * 100, sum(rdrate) / sum(pyrate) * 100))

        # Compute and plot CDF
        plt.figure('%d' % K)
        plt.style.use('seaborn-deep')

        self.plot_cdf(pyrate, 'WMMSE')
        self.plot_cdf(nnrate, 'DNN')
        self.plot_cdf(mprate, 'Max Power')
        self.plot_cdf(rdrate, 'Random Power')

        plt.legend(loc='lower right')
        plt.xlim([0, max(pyrate)])
        plt.xlabel('Sum-rate')
        plt.ylabel('CDF')
        plt.title('CDF vs Sum-rate for WMMSE, DNN, Max Power, and Random Power')
        plt.grid(True)
        plt.savefig('CDF_vs_Sumrate_%d.eps' % K, format='eps', dpi=1000)
        plt.show()

        print('Sum-rate: WMMSE: %0.3f, DNN: %0.3f, Max Power: %0.3f, Random Power: %0.3f' % (
            sum(pyrate) / num_sample, sum(nnrate) / num_sample, sum(mprate) / num_sample, sum(rdrate) / num_sample))
        print('Ratio: DNN: %0.3f%%, Max Power: %0.3f%%, Random Power: %0.3f%%\n' % (
            sum(nnrate) / sum(pyrate) * 100, sum(mprate) / sum(pyrate) * 100, sum(rdrate) / sum(pyrate) * 100))

        return 0

    def generate_Gaussian(self, K, num_H, Pmax=1, Pmin=0, seed=2017):
        print('Generate Data ... (seed = %d)' % seed)
        np.random.seed(seed)
        Pini = Pmax * np.ones(K)
        var_noise = 1
        X = np.zeros((K ** 2, num_H))
        Y = np.zeros((K, num_H))
        total_time = 0.0
        for loop in range(num_H):
            CH = 1 / np.sqrt(2) * (np.random.randn(K, K) + 1j * np.random.randn(K, K))
            H = abs(CH)
            X[:, loop] = np.reshape(H, (K ** 2,), order="F")
            H = np.reshape(X[:, loop], (K, K), order="F")
            mid_time = time.time()
            self.p_int = Pini
            self.H = H
            self.Pmax = Pmax
            self.var_noise = var_noise
            Y[:, loop] = self.WMMSE_sum_rate()
            total_time = total_time + time.time() - mid_time
        # print("wmmse time: %0.2f s" % total_time)
        return X, Y, total_time

    def generate_Gaussian_half(self, K, num_H, Pmax=1, Pmin=0, seed=2017):
        print('Generate Testing Data ... (seed = %d)' % seed)
        np.random.seed(seed)
        Pini = Pmax * np.ones(K)
        var_noise = 1
        X = np.zeros((K ** 2 * 4, num_H))
        Y = np.zeros((K * 2, num_H))
        total_time = 0.0
        for loop in range(num_H):
            CH = 1 / np.sqrt(2) * (np.random.randn(K, K) + 1j * np.random.randn(K, K))
            H = abs(CH)
            print('H is', H)
            print('Pini is', Pini)
            mid_time = time.time()
            self.p_int = Pini
            self.H = H
            self.Pmax = Pmax
            self.var_noise = var_noise
            Y[0: K, loop] = self.WMMSE_sum_rate()
            total_time = total_time + time.time() - mid_time
            OH = np.zeros((K * 2, K * 2))
            OH[0: K, 0:K] = H
            X[:, loop] = np.reshape(OH, (4 * K ** 2,), order="F")
        # print("wmmse time: %0.2f s" % total_time)
        return X, Y, total_time  ##X is the channel matrix vectorized to form a column for the NN,
        ## If you set K = 5 (number of users), X will have a length of 5x5
        ## Y is vector of the power allocated to the 5 users which will serve
        ## as a label for the NN. num_H is the number of dataset like snapshots

    def generate_IMAC(self, num_BS, num_User, num_H, Pmax=1, var_noise=1):
        # Load Channel Data
        CH = sio.loadmat('IMAC_%d_%d_%d' % (num_BS, num_User, num_H))['X']
        Temp = np.reshape(CH, (num_BS, num_User * num_BS, num_H), order="F")
        H = np.zeros((num_User * num_BS, num_User * num_BS, num_H))
        for iter in range(num_BS):
            H[iter * num_User:(iter + 1) * num_User, :, :] = Temp[iter, :, :]

        # Compute WMMSE output
        Y = np.zeros((num_User * num_BS, num_H))
        Pini = Pmax * np.ones(num_User * num_BS)
        start_time = time.time()
        for loop in range(num_H):
            self.p_int = Pini
            self.H = H[:, :, loop]
            self.Pmax = Pmax
            self.var_noise = var_noise
            Y[:, loop] = self.WMMSE_sum_rate()
        wmmsetime = (time.time() - start_time)
        # print("wmmse time: %0.2f s" % wmmsetime)
        return CH, Y, wmmsetime, H
