# import library
import scipy.io as sio  # import scipy.io for .mat file I/O
import scipy
from scipy import stats
import numpy as np  # import numpy
import matplotlib.pyplot as plt  # import matplotlib.pyplot for figure plotting
import torch
#import function_wmmse_powercontrol as wmmse_pc
from function_wmmse_powercontrol import ObjSumrate
from subnetwork_generator import MainGenerateSamples
import time
import math

class NetworkParameters:
    def __init__(self, shadowing_std_dev, random_seed, num_subnetworks, testing_sanp):
        self.num_subnetworks = num_subnetworks
        self.factory_area_size = 20  # Length and breadth of the factory area (m)
        self.subnetwork_radius = 2  # Radius of the subnetwork cell (m)
        self.min_device_to_controller_dist = 0.5  # Minimum distance from device to controller (access point) (m)
        self.min_controller_dist = self.subnetwork_radius  # Minimum controller to controller distance (m)
        self.shadowing_std_dev = shadowing_std_dev  # Shadowing standard deviation
        self.transmit_power = 1  # Normalized transmit power mW
        self.random_state = np.random.RandomState(random_seed)
        self.bandwidth = 5e6  # Bandwidth (Hz)
        self.frequency = 6e9  # Operating frequency (Hz)
        self.lambdA = 3e8 / 6e9  # Wavelength
        self.path_loss_exponent = 2.7  # Path loss exponent
        self.num_h = testing_sanp
        self.Pmax = 1
        self.Pmin = 0
        self.label_1 = "WMMSE"
        self.label_2 = "Max Power"
        self.label_3 = "Random Power"

    def wmmse_powers_sum_rate(self):
        Pini = self.Pmax * np.ones(self.num_subnetworks)
        var_noise = noise_power
        X = np.zeros((self.num_subnetworks ** 2, self.num_h))
        Y = np.zeros((self.num_subnetworks, self.num_h))
        iter_ = np.zeros(self.num_h)
        total_time = 0.0
        for loop in range(self.num_h):
            H = test_powers[loop, :, :]
            X[:, loop] = np.reshape(H, (self.num_subnetworks ** 2,), order="F")
            H = np.reshape(X[:, loop], (self.num_subnetworks, self.num_subnetworks), order="F")
            mid_time = time.time()

            # Create an instance of ObjSumrate
            obj_sumrate = ObjSumrate(Pini, H, self.Pmax, var_noise)
            Y[:, loop], iter_[loop] = obj_sumrate.WMMSE_sum_rate()
            #Y[:, loop], iter_[loop] = wmmse_pc.WMMSE_sum_rate(Pini, H, self.Pmax, var_noise)
            total_time = total_time + time.time() - mid_time
        # print("wmmse time: %0.2f s" % total_time)
        return X, Y, total_time, iter_

    def generate_capacity_and_powers(self, weights, data):
        weights = weights.reshape([-1, self.num_subnetworks, 1, 1])

        power_mat = data.reshape([-1, self.num_subnetworks, self.num_subnetworks, 1])

        weighted_powers = torch.mul(weights, power_mat)  # received powers

        eye = torch.eye(self.num_subnetworks)

        desired_rcv_power = torch.sum(torch.mul(weighted_powers.squeeze(-1), eye), dim=1)

        Interference_power = torch.sum(torch.mul(weighted_powers.squeeze(-1), 1 - eye), dim=1)

        signal_interference_ratio = torch.divide(desired_rcv_power, Interference_power + noise_power)

        capacity = torch.log2(1 + signal_interference_ratio)

        return capacity, weighted_powers

    @staticmethod
    def generate_cdf(values, bins_):
        data = np.array(values)
        count, bins_count = np.histogram(data, bins=bins_)
        pdf = count / sum(count)
        cdf = np.cumsum(pdf)
        return bins_count[1:], cdf

    def plot_generate(self, x, y):
        plt.plot(x, y, label=self.label_1)

        # Generate CDF for Max Power
        x_max, y_max = self.generate_cdf(torch.sum(capacities_max, 1) / 20, 1000)
        plt.plot(x_max, y_max, label=self.label_2)

        # Generate CDF for Random Power
        x_random, y_random = self.generate_cdf(torch.sum(capacities_random, 1) / 20, 1000)
        plt.plot(x_random, y_random, label=self.label_3)
        plt.legend()
        plt.ylabel('Cumulative Distribution Function')
        plt.xlabel('Spectral efficiency (bits/s/Hz)')
        plt.grid()
        plt.show()


if __name__ == '__main__':
    # Initialize network parameters
    shadowing_std_dev = 7
    num_subnetworks = 20
    testing_snapshots = 50000
    network_config = NetworkParameters(shadowing_std_dev, 0, num_subnetworks, testing_snapshots)
    sample_obj = MainGenerateSamples(network_config, testing_snapshots)
    # Generate testing data
    test_powers, _ = sample_obj.generate_samples()
    noise_power = np.power(10, ((-174 + 10 + 10 * np.log10(network_config.bandwidth)) / 10))
    
    # Compute capacities and powers for WMMSE
    X_sr, Y_sr, total_time, _ = network_config.wmmse_powers_sum_rate()
    WMMSE_capacities_sr, powers_sr = network_config.generate_capacity_and_powers(torch.tensor(np.transpose(Y_sr)), torch.tensor(test_powers))

    # Compute capacities and powers for Max Power

    max_power_values = torch.max(torch.tensor(np.transpose(Y_sr)), dim=1)[0]
    weights_max = torch.tensor(np.transpose(max_power_values))
    weights_max_reshaped = weights_max.unsqueeze(-1).unsqueeze(-1).repeat(1, network_config.num_subnetworks, 1, 1)
    test_powers_tensor = torch.tensor(test_powers)
    weights_max_reshaped = weights_max_reshaped.to(test_powers_tensor.dtype).to(test_powers_tensor.device)

    capacities_max, Max_pow = network_config.generate_capacity_and_powers(weights_max_reshaped, test_powers_tensor)

    # Compute capacities and powers for Random Power

    random_power_values = torch.rand(size=(len(max_power_values),))
    weights_random = torch.tensor(np.transpose(random_power_values))
    weights_random_reshaped = weights_random.unsqueeze(-1).unsqueeze(-1).repeat(1, network_config.num_subnetworks, 1, 1)
    test_powers_tensor = torch.tensor(test_powers)
    weights_random_reshaped = weights_random_reshaped.to(test_powers_tensor.dtype).to(test_powers_tensor.device)

    capacities_random, Random_pow = network_config.generate_capacity_and_powers(weights_random_reshaped, test_powers_tensor)

    # Generate CDF for WMMSE
    x_wmmse, y_wmmse = network_config.generate_cdf(torch.sum(WMMSE_capacities_sr, 1) / 20, 1000)
    network_config.plot_generate(x_wmmse, y_wmmse)
