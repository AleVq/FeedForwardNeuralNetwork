import numpy as np
import math

class Selector:
    x_mean = 0
    y_mean = 0
    def __init__(self, matrix):
        self.dataSet = np.array(matrix)
        self.x_mean = np.nanmean(self.dataSet, axis=0)
        self.y_mean = self.x_mean[self.x_mean.size-1]
        self.x_mean = np.delete(self.x_mean, self.x_mean.size-1)

    def PearsonCoefficient(ds, x_mean, y_mean, i):
        k = 0
        sum_numerator = 0
        sum_denominator_1 = 0
        sum_denominator_2 = 0
        while k < ds.shape[0] - 1:
            if not np.isnan(ds[k][i]):
                sum_numerator += (ds[k][i] - x_mean[i]) * (ds[k][ds[k].size-1] - y_mean)
                sum_denominator_1 += (ds[k][i] - x_mean[i]) ** 2
                sum_denominator_2 += (ds[k][ds[k].size-1] - y_mean) ** 2
            k += 1
        R_i = sum_numerator / (math.sqrt(sum_denominator_1 * sum_denominator_2))
        return R_i

