import csv
import numpy as np
import math

class Selector:

    def __init__(self, fileName):
        self.dataSet = self.parse(fileName)
        self.x_mean = np.nanmean(self.dataSet, axis=0)#each attribute's average value
        self.y_mean = self.x_mean[self.x_mean.size-1]#target average value
        self.x_mean = np.delete(self.x_mean, self.x_mean.size-1) #delete target column

    def parse(self, filename):
        temp = []
        f = open(filename, 'r')
        reader = csv.reader(f, delimiter = ';')
        i = 0
        for row in reader:
            if i == 0:
                self.attributes = row
                i = 1
            else:
                temp.append(row)
        temp = np.array(temp, dtype=float)
        temp[temp==-1] = np.NaN
        return temp

    def apply_feature_selection(self, n): #numOfAttr: #arbitrarily choosing the n most relevant attributes w.r.t. the target
        ranks = self.ranking_list(n)
        return self.slicer(ranks)

    def slicer(self, ranks): #delete not high-ranked attribute's column
        orderedAttrNum = np.sort(ranks, axis=0)[:,0]
        orderedAttrNum = orderedAttrNum.astype(int)
        #slicing columns
        k=0
        j=0
        for i in orderedAttrNum:#
            if k == 0:
                featureSelectedDS = np.column_stack((self.dataSet[:, i]))
                k = 1
            elif k == 1:
                featureSelectedDS = np.column_stack((featureSelectedDS[0], self.dataSet[:, i]))
                k = 2
            else:
                featureSelectedDS = np.column_stack((featureSelectedDS[:, 0:k], self.dataSet[:, i]))
                k = k + 1
        return featureSelectedDS

    def ranking_list(self, n):
        i = 0
        score = np.zeros(shape=(self.dataSet.shape[1]-1,2))
        while i < score.shape[0]: #loop through attributes except the target in the last column
            score[i] = [i, math.fabs(self.pearson_coefficient(i))] #array of couple <score, i-th attribute>
            i += 1
        orderedScore =score[score[:,1].argsort(axis=0)] #in rank-ascending order
        reversedScore = orderedScore[::-1]
        return reversedScore[0:n]

    def pearson_coefficient(self, i):
        ds = self.dataSet
        x_mean = self.x_mean
        if x_mean[i] == 0:
            return 0 #if in all instances this attribute is 0, for example
        y_mean = self.y_mean
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

