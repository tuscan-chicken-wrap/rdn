import numpy.random as rnd
import pandas
import numpy as np

from rmllib.data.base import Dataset
from rmllib.data.base import class_transform_to_dataframe
#from rmllib.data.generate import matched_edge_generator
from matched_edge_generator import matched_edge_generator
from random_edge_generator import random_edge_generator
from sklearn.preprocessing import OneHotEncoder

class Cora(Dataset):

    def __init__(self, subfeatures=None, **kwargs):
        super().__init__(**kwargs)
        data1 = np.loadtxt("dataset/cora/cora.content", dtype = str)
        data2 = np.loadtxt("dataset/cora/cora.cites").astype(int)

        data_map = dict(zip(data1[:,0].astype(int),range(len(data1[:,0]))))
        matrix = np.zeros((data1.shape[0],data1.shape[0]))
        for i in data2:
            matrix[data_map[i[0]],data_map[i[1]]] = 1
            matrix[data_map[i[1]],data_map[i[0]]] = 1

        data1 = np.loadtxt("dataset/cora/cora.content", dtype = str)[:,:]
        features = data1[:,1:-1].astype(int)
        labels = np.array(list(set(data1[:,-1])))
        label_map = dict(zip(labels,range(len(labels))))
        labels = np.array([label_map[x] for x in data1[:,-1]])
        temp = []
        for i in labels:
            if(i == 1 or i == 2 or i ==3):
            # if(i == 1 ):
                temp.append(1)
            else:
                temp.append(0)
        temp = np.array(temp)

        cora_= {'data' : features, 'target' : temp, 'feature_names' : np.array(range(features.shape[1])).astype(str)}

        init_features = pandas.DataFrame(cora_['data'], columns=cora_['feature_names'])

        if subfeatures:
            init_features = init_features[subfeatures]


        init_labels = pandas.DataFrame(cora_['target'], columns=['Y'])
        init_labels.Y = np.where(init_labels.Y > 0.5, 1, 0).astype(int)


        # Booleanize feat by medians
        init_features[init_features < 0.5] = 0
        init_features[init_features > 0.5] = 1
        init_features = init_features.astype(int)

        # Create dataframe
        self.labels = class_transform_to_dataframe(init_labels.Y.values, islabel=True)

        self.features = class_transform_to_dataframe(init_features.values, islabel=False, classes=init_features.columns.values)

        # Simple correlation for edges
        #self.edges = matched_edge_generator(self.labels, mu_match = 0.5, mu_nomatch = 0.5, **kwargs)
        #self.edges = matched_edge_generator(self.labels, **kwargs)
        self.edges = matrix
        # self.edges = random_edge_generator(self.labels, **kwargs)

        return
