'''
    Author: SC
    Date Created: 11/11/2018
    Python Version: 3.6
'''
import sklearn.datasets
import pandas
import numpy as np

from rmllib.data.base import Dataset
from rmllib.data.base import class_transform_to_dataframe
from rmllib.data.generate import matched_edge_generator

class InfluencerMedians(Dataset):
    '''
    Simple boston dataset with randomized edge data
    '''
    def __init__(self, subfeatures=None, **kwargs):
        '''
        Builds our dataset by
        (a) loading sklearn Boston dataset
        (b) binarizing it via the median of the feature values
        (c) generating random edges

        :subfeatures: Subsets of features available in the boston dataset.  Primarily for simulating weakened feature signals.
        :kwargs: Arguments for matched_edge_generator
        '''
        super().__init__(**kwargs)

        data = pandas.read_csv('dataset/influencer/train.csv')

        # label
        init_labels = pandas.DataFrame(data['Choice'])
        init_labels = init_labels.rename(columns={'Choice': 'Y'})

        # features
        init_features = data.drop('Choice', axis=1)
        init_features = init_features - init_features.median(axis=0)
        init_features[init_features < 0] = 0
        init_features[init_features > 0] = 1
        init_features = init_features.astype(int)

        # Create dataframe
        self.labels = class_transform_to_dataframe(init_labels.Y.values, islabel=True)
        self.features = class_transform_to_dataframe(init_features.values, islabel=False, classes=init_features.columns.values)
        # Simple correlation for edges       
        self.edges = matched_edge_generator(self.labels)
        return


