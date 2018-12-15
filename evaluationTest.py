import code
import time
import sklearn.metrics
import numpy as np

from influencer import InfluencerMedians
from relational_naive_bayes_new import RelationalNaiveBayes
from evaluation import rocCompareWithAlgos
from cora import Cora

from loadData import AcademicPerformance

if __name__ == '__main__':

    datasets = []
    datasets.append(InfluencerMedians(name='Influencer Medians', sparse=False).node_sample_mask(.7))
    datasets.append(AcademicPerformance(name='Academic Performance', subfeatures=None, sparse=False).node_sample_mask(.7))
	
	# Use this Cora Dataset instead if you find that the code runs too slow
    #datasets.append(Cora(name="Cora", subfeatures=list(np.array(range(50)).astype(str)), sparse=False).node_sample_mask(.7) )
	
    datasets.append(Cora(name="Cora", subfeatures=None, sparse=False).node_sample_mask(.7) )

    rdnModels = []
	
	#Append the Naive bayes and Relational Naive Bayes inference models
    rdnModels.append(RelationalNaiveBayes(name='NB_0', learn_method='iid', infer_method='iid', calibrate=False, neighbor_flag = 0))
    rdnModels.append(RelationalNaiveBayes(name='RNB_0', learn_method='r_iid', infer_method='r_iid', calibrate=False, neighbor_flag = 0))
	
    #For each dataset, train and run inference on the model, and compare it against other classifier algorithmns
    for dataset in datasets:

        rdnPredictionResults = {}
        for rdnModel in rdnModels:
            print("start Evaluation of " + rdnModel.name  + 'rdnModel on ' + dataset.name + ' dataset')
            train_data = dataset.create_training().copy()
            start_time = time.time()
            rdnModel.fit(train_data)
            rdnModel.predictions = rdnModel.predict_proba(train_data)
            rdnModel.listTopKfeatures(train_data, 5)
            #print(rdnModel.predictions)
            print("(" + dataset.name + ") " + rdnModel.name, 'Total Time:', time.time() - start_time)            
            print("(" + dataset.name + ") " + rdnModel.name, 'Average Prediction:', rdnModel.predictions[:, 1].mean(), 'AUC:', sklearn.metrics.roc_auc_score(dataset.labels.Y[dataset.mask.Unlabeled][1], rdnModel.predictions[:, 1]))
            print("(" + dataset.name + ") " + rdnModel.name + ": End Train")

            rdnPredictionResults[rdnModel.name] = rdnModel.predictions
            

        rocCompareWithAlgos(dataset, rdnPredictionResults)

    