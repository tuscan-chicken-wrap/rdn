import code
import time
import sklearn.metrics

#from rmllib.data.load import InfluencerMedians
from influencer import InfluencerMedians
from relational_naive_bayes import RelationalNaiveBayes
from evaluation import rocCompareWithAlgos
#from rmllib.data.generate import Cora
from cora import Cora

from loadData import AcademicPerformance

if __name__ == '__main__':

    datasets = []
    datasets.append(InfluencerMedians(name='Influencer Medians', sparse=False).node_sample_mask(.7))
    datasets.append(AcademicPerformance(name='Academic Performance', subfeatures=None, sparse=False).node_sample_mask(.7))
    datasets.append(Cora(name="Cora", subfeatures=None, sparse=False).node_sample_mask(.7) )

    rdnModels = []
    rdnModels.append(RelationalNaiveBayes(name='NB_0', learn_method='iid', infer_method='iid', calibrate=False, neighbor_flag = 0))
    rdnModels.append(RelationalNaiveBayes(name='RNB_0', learn_method='r_iid', infer_method='r_iid', calibrate=False, neighbor_flag = 0))
	
    #rdnModels.append(RelationalNaiveBayes(name='NB_1', learn_method='iid', infer_method='iid', calibrate=False, neighbor_flag = 1))
    #rdnModels.append(RelationalNaiveBayes(name='RNB_1', learn_method='r_iid', infer_method='r_iid', calibrate=False, neighbor_flag = 1))

    for dataset in datasets:

        rdnPredictionResults = {}
        for rdnModel in rdnModels:
            print("start Evaluation of " + rdnModel.name  + 'rdnModel on ' + dataset.name + ' dataset')
            train_data = dataset.create_training().copy()
            start_time = time.time()
            rdnModel.fit(train_data)
            rdnModel.predictions = rdnModel.predict_proba(train_data)
			
            print("(" + dataset.name + ") " + rdnModel.name, 'Total Time:', time.time() - start_time)            
            print("(" + dataset.name + ") " + rdnModel.name, 'Average Prediction:', rdnModel.predictions[:, 1].mean(), 'AUC:', sklearn.metrics.roc_auc_score(dataset.labels.Y[dataset.mask.Unlabeled][1], rdnModel.predictions[:, 1]))
            print("(" + dataset.name + ") " + rdnModel.name + ": End Train")

            rdnPredictionResults[rdnModel.name] = rdnModel.predictions
            #print(rdnModel.predictions)

        rocCompareWithAlgos(dataset, rdnPredictionResults)

    