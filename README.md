#################################################
    How to install the required dependencies:
	
	"pip install rmllib"
	
	And that should be all you need

##################################################
	
	Which of these gosh darn files do I run?
	
	You only need to run "evaluationTest.py"
	
	It will automatically evaluate each dataset and compare
	against other learning algorithms
	
####################################################

	Notes on the file structure:
	
		- The dataset folder contains the datasets we used in this project (text and csv files)
		- The rmllib/models folder contains all the RMLLib code for running various kinds of training and inference
			However, the extended versions appear in this top level directory (i.e. relational_naive_bayes_new.py)
		- The rmllib/data/generate folder contains RMLLib code for creating edges between data nodes.
			However, like above we have changed the original code and added it to this top level directory (matched_edge_generatory.py)
		- The rmllib/data/load folder contains the files for loading data from csvs.
			However, we added our own to the top level directory:
				- influencer.py for Social Network Influencers
				- loadData.py for Academic Performance
				- cora.py for Cora
	