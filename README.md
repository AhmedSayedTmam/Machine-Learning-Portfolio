# Machine-Learning-Portfolio

This Repository contains portfolio of Machine Learning projects completed by me for academic, self learning and work purposes, presented in the form of iPython Notebooks.


## Contents

- ### Data Analysis and Visualisation
	-
		- [Wrangle and Analyze The WeRateDogs Twitter archive](): The dataset that is wrangled is the tweet archive of Twitter user @dog_rates, also known as WeRateDogs, starting with Gathering data then Assessing data and Cleaning data.
		- [Investigate a Dataset (TMDb Movie)](): Investigate and explore this data set by proposing the answers of some questions.
		- [Analyze A/B Test Results](): The goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision using an A/B test.
		- [Loan Data from Prosper Exploration](): Exploratory Data Analysis of  data set that contains 113,937 loans with 81 variables on each loan, including loan amount, borrower rate (or interest rate), current loan status, borrower income, and many others.
        
	_Tools: Pandas, Seaborn and Matplotlib_
    

- ### Simple Machine Learning NoteBooks: 
	-
		- [ML with Regression](): Simple Linear Regression, Multiple Linear Regression, Polynomial Regression, SVR, Decision Tree Regression, Random Forest Regression.
		- [ML with Classification](): Logistic Regression, K-NN, SVM, Kernel SVM, Naive Bayes, Decision Tree Classification, Random Forest Classification.
		- [ML with Clustering](): K-Means, Hierarchical Clustering.
		- [ML with Association Rule Learning](): Apriori, Eclat. 
		- [ML with Reinforcement Learning](): Upper Confidence Bound, Thompson Sampling. 
		- [ML with Natural Language Processing](): Bag-of-words model and algorithms for NLP. 
		- [ML with Dimensionality Reduction](): PCA, LDA, Kernel PCA. 
		- [ML with Model Selection & Boosting](): k-fold Cross Validation, Parameter Tuning, Grid Search, XGBoos
		- [Deep Learning](): Artificial Neural Networks, Convolutional Neural Networks.

	_Tools: Pandas, scikit-learn , Seaborn and Matplotlib_


- ### AWS SageMaker: 
	-
		- [Xgboost_Customer_Churn](): Using ML to automate the identification of unhappy customers, also known as customer churn prediction.
		- [Recommend Movies or Shows to Users](): Leveraging machine learning to create a recommendation engine to be used on the user website. We can use the data set to train a machine learning model to recommend movies/shows to watch.
		- [Predicting Credit Card Fraud ](): Leveraging machine learning to identify fraudulent credit card transactions before they have a larger impact on a company. We can use a dataset of past credit card transactions to train a machine learning model to predict if transactions are fraudulent or not.
		- [Predicting Airplane Delays](): Leveraging machine learning to identify whether the flight will be delayed due to weather. We can use the a dataset of on-time performance of domestic flights operated by large air carriers to train a machine learning model to predict if the flight is going to be delayed for the busiest airports.
		- [UFO Sightings K-Means Clustering](): Analyze where Mr. K should build his extraterrestrial life facilities using the K-Means algorithm
		- [UFO Sightings Algorithms Lab](): Build out models to use for predicting the legitimacy of a UFO sighting using the XGBoost and Linear Learner algorithm.
		- [UFO Sightings Implementation and Operations](): Train and deploy our model into SageMaker online hosting with 1 variant.

	_Tools: SageMaker session, IAM role, S3 bucket, Pandas, scikit-learn , Seaborn and Matplotlib_


- ### AWS SageMaker Deep Dive Notebooks: 
	-
		- [1.Blazingtext_text_classification_dbpedia](): Train the text classification model on the DBPedia Ontology Dataset as done by Zhang et al. The DBpedia ontology dataset is constructed by picking 14 nonoverlapping classes from DBpedia 2014. It has 560,000 training samples and 70,000 testing samples. The fields we used for this dataset contain title and abstract of each Wikipedia article.
		- [2.Hpo_xgboost_direct_marketing_sagemaker_python_sdk](): Train a model which can be used to predict if a customer will enroll for a term deposit at a bank, after one or more phone calls. Hyperparameter tuning will be used in order to try multiple hyperparameter settings and produce the best model.
		- [3.Hpo_image_classification_warmstart](): Demonstrating how to iteratively tune an image classifer leveraging the warm start feature of Amazon SageMaker Automatic Model Tuning. The caltech-256 dataset will be used to train the image classifier.
		- [4.Tensorflow_script_mode_training_and_serving](): Using a training script format for TensorFlow that lets you execute any TensorFlow training script in SageMaker with minimal modification
		- [5.Scikit_learn_estimator_example_with_batch_transform](): Using Scikit-learn with Sagemaker by utilizing the pre-built container which is a popular Python machine learning framework. It includes a number of different algorithms for classification, regression, clustering, dimensionality reduction, and data/feature pre-processing.
		- [6.Inference Pipeline with Scikit-learn and Linear Learner](): Demonstrating how you can build your ML Pipeline leveraging the Sagemaker Scikit-learn container and SageMaker Linear Learner algorithm & after the model is trained, deploy the Pipeline (Data preprocessing and Lineara Learner) as an Inference Pipeline behind a single Endpoint for real time inference and for batch inferences using Amazon SageMaker Batch Transform.
		- [7.Multiprocess Ensembler](): After we already have train, test, and validation data. Then we can train & tune a large number of models, and pull the results back in using an ensembling approach that takes the maximum prediction out of each classifier.Finally, we'll use SageMaker Search to find the best performing models from our bucket, and run parallel batch transform jobs to run inference on all of your newly trained models.
		- [8.Xgboost_multi_model_endpoint_home_value](): To demonstrate how multi-model endpoints are created and used, we use a set of XGBoost models that each predict housing prices for a single location. This domain is used as a simple example to easily experiment with multi-model endpoints.

	_Tools: SageMaker session, IAM role, S3 bucket, Pandas, scikit-learn , Seaborn and Matplotlib_


- ### Amazon SageMaker Neo: 
	-
		- [Image-classification-fulltraining-highlevel-neo](): Using the Amazon SageMaker Image Classification algorithm to train on the caltech-256 dataset and then we will demonstrate Amazon SageMaker Neo's ability to optimize models.
        
	_Tools: SageMaker session, IAM role and S3 bucket_


- ### Use Cases: 
	-
		- [Use Case 1 : Default of credit card clients](): Build, train to predict the target label Y Did the person pay default payment next month (Yes=1 or No=0 ). 
		- [Use Case 2 : Amazon Product Reviews](): Classify reviews as positive or negative for dataset about amazon product reviews, so we will make NLP then build, train and evaluate ML model to predict customers reviews as positive or negative. 
		- [Use Case 3 : Object-Detection](): Using a dataset from the inaturalist.org This dataset contains 500 images of bees that have been uploaded by inaturalist users for the purposes of recording the observation and identification.

	_Tools: SageMaker session, IAM role, S3 bucket, Pandas, scikit-learn , Seaborn and Matplotlib_



