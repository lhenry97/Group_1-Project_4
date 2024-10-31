### Cancer Diagnosis Prediction App

## Disclaimer
This machine learning model is designed for educational and research purposes only. It is not intended to diagnose, treat, cure, or prevent any disease. Always consult a healthcare professional for medical advice, diagnosis, or treatment.

# Goal of the Project:
The purpose for this project is to assist with breast cancer survial through early intervention. An application has been created for the use of researchers which contains user inputted data based on the visual characteristics of a cancer. A machine learning model has been selected to be used to predict a diagnosis on whether the identified cancer is Benign or Malignant based on these visual characteristics. The features that are used to assist in determining the diagnosis include: mean radius, mean texture, mean perimeter, mean area, mean smoothness, mean compactness, mean concavity and mean concave points. The dataset identifed for this project is the Breast Cancer Wisconsin (Diagnostic) Data Set. 

## Dataset Information: 
This dataset is provided under the **Creative Commons Attribution-NonCommercial-ShareAlike (CC BY-NC-SA) 4.0** license.

### Attribution
The original dataset was provided by the [UC Irvine Machine Learning Repository](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

### Usage
You are free to use this dataset for non-commercial purposes. If you modify the dataset, you must distribute the modified dataset under the same license.

### License
For more information, please refer to the [CC BY-NC-SA 4.0 License] (https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Ethical Considerations

### Ethical Use of Dataset
This dataset has been utilized with the utmost consideration for ethical standards and practices. The data is anonymized, ensuring that no Personally Identifiable Information (PII) is included.

### Purpose and Use
The dataset has been used exclusively for educational and research purposes to develop a machine learning model for predicting whether a tumor is benign or malignant.

### Privacy and Confidentiality
The dataset includes anonymized IDs to protect the privacy of individuals. All efforts have been made to ensure data confidentiality and compliance with privacy regulations.

By including this statement, we aim to uphold ethical standards in the development and use of machine learning models and protect the privacy and rights of individuals.

# Slide Deck:
https://www.canva.com/design/DAGUqhcvQm0/zgpipBtmAcZnmDA6aaSEtw/edit?utm_content=DAGUqhcvQm0&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

# Inspiration:
<img src="https://github.com/CathyMatthee/predictor_app/blob/main/images/proposal.png" alt="Proposal Inspiration" width="300"/>

# Method
The data underwent cleaning and removal of any duplications or unnecessary data. Feature engineering was then conducted on the dataset to ensure the data is in a usable state for machine learning. Postgres was used to manage the database and the app.py flask app was used to connect to the database to enable a website to call information from it. A number of machine learning models were tested and evaluated on their prediction of the cancer diagnosis. The models include logistic regression, SVM, random forest and Deep Neural Network (utilising keras tuner). This was conducted using jupyter notebook for each machine learning model and then the workflow was collated into one complete notebook for ease of use. Each model has undergone optimisation and evaluated to select the best model.  The selected machine learning model has been used in the final website. The website took inspiration from the above screenshot to enable a user to alter different visual characteristics and the model will output a predicted diagnosis.


# Workflow for Reproducibility

## Data Fetching and API Integration

### To deploy the Flask app locally start:
1. load csv into pgAdmin4 as per Database Setup instructions further below.
2. run app.py (located in the root folder) in the terminal/vscode AFTER setting up the database in pgAdmin 4.
3. running the jupyter notebook Collated_Machine_Learning_Notebook.ipynb retrieves json data served by the flask app and saves the ML models.

When the Flask app starts, it establishes a connection to the PostgreSQL database using SQLAlchemy. This connection allows the app to interact with the database, querying the database and serving the results in JSON format to the machine learning training code in a jupyter notebook. When the endpoint is called, any updates made to the database will be reflected in the JSON response automatically without needing to restart the Flask app.
<img src="https://github.com/CathyMatthee/predictor_app/blob/main/images/data%20architecture.png?raw=true" alt="data architecture" width="300"/>

Trained machine learning models are saved on the backend whenever the jupyter notebook is run. The app loads the saved models for use when an end/user such as a doctor or other healthcare professional attempts to make a prediction.

This section also explains how to connect to a PostgreSQL database using psycopg2 in python.

### Database Setup (pgAdmin4)
1. Download/clone the repository from https://github.com/lhenry97/Group_1-Project_4.git.
2. Open pgAdmin 4 and create a new database called Cancer_db
3. Right click on Cancer_db and select Query Tool
4. Select the Folder icon to open a file called "cancer_data.sql" from the data folder
5. Run the "create table" query to load the table columns
6. Refresh the Table in Schemas, right-click on Cancer_Data table and select "Import/Export data"
7. Find the file called Cancer_Data.csv also in the "data" folder and open this file.
8. In Options menu set Header to "active" and Delimiter as ",".
9. Optionally, run the json_agg query in the "cancer_data.sql" to produce the data in json format. The app.py file will pull this data once it is run.
<img src="https://github.com/CathyMatthee/predictor_app/blob/main/images/pgAdmin%204.png?raw=true" alt="pgAdmin 4" width="300"/>

### Fetching and API Integration
1. From the root directory of the repo open app.py file
2. Install psycopg2 and numpy if you need to: use pip to install the libraries
3. Create a new file in the root directory called "config.py" which is where you provide your pgAdmin password in a safe manner. 
    - Add this text to the file: "password = "your_password_here" and replace "your_password_here" with your real password.
    - Add: db_host = "localhost"
<img src="https://github.com/CathyMatthee/predictor_app/blob/main/images/config.png?raw=true" alt="config.py" width="300"/>
    Save this in root directory of the cloned repo.
    This "config.py" file is referenced in the .gitignore file for safety reasons and is not present in the github repo. 
4. In git bash terminal activate your dev environment from the local repo and run "python app.py" to make a connection to the database wher the Flask app will serve the database data in JSON, dynamically to the machine learning models, ensuring they are trained on the most up-to-date data. 
6. Select the CTRL+click on the link that is output in the bash terminal that deploys the Flask locally in a window.
7. Select the "Predictor_App" option in the top navigation bar to go straight to the machine learning app that will predict cancer.
<img src="https://github.com/CathyMatthee/predictor_app/blob/main/images/app%20fields.png?raw=true" alt="Predictor App Fields" width="300"/>
<img src="https://github.com/CathyMatthee/predictor_app/blob/main/images/app_result.png?raw=true" alt="Predictor App Results" width="300"/>

# Machine Learning Model Builds

## Keras Tuner Model 4
### Overview
A Keras Tuner model was trained on the cancer dataset to predict whether a cancer is likely benign (B) or malignant (M) based on its visual characteristics. Through hyperparameter tuning, the model was optimised to select the best parameters for the best performing model. 
The notebook, ML_KT_Model documents the workflow of the keras tuner including parameter selection, model training, optimisation of the model and evaluating and testing with the model created. It includes an initial model that specifies the specific parameters used in the model and then a second model that utilises a function to perform hyperparameter tuning for the model. 

### Contents
-**Kt_intial_model.pkl:** This is the unoptimised saved version of the keras tuner model

-**Kt_best_model.pkl:** This is the saved optimised version of the keras tuner model which utilises a function for hyperparameter tuning.

### Steps
1. **Initial Model Training:** Initially a keras tuner model was trained that contained two hidden layers with 8 neurons in the first layer and 5 in the second. The model was then compiled using Adam as the optimizer and the epochs set to 100.
2. **Optimised Model Training:** A second keras tuner model was then trained utilising a function to identify the best parameters for the model. It identified three hidden layers with the first having 32 neurons, the second also 32 and the third 16 neurons. The optimizer was also identified to be Adam and the epoch was set to 50.

### Model Evaluation
The model was evaluated using accuracy, precision, recall, F1 score, and confusion matrices using test data.

### Results
The model's performance slightly improved after hyperparameter tuning, which is reflected in:
- A small increase in accuracy (from 97.18% to 98.59%).
- Minor increase in precision and F1-scores for class 0 (benign).
- A slight decrease in misclassifications for class 1 (malignant) from 3 to 1.

Therefore, hyperparameter tuning slightly increased the model's ability to differentiate between the classes.

## Logistic Regression Model 1

### Overview
This part of the project focuses on building a logistic regression model to classify outcomes based on the dataset. The model was developed and trained to predict binary outcomes (such as B for Benign and M for Malignant in cancer diagnosis). Key steps included data preprocessing, training the model using default settings, and further improving performance through hyperparameter tuning with GridSearchCV.

Additionally, feature selection was attempted, followed by tuning and evaluation to compare the performance of the model with selected features against the one trained with all features. Here, we aimed to determine if the feature-selected model performed similarly for conversion into an app.

### Contents
- **Logistic_Regression_Model_1_ZR.ipynb**: This notebook contains the entire workflow of the logistic regression analysis, including data preprocessing, feature selection, model training, and evaluation results.
- **tuned_logistic_regression_model.pkl**: A saved version of the trained logistic regression model, which can be used for future predictions.

### Steps
1. **Data Preprocessing**: The dataset was cleaned by removing irrelevant columns, treating skewness of the data, and removing outliers. Categorical variables were encoded, and numerical features were scaled.
  
2. **Feature Selection**: Features were selected using Recursive Feature Elimination (RFE) to enhance model performance.

3. **Model Training**: The logistic regression model was trained using the scikit-learn library. Hyperparameter tuning was performed to optimize the model's performance using GridSearchCV.

### Model Evaluation
The model was evaluated using accuracy, precision, recall, F1 score, and confusion matrices using test data.

### Results
The model's performance slightly decreased after feature selection, which is reflected in:
- A small drop in accuracy (from 98% to 96%).
- Minor drops in precision and F1-scores for class 1 (malignant).
- A slight increase in misclassifications for class 0 (benign) from 1 to 2.

Therefore, feature selection with RFE slightly reduced the model's ability to differentiate between the classes. The full feature model performed better than the feature-selected model for logistic regression.

### Next Steps
We explored other machine learning algorithms such as Random Forest, SVM, or Neural Networks to compare performance.

## Random Forest Model 3

### Goal
This part of the project leverages a Random Forest model to classify outc
omes in a medical dataset, particularly predicting benign (B) or malignan
t (M) cancer diagnoses. By utilizing the feature importance from the Ran
dom Forest, we aim to identify the most crucial features contributing to t
hese predictions. We will then refine the dataset by selecting these critica
l features, enhancing both the model's efficiency and interpretability.
 
### Contents
This notebook, ML_RF_Models_AL.ipynb, documents the complete workfl
ow of the Random Forest analysis, feature selection using feature import
ance, model training, optimizing the model  and evaluating and  testing 
with the model created.
 
rf_model.pkl: This is the saved version of the trained Random Forest 
model, ready for future predictions.
 
best_model.pkl: This is the saved and optimized Random Forest model, 
refined with the top 11 features, and is suitable for future predictions 
with fewer features.
 
### Steps
Initially, a basic Random Forest model was trained and the feature import
ance was 
accessed . Based on this analysis, the top 11 features were selected to co
nstruct a refined second model. This second 
model was then optimized using GridSearchCV. To evaluate the performa
nce of the optimized model, metrics such as accuracy score, confusion m
atrix, and cross-validation scores were employed. Finally, the model's per
formance was validated by testing it with unseen data.
 
### Key Findings:
1. Reduced Accuracy: The model's accuracy decreased from 96% to 9
5% after reducing the number of features. This suggests a slight im
pact on overall performance.
2. Importance of Removed Features: The increase in false negatives in
dicates that some of the removed features were essential for correc
tly identifying malignant cases. Their exclusion led to a decrease in 
the model's ability to detect these critical instances.
Lesson Learned:
To conclude, it's apparent that retaining a broader set of features is bene
ficial for achieving optimal results. This ensures that crucial information i
sn't lost, enhancing the model’s ability to make accurate and reliable pre
dictions. This experience underscores the importance of balancing model 
simplicity with the retention of essential predictive information for high-
stakes applications like medical diagnosis.


## Support Vector Machine Model 1

## Keras Tuner Model 4
### Overview
A Keras Tuner model was trained on the cancer dataset to predict whether a cancer is likely benign (B) or malignant (M) based on its visual characteristics. Through hyperparameter tuning, the model was optimised to select the best parameters for the best performing model. 
The notebook, ML_KT_Model documents the workflow of the keras tuner including parameter selection, model training, optimisation of the model and evaluating and testing with the model created. It includes an initial model that specifies the specific parameters used in the model and then a second model that utilises a function to perform hyperparameter tuning for the model. 

### Contents
-**Kt_intial_model.pkl:** This is the unoptimised saved version of the keras tuner model

-**Kt_second_model.pkl:** This is the saved optimised version of the keras tuner model which utilises a function for hyperparameter tuning.

-**kt_best_model.pkl:** This is the model containing the higher accuracy value of the two models. This has been used in the application.

### Steps
1. **Initial Model Training:** Initially a keras tuner model was trained that contained two hidden layers with 8 neurons in the first layer and 5 in the second. The model was then compiled using Adam as the optimizer and the epochs set to 100.
2. **Optimised Model Training:** A second keras tuner model was then trained utilising a function to identify the best parameters for the model. It identified three hidden layers with the first having 32 neurons, the second also 32 and the third 8 neurons. The optimizer was also identified to be Adam and the epoch was set to 50.

### Model Evaluation
The model was evaluated using accuracy, precision, recall, F1 score, and confusion matrices using test data.

### Results
In the case of when the models were run in the notebook the following results were shown:
First Model:
    Accuracy = 0.9859
    Class 0:
        precision = 0.98
        recall = 1.00
        f1-core = 0.99
    Class 1: 
        precision = 1.00
        recall = 0.96
        f1-core = 0.98

Second Model (Optimised):
    Accuracy = 0.9789
    Class 0:
        precision = 0.98
        recall = 0.99
        f1-core = 0.98
    Class 1: 
        precision = 0.98
        recall = 0.96
        f1-core = 0.97

It was identifed that re-running the keras tuner resulted in accuracy fluctuations within ~2% for both models.The two models appeared to perform similarly as when re-running the models, the models would often alternate between which performed better or worse. 
This is considered normal and is likely due to the nature of the keras tuner model and its inherent randomness in the training process. Due to this, as higher accuracy scores were found to usually correlate with higher precision, recall and f1-scores, the model saved is based on the highest accuracy value.

# Ensemble, Looking Forward and Conclusion

An ensemble approach is a method that uses all the models where the ideal algorithm has low bias (accurately models the true relationship in the data) and low variability (producing consistent predictions across different datasets).

Since all of the 4 models produced strong perfomance metrics, we considered leveraging the strengths of all the models with a max count ensemble method or majority rules approach. When resolving any ties, the SVM model was utilised because it resulted in the best performance metrics for the ensemble. Additionally, our SVM model is able to reduce computational resources by reducing dimensionality to only using 5 features. By focussing on these 5 features simplifies the interpretability of the model and can guide future feature engineering efforts to optimise the model further.

The ensemble method produced excellent perfromance metrics with an overall accuracy of 0.98. The precision score of 1.0 indicates that predicted malignancies were 100% accurate. The recall score of 0.94 indicates that 94% of actual malignancies were accurately predicted.

Even though the low error rate of 3 false negatives is a good result, in a medical context the consequences of not identyfing malignancies such as these 3 instances can have serious and longstanding implications.

Collaborating with the healthcare community will enhance the model's performance and reduce error rates. By incorporating carefully curated input from healthcare professionals into the model's prediction app, we can improve its accuracy and relevance.

This prediction app will not only assist healthcare professionals in their decision-making processes but also in the future facilitate the collection of additional data. This continuous feedback loop will ensure that the models remain up-to-date and provide reliable support for clinical decisions.

### Sources   
 - W3 Schools code used to build app navigation https://www.w3schools.com/bootstrap5/bootstrap_navs.php
 - Chat gpt and Codepen used to convert README.md file and app gui html and css formating for the web app pages including the predictor app.
 - Flask linking html pages: https://www.youtube.com/watch?v=VyICzbnf6q4
 - Ensemble method and code https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/
 - Flask psycopg2 crendential connection: https://medium.com/@shahrukhshl0/building-a-flask-crud-application-with-psycopg2-58de201e3c14 and chatgpt.
