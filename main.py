import pandas as pd
import numpy as np
import csv
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# === Pre-processing data ===================================================================================================-

#Checks if sparsity rate is larger then threshold(0.1). Returns boolean value
def checkSparsity(df, col):
    if df[col].isna().sum()/len(df[col]) > 0.1:
        return True
    else:
        return False

#If numerical columns we impute with mean value, else we impute the nominal ones with the most frequent value
def imputation(df, col):
    if col.startswith('Num'):
        df[col]= df[col].fillna(df[col].mean())
    else:
        df[col]= df[col].fillna(df[col].value_counts().idxmax())
    return df

#Function handling NaN values
def handle_NaN_values(df):
    for col in df.columns:
        if checkSparsity(df, col):
            del df[col]
        else:
            imputation(df, col)
    return df

#Function to detect outliers, given threshold on 2.25. Only performed with training dataset
def outlierDetection(df, df_num):
    outliers= df[(np.abs(stats.zscore(df_num)) < 2.25).all(axis=1)]
    return outliers

#Normalize the dataset using min_max scaling
def min_max(df, num):
    for column in num.columns:
        df[column]= (df[column]-df[column].min())/(df[column].max()-df[column].min())
    return df

#Pre-processing of training dataset
def pre_processing_train_data(df):
    df_NaN= handle_NaN_values(df)
    df_Num= df_NaN.iloc[:, 0: 103]
    outlierDF= outlierDetection(df_NaN, df_Num)
    outlierNum= outlierDF.iloc[:, 0: 103]
    df_min_max= min_max(outlierDF, outlierNum)
    df_min_max.to_csv('datasets/Ecoli_train_preprocessed.csv', index= False)

#Pre-processing of test dataset
def pre_processing_test_data(df):
    df_NaN= handle_NaN_values(df)
    df_Num= df_NaN.iloc[:, 0: 103]
    df_min_max= min_max(df_NaN, df_Num)
    df_min_max.to_csv('datasets/Ecoli_test_preprocessed.csv', index= False)

# === Training and prediction phase ===================================================================================================
#Train the classifier and perform CV on the training dataset
def train_data(df):
    X= df.drop('Target (Col 107)', axis=1)
    y= df['Target (Col 107)']
    classifier= KNeighborsClassifier(n_neighbors= 9, p=3)
    clf= classifier.fit(X, y)
    f1 = round(cross_val_score(clf, X, y, cv=10, scoring='f1').mean(), 3)
    accuracy= round(cross_val_score(clf, X, y, cv=10, scoring='accuracy').mean(), 3)
    return clf, accuracy, f1

#Predict data and append results to csv file
def predict_data(df_test, list):
    f= open("s4760317.csv", "w", newline='')
    writer= csv.writer(f)
    clf= list[0]
    y_pred= clf.predict(df_test).tolist()
    for element in y_pred:
        i= int(element)
        s= str(i)
        writer.writerow(s)
    lastrow=[]
    lastrow.append(list[1])
    lastrow.append(list[2])
    writer.writerow(lastrow)

if __name__ == "__main__":
    df_train= pd.read_csv('datasets/Ecoli.csv')
    df_test= pd.read_csv('datasets/Ecoli_test.csv')
    pre_processing_train_data(df_train)
    pre_processing_test_data(df_test)
    df_train_preprocessed= pd.read_csv('datasets/Ecoli_train_preprocessed.csv')
    df_test_preprocessed= pd.read_csv('datasets/Ecoli_test_preprocessed.csv')
    train= train_data(df_train_preprocessed)
    predict_data(df_test_preprocessed, train)
