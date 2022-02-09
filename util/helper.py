#Filename:	helper.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sab 02 Jan 2021 06:50:34  WIB

import warnings
warnings.filterwarnings('ignore')

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

def load_adult_income(filename):

    income_df = pd.read_csv(filename)
    income_df.replace("?", np.nan, inplace= True)

    for col in income_df.columns:
        if income_df[col].dtype == np.float64:
            income_df[col].fillna(income_df[col].mean()[0], inplace = True)
        elif income_df[col].dtype  == object:
            income_df[col].fillna(income_df[col].mode()[0], inplace = True)
        else:
            continue
    
    #income_df.drop(["fnlwgt", "educational-num"], axis = 1, inplace = True)
    income_df.drop(["fnlwgt"], axis = 1, inplace = True)
    income_df.at[income_df[income_df['income'] == '>50K'].index, 'income'] = 1
    income_df.at[income_df[income_df['income'] == '<=50K'].index, 'income'] = 0

    income_df['education'].replace('Preschool', 'dropout',inplace=True)
    income_df['education'].replace('10th', 'dropout',inplace=True)
    income_df['education'].replace('11th', 'dropout',inplace=True)
    income_df['education'].replace('12th', 'dropout',inplace=True)
    income_df['education'].replace('1st-4th', 'dropout',inplace=True)
    income_df['education'].replace('5th-6th', 'dropout',inplace=True)
    income_df['education'].replace('7th-8th', 'dropout',inplace=True)
    income_df['education'].replace('9th', 'dropout',inplace=True)
    income_df['education'].replace('HS-Grad', 'HighGrad',inplace=True)
    income_df['education'].replace('HS-grad', 'HighGrad',inplace=True)
    income_df['education'].replace('Some-college', 'CommunityCollege',inplace=True)
    income_df['education'].replace('Assoc-acdm', 'CommunityCollege',inplace=True)
    income_df['education'].replace('Assoc-voc', 'CommunityCollege',inplace=True)
    income_df['education'].replace('Bachelors', 'Bachelors',inplace=True)
    income_df['education'].replace('Masters', 'Masters',inplace=True)
    #income_df['education'].replace('Prof-school', 'Masters',inplace=True)
    income_df['education'].replace('Prof-school', 'Doctorate',inplace=True)
    income_df['education'].replace('Doctorate', 'Doctorate',inplace=True)

    income_df = income_df.replace({'workclass': {'Without-pay': 'Other/Unknown', 'Never-worked': 'Other/Unknown'}})
    income_df = income_df.replace({'workclass': {'Federal-gov': 'Government', 'State-gov': 'Government', 'Local-gov':'Government'}})
    income_df = income_df.replace({'workclass': {'Self-emp-not-inc': 'Self-Employed', 'Self-emp-inc': 'Self-Employed'}})
    income_df = income_df.replace({'workclass': {'Never-worked': 'Self-Employed', 'Without-pay': 'Self-Employed'}})
    income_df = income_df.replace({'workclass': {'?': 'Other/Unknown'}})


    occupation_map = {
        "Adm-clerical": "Admin", "Armed-Forces": "Military",
        "Craft-repair": "Blue-Collar", "Exec-managerial": "White-Collar",
        "Farming-fishing": "Blue-Collar", "Handlers-cleaners":
            "Blue-Collar", "Machine-op-inspct": "Blue-Collar", "Other-service":
            "Service", "Priv-house-serv": "Service", "Prof-specialty":
            "Professional", "Protective-serv": "Other", "Sales":
            "Sales", "Tech-support": "Other", "Transport-moving":
            "Blue-Collar"
    }

    income_df['occupation'] = income_df['occupation'].map(occupation_map)

    income_df['marital-status'].replace('Never-married', 'NotMarried',inplace=True)
    income_df['marital-status'].replace(['Married-AF-spouse'], 'Married',inplace=True)
    income_df['marital-status'].replace(['Married-civ-spouse'], 'Married',inplace=True)
    income_df['marital-status'].replace(['Married-spouse-absent'], 'NotMarried',inplace=True)
    income_df['marital-status'].replace(['Separated'], 'Separated',inplace=True)
    income_df['marital-status'].replace(['Divorced'], 'Separated',inplace=True)
    income_df['marital-status'].replace(['Widowed'], 'Widowed',inplace=True)

    income_df['native-country'] = income_df['native-country'].apply(lambda el: 1 if el.strip() == "United-States" else 0)
    income_df['education'] = income_df['education'].map({'dropout':1, 'HighGrad':2, 'CommunityCollege':3, 'Bachelors':4, 'Masters':5, "Doctorate":6})

    return income_df

def load_heloc(filename):

    df = pd.read_csv(filename)
    x_cols = list(df.columns.values)
    for col in x_cols:
        df[col][df[col].isin([-7, -8, -9])] = 0
    # Get the column names for the covariates and the dependent variable
    df = df[(df[x_cols].T != 0).any()]
    df['RiskPerformance'] = df['RiskPerformance'].map({'Good':1, 'Bad':0})
    return df

def load_lending_club(filename):

    loans_2007 = pd.read_csv(filename)
    loans_2007 = loans_2007[(loans_2007['loan_status'] == 'Fully Paid') | (loans_2007['loan_status'] == 'Charged Off')]
    mapping_dict = {
        'loan_status':{
            'Fully Paid':1,
            'Charged Off':0
        }
    }
    loans_2007.replace(mapping_dict, inplace=True)

    cols_drop_1 = ['id', 'member_id', 'funded_amnt', 'funded_amnt_inv', 'grade',
               'sub_grade', 'emp_title', 'issue_d', 'zip_code']

    loans_2007.drop(cols_drop_1, axis=1, inplace=True)
    cols_drop_2 = ['out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp',
              'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
              'last_pymnt_d', 'last_pymnt_amnt']
    loans_2007.drop(cols_drop_2, axis=1, inplace=True)

    for col in loans_2007.columns:
        # Drop any null values
        non_null = loans_2007[col].dropna()
        # Check the number of unique values
        unique_non_null = non_null.unique()
        num_true_unique = len(unique_non_null)
        # Remove the col if there is only 1 unique value
        if num_true_unique == 1:
            loans_2007.drop(col, axis=1, inplace=True)

    more_than_1pct = ['pub_rec_bankruptcies']
    loans_2007.drop(more_than_1pct, axis=1, inplace=True)
    loans_2007.dropna(inplace = True)

    cols_drop = ['addr_state', 'title']
    loans_2007.drop(cols_drop, axis=1, inplace=True)

    for col in ['last_credit_pull_d', 'earliest_cr_line']:
        loans_2007.loc[:, col] = pd.DatetimeIndex(loans_2007[col]).astype(np.int64)*1e-9

    float_cols = ['int_rate', 'revol_util']
    for col in float_cols:
        loans_2007[col] = loans_2007[col].str.rstrip('%').astype(float)
    
    mapping_dict = {
        'emp_length': {
            '10+ years': 10,
            '9 years': 9,
            '8 years': 8,
            '7 years': 7,
            '6 years': 6,
            '5 years': 5,
            '4 years': 4,
            '3 years': 3,
            '2 years': 2,
            '1 year': 1,
            '< 1 year': 0,
            'n/a': 0
        }
    }

    loans_2007.replace(mapping_dict, inplace=True)
    nan_columns = ['emp_length', 'revol_util']
    loans_2007['revol_util'].fillna(loans_2007['revol_util'].mean(), inplace = True)
    loans_2007['emp_length'].fillna(loans_2007['emp_length'].mode()[0], inplace = True)

    loan_status = loans_2007['loan_status']
    idx1 = np.argwhere(loan_status.values == 1).squeeze()
    idx2 = np.argwhere(loan_status.values == 0).squeeze()
    idx3 = idx1[0:len(idx2)]
    idx4 = np.concatenate((idx2, idx3))
    np.random.seed(0)
    np.random.shuffle(idx4)
    idx5 = list(set(np.arange(len(df))).difference(idx4))

    data1 = loans_2007.iloc[idx4]
    data2 = loans_2007.iloc[idx5]

    return data1, data2

def load_bank_market(filename):

    data = pd.read_csv(filename, sep = ";")
    data = data.drop_duplicates()
    data.drop("duration", axis = 1, inplace = True)
    data['y'].replace({"no":0, "yes":1}, inplace = True)

    cat_columns = data.select_dtypes("object").columns.tolist()
    y = data['y']
    idx1 = np.argwhere(y.values == 0).squeeze()
    idx2 = np.argwhere(y.values == 1).squeeze()
    idx3 = idx1[0:5000]
    idx4 = np.concatenate((idx2, idx3))
    np.random.shuffle(idx4)
    idx5 = list(set(np.arange(len(data))).difference(idx4))

    data1 = data.iloc[idx4]
    data2 = data.iloc[idx5]

    return data1, data2

def load_GMSC(filename):

    df = pd.read_csv(filename)
    df.SeriousDlqin2yrs = 1 - df.SeriousDlqin2yrs # for simply computing the CF
    df = df.drop("Unnamed: 0", axis=1) # drop id column
    df = df.loc[df["DebtRatio"] <= df["DebtRatio"].quantile(0.975)]
    df = df.loc[(df["RevolvingUtilizationOfUnsecuredLines"] >= 0) & (df["RevolvingUtilizationOfUnsecuredLines"] < 13)]
    df = df.loc[df["NumberOfTimes90DaysLate"] <= 17]
    dependents_mode = df["NumberOfDependents"].mode()[0] # impute with mode
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(dependents_mode)
    income_median = df["MonthlyIncome"].median()
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(income_median)
    mean = df["MonthlyIncome"].mean()
    std = df["MonthlyIncome"].std()
    df.loc[df["MonthlyIncome"].isnull()]["MonthlyIncome"] = np.random.normal(loc=mean, scale=std, size=len(df.loc[df["MonthlyIncome"].isnull()]))

    y = df['SeriousDlqin2yrs']
    idx1 = np.argwhere(y.values == 1).squeeze()
    idx2 = np.argwhere(y.values == 0).squeeze()
    idx3 = idx1[0:len(idx2)]
    idx4 = np.concatenate((idx2, idx3))
    np.random.seed(0)
    np.random.shuffle(idx4)
    idx5 = list(set(np.arange(len(df))).difference(idx4))

    data1 = df.iloc[idx4]
    data2 = df.iloc[idx5]

    return data1, data2
    
def data_loader_torch(X_train, X_test, y_train, y_test, batch_size = 128):

    Train = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    Test = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(Train, batch_size = batch_size)
    test_loader = DataLoader(Test, batch_size = batch_size)

    return train_loader, test_loader
