"""
CONCEPTS: 
- The dataset will be divided in two. One for training and the other for testing

- Machine learning is divided in Supervised and Unsupervised learning.

"""

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Print the files names found on the dataset folder
import os

for dirname, _, filenames in os.walk("./dataset"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno


# Leear el archivo
df = pd.read_csv("./dataset/heart.csv")

# Change the column names
new_columns_headers = [
    "age",
    " sex",
    "cp",
    "trtbps",
    "chol",
    "fbs",
    " rest_ecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]
df.columns = new_columns_headers

# * Print information about the dataset
print(df.head(), "\n")  # Print the first 5 rows
print("Shape", df.shape, "\n")  # Print the shape of the dataset
print(df.info(), "\n")  # Print the information of the dataset
print(df.isnull(), "\n")  # Print the null values of the dataset
print(df.isnull().sum(), "\n")  # Print the sum of the null values of the dataset


# * Validate the null values of the dataset
def validate_null_variables():

    isnull_number = []
    for i in df.columns:
        isnull_number.append(df[i].isnull().sum())
    print(pd.DataFrame(isnull_number, index=df.columns, columns=["Null Values"]))

    msno.bar(df, color="b")
    plt.show()


# validate_null_variables()

# * List unique values
# This is usefull to know which columns are categorical and which are numerical
unique_values = []
for i in df.columns:
    x = df[i].value_counts().count()
    unique_values.append(x)
pd.DataFrame(unique_values, index=df.columns, columns=["Unique Values"])


# TODO. FALTA COPIAR EL CODIGO DE ESTE CAPITULO/CURSO
# * Analyze the numerical and categorical variables
numeric_var = [
    "age",
    "trtbps",
    "chol",
    "thalach",
    "oldpeak",
]  # This is to separate the numerical variables
categoric_var = [
    "sex",
    "cp",
    "fbs",
    "rest_ecg",
    "exang",
    "slope",
    "ca",
    "thal",
    "target",
]


def analyze_numerical_and_categorical():
    print(df[numeric_var].describe())

    # * Show the distribution of the numerical variables
    """sns.distplot(df["age"], hist_kws=dict(linewidth=1, edgecolor="k"))  # deprecated
    sns.histplot(df["age"], kde=True, bins=20)  # new

    sns.distplot(
        df["trtbps"], hist_kws=dict(linewidth=1, edgecolor="k"), bins=20
    )  # deprecated

    sns.distplot(df["chol"], hist=False)

    x, y = plt.subplots(figsize=(8, 6))
    sns.distplot(df["thalach"], hist=False, ax=y)
    y.axvline(df["thalach"].mean(), color="r", linestyle="--")

    x, y = plt.subplots(figsize=(8, 6))

    # show all the sns plots
    plt.show()"""

    # * CREATING BAR CHARTS
    # * Showing numeric variables in cycle
    """numeric_axis_name = [
        "Age of the patient",
        "Resting Blood Pressure",
        "Cholesterol",
        "Maximum Heart Rate Achieve",
        "ST Depression",
    ]
    var_names = list(zip(numeric_var, numeric_axis_name))

    title_font = {"family": "arial", "color": "darkred", "weight": "bold", "size": 15}
    axis_font = {"family": "arial", "color": "darkblue", "weight": "bold", "size": 13}
    for i, z in var_names:
        plt.figure(figsize=(8, 6), dpi=80)
        sns.distplot(df[i], hist_kws=dict(linewidth=1, edgecolor="k"), bins=20)
        plt.title(i, fontdict=title_font)
        plt.xlabel(z, fontdict=axis_font)
        plt.ylabel("Density", fontdict=axis_font)

        plt.tight_layout()
        plt.show()"""
        
    # * CREATING PIE CHARTS
    # * Showing categorical variables in cycle
    categoric_axis_name = [
        "Gender",
        "Chest Pain Type",
        "Fasting Blood sugar",
        "Resting Electrocardiographic Results",
        "Exercise Induced Angina",
        "The Slope of ST Segment",
        "Number of Major Vessels",
        "Thal",
        "Target",
    ]
    var_names = list(zip(categoric_var, categoric_axis_name))
    
    title_font = {"family": "arial", "color": "darkred", "weight": "bold", "size": 15}
    axis_font = {"family": "arial", "color": "darkblue", "weight": "bold", "size": 13}
    
    for i, z in var_names:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        observation_values = list(df[i].value_counts().index)
        total_observation_values = list(df[i].value_counts())
        
        ax.pie(total_observation_values, labels=observation_values, autopct="%1.1f%%", startangle=110, labeldistance=1.1)
        ax.axis("equal")
        
        plt.title(f"{i}( {z} )", fontdict=title_font)
        plt.legend()
        plt.show()
        


analyze_numerical_and_categorical()


# Lee los archivos, divide el contenido en dos partes, una para entrenar y otra para testear
# Luego has un modelo de regresión lineal y lo entrena con los datos de entrenamiento y validas con los datos de testeo
# Finalmente, muestra el resultado de la predicción , relaliza predicciones
