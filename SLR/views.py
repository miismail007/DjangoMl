from django.shortcuts import render
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from django.contrib.staticfiles.storage import staticfiles_storage

# Create your views here.


def index(request):
    return render(request, 'index.html')


def predict(request):
    exp = int(request.GET['exp'])
    rawdata = staticfiles_storage.path('Salary_Data.csv')
    dataset = pd.read_csv(rawdata)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 1].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=0)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    yet_to_predict = np.array([[exp]])
    y_pred = regressor.predict(yet_to_predict)
    accuracy = regressor.score(X_test, y_test)
    accuracy = accuracy*100
    accuracy = int(accuracy)
    return render(request, 'index.html', {'result': "The Salary will be" + str(int(y_pred))})
