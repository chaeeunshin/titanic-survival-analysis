#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

titanic = pd.read_csv('Titanic.csv')
df = pd.DataFrame(titanic) # import dataset

# Data cleansing
excluded = ['PassengerId', 'Name', 'Ticket', 'Cabin'] # Remove irrelevant information
df = df.drop(columns=excluded)
df = df.dropna(subset=['Age']) # Remove na
df = df.dropna(subset=['Embarked'])

# Modify categorical data 
df['Sex'] = df['Sex'].replace({'male':0,'female':1})
df['Embarked'] = df['Embarked'].replace({'C':0,'Q':1,'S':2})

# Split predictor and response variables
x = df.drop('Survived', axis = 1)
y = df['Survived']

# Scale numeric data in predictor to improve accuracy in logistic regression and neural network
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)