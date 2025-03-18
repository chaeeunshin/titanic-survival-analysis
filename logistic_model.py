#!/usr/bin/env python
# coding: utf-8

import Preprocessing
from sklearn.linear_model import LogisticRegression
df = Preprocessing.df

LogisticModel = LogisticRegression()
LogisticModel.fit(df.x_scaled, df.y)