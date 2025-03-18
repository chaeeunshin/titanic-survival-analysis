#!/usr/bin/env python
# coding: utf-8

import Preprocessing
from sklearn.ensemble import RandomForestRegressor

df = Preprocessing.df

RandomForestModel = RandomForestRegressor(n_estimators=300, random_state = 42)
RandomForestModel.fit(df.x, df.y)