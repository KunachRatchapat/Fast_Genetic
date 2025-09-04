import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

data = {'dad_x':[0,0,0,0,1,1],
        'mom_x':[0,1,2,0,1,2],
        'gender':[0,0,0,1,1,1],
        'outcome':[0.0, 0.50, 1.0, 0.0, 0.0, 0.50]}

df = pd.DataFrame(data)

model_xr = LinearRegression()
model_xr.fit(df[['dad_x', 'mom_x', 'gender']], df['outcome'])


joblib.dump(model_xr, 'model_xr.pkl')
