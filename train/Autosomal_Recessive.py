import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

data = {'dad':[0,0,0,1,1,1,2,2,2],
        'mom':[0,1,2,0,1,2,0,1,2],
        'outcome':[0.0, 0.0, 0.0, 0.0, 0.25, 0.50, 0.0, 0.50, 1.0]} # 0=AA, 1=Aa, 2=aa
df = pd.DataFrame(data)

#--- Train Model ---
model_ar = LinearRegression()
model_ar.fit(df[['dad', 'mom']], df['outcome'])

#--- Save File pkl ---
joblib.dump(model_ar, 'model_ar.pkl') 