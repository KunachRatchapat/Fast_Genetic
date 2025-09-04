import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

#--- Dummy Data ---
data = {'dad': [0,0,0,1,1,1,2,2,2],
        'mom': [0,1,2,0,1,2,0,1,2],
        'outcome':[1.0, 1.0, 1.0, 1.0, 0.75, 0.50, 1.0, 0.50, 0.0]}  # โอกาสที่ลูกจะเป็นโรค (AA,Aa)

df = pd.DataFrame(data)


#--- Train Model ---
model_ad = LinearRegression()
model_ad.fit(df[['dad', 'mom',]], df['outcome'])

#--- Save File pkl ---
joblib.dump(model_ad, 'model.ad.pk')