from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib



#--- Models ---
models = {
   'Autosomal_Dominant': joblib.load('models/model_ad.pk'),
   'Autosomal_Recessive':joblib.load('models/model_ar.pkl'),
   'X-linked_Dominant': joblib.load('models/model_xd.pkl'),
   'X-linked_Recessive' : joblib.load('models/model_xr.pkl')
}

app = FastAPI()

class GenotypeData(BaseModel):
    inheritance_type:str
    dad_genotype:int 
    mom_genotype:int 
    

@app.post("/predict")
def predict_risk(data: GenotypeData):
    #--- Check model in Data ---
    if data.inheritance_type not in models:
        raise HTTPException(status_code=404, detail="Model not found for this inheritance type")
    
    #--- use true Model in data ---
    model = models[data.inheritance_type]
    
    prediction = model.predict([[data.dad_genotype, data.mom_genotype]])[0]
    
    return {
        "inheritance_type" : data.inheritance_type,
        "predicted_risk" : float(prediction)
    }