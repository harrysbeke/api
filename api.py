from fastapi import FastAPI
import uvicorn
import pickle
import pandas as pd

data = pd.read_csv('data_op.csv')
model = pickle.load(open('model.pkl', 'rb'))
pipe = pickle.load(open('pipe.pkl', 'rb'))
#Initialisation de l'api
api = FastAPI()

def get_prediction(user_id):
    df = data[data['SK_ID_CURR']==user_id]
    df = pipe.transform(df)
    dt = model.predict_proba(df)
    prediction = model.predict(df)[0]
    probability = dt[0][prediction]
    return int(prediction), float(probability)

# Récupération
@api.get('/')
async def test():
    return {'presentation':'Fermer'}

@api.get('/test/{entier}')
async def preview(entier:int):
    result = get_prediction(entier)
    result = int(result)
    return {"id":result}

@api.get('/prediction/{user_id}')
async def prediction(user_id:int):
    user_id = int(user_id)
    prediction, probability = get_prediction(user_id)
    #prediction = int(prediction)
    #probability = float(probability)
    return {'prediction':prediction, 'probability':probability}

if __name__ == '__main__':
    uvicorn.run(api,host='127.0.0.1',port=8000)