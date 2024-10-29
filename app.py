from pipeline import TrainingPipeline
from src.networksecurity.exception import NSException
from src.networksecurity.loggings import logger
from src.networksecurity.utils import load_object, NetworkModel
from src.networksecurity.constants import training_pipeline

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, Request
from uvicorn import run as app_run
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse

import pymongo

import os, sys
import certifi
import uvicorn
import pandas as pd

from dotenv import load_dotenv


load_dotenv()

logger=logger()

MONGO_DB_UN=os.getenv('MONGO_DB_UN')
MONGO_DB_PWD=os.getenv('MONGO_DB_PWD')

ca = certifi.where()

mongo_uri = f"mongodb+srv://{MONGO_DB_UN}:{MONGO_DB_PWD}@networksecuritycluster.bp8lc.mongodb.net/?"
mongo_client = pymongo.MongoClient(mongo_uri, tlsCAFile=ca)



database = mongo_client[training_pipeline.DATA_INGESTION_DATABASE_NAME]
collection = database[training_pipeline.DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()

origin=["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origin,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

templates = Jinja2Templates(directory="./templates")

@app.get("/",tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        training_pipeline =TrainingPipeline()
        training_pipeline.run_pipeline()
        return Response("Training is successfull!")
    except Exception as e:
        logger.info(NSException(e,sys))
        raise NSException(e,sys)
    
@app.post("/predict")
async def predict_route(request:Request, file:UploadFile=File(...)):
    try:
        df=pd.read_csv(file.file)
        #print(df)
        preprocesor=load_object("final_model/preprocessor.pkl")
        final_model=load_object("final_model/model.pkl")
        network_model = NetworkModel(preprocessor=preprocesor,model=final_model)
        print(df.iloc[0])
        y_pred = network_model.predict(df)
        print(y_pred)
        df['predicted_column'] = y_pred
        print(df['predicted_column'])
        #df['predicted_column'].replace(-1, 0)
        #return df.to_json()
        os.makedirs("prediction_output",exist_ok=True)
        df.to_csv('prediction_output/output.csv')
        table_html = df.to_html(classes='table table-striped')
        #print(table_html)
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
    except Exception as e:
        logger.info(NSException(e,sys))
        raise NSException(e,sys)


if __name__=="__main__":
    app_run(app, host="0.0.0.0",port=8000)