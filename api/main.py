import logging
import random
from contextlib import asynccontextmanager

import PIL
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from utils.model_func import (
    class_id_to_label, load_pt_model,
    load_sklearn_model, transform_image
)

logger = logging.getLogger('uvicorn.info')

class ImageResponse(BaseModel):
    class_name: str
    class_index: int
    
class TextInput(BaseModel):
    text: str
    
class TextResponse(BaseModel):
    label: str
    prob: float
    
class TableInput(BaseModel):
    feature1: float
    feature2: float
    
class TableOutput(BaseModel):
    prediction: float
    
pt_model = None
sk_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pt_model
    global sk_model
    pt_model = load_pt_model()
    logger.info('Torch model loaded')
    
    sk_model = load_sklearn_model()
    logger.info('Sklearn model loaded')
    yield
    
    del pt_model, sk_model
    
app = FastAPI(lifespan=lifespan)

@app.get('/')
def return_info():
    return 'Hello FastAPI'

@app.post('/clf_image')
def classify_image(file: UploadFile):
    image = PIL.Image.open(file.file)
    transformed_image = transform_image(image)
    logger.info(f'{transformed_image.shape}')
    
    with torch.inference_mode():
        pred_index = pt_model(transformed_image).numpy().argmax()
        
    imagenet_class = class_id_to_label(pred_index)
    response = ImageResponse(
        class_name=imagenet_class,
        class_index=pred_index
    )
    
    return response

@app.post('/clf_table')
def predict(x: TableInput):
    prediction = sk_model.predict(
        np.array([x.feature1, x.feature2]).reshape(1,2)
    )
    result = TableOutput(prediction=prediction[0])
    return result

@app.post('/clf_text')
def clf_text(data: TextInput):
    pred_class = random.choice(['positive', 'negative'])
    probability = random.random()
    
    response = TextResponse(
        label=pred_class,
        prob=probability
    )
    return response


if __name__ == '__main__':
    uvicorn.run('main:app', host='127.0.0.1', port=8000, reload=True)        
        