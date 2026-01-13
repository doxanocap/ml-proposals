import base64
import uvicorn
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.models import network

app = FastAPI()
model = network.Network
image_title = "curr"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImageRequest(BaseModel):
    base64: str


class ErrorResponse(BaseModel):
    message: str


@app.post("/image")
async def process_image(image_request: ImageRequest):
    global model, image_title
    try:
        image_bytes = base64.b64decode(image_request.base64)

        output, image_id = model.evaluate_img(image_bytes)

        return {
            "number": output,
            "image_id": image_id
        }

    except Exception as e:
        print("err", str(e))
        response = ErrorResponse(message="internal server error")
        return JSONResponse(content=jsonable_encoder(response), status_code=500)


@app.options("/image")
async def options_image():
    return {"message": "OPTIONS request received. Please use POST method for image upload."}


def run(init_model):
    global model, image_title
    model = init_model
    uvicorn.run("src.rest:app", host="0.0.0.0", port=8000, reload=False)
