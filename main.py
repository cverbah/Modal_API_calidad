from modal import Image, Secret, Mount, Volume, App, asgi_app, gpu
from fastapi import FastAPI, Response, Query, BackgroundTasks, Request
import matplotlib.pyplot as plt
from typing import List, Annotated, Union
from utils import thresholding_display, check_url, similarity_score, get_available_gpus, model_resnet50_v2_avg, model_vit
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
import time

app = App(name="fastapi-image-similarity")
image = (Image.micromamba()
         .micromamba_install("cudatoolkit=11.2", "cudnn=8.1.0", "cuda-nvcc",
         channels=["conda-forge", "nvidia"],
        )
        .pip_install("pandas==2.2.2", "numpy==1.24.3", "matplotlib==3.7.1", "requests", "Pillow==10.3.0",
                     "opencv-python-headless==4.10.0.84","jax", "jaxlib", "transformers==4.40.0", "nvidia-tensorrt",
                     "fastapi==0.111.0", "tensorflow~=2.9.1"))

class MyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers['X-Process-Time'] = str(process_time)
        return response


img_app = FastAPI(title='ImageSimilarityAPI',
                        summary="Image Similarity API", version="1.1",
                        contact={"name": "Cristian Vergara",
                                 "email": "cvergara@geti.cl"})

img_app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'])
img_app.add_middleware(MyMiddleware)


@img_app.get("/")
async def read_root():
    return {"Hello": "World"}


@img_app.get("/plot")
async def plot_imgs(url_img_cliente: str, url_img_retail: str, background_tasks: BackgroundTasks):
    plt.style.use('seaborn-white')
    try:
        for index, url in enumerate([url_img_cliente, url_img_retail]):
            try:
                check_url(url)
            except Exception as e:
                response = {
                    "error": str(e),
                    "url with error": str(url),
                }
                return response

        img_buf = thresholding_display(url_img_cliente, url_img_retail)
        background_tasks.add_task(img_buf.close)
        headers = {'Content-Disposition': 'inline; filename="out.png"'}

    except Exception as e:
        response = {
            "error": str(e),
        }, 404
        return response

    return Response(img_buf.getvalue(), headers=headers, media_type='image/png')


@img_app.get("/predict")
async def predict_score(url_img_cliente: str, url_img_retail: str, crop: int = Query(1, enum=[1, 0]),
                        model: str = Query("VIT_Transformer", enum=["Resnet50_v2", "VIT_Transformer"])):
    try:
        for url in [url_img_cliente, url_img_retail]:
            try:
                check_url(url)
            except Exception as e:
                response = {
                    "error": str(e),
                    "url with error": str(url),
                }, 404
                return response

        if model == 'Resnet50_v2':
            model_used = model_resnet50_v2_avg

        elif model == 'VIT_Transformer':
            model_used = model_vit

        score = similarity_score(url_img_cliente, url_img_retail, model_used, crop=crop)

        response = {
            "similarity_score": round(float(score), 4),
            "model": str(model),
            "crop": int(crop),
            "url_cliente": str(url_img_cliente),
            "url_retail": str(url_img_retail),
        }
        return response

    except Exception as e:
        response = {
            "error": str(e)
        }
        return response


@app.function(image=image,
              gpu=gpu.T4(count=1),
              secret=Secret.from_name("automatch-secret-keys"),)
@asgi_app(label='fastapi-image-similarity')
def img_similarity_app():
    import os
    import warnings
    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # no mostrar los warnings
    print(get_available_gpus())
    print('### Img Similarity Scores ###')

    return img_app