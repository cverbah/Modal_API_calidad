from modal import Image, Secret, Mount, Volume, App, asgi_app, gpu
from fastapi import FastAPI, Response, Query, BackgroundTasks
import matplotlib.pyplot as plt
from typing import Union
from utils import thresholding_display, check_url, similarity_score,\
                      get_available_gpus, model_resnet50_v2_avg, model_vit

img_app = App(name="fastapi-image-similarity")
image = (Image.micromamba()
         .micromamba_install("cudatoolkit=11.2", "cudnn=8.1.0", "cuda-nvcc",
         channels=["conda-forge", "nvidia"],
        )
        .pip_install("pandas", "numpy", "matplotlib", "requests", "Pillow", "opencv-python-headless",
                     "jax", "jaxlib", "transformers", "nvidia-tensorrt", "tensorflow~=2.9.1"))



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


@img_app.function(image=image, gpu=gpu.T4(count=1), keep_warm=1)
@asgi_app(label='fastapi-image-similarity')
def img_similarity_app():
    import os
    import warnings
    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # no mostrar los warnings
    print(get_available_gpus())

    return img_app