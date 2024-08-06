# libraries
import pandas as pd
import numpy as np
from PIL import Image as PIL_Image
import cv2
import requests
import matplotlib.pyplot as plt
import io
import tensorflow as tf
from transformers import TFViTModel, ViTImageProcessor
from numpy.linalg import norm
import os
import warnings
from tensorflow.python.client import device_lib
from dotenv import load_dotenv
plt.style.use('seaborn-white')
warnings.filterwarnings('ignore')


# envs variables
#load_dotenv()
#os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'key.json'
CACHE_DIR = '/Users/cvergarabah/.cache/huggingface/hub'


# Resnet50v2-avg Pooling
model_resnet50_v2_avg = tf.keras.applications.ResNet50V2(
    include_top=False,
    weights='imagenet',  # 'imagenet' (pre-training on ImageNet).
    input_tensor=None,
    input_shape=None,
    pooling='avg',# global avg pooling will be applied
)

# VIT Model
preprocess_img = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model_vit = TFViTModel.from_pretrained("google/vit-base-patch16-224-in21k", cache_dir=CACHE_DIR)

# ########### Functions #####################

def crop_product(url_img, blur_kernel=15):
    '''crops a product from an image'''
    try:
        headers = {'User-Agent': 'Mozilla/5.0 ...'}
        # fix possible errors in url: <--- PUEDE SER MEJORADO ESTA PARTE. para casos con la url con ruido
        if url_img[:8] == 'https://':
            url_img = url_img.split('https://')[-1]
            url_img = 'https://' + url_img

        elif url_img[:7] == 'http://':
            url_img = url_img.split('http://')[-1]
            url_img = 'http://' + url_img

        elif url_img[:8] != 'https://':
            url_img = 'https:' + url_img

        response = requests.get(url_img, stream=True, headers=headers)
        img = PIL_Image.open(io.BytesIO(response.content))
        img_mode = img.mode

        # change channels to RGB
        if img_mode == 'RGBA':
            img_aux = PIL_Image.new("RGB", img.size, (255, 255, 255))
            img_aux.paste(img, mask=img.split()[3])
            img = img_aux

        if img_mode == 'CMYK':
            img = img.convert('RGB')

        if img_mode == 'P':
            img = img.convert('RGB', palette=PIL_Image.ADAPTIVE, colors=256)

        if img_mode == 'L':
            img = img.convert('RGB')
        else:
            img = img.convert('RGB')

        img = np.array(img)
        #thresholding
        gray = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY)
        gray = cv2.blur(gray, (blur_kernel, blur_kernel))
        thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)[1] #min 252 antes
        #thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY +cv2.THRESH_OTSU)[1] # #BUENO para el caso en que el fondo no es blanco y el color es muy distinto al del objeto
        alpha = 1  # for undefined cases : x/0 (no white pixels)
        ratio = cv2.countNonZero(thresh)/((img.shape[0] * img.shape[1]) - cv2.countNonZero(thresh) + alpha)

        if ratio > 2:  # no crop. ratio=2 good enough?
            cropped = img
            return img, cropped, thresh, ratio

        # ratio<2,  getting the max countour from img (product)
        contours, hierarchy = cv2.findContours(thresh, 1, 2)
        max_a = 0
        for contour in contours:
            x_aux, y_aux, w_aux, h_aux = cv2.boundingRect(contour)
            a = w_aux * h_aux
            if a > max_a:
                max_a = a
                x, y, w, h = x_aux, y_aux, w_aux, h_aux

        cropped = img.copy()[y:y + h, x:x + w]

    except Exception as e:
        return e

    return img, cropped, thresh, ratio


def thresholding_display(img1, img2):
    '''plots the base image, thresholded and cropped'''
    try:

        img1_display, img1_cropped, img1_threshold, img1_ratio = crop_product(img1)
        img1_ratio = round(img1_ratio, 3)

        img2_display, img2_cropped, img2_threshold, img2_ratio = crop_product(img2)
        img2_ratio = round(img2_ratio, 3)

        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 8)) # define subplots
        #Customer
        #original
        ax[0,0].set_xlabel('pixels')
        ax[0,0].set_ylabel('pixels')
        ax[0,0].set_title("Original Cliente, crop=0")
        ax[0,0].imshow(img1_display)
        #thresholded
        ax[0,1].set_title("Thresholded")
        ax[0,1].set_xlabel(f'Ratio: {img1_ratio}')
        ax[0,1].imshow(img1_threshold)
        #cropped
        ax[0,2].set_title("Cropped, crop=1")
        ax[0,2].imshow(img1_cropped)

        #Retail
        ax[1,0].set_xlabel('pixels')
        ax[1,0].set_ylabel('pixels')
        ax[1,0].set_title("Original Retail, crop=0")
        ax[1,0].imshow(img2_display)

        ax[1, 1].set_title("Thresholded")
        ax[1, 1].set_xlabel(f'Ratio: {img2_ratio}')
        ax[1, 1].imshow(img2_threshold)

        ax[1,2].set_title("Cropped, crop=1")
        ax[1,2].imshow(img2_cropped)

        fig.tight_layout()

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        plt.close(fig)

    except Exception as e:
        return e

    return img_buf


def check_url(url):
    ''''returns code status from requesting a url'''
    code = requests.head(url).status_code
    return code


def cosine_distance(url_img1, url_img2, model, crop=0):
    '''calculates cosine distance between 2 images'''
    assert (model == model_resnet50_v2_avg or model == model_vit), 'wrong input for model'
    assert crop in {0, 1}, 'no crop:0, crop:1'
    try:
        if crop:
            img1 = url_img1
            img2 = url_img2

        if not crop:  # change channels to RGB
            headers = {'User-Agent': 'Mozilla/5.0 ...'}

            response_img1 = requests.get(url_img1, stream=True, headers=headers)
            img1 = PIL_Image.open(io.BytesIO(response_img1.content))

            response_img2 = requests.get(url_img2, stream=True, headers=headers)
            img2 = PIL_Image.open(io.BytesIO(response_img2.content))

            imgs = []
            for img in [img1, img2]:
                # change channels to RGB
                if img.mode == 'RGBA':
                    img_aux = PIL_Image.new("RGB", img.size, (255, 255, 255))
                    img_aux.paste(img, mask=img.split()[3])
                    img = img_aux

                if img.mode == 'CMYK':
                    img = img.convert('RGB')

                if img.mode == 'P':
                    img = img.convert('RGB', palette=PIL_Image.ADAPTIVE, colors=256)

                if img.mode == 'L':
                    img = img.convert('RGB')
                else:
                    img = img.convert('RGB')

                imgs.append(img)
            img1 = imgs[0]
            img2 = imgs[1]

        # Generating the embedding for each img depending on the model used.
        embeddings = []
        for img in [img1, img2]:

            if model == model_resnet50_v2_avg:
                # preprocessing
                img = np.array(img)
                img = tf.keras.applications.resnet_v2.preprocess_input(img)
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
                img = np.expand_dims(img, axis=0)
                # embedding
                embedding = model.predict(img)[0]
                embeddings.append(embedding)

            if model == model_vit:
                # preprocessing
                inputs = preprocess_img(img, return_tensors="tf")
                # embedding
                embedding = model(**inputs).last_hidden_state[0][0].numpy()
                embeddings.append(embedding)

        embedding1 = embeddings[0]
        embedding2 = embeddings[1]
        # cosine_distance
        distance = np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))

    except Exception as e:
        print(e)
        return -1

    if distance > 1:
        return 1
    if distance < 0:
        return 0

    return distance


def similarity_score(url_img1, url_img2, model, crop=1, blur_kernel=15):
    '''calculates the similarity score between 2 imgs using a defined model'''
    if crop:
        img_cliente = crop_product(url_img1, blur_kernel=blur_kernel)[1]
        img_retail = crop_product(url_img2, blur_kernel=blur_kernel)[1]
        score = cosine_distance(img_cliente, img_retail, model, crop=crop)

    if not crop:
        score = cosine_distance(url_img1, url_img2, model, crop=crop)

    return score


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
