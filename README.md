# ImageSimilarity_API
1 - Computes the similarity between 2 images using the cosine distance </br>
2 - Plots the thresholded and cropped images from the images urls </br> </br>

## Built with
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/) <br />
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)


#Available Models used for generating the images embeddings: 
- <b>Vision Transformer(VIT)</b> 
- <b>Resnet50v2</b> 

Required Inputs: </br>
- Customer Image URL
- Retail Image URL


You need:
- To set up a modal account [Modal: Serverless platform for AI teams](https://modal.com/)

For running the app: <br>
1. `pip install -r requirements.txt`
2. `modal serve main.py`    For serving the app OR
3. `modal deploy main.py` For deploying the app in the modal cloud

