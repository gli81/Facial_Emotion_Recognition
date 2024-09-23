# -*- coding: utf-8 -*-

from flask import request, jsonify, make_response
from flask_restx import fields, Resource, Namespace
import torch
import torchvision.models as models
import os
# import pandas as pd
import numpy as np
import base64
from PIL import Image
from io import BytesIO
# image_files = pd.DataFrame(columns=range(784)).add_prefix('pixels_')
# for i in range(1, 6):
#     r_image = cv2.imread(f'images/{i}.JPG')
#     numpy_image = cv2.cvtColor(r_image, cv2.COLOR_BGR2GRAY)
#     image = cv2.resize(numpy_image, (28, 28)).astype(np.float32)
#     image = image.reshape(-1)
#     image_files.loc[f'image_{i}', 'pixels_0':] = image
# print(image_files.head())

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

inf_ns = Namespace(
    "Inference",
    description="NS for inference operations",
    path="/"
)


@inf_ns.route("/inference")
class Inference(Resource):
    def post(self):
        data = request.get_json()
        ##  get an image and a model name
        image = data.get("image")
        if "data:image" in image:
            image = image.split(",")[1]
        image_bytes = base64.b64decode(image)
        imgstream = BytesIO(image_bytes)
        image = Image.open(imgstream)
        # image = image.convert("L")
        image = image.convert("RGB")
        image = image.resize((28, 28))
        # image.show() ## so blurry but that is 28*28 grayscale so fine
        # image.show()
        numpy_image = np.array(image, dtype=np.float32)
        # print(numpy_image.shape)
        imginput = torch.tensor(numpy_image).permute(2, 0, 1).unsqueeze(0).to(device)
        # print(imginput.shape)
        model_name = data.get("model_name")
        ## load model based on model name
        resnet50 = models.resnet50()
        num_classes = 7 
        resnet50.fc = torch.nn.Linear(resnet50.fc.in_features, num_classes)
        pretrained = torch.load(
            BASE_DIR + f"/saved_model/{model_name}.pth",
            map_location=torch.device(device)
        )
        resnet50.load_state_dict(pretrained["state_dict"])
        resnet50.to(device)
        resnet50.eval()
        ## make prediction
        ## return prediction
        rslt = resnet50(imginput)
        return make_response(jsonify({"msg": "Success", "data": {"prediction": torch.argmax(rslt).item()}}), 200)
        # return resnet50(image)
