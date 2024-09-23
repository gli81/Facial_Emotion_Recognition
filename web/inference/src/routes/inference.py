# -*- coding: utf-8 -*-

from flask import request, jsonify, make_response
from flask_restx import fields, Resource, Namespace
import torch
import torchvision.models as models
import os

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
        model_name = data.get("model_name")
        ## load model based on model name
        resnet50 = models.resnet50()
        num_classes = 7 
        resnet50.fc = torch.nn.Linear(resnet50.fc.in_features, num_classes)
        pretrained = torch.load(
            BASE_DIR + f"saved_model/{model_name}.pth"
        )
        resnet50.load_state_dict(pretrained["state_dict"])
        resnet50.to(device)
        resnet50.eval()
        ## make prediction
        ## return prediction
        return resnet50(image)
