# -*- coding: utf-8 -*-

"""
defines the app factory function here
"""

from flask import Flask
from flask_restx import Api
from .config import DevConfig, ProdConfig, TestConfig

config_dict = {
    "dev": DevConfig,
    "prod": ProdConfig,
    "test": TestConfig
}

def create_app(conf: "str"="dev"):
    """
    factory function to create app
    """
    app = Flask(__name__)
    app.config.from_object(config_dict[conf])
    api = Api(app)
    from .routes.inference import inf_ns
    api.add_namespace(inf_ns)
    return app
