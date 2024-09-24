# -*- coding:utf-8 -*-

"""
defines some configurations
"""

import os

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

class Config:
    pass

class DevConfig(Config):
    pass

class ProdConfig(Config):
    pass

class TestConfig(Config):
    TESTING=True