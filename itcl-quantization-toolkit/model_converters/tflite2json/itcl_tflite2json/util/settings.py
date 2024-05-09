import os
import yaml
from typing_extensions import TypedDict

class _Lut(TypedDict):
    lut_depth: int 
    min_removal: int
    asymmetric: str

class _Settings(TypedDict):
    lut: _Lut

class _SettingsRoot(TypedDict):
    settings: _Settings




# this attribute should be readonly 
settings: _SettingsRoot

with open(f"{os.path.dirname(__file__)}/../settings.yaml", "r") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    settings = data; 



# Generated With: https://json2pyi.pages.dev/#TypedDictClass






