import requests 
import numpy as np
import json
import base64
import matplotlib.image
from requests.auth import HTTPBasicAuth 
import matplotlib.pyplot as plt
import os
import configparser


#URL= "http://ec2-23-23-16-131.compute-1.amazonaws.com/"
URL= "http://18.209.245.202/"
#URL = "http://127.0.0.1:5000/"

config = configparser.ConfigParser()
config.read('utlis/config.ini')

key = config['APIKEY']['secret_api_key']

def encode(path):
    f1 = open(path, 'rb').read()
    f = str(base64.b64encode(f1))
    f = f[2:len(f)-1]
    return f


original_path = "/media/youssef/DVolume/AI/home/impactyn/proj/Social-Media-Image-Analyzer/test/ai matching/15.jpg"
screenshot_path = "/media/youssef/DVolume/AI/home/impactyn/proj/Social-Media-Image-Analyzer/test/ai matching/16.jpg"
headers = {'content-type': 'application/json', 'API-KEY': key}


###############################################################################
read = {"OriginalPost": encode(original_path), "Screenshot": encode(screenshot_path),
        "SocialAccount": "Instagram", "Type": "Story", "Matching": 1, "PostId": "123456789"}

r = requests.post(url=URL + "ai", json=json.dumps(read), headers=headers)

print("Response code is: ", r)
data = r.json()
data.keys()

print(data)