import numpy as np
from PIL import Image
import base64
from io import BytesIO


def encode(img):
    encoded_string = base64.b64encode(img)
    return encoded_string


def decode(base64_string):
    image = Image.open(BytesIO(base64.b64decode(base64_string)))
    image = image.convert("RGB")
    #resize image
    image = image.resize((1024, 1024))

    return np.array(image)
