# Create a flask application and run it on aws ec2 instance
from flask import Flask, request, jsonify, render_template
from utlis import facebook as fb
from utlis import instagram as insta

from utlis import image_match as im
from utlis.helper import decode, encode
import torch
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2L, preprocess_input
from tensorflow.keras.models import Model
import json
from sentence_transformers import SentenceTransformer
from functools import wraps
#from PIL import Image

app = Flask(__name__)
# Load all machine learning models
facebook_post_model = torch.hub.load('ultralytics/yolov5', 'custom',
                                     path='models/facebook_pos4_m6.pt')

facebook_story_model = torch.hub.load('ultralytics/yolov5', 'custom',
                                      path='models/FB_Story_x.pt')

matching_detection_model = torch.hub.load('ultralytics/yolov5', 'custom',
                                          'models/photo_class_98.pt')

base_efficientnet_model = EfficientNetV2L(weights='imagenet')
efficientnet_model = Model(inputs=base_efficientnet_model.input,
                           outputs=base_efficientnet_model.get_layer('top_dropout').output)

# First, we load the respective CLIP model
matching_model_transformer = SentenceTransformer('clip-ViT-B-32')

social_models = {"Post": {'Facebook': {"likes": facebook_post_model, "match": matching_detection_model},
                          'Instagram': {"match": matching_detection_model}},
                 "Story": {'Facebook': {"likes": facebook_story_model}}}

secret_token = "sdlkhfkuherflnskdfjapsdfijsifewro1293u8euji23dfloj9r4fdkm"


def token_required(f):
    @wraps(f)
    def decorated():
        token = None

        if 'API-KEY' in request.headers:
            token = request.headers['API-KEY']
            print(type(token))

        if not token:
            return jsonify({'Message': 'Token is missing!', 'Success': False, "Code": "5001"})

        if token != secret_token:
            return jsonify({'Message': 'Token is invalid!', 'Success': False, "Code": "5001"})

        return f()

    return decorated


@app.route('/hi', methods=['GET'])
def ap():
    return "Welcome to Impactyn AI Engine :)"


@app.route('/ai', methods=['POST'])
@token_required
def ai():
    read = request.get_json()
    if type(read) == str:
        read = json.loads(read)
    try:
        screenshot_b64 = read['Screenshot']
        screenshot_bytes = bytes(screenshot_b64, 'utf-8')
        screenshot = decode(screenshot_b64)
        social_media = read['SocialAccount']
        img_type = read['Type']
        match = read['Matching']
        post_id = read['PostId']

    except Exception as e:
        print(e)
        "return Incorrect parameters Message and status code 5001 and success false"
        return jsonify({"Message": "Incorrect parameters", "Code": 5001, "Success": False})

    fb_post_data_extractor = fb.PostDataExtractor(facebook_post_model)
    insta_post_data_extractor = insta.InstaPostExtractor()
    fb_story_data_extractor = fb.StoryDataExtractor(facebook_story_model)
    insta_story_data_extractor = insta.InstaStoryExtractor()

    find_match = im.ImageMatch(object_model=matching_detection_model,
                               matching_model=matching_model_transformer, imagenet_model=efficientnet_model, size=480)

    if img_type == "Post" and social_media == "Facebook":
        result = fb_post_data_extractor.get_data(screenshot)

    elif img_type == "Post" and social_media == "Instagram":
        result = insta_post_data_extractor.get_data(screenshot)

    elif img_type == "Story" and social_media == "Facebook":
        result = fb_story_data_extractor.get_data(screenshot)

    elif img_type == "Story" and social_media == "Instagram":
        result = insta_story_data_extractor.get_data(screenshot)

    result['PostId'] = post_id

    if match == 1:

        original_image_b64 = read['OriginalPost']
        original_image = decode(original_image_b64)
        if img_type == "Post":
            screenshot_ = find_match.get_cropped_image(screenshot)
        else:
            screenshot_ = screenshot
        #similarity = find_match.clip_transformer_matching(post_image=Image.fromarray(screenshot_),original_image=Image.fromarray(original_image))
        similarity = find_match.cosine_image_match(post_image=screenshot_,
                                                          original_image=original_image)
        result["MatchPercentage"] = str(similarity)
        result['Success'] = True
        if img_type == "Story" and social_media == "Instagram":
            res = {'MatchPercentage': str(similarity), 'Success': True}
            return jsonify(res)
        return jsonify(result)

    elif match == 0:
        result['Success'] = True
        return jsonify(result)


if __name__ == "__main__":
    app.run(host='0.0.0.0')