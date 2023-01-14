import torch
import cv2
import boto3
import configparser
from instagram import InstaStoryExtractor

config = configparser.ConfigParser()
config.read('utlis/config.ini')

AccessKeyID = config['AWS']['access_key']
SecretAccessKey = config['AWS']['secret_key']

rekognition_client = boto3.client('rekognition',
                                  aws_access_key_id=AccessKeyID,
                                  aws_secret_access_key=SecretAccessKey,
                                  region_name='us-east-1')

# ToDo: Retrain  the model with more custom  privacy data
class PostDataExtractor:
    def __init__(self, model):
        self.model = model
        # use converter for normalizaion
        self.converter = {"٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4", "٥": "5",
                          "٦": "6", "٧": "7", "٨": "8", "٩": "9", "أ": "ا", "آ": "ا", }

    def get_text(self, img):
        # read image as byte object if the input is image path
        if type(img) == str:
            with open(img, 'rb') as image_file:
                image_bytes = image_file.read()

        # convert the image to byte object if the input is image
        elif type(img) != bytes:
            success, encoded_image = cv2.imencode('.png', img)
            image_bytes = encoded_image.tobytes()

        else:
            image_bytes = img
        # get text in image using AWs rekognition
        response = rekognition_client.detect_text(Image={'Bytes': image_bytes})
        result = response['TextDetections'][0]['DetectedText']
        return result

    def get_data(self, image):
        # {0: 'Date', 1: 'Custom', 2: 'Likes', 3: 'Public', 4: 'Friends', 5: 'onlyme'}

        data = {}
        result = self.model(image, size=1280)
        all_classes = result.crop(save=False)

        for c in all_classes:
            c['cls'] = c['cls'].cpu()
            if c['cls'].numpy() == 0:
                date_img = c['im']
                date = self.get_text(date_img.copy())
                data["AddedFrom"] = date

            elif c['cls'].numpy() == 1:
                data['Privacy'] = "custom"

            elif c['cls'].numpy() == 2:
                likes_img = c['im']
                likes = self.get_text(likes_img.copy())
                data["Likes"] = likes

            elif c['cls'].numpy() == 3:
                data['Privacy'] = "public"

            elif c['cls'].numpy() == 4:
                data['Privacy'] = "friends only"

            elif c['cls'].numpy() == 5:
                data['Privacy'] = "onlyme"

        return data

# ToDo: Retrain  the model to extract both date and username
class StoryDataExtractor:
    def __init__(self, model):
        self.model = model
        # use converter for normalizaion
        self.converter = {"٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4", "٥": "5",
                          "٦": "6", "٧": "7", "٨": "8", "٩": "9", "أ": "ا", "آ": "ا", }

    def get_text(self, img):
        # read image as byte object if the input is image path
        if type(img) == str:
            with open(img, 'rb') as image_file:
                image_bytes = image_file.read()

        # convert the image to byte object if the input is image
        elif type(img) != bytes:
            success, encoded_image = cv2.imencode('.png', img)
            image_bytes = encoded_image.tobytes()

        else:
            image_bytes = img
        # get text in image using AWs rekognition
        response = rekognition_client.detect_text(Image={'Bytes': image_bytes})
        result = response['TextDetections'][0]['DetectedText']
        return result

    def get_data(self, image):
        # {0: 'time', 1: 'public', 2: 'views', 3: 'custom', 4: 'friends'}

        data = {}
        result = self.model(image, size=1280)
        all_classes = result.crop(save=False)

        for c in all_classes:
            c['cls'] = c['cls'].cpu()
            if c['cls'].numpy() == 0:
                date_img = c['im']
                date = self.get_text(date_img.copy())
                data["AddedFrom"] = date

            elif c['cls'].numpy() == 1:
                data['Privacy'] = "public"

            elif c['cls'].numpy() == 2:
                views_img = c['im']
                views = self.get_text(views_img.copy())
                data["Views"] = views

            elif c['cls'].numpy() == 3:
                data['Privacy'] = "custom"


            elif c['cls'].numpy() == 4:
                data['Privacy'] = "friends only"

        return data


if __name__ == "__main__":
    import glob
    imgs = glob.glob("test\\facebook_story\\*")

    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path='models/FB_Story_x.pt')
    post_extractor = StoryDataExtractor(model)
    insta_extractor = InstaStoryExtractor()

    for img in imgs:
        result = post_extractor.get_data(img)
        if not result.get("AddedFrom"):
            print("Failed to get date")
            result = insta_extractor.get_date(img, result)

        if not result.get("Views"):
            print("Failed to get viewers")
            result = insta_extractor.get_viwers(img, result)

        print(f"img: {img} result: {result}")

'''
if __name__ == "__main__":
    import glob
    imgs = glob.glob("test\\facebook_post\\*")

    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path='models/facebook_pos4_X6.pt')
    post_extractor = PostDataExtractor(model)
    for img in imgs:
        result = post_extractor.get_data(img)
        print(f"img: {img} result: {result}")
        
'''