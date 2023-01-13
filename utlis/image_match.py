from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
# load Yolov5 model from torch hub
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# Create a Class take two different images crop the object from one of them and match it with the other image


class ImageMatch:
    def __init__(self, object_model, size, matching_model=None, imagenet_model=None):
        self.objects_model = object_model
        self.imagenet_model = imagenet_model
        self.size = size
        self.matching_model = matching_model

    def get_cropped_image(self, image1):
        # get the object from the image
        results = self.objects_model(image1)
        # get the cropped image
        cropped_images = results.crop(save=False)
        # choose the cropped image if the confidence is more than 0.7
        for cropped_image in cropped_images:
            if float(cropped_image['conf']) > 0.4:
                img = cropped_image['im']
                return img

    def get_image_features(self, img, resize):
        img = img.resize((resize, resize))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis
        # Extract Features
        feature = self.imagenet_model.predict(img_array, verbose=0)[0]
        return feature / np.linalg.norm(feature)

    def cosine_image_match(self, post_image, original_image):
        # get the cropped image
        #cropped_image = self.get_cropped_image(post_image)
        # get the features of the cropped image
        post_image = Image.fromarray(post_image)
        cropped_image_features = self.get_image_features(post_image, self.size)
        # get the features of the second image
        # read numpy array using Image

        original_img = Image.fromarray(original_image)
        image2_features = self.get_image_features(original_img, self.size)
        # calculate the cosine similarity between the cropped image and the second image
        cos_similarity = np.dot(cropped_image_features, image2_features.T) / (
                np.linalg.norm(cropped_image_features) * np.linalg.norm(image2_features))
        return cos_similarity

    def orb_matching(self, post_image, original_image):
        # get the cropped image
        cropped_image = self.get_cropped_image(post_image)
        cropped_image = np.array(cropped_image)
        cropped_image = cropped_image[:, :, ::-1].copy()

        original_img = cv2.imread(original_image)
        # Resize the images
        # Resize the images
        image1 = cv2.resize(cropped_image, (480, 480))
        image2 = cv2.resize(original_img, (480, 480))

        # Convert the images to grayscale
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Find the keypoints and descriptors with ORB
        orb = cv2.ORB_create()
        keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

        # Create a Brute-Force matcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Perform the matching
        matches = bf.match(descriptors1, descriptors2)

        # Calculate the average distance between the matches
        total_distance = 0
        for matchh in matches:
            total_distance += matchh.distance
        average_distance = total_distance / len(matches)

        return average_distance

    def clip_transformer_matching(self, post_image, original_image):
        # resize images
        post_image = post_image.resize((224, 224))
        original_image = original_image.resize((224, 224))
        img_emb1 = self.matching_model.encode([post_image], batch_size=128, convert_to_tensor=True)
        img_emb2 = self.matching_model.encode([original_image], batch_size=128, convert_to_tensor=True)
        similarity = cosine_similarity(img_emb1.cpu(), img_emb2.cpu())[0][0]
        sim = round(similarity * 100, 3)

        return sim