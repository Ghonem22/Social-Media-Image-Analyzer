import cv2
import boto3
import re

import configparser

config = configparser.ConfigParser()
config.read('utlis/config.ini')

AccessKeyID = config['AWS']['access_key']
SecretAccessKey = config['AWS']['secret_key']

rekognition_client = boto3.client('rekognition',
                                  aws_access_key_id=AccessKeyID,
                                  aws_secret_access_key=SecretAccessKey,
                                  region_name='us-east-1')


class InstaPostExtractor:
    def __init__(self):
        self.date_sumbols = ["january", "february", "march", "april", "may", "june", "july", "august",
                             "september", "october", "november", "december", "day", "month", "ساع", "اعجاب", "يوم",
                             "ايام", "اسبوع", "اسابيع", "شهر", "اشهر",
                             "يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس", "سبتمبر", "اكتوبر",
                             "نوفمبر", "ديسمبر"]

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
        return rekognition_client.detect_text(Image={'Bytes': image_bytes})

    def get_data(self, img):
        response = self.get_text(img)
        data = {}
        for i in range(len(response["TextDetections"])):
            text = response["TextDetections"][i]['DetectedText']
            # use text_checker for detection
            text_checker = text.lower().replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
            if ("like" in text_checker or "اعجاب" in text_checker) and text_checker != "likes" and \
                    text_checker != "liked" and not data.get("Likes"):
                data["Likes"] = text.strip()

            # iterate over date_sumbols to get date
            for date_sumbol in self.date_sumbols:
                if date_sumbol.lower().strip() in text.lower() and not data.get("AddedFrom"):
                    added_from = text.replace("see", "").replace("translation", "").replace("See", "").strip()
                    # in some cases we find date symbol in a text but it's just notmal sentence, this will
                    # minimize probability of error
                    if len(added_from) < 15:
                        data["AddedFrom"] = added_from
                        break

            if data.get("AddedFrom") and data.get("Likes"):
                # break if we found likes and dates
                break
        return data


class InstaStoryExtractor:
    def __init__(self):
        self.dates = ("d", "s", "س", "ث", "h", "ي", "m")
        self.converter = {"٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4", "٥": "5",
                          "٦": "6", "٧": "7", "٨": "8", "٩": "9", "س": "h", "ث": "s", "ي": "d"}

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
        return rekognition_client.detect_text(Image={'Bytes': image_bytes})

    def is_num(self, txt):
        try:
            float(txt)
            return True
        except:
            return False

    def get_viwers(self, img, data = None):
        flag = False
        response = self.get_text(img)["TextDetections"]
        if data is None:
            data = {}

        for i in range(len(response)):
            text = response[i]['DetectedText']
            text_checker = text.lower().replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")

            if ("viewers" in text_checker or "مشاهد" in text_checker):
                num = text_checker.replace("viewers", "").replace("مشاهد", "").replace(",", "").strip()
                if self.is_num(num):
                    data["Views"] = int(num)
                    break
                print(text_checker)
                # check the previous last 5 numbers
                for j in range(1,6):
                    if self.is_num(response[i - j]['DetectedText'].strip()):
                        data["Views"] = int((response[i - j]['DetectedText']).strip())
                        flag = True
                        break
                    if flag:
                        break
                break
        return data


    def check_date(self, txt):
        pp = txt.split()
        for i in range(len(pp)):
            if i > 0:
                # if there's a date symbol and the last item is num, we can consider that as a date
                if pp[i].strip() in self.dates and self.is_num(pp[i - 1]):
                    return pp[i - 1].strip() + " " + pp[i].strip()

                # if there's a num and the last item is date symbol, we can consider that as a date
                if self.is_num(pp[i].strip()) and pp[i - 1].strip() in self.dates:
                    return pp[i - 1].strip() + " " + pp[i].strip()

            # if this item is consist of number and date symbol
            num = re.findall(r'\d+', pp[i])
            symbol = re.sub("\d+", " ", pp[i]).strip()
            if symbol in self.dates:
                num = re.findall(r'\d+', pp[i])
                if len(num) > 0:
                    return num[0] + " " + symbol

        return False

    def get_date(self, img, data = None):
        if data is None:
            data = {}

        response = self.get_text(img)
        # iterate over date_sumbols to get date
        for i in range(len(response["TextDetections"])):
            text = response["TextDetections"][i]['DetectedText']
            result = self.check_date(text)
            if result:
                data["AddedFrom"] = result
                return data
        return data


if __name__ == "__main__":
    im1 = "test\\facebook_story\\9ed0f0166b7e5b00225536d3309b8335.jpg"
    im2 = "test\\facebook_story\\9ed0f0166b7e5b00225536d3309b8335.jpg"
    insta_extractor = InstaStoryExtractor()
    result = insta_extractor.get_viwers(im1)
    print(result)
    result = insta_extractor.get_date(im2, result)
    print(result)