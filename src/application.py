from __future__ import division, print_function
import sys
import os
import glob
import re
from pathlib import Path
from io import BytesIO
import base64
import requests
import numpy
import cv2
from scipy.spatial import distance##########
import io


# # Import fast.ai Library
# from fastai import *
# from fastai.vision import *

# Flask utils
from flask import Flask, redirect, url_for, render_template, request
# from PIL import Image as PILImage

# Define a flask app
application = Flask(__name__)

# NAME_OF_FILE = 'model_best' # Name of your exported file
# PATH_TO_MODELS_DIR = Path('C:/Users/Aran/Desktop/TUD Work/Year 4/Final Year Project/Code/Testing on Python 3.9/SkinCancerDetection-WebApp') # by default just use /models in root dir
# classes = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis',
#            'Dermatofibroma', 'Melanocytic nevi', 'Melanoma', 'Vascular lesions']

# def setup_model_pth(path_to_pth_file, learner_name_to_load, classes):
#     data = ImageDataBunch.single_from_classes(
#         path_to_pth_file, classes, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
#     learn = cnn_learner(data, models.densenet169, model_dir='models')
#     learn.load(learner_name_to_load, device=torch.device('cpu'))
#     return learn

# learn = setup_model_pth(PATH_TO_MODELS_DIR, NAME_OF_FILE, classes)

# def encode(img):
#     img = (image2np(img.data) * 255).astype('uint8')
#     pil_img = PILImage.fromarray(img)
#     #IP(pil_img)
#     #pil_img = IP(pil_img)
#     #pil_img.show()
#     buff = BytesIO()
#     pil_img.save(buff, format="JPEG")
#     return base64.b64encode(buff.getvalue()).decode("utf-8")
	
# def model_predict(img):
#     img = open_image(BytesIO(img))
#     pred_class,pred_idx,outputs = learn.predict(img)
#     formatted_outputs = ["{:.1f}%".format(value) for value in [x * 100 for x in torch.nn.functional.softmax(outputs, dim=0)]]
#     pred_probs = sorted(
#             zip(learn.data.classes, map(str, formatted_outputs)),
#             key=lambda p: p[1],
#             reverse=True
#         )
	
#     img_data = encode(img)
#     result = {"class":pred_class, "probs":pred_probs, "image":img_data}
#     return render_template('result.html', result=result)

# def IP(img):
#     # Convert image to openCV 
#     img = open_image(BytesIO(img))
#     img = (image2np(img.data) * 255).astype('uint8')
#     Image = PILImage.fromarray(img)

#     Image = numpy.array(img) 
#     Image = Image[:, :, ::-1].copy()#cv2.cvtColor(open_cv_image, cv2.cv.CV_BGR2RGB) 
#     final_image = Image.copy()

#     #Automatically finding best threshold and creating mask
#     Image_Gray = cv2.cvtColor(Image,cv2.COLOR_BGR2GRAY)
#     Threshold = np.mean(Image_Gray) - np.std(Image_Gray)
#     Threshold, Mask = cv2.threshold(Image_Gray, thresh = Threshold, 
#                                 maxval = 255, type = cv2.THRESH_BINARY)

#     # Creating inverse of mask to create ROI
#     inverse_mask = cv2.bitwise_not(Mask)

#     # Applying morphology to get rid of blotched and fill in gaps.
#     shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,4))
#     inverse_mask = cv2.morphologyEx(inverse_mask,cv2.MORPH_CLOSE,shape)
#     inverse_mask = cv2.morphologyEx(inverse_mask,cv2.MORPH_OPEN,shape)

#     # Creating ROI
#     ROI = cv2.bitwise_and(Image,Image,mask=inverse_mask)

#     contours, hierarchy = cv2.findContours(inverse_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

#     h, w = inverse_mask.shape
#     h = h/2
#     w = w/2

#     h = int(h)
#     w = int(w)

#     cv2.circle(Image, center = (w,h), radius =10, color =(205, 114, 101), thickness=1)
#     center = (w,h)

#     buildings = []

#     for contour in contours:
#         # find center of each contour
#         M = cv2.moments(contour)
#         if M["m10"] == 0:
#             M["m10"] = M["m10"] + 2

#         if M["m00"] == 0:
#             M["m00"] = M["m00"] + 2
        
#         center_X = int(M["m10"] / M["m00"])
#         center_Y = int(M["m01"] / M["m00"])
#         contour_center = (center_X, center_Y)

#         # calculate distance to image_center
#         distances_to_center = (distance.euclidean(center, contour_center))
        
#         # save to a list of dictionaries
#         buildings.append({'contour': contour, 'center': contour_center, 'distance_to_center': distances_to_center})
        
#         # draw each contour (red)
#         cv2.drawContours(Image, [contour], 0, (0, 50, 255), 2)
#         M = cv2.moments(contour)
#         # draw center of contour (green)
#         cv2.circle(Image, contour_center, 3, (100, 255, 0), 2)

#     # sort the buildings
#     sorted_buildings = sorted(buildings, key=lambda i: i['distance_to_center'])
        
#     # find contour of closest building to center and draw it (blue)
#     center_building_contour = sorted_buildings[0]['contour']
#     cv2.drawContours(Image, [center_building_contour], 0, (255, 0, 0), 2)

#     # Creating a boundry rectangle to crop the image from
#     cont = center_building_contour
#     x,y,width,height = cv2.boundingRect(cont)

#     x = x-15
#     y = y-15
#     width = width+30
#     height = height+30
#     # Drawing rectangle
#     cv2.rectangle(Image,(x,y),(x+width,y+height),(0,255,0),2)

#     final_image = final_image[y:y+height,x:x+width]

#     # You may need to convert the color.
#     # Converting image back to Pillow 
#     final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
#     img = PILImage.fromarray(final_image)

#     # Converting pillow image to bytes
#     buf = io.BytesIO()
#     img.save(buf, format='JPEG')
#     img = buf.getvalue()

#     return img


@application.route('/', methods=['GET', "POST"])
def homepage():
    # Main page
    return render_template('homepage.html')

# @application.route('/index', methods=['GET', "POST"])
# def index():
#     # Main page
#     return render_template('index.html')

# @application.route('/upload', methods=["POST", "GET"])
# def upload():
#     if request.method == 'POST':
#         # Get the file from post request
#         img = request.files['file'].read()

#         IP(img)
#         img = IP(img)
#         if img != None:
#         # Make prediction
#             preds = model_predict(img)
#             return preds
#     return 'OK'
	
# @application.route("/classify-url", methods=["POST", "GET"])
# def classify_url():
#     if request.method == 'POST':
#         url = request.form["url"]
#         if url != None:
#             response = requests.get(url)
#             preds = model_predict(response.content)
#             return preds
#     return 'OK'

# @application.route('/info', methods=['GET', "POST"])
# def info():
#     # Information page
#     return render_template('info.html')


if __name__ == '__main__':
    port = os.environ.get('PORT', 8008)

    if "prepare" not in sys.argv:
        application.run(debug=False, host='0.0.0.0', port=port)



# from flask import Flask
# application = Flask(__name__)

# @application.route("/")
# def hello():
#     return "Is this working?"