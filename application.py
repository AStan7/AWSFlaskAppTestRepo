# Import modules and packages
from flask import Flask, request, render_template, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy


import pickle
import numpy as np
from scipy.spatial import distance
from pathlib import Path
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
import io
from PIL import Image as PILImage
from datetime import date, datetime, timedelta
from flask_login import current_user, LoginManager
import bcrypt


# Import fast.ai Library
from fastai import *
from fastai.vision import *



application = Flask(__name__)

application.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://admin:password@flaskdb.cy3lotqpgdfu.us-east-1.rds.amazonaws.com/testingdb_aws'
application.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
application.secret_key = "somethingunique"

db = SQLAlchemy(application)


class Book(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    author = db.Column(db.String(100), nullable=False)
    price = db.Column(db.Float)

    def __init__(self, title, author, price):
        self.title = title
        self.author = author
        self.price = price

class accounts(db.Model): #usermixin
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    schelduled_test = db.Column(db.String(100), nullable=False)
    test_count = db.Column(db.Integer, nullable=False)
    account_created = db.Column(db.String(100), nullable=False)
    sex = db.Column(db.String(10), nullable=False)
    age = db.Column(db.String, nullable=False)

    def __repr__(self):
        return f'<User(username={self.username}, password={self.password}, email={self.email})>'


class imagetest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    #image = db.Column(db.LargeBinary, nullable=False)
    image = db.Column(db.String(10000), nullable=False)
    prediction = db.Column(db.String(100), nullable=False)
    #advice = db.Column(db.String(500), nullable=False)
    date = db.Column(db.String(100), nullable=False)
    time = db.Column(db.String(100), nullable=False)
    username = db.Column(db.String(50), nullable=False)
    next_test = db.Column(db.String(100), nullable=False)

    
class model_training(db.Model):
    lesion_id = db.Column(db.String(50), nullable=False)
    image_id = db.Column(db.String(50), primary_key=True)
    dx = db.Column(db.String(50), nullable=False)
    dx_type = db.Column(db.String(50), nullable=False)
    age = db.Column(db.String, nullable=False)
    sex = db.Column(db.String(50), nullable=False)
    localization = db.Column(db.String(50), nullable=False)
    lesion = db.Column(db.String(50), nullable=False)

class image_training(db.Model):
    image_id = db.Column(db.Integer, primary_key=True)
    image = db.Column(db.String(10000), nullable=False)



NAME_OF_FILE = 'model_best' # Name of your exported file
PATH_TO_MODELS_DIR = Path('') # by default just use /models in root dir
classes = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis',
           'Dermatofibroma', 'Melanocytic nevi', 'Melanoma', 'Vascular lesions']

def setup_model_pth(path_to_pth_file, learner_name_to_load, classes):
    data = ImageDataBunch.single_from_classes(
        path_to_pth_file, classes, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
    learn = cnn_learner(data, models.densenet169, model_dir='models')
    learn.load(learner_name_to_load, device=torch.device('cpu'))
    return learn

learn = setup_model_pth(PATH_TO_MODELS_DIR, NAME_OF_FILE, classes)

def encode(img):
    img = (image2np(img.data) * 255).astype('uint8')
    pil_img = PILImage.fromarray(img)
    #IP(pil_img)
    #pil_img = IP(pil_img)
    #pil_img.show()
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

def model_predict(img):
    imagecopy = img
    img = open_image(BytesIO(img))
    pred_class,pred_idx,outputs = learn.predict(img)
    formatted_outputs = ["{:.1f}%".format(value) for value in [x * 100 for x in torch.nn.functional.softmax(outputs, dim=0)]]
    pred_probs = sorted(
            zip(learn.data.classes, map(str, formatted_outputs)),
            key=lambda p: p[1],
            reverse=True
        )

    img_data = encode(img)

#need to add advice

    if pred_probs[0][1] < '20%':
        pred_class = 'No Cancer Detected'


    if "logged_in" not in session:
        result = {"class":pred_class, "probs":pred_probs, "image":img_data}
        return render_template('result.html', result=result)
  
    localtime = datetime.now()
    it = imagetest(image = img_data,
                        prediction = pred_class,
                        #advice = adv,
                        date = date.today().strftime("%B %d, %Y"),
                        #date.today().strftime("%B %d, %Y")
                        time = localtime.strftime("%H:%M:%S"),
                        username = session['username'],
                        next_test = (date.today() + timedelta(days=30)).strftime('%B %d, %Y')
    )
    
    loggedin = accounts.query.filter_by(username=session['username']).first()
    loggedin.schelduled_test = (date.today() + timedelta(days=30))
    loggedin.test_count = loggedin.test_count + 1

    db.session.add(it)
    db.session.commit()
    
    if pred_probs[0][1] > '20%':
        generateid = datetime.now()
        time_string = generateid.strftime("%Y%m%d%H%M%S")
        collectimage = image_training(image_id = time_string,
                                        image = img_data
                                    )
        db.session.add(collectimage)
        db.session.commit()


        for_training = model_training(lesion_id = 'NA',
                                        image_id = time_string,
                                        dx = 'NA',
                                        dx_type = 'NA',
                                        age = loggedin.age,
                                        sex = loggedin.sex,
                                        localization = 'NA',
                                        lesion = pred_class
                                        )


        db.session.add(for_training)
        db.session.commit()

    result = {"class":pred_class, "probs":pred_probs, "image":img_data}
    return render_template('result.html', result=result)



def IP(img, clarity):
    # Convert image to openCV 
    img = open_image(BytesIO(img))
    img = (image2np(img.data) * 255).astype('uint8')
    Image = PILImage.fromarray(img)

    Image = numpy.array(img) 
    Image = Image[:, :, ::-1].copy()#cv2.cvtColor(open_cv_image, cv2.cv.CV_BGR2RGB) 
    final_image = Image.copy()

    #Automatically finding best threshold and creating mask
    Image_Gray = cv2.cvtColor(Image,cv2.COLOR_BGR2GRAY)
    Threshold = np.mean(Image_Gray) - np.std(Image_Gray)
    Threshold, Mask = cv2.threshold(Image_Gray, thresh = Threshold, 
                                maxval = 255, type = cv2.THRESH_BINARY)

    # Creating inverse of mask to create ROI
    inverse_mask = cv2.bitwise_not(Mask)

    # Applying morphology to get rid of blotched and fill in gaps.
    shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,4))
    inverse_mask = cv2.morphologyEx(inverse_mask,cv2.MORPH_CLOSE,shape)
    inverse_mask = cv2.morphologyEx(inverse_mask,cv2.MORPH_OPEN,shape)

    # Creating ROI
    ROI = cv2.bitwise_and(Image,Image,mask=inverse_mask)

    contours, hierarchy = cv2.findContours(inverse_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    h, w = inverse_mask.shape
    h = h/2
    w = w/2

    h = int(h)
    w = int(w)

    cv2.circle(Image, center = (w,h), radius =10, color =(205, 114, 101), thickness=1)
    center = (w,h)

    buildings = []

    for contour in contours:
        # find center of each contour
        M = cv2.moments(contour)
        if M["m10"] == 0:
            M["m10"] = M["m10"] + 2

        if M["m00"] == 0:
            M["m00"] = M["m00"] + 2
        
        center_X = int(M["m10"] / M["m00"])
        center_Y = int(M["m01"] / M["m00"])
        contour_center = (center_X, center_Y)

        # calculate distance to image_center
        distances_to_center = (distance.euclidean(center, contour_center))
        
        # save to a list of dictionaries
        buildings.append({'contour': contour, 'center': contour_center, 'distance_to_center': distances_to_center})
        
        # draw each contour (red)
        cv2.drawContours(Image, [contour], 0, (0, 50, 255), 2)
        M = cv2.moments(contour)
        # draw center of contour (green)
        cv2.circle(Image, contour_center, 3, (100, 255, 0), 2)

    # sort the buildings
    sorted_buildings = sorted(buildings, key=lambda i: i['distance_to_center'])
        
    # find contour of closest building to center and draw it (blue)
    center_building_contour = sorted_buildings[0]['contour']
    cv2.drawContours(Image, [center_building_contour], 0, (255, 0, 0), 2)

    # Creating a boundry rectangle to crop the image from
    cont = center_building_contour
    x,y,width,height = cv2.boundingRect(cont)

    x = x-15
    y = y-15
    width = width+30
    height = height+30
    # Drawing rectangle
    cv2.rectangle(Image,(x,y),(x+width,y+height),(0,255,0),2)

    final_image = final_image[y:y+height,x:x+width]

    img = final_image

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate the Laplacian variance of the grayscale image
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Define a threshold for the Laplacian variance
    threshold = 100

    # Check if the Laplacian variance is below the threshold, indicating that the image is blurry
    if lap_var < threshold:
        clarity = 'Unclear'
    else:
        clarity = 'Clear'

    # if clarity == 'Unclear':
    #     return 

    # You may need to convert the color.
    # Converting image back to Pillow 
    final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
    img = PILImage.fromarray(final_image)

    # Converting pillow image to bytes
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    img = buf.getvalue()

    return img, clarity



@application.route('/')
def index():
    return render_template('index.html')

@application.route('/test', methods=['GET', "POST"])
def test():
    # Main page
    return render_template('test.html')


@application.route('/upload', methods=["POST", "GET"])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        img = request.files['file'].read()
        clarity = 'NULL'
        IP(img, clarity)
        img, clarity = IP(img, clarity)

        if clarity == 'Unclear':
            flash('The image uploaded was not readable.  Please upload a clearer image with the area of concern at the center.')
            return redirect(url_for('test'))

        if img != None:
            # Make prediction
            preds = model_predict(img)
            return preds
      
    return 'OK'


@application.route("/login",methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        loggedin = accounts.query.filter_by(username=username).first()
        
        if loggedin is not None and bcrypt.checkpw(password.encode('utf-8'), loggedin.password.encode('utf-8')):
            session["logged_in"] = True
            session['username'] = loggedin.username
            session['id'] = loggedin.id
            return redirect(url_for("index"))

        flash("Invalid username or password")
        session["logged_in"] = False
    return render_template("login.html")



@application.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        sex = request.form['sex']
        age = request.form['age']

        user=accounts.query.filter_by(email=email).first() #checking if account exists
        if user:
            flash("Account with same email already exists")
            return redirect(url_for("register"))
        
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        register = accounts(username = username, email = email, password = hashed_password, test_count = 0, account_created = date.today().strftime("%B %d, %Y"), sex = sex, age = age)
        db.session.add(register)
        db.session.commit()

        return redirect(url_for("login"))
    return render_template("register.html")



@application.route('/logout')
def logout():
     # Remove session data, this will log the user out
    session.pop('logged_in', None)
    session.pop('id', None)
    session.pop('username', None)
    # Redirect to login page
    return redirect(url_for('index'))


@application.route('/info', methods=['GET', "POST"])
def info():
    # Information page
    return render_template('info.html')


@application.route('/predict')
def predict():
    books = Book.query.all()
    return render_template('predict.html', books=books)


# @application.route('/add', methods =['POST'])
# def insert_book():
#     if request.method == "POST":
#         book = Book(
#             title = request.form.get('title'),
#             author = request.form.get('author'),
#             price = request.form.get('price')
#         )
#         db.session.add(book)
#         db.session.commit()
#         flash("Book added successfully")
#         return redirect(url_for('predict'))


@application.route('/account')
def account():
    account = accounts.query.filter_by(username=session['username'])

    return render_template('account.html', account=account)


@application.route('/update', methods = ['POST'])
def update():
    if request.method == "POST":
        my_data = accounts.query.get(request.form.get('id'))

        my_data.username = request.form['username']
        my_data.email = request.form['email']
        my_data.password = request.form['password']
        hashed_password = bcrypt.hashpw(my_data.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        my_data.password = hashed_password


        session['username'] = my_data.username

        db.session.commit()
        flash("Account is updated")
        return redirect(url_for('account'))


@application.route('/account1', methods=['GET', "POST"])
def account1():
    # Account page
    test = imagetest.query.filter_by(username=session['username'])
    return render_template('account1.html', test=test)


@application.route('/delete/<id>/', methods = ['GET', 'POST'])
def delete(id):
    my_data = imagetest.query.get(id)
    db.session.delete(my_data)
    db.session.commit()
    flash("Test is deleted")
    return redirect(url_for('account1'))


if __name__ == '__main__':

    if "prepare" not in sys.argv:
        application.run(host='0.0.0.0', port=80, debug=False)

