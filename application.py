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
#import MySQLdb.cursors
#from flask_mysqldb import MySQL
#import pymysql
#from passlib.hash import sha256_crypt

# Import fast.ai Library
from fastai import *
from fastai.vision import *

# import pymysql

# conn = pymysql.connect(
#         host= 'mysql://admin:password@flaskdb.cy3lotqpgdfu.us-east-1.rds.amazonaws.com', 
#         port = 3306,
#         user = 'admin', 
#         password = 'password',
#         db = 'testingdb_aws'
        
#         )

# application = Flask(__name__)

# # # Change this to your secret key (can be anything, it's for extra protection)
# application.secret_key = 'your secret key'

# # # Enter your database connection details below
# application.config['MYSQL_HOST'] = 'mysql://admin:password@flaskdb.cy3lotqpgdfu.us-east-1.rds.amazonaws.com'
# application.config['MYSQL_USER'] = 'admin'
# application.config['MYSQL_PASSWORD'] = 'password'
# application.config['MYSQL_DB'] = 'testingdb_aws'

# # Intialize MySQL
# mysql = MySQL(application)

# app = Flask(__name__)

# # Change this to your secret key (can be anything, it's for extra protection)
# app.secret_key = 'your secret key'

# # Enter your database connection details below
# app.config['MYSQL_HOST'] = 'localhost'
# app.config['MYSQL_USER'] = 'root'
# app.config['MYSQL_PASSWORD'] = ''
# app.config['MYSQL_DB'] = 'pythonlogin'

# # Intialize MySQL
# mysql = MySQL(app)

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

class accounts(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)

    # def __init__(self, username, password, email):
    #     self.title = username
    #     self.author = password
    #     self.price = email


# C:/Users/Aran/Desktop/TUD Work/Year 4/Final Year Project/Code/Testing on Python 3.9/SkinCancerDetection-WebAppNAME_OF_FILE = 'model_best' # Name of your exported file
NAME_OF_FILE = 'model_best' # Name of your exported file
#PATH_TO_MODELS_DIR = Path('C:/Users/Aran/Desktop/TUD Work/Year 4/Final Year Project/Code/Testing on Python 3.9/SkinCancerDetection-WebApp') # by default just use /models in root dir
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
    img = open_image(BytesIO(img))
    pred_class,pred_idx,outputs = learn.predict(img)
    formatted_outputs = ["{:.1f}%".format(value) for value in [x * 100 for x in torch.nn.functional.softmax(outputs, dim=0)]]
    pred_probs = sorted(
            zip(learn.data.classes, map(str, formatted_outputs)),
            key=lambda p: p[1],
            reverse=True
        )
	
    img_data = encode(img)
    result = {"class":pred_class, "probs":pred_probs, "image":img_data}
    return render_template('result.html', result=result)

def IP(img):
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

    # You may need to convert the color.
    # Converting image back to Pillow 
    final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
    img = PILImage.fromarray(final_image)

    # Converting pillow image to bytes
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    img = buf.getvalue()

    return img

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
        IP(img)
        img = IP(img)
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
        
        login = accounts.query.filter_by(username=username, password=password).first()
        if login is not None:
            session["logged_in"] = True
            return redirect(url_for("index"))
    session["logged_in"] = False
    return render_template("login.html")



@application.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        register = accounts(username = username, email = email, password = password)
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

@application.route('/account', methods=['GET', "POST"])
def account():
    # Main page
    return render_template('account.html')

@application.route('/info', methods=['GET', "POST"])
def info():
    # Information page
    return render_template('info.html')


@application.route('/predict')
def predict():
    books = Book.query.all()
    return render_template('predict.html', books=books)



@application.route('/add', methods =['POST'])
def insert_book():
    if request.method == "POST":
        book = Book(
            title = request.form.get('title'),
            author = request.form.get('author'),
            price = request.form.get('price')
        )
        db.session.add(book)
        db.session.commit()
        flash("Book added successfully")
        return redirect(url_for('predict'))



@application.route('/delete/<id>/', methods = ['GET', 'POST'])
def delete(id):
    my_data = Book.query.get(id)
    db.session.delete(my_data)
    db.session.commit()
    flash("Book is deleted")
    return redirect(url_for('predict'))


if __name__ == '__main__':

    if "prepare" not in sys.argv:
        application.run(host='0.0.0.0', port=80, debug=False)




# # http://localhost:5000/pythonlogin/ - the following will be our login page, which will use both GET and POST requests
# @application.route('/login/', methods=['GET', 'POST'])
# def login():
#     # Output message if something goes wrong...
#     msg = ''
#     # Check if "username" and "password" POST requests exist (user submitted form)
#     if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
#         # Create variables for easy access
#         username = request.form['username']
#         password = request.form['password']
#         # Check if account exists using MySQL
#         cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
#         #cursor = db.connection.cursor(MySQLdb.cursors.DictCursor)
#         cursor.execute('SELECT * FROM accounts WHERE username = %s AND password = %s', (username, password,))
#         # Fetch one record and return result
#         account = cursor.fetchone()
#                 # If account exists in accounts table in out database
#         if account:
#             # Create session data, we can access this data in other routes
#             session['loggedin'] = True
#             session['id'] = account['id']
#             session['username'] = account['username']
#             # Redirect to home page
#             return 'Logged in successfully!'
#         else:
#             # Account doesnt exist or username/password incorrect
#             msg = 'Incorrect username/password!'
#     # return render_template('index.html', msg='')
#     return render_template('login.html', msg='')

# # http://localhost:5000/python/logout - this will be the logout page
# @application.route('/login/logout')
# def logout():
#     # Remove session data, this will log the user out
#    session.pop('loggedin', None)
#    session.pop('id', None)
#    session.pop('username', None)
#    # Redirect to login page
#    return redirect(url_for('login'))


# # http://localhost:5000/pythinlogin/register - this will be the registration page, we need to use both GET and POST requests
# @application.route('/login/register', methods=['GET', 'POST'])
# def register():
#     # Output message if something goes wrong...
#     msg = ''
#     # Check if "username", "password" and "email" POST requests exist (user submitted form)
#     if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
#         # Create variables for easy access
#         username = request.form['username']
#         password = request.form['password']
#         email = request.form['email']
#         # Check if account exists using MySQL
        
#         cursor=conn.cursor()
#         #cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
#         #cursor = db.connection.cursor(MySQLdb.cursors.DictCursor)
#         cursor.execute('SELECT * FROM accounts WHERE username = %s', (username,))
#         account = cursor.fetchone()
#         # If account exists show error and validation checks
#         if account:
#             msg = 'Account already exists!'
#         elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
#             msg = 'Invalid email address!'
#         elif not re.match(r'[A-Za-z0-9]+', username):
#             msg = 'Username must contain only characters and numbers!'
#         elif not username or not password or not email:
#             msg = 'Please fill out the form!'
#         else:
#             # Account doesnt exists and the form data is valid, now insert new account into accounts table
#             cursor.execute('INSERT INTO accounts VALUES (NULL, %s, %s, %s)', (username, password, email,))
#             conn.commit()
#             #mysql.connection.commit()
#             #db.connection.commit()
#             msg = 'You have successfully registered!'
#     elif request.method == 'POST':
#         # Form is empty... (no POST data)
#         msg = 'Please fill out the form!'
#     # Show registration form with message (if any)
#     return render_template('register.html', msg=msg)

# http://localhost:5000/pythinlogin/register - this will be the registration page, we need to use both GET and POST requests
# @application.route('/register', methods=['GET', 'POST'])
# def register():
#     # Check if "username", "password" and "email" POST requests exist (user submitted form)
#     if request.method == 'POST':# and 'username' in request.form and 'password' in request.form and 'email' in request.form:
#         # Create variables for easy access
#         user = User(
#         username = request.form['username'],
#         password = request.form['password'],
#         email = request.form['email']
#         )
#         #secure_password=sha256_crypt.encrypt(str(password))
#         # Check if account exists using MySQL
        
#         usernamedata=db.execute("SELECT username FROM users WHERE username=:username",{"username":user.username}).fetchone()
#         #usernamedata=str(usernamedata)
        
        
#         if usernamedata == None:
#             if user.password==user.password:
#                 db.execute("INSERT INTO users (username, password, email) VALUES (:username, :password, :email)",{"username":user.username, "password":user.password, "email":user.email})
#                 db.commit()
#                 flash("You are registered and can now login","success")
#                 return redirect(url_for('login'))
#             else:
#                  flash("password does not match")
#                  return render_template('register.html')
    
#         else:
#             flash("username already exists")
#             return render_template(url_for('login'))

#     return render_template('register.html')


# @application.route('/login/', methods=['GET', 'POST'])
# def login():
#     if request.method=="POST":
#         username=request.form['username']
#         password=request.form['password']
#         usernamedata=db.execute("SELECT username FROM users WHERE username=:username",{"username":username}).fetchone()
#         passworddata=db.execute("SELECT password FROM users WHERE username=:username",{"username":username}).fetchone()

#         if usernamedata in None:
#             flash("username does not exist")
#             return render_template('login.html')
#         else:
#             if password==passworddata:
#                 flash("You are logged in")
#                 return redirect(url_for('index'))
#             else:
#                 flash("password does not match")
#                 return render_template('login.html')

# @application.route('/', methods=['POST'])
# def get_input_values():
#     val = request.form['my_form']


# @application.route('/predict', methods=['POST', 'GET'])
# def predict():
#     if request.method == 'GET':
#         return 'The URL /predict is accessed directly. Go to the main page firstly'

#     if request.method == 'POST':
#         input_val = request.form

#         if input_val != None:
#             # collecting values
#             vals = []
#             for key, value in input_val.items():
#                 vals.append(float(value))

#         # Calculate Euclidean distances to freezed centroids
#         with open('freezed_centroids.pkl', 'rb') as file:
#             freezed_centroids = pickle.load(file)

#         assigned_clusters = []
#         l = []  # list of distances

#         for i, this_segment in enumerate(freezed_centroids):
#             dist = distance.euclidean(*vals, this_segment)
#             l.append(dist)
#             index_min = np.argmin(l)
#             assigned_clusters.append(index_min)

#         return render_template(
#             'predict.html', result_value=f'Segment = #{index_min}'
#             )
