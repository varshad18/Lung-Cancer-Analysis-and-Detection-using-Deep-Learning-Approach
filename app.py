import numpy as np
import pandas as pd
import pickle
import json
import plotly
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os
import cv2
import tensorflow as tf

import urllib.request
from PIL import Image
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

CATEGORIES = ['Bengin case', 'Malignant case', 'Normal case']

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

model = pickle.load(open('output.pkl', 'rb'))
model1 = pickle.load(open('result.pkl', 'rb'))
MODEL_PATH ='cnn_model1.h5'
model2= load_model(MODEL_PATH)
model2.make_predict_function()

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def model2_predict(filename):
    IMG_SIZE=256
    img =cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    img_array = img / 255.0
    new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
    img_tensor = new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)                             
    return img_tensor


@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/stage')
def stage():
    return render_template('stage.html')     


@app.route('/upload_image/', methods=['GET', 'POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
       
        flash('Image successfully uploaded and displayed below')
        
        #Classification
        img_path= os.path.join(app.config['UPLOAD_FOLDER'], filename)
        new_image = model2_predict(img_path)   
        prediction = model2.predict(new_image)
        #flash(prediction)
        predictions = np.argmax(prediction,-1)
        #flash(predictions[0])
        pred = CATEGORIES[int(predictions[0])]
        flash(pred)
        return render_template('stage.html', filename=filename,prediction_text='The classification is {}'.format(pred))
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 
@app.route('/lungcancer')
def lungcancer():
    return render_template('lungcancerdetection.html')

@app.route('/chart1//')
def chart1():
    data = pd.read_csv('dataset1.csv')
    trace0 = go.Histogram(x=data['GENDER'],name="Gender")
    trace1 = go.Histogram(x=data['AGE'],name="Age")
    trace2 = go.Histogram(x=data['SMOKING'],name="Smoking")
    trace3 = go.Histogram(x=data['YELLOW_FINGERS'],name="Yellow Fingers")
    trace5 = go.Histogram(x=data['ANXIETY'],name="Anxiety")
    trace6 = go.Histogram(x=data['PEER_PRESSURE'],name="Peer Pressure")
    trace7 = go.Histogram(x=data['CHRONIC_DISEASE'],name="Chronic Disease")
    trace8 = go.Histogram(x=data['FATIGUE'],name="Fatigue")
    trace9 = go.Histogram(x=data['ALLERGY'],name="Allergy")
    trace10 = go.Histogram(x=data['WHEEZING'],name="Wheezing")
    trace11= go.Histogram(x=data['ALCOHOL_CONSUMING'],name="Alcohol Consuming")
    trace12= go.Histogram(x=data['COUGHING'],name="Coughing")
    trace13= go.Histogram(x=data['SHORTNESS_OF_BREATH'],name="Shortness Of Breath")
    trace14= go.Histogram(x=data['SWALLOWING_DIFFICULTY'],name="Swallowing Difficulty")
    trace15= go.Histogram(x=data['CHEST_PAIN'],name="Chest Pain")
    trace16 = go.Histogram(x=data['LUNG_CANCER'],name="Lung Cancer")
    data1 = pd.read_csv('dataset2.csv')
    corrmat = data1.corr()
    trace17 = go.Heatmap( z = corrmat.values, x = list(corrmat.columns),y = list(corrmat.index),colorscale = 'Viridis',showscale=False)
    models = pd.read_csv('models.csv')
    trace18 = go.Bar(x=models['Model'], y= models['Score'],name="Model Comparison")

    fig = plotly.tools.make_subplots(
        rows=9,
        cols=3,
        specs=[[{}, {}, {}],[{},{},{}],[{}, {'colspan': 2, 'rowspan': 3}, None], [{} , None, None],[{} , None, None],[{}, {}, {}],[{},{},{}],[{'colspan': 3, 'rowspan': 1},None, None],[{'colspan': 3, 'rowspan': 1},None, None]],
        subplot_titles=('Gender','Allergy', 'Smoking',"Yellow Fingers", "Anxiety","Peer Pressure","Chronic Disease","Age","Fatigue","Lung Cancer","Wheezing","Alcohol Consuming","Coughing","Shortness Of Breath","Swallowing Difficulty","Chest Pain","Correlation Matrix"," Model Comparison")
        
    )
    fig.update_layout(width=1100,height=3000,title="Data Distribution")

    fig.append_trace(trace0, 1, 1)
    fig.append_trace(trace9, 1, 2)
    fig.append_trace(trace2, 1, 3)
    fig.append_trace(trace3, 2, 1)
    fig.append_trace(trace5, 2, 2)
    fig.append_trace(trace6, 2, 3)
    fig.append_trace(trace7, 3, 1)
    fig.append_trace(trace1, 3, 2)
    fig.append_trace(trace8, 4, 1)
    fig.append_trace(trace16, 5, 1)
    fig.append_trace(trace10, 6, 1)
    fig.append_trace(trace11, 6, 2)
    fig.append_trace(trace12, 6, 3)
    fig.append_trace(trace13, 7, 1)
    fig.append_trace(trace14, 7, 2)
    fig.append_trace(trace15, 7, 3)
    fig.append_trace(trace17, 8,1)
    fig.append_trace(trace18, 9,1)
    fig.update_layout(bargap=0.2)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    header="Lung Cancer Analysis"
    description="The Random Forest Classifer is the most accurate model for predicting Lung Cancer"
    return render_template('notdash.html', graphJSON=graphJSON,header=header,description=description)

@app.route('/chart2')
def chart2():
    data = pd.read_csv('insurance.csv')
    trace0 = go.Histogram(x=data['age'],name="AGE")
    trace1 = go.Histogram(x=data['sex'],name="SEX")
    trace2 = go.Histogram(x=data['bmi'],name="BMI")
    trace3 = go.Histogram(x=data['children'],name="CHILDREN")
    trace4 = go.Histogram(x=data['smoker'],name="SMOKER")
    trace5 = go.Histogram(x=data['region'],name="REGION")
    trace6 = go.Histogram(x=data['charges'],name="CHARGES")
    trace11 = go.Scatter(x=data['age'], y=data['bmi'], name="AGExBMI",mode='markers')
    trace12 = go.Scatter(x=data['age'], y=data['children'], name="AGExCHILDREN", mode='markers')
    trace13= go.Scatter(x=data['age'], y=data['charges'], name="AGExCHARGES", mode='markers')
    trace14= go.Scatter(x=data['bmi'], y=data['charges'],  name="BMIxCHARGES",mode='markers')
    trace15= go.Scatter(x=data['children'], y=data['charges'],  name="CHILDRENxCHARGES",mode='markers')
    trace16= go.Scatter(x=data['smoker'], y=data['charges'],  name="SMOKERxCHARGES",mode='markers')
    corrmat = data.corr()
    trace21=  go.Heatmap( z = corrmat.values, x = list(corrmat.columns),y = list(corrmat.index),colorscale = 'Viridis',showscale=False)
    
    models = pd.read_csv('modelsinsurance.csv')
    trace22 = go.Bar(x=models['Model'], y= models['Score'],name="Model Comparison",marker = {'color' : 'teal'})



    fig = plotly.tools.make_subplots(
        rows=11,
        cols=3,
        specs=[[{'rowspan':2},{},{'rowspan':2}],[None,{},None],[{},{},{}],[{'colspan':3,'rowspan':3},None,None],[None,None,None],[None,None,None],[{},{},{}],[{},{},{}],[{'colspan':3,'rowspan':3},None,None],[None,None,None],[None,None,None]],
        subplot_titles=('CHARGES','SEX','BMI','CHILDREN','SMOKER','REGION','AGE','CORRELATION MATRIX','AGExBMI','AGExCHILDREN','AGExCHARGES','BMIxCHARGES','CHILDRENxCHARGES','SMOKERxCHARGES','MODELS COMPARISON')
        
    )
    
    fig.update_layout(width=1300,height=1300)
    fig.append_trace(trace0, 3, 3)
    fig.append_trace(trace1, 1, 2)
    fig.append_trace(trace2, 1, 3)
    fig.append_trace(trace3, 3, 1)
    fig.append_trace(trace4, 2, 2)
    fig.append_trace(trace5, 3, 2)
    fig.append_trace(trace6, 1, 1)
    fig.append_trace(trace11, 7, 1)
    fig.append_trace(trace12, 7, 2)
    fig.append_trace(trace13, 7, 3) 
    fig.append_trace(trace14, 8, 1)
    fig.append_trace(trace15, 8, 2)  
    fig.append_trace(trace16, 8, 3)
    fig.append_trace(trace21, 4, 1)
    fig.append_trace(trace22, 9, 1)

    fig.update_layout(bargap=0.2,title="Data Distribution")
    header="Medical Insurance for Lung Cancer"
    description="The Random Forest Regression is the best for obtaining medical insurance"
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('notdash.html', graphJSON=graphJSON,header=header,description=description)

@app.route('/predict',methods=['POST'])
def predict():
  int_features = [int(x) for x in request.form.values()]
  if(len(int_features)>0):
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = prediction[0]
    print(output)
    if output == 1:
        res_val = "Lung cancer"
        return render_template('insurance.html', prediction_text='Patient may have {}'.format(res_val))
    else:
        res_val = "Lung cancer"
        return render_template('healthy.html', prediction_text='Patient may not have {}'.format(res_val))
  else:
        return render_template('stage.html')

@app.route('/predict1',methods=['POST'])
def predict1():
  int_features1 = [int(x) for x in request.form.values()]
  final_features1 = [np.array(int_features1)]
  prediction = model1.predict(final_features1)
  output = prediction[0]
  return render_template('insuranceoutput.html', prediction_text='Medical insurance cost is {}'.format(round(output,2)))


  
if __name__ == '__main__':
    app.run(host="localhost", port=8000, debug=True)
