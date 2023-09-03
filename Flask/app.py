from keras.applications.xception import preprocess_input
from keras.preprocessing import image
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask,render_template,request
app=Flask(__name__)

model=load_model(r"xception_covid.h5",compile=False)
@app.route('/')
def index():
    return render_template("index.html")
@app.route('/precaution/')
def next1():
    return render_template("precaution.html")
@app.route('/about/')
def next3():
    return render_template("about.html")
@app.route('/test/')
def next2():
    return render_template("test.html")

@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        print("Hello")
        filepath=os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)
        img=image.load_img(filepath,target_size=(224,224))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)

        img_data=preprocess_input(x)
        pred=np.argmax(model.predict(img_data),axis=1)

        print(pred)
        index=['COVID','Lung_Opacity','Normal','Viral Pneumonia']
        text="The result of x-ray : " +str(index[pred[0]])
    return text

if __name__=='__main__':
    app.run(debug=True)
