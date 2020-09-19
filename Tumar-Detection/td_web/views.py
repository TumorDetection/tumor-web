from td_web import app
from flask import render_template, redirect, flash, url_for, session, abort, request
import os
from werkzeug.utils import secure_filename
from bootstrap import tumor_detection_factory

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        file_path = file_path.replace("\\\\","\\" )
        f.save(file_path)

        # Make prediction
        td = tumor_detection_factory()
        preds = td.predict(file_path)
        if preds[0][0] == 0 :
            ans = "yes it has tumor"
        else :
            ans  = "no it has not tumor"
        return ans
