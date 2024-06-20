
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import pandas as pd
from werkzeug.utils import secure_filename
import os
from tqdm import tqdm
import numpy as np
import json
from PIL import Image
import torch
import random
import warnings
from transformers import AutoProcessor
from transformers import AutoModelForTokenClassification
from google.cloud import vision_v1p3beta1 as vision
from google.cloud import vision_v1p3beta1 as vision
from google.cloud import vision_v1
from pdf2img import pdf2img_bp
from reviewingTool import reviewingTool_bp
from annotatingTool import annotatingTool_bp

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 2000 * 1024 * 1024  # 500 Megabytes
app.register_blueprint(pdf2img_bp)
app.register_blueprint(reviewingTool_bp)
app.register_blueprint(annotatingTool_bp)

images_path = os.path.join(app.config['UPLOAD_FOLDER'], 'images')
annotations_path = os.path.join(app.config['UPLOAD_FOLDER'], 'annotations')
predictions_path = os.path.join(app.config['UPLOAD_FOLDER'], 'predictions')
reviewed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'reviewed')
pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], 'pdf')
certificate_path = 'certificates'
for path in [images_path, annotations_path, predictions_path, reviewed_path, pdf_path, certificate_path]:
    if not os.path.exists(path):
        os.makedirs(path)

@app.route('/uploads')
def home():
    return render_template('upload.html')

@app.route('/')
def welcome():
    return render_template('welcome.html')


@app.route('/uploads/images/<filename>')
def uploaded_image(filename):
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], 'images'), filename)
   
   
@app.route('/upload', methods=['POST'])
def upload_files():
    # Create directories for uploaded file types
    image_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'images')
    for directory in [image_dir, annotations_path, predictions_path, reviewed_path]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Save uploaded files
    for image in request.files.getlist('images'):
        image.save(os.path.join(image_dir, secure_filename(image.filename)))
    for annotation in request.files.getlist('annotations'):
        annotation.save(os.path.join(annotations_path, secure_filename(annotation.filename)))
    for prediction in request.files.getlist('predictions'):
        prediction.save(os.path.join(predictions_path, secure_filename(prediction.filename)))

    # Redirect to the main page (where you'll need to process these files)
    return redirect(url_for('main_page'))

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store'
    return response

if __name__ == '__main__':
    app.run(debug=True)

