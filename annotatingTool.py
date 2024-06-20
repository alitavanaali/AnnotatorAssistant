
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, Blueprint
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

annotatingTool_bp = Blueprint('annotate', __name__, template_folder='templates')

UPLOAD_FOLDER = 'uploads'
data = {}


images_path = os.path.join(UPLOAD_FOLDER, 'images')
annotations_path = os.path.join(UPLOAD_FOLDER, 'annotations')
predictions_path = os.path.join(UPLOAD_FOLDER, 'predictions')
reviewed_path = os.path.join(UPLOAD_FOLDER, 'reviewed')

#id2label= {0: 'client_code', 1: 'client_id', 2: 'client_reference', 3: 'client_vat', 4: 'delivery_place', 5: 'detail_desc', 6: 'detail_packnr', 7: 'detail_packtype', 8: 'detail_qty', 9: 'detail_weight', 10: 'detail_weight_um', 11: 'doc_date', 12: 'doc_nr', 13: 'doc_type', 14: 'issuer_addr', 15: 'issuer_cap', 16: 'issuer_city', 17: 'issuer_contact', 18: 'issuer_contact_email', 19: 'issuer_contact_phone', 20: 'issuer_fax', 21: 'issuer_name', 22: 'issuer_prov', 23: 'issuer_state', 24: 'issuer_tel', 25: 'issuer_vat', 26: 'operation_code', 27: 'order_date', 28: 'order_nr', 29: 'others', 30: 'pickup_date', 31: 'pickup_place', 32: 'receiver_addr', 33: 'receiver_cap', 34: 'receiver_city', 35: 'receiver_fax', 36: 'receiver_name', 37: 'receiver_prov', 38: 'receiver_state', 39: 'receiver_tel', 40: 'receiver_vat', 41: 'recipient_name', 42: 'ref_nr', 43: 'sender_name', 44: 'service_date', 45: 'service_date-end', 46: 'service_key', 47: 'service_order', 48: 'service_value', 49: 'shipment_nr', 50: 'time', 51: 'tot_value', 52: 'correspondent_ref', 53: 'vehicle_plate'}
#id2label = {0: 'box_id', 1: 'client_id', 2: 'client_name', 3: 'client_address', 4: 'giro', 5: 'date', 6: 'time', 7: 'others', 8: 'picking', 9: 'client_city', 10: 'client_cap', 11: 'client_prov', 12: 'sender_name'}
id2label = {}
# replace your Huggingface access token here if you want to directly load page
#auth_token = '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'
#model_path = 'DataIntelligenceTeam/pharma_label_v3.2'

global model
global processor

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'certificates/GCP_APIKEY.json'

@annotatingTool_bp.route('/annotate', methods=['POST'])
def annotatePage():
    model_path = request.form['modelPath']
    auth_token = request.form['token']
    labels_with_colors = json.loads(request.form['labels'])
    
    # Initialize id2label and label2color from received data
    label2color = {}
    for idx, label_info in enumerate(labels_with_colors):
        label_name = label_info['name']
        label_color = label_info['color']
        id2label[idx] = label_name
        if label_name == 'others':
            label2color[label_name] = '#0000004d'
        else:
            label2color[label_name] = label_color


    if model_path and auth_token:
        global model
        global processor
        processor = AutoProcessor.from_pretrained(model_path, apply_ocr=False, token=auth_token)
        model = AutoModelForTokenClassification.from_pretrained(model_path, token=auth_token)

    data = loadData()
    print(f'annotatePage has been called. length of data: {len(data)}')
    #label2color = populateLabel2Color(id2label)
    return render_template('annotate.html', data=data, label2color=json.dumps(label2color), model_path=model_path, auth_token=auth_token)

def loadData():
    # Assuming filenames without extensions are the same across images, annotations, and predictions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    data = {}
    for filename in os.listdir(images_path):
        # Check if the file has an allowed image extension and is not a .DS_Store file
        if filename != '.DS_Store' and any(filename.lower().endswith(ext) for ext in image_extensions):
            data[filename] = {}
    return data

def detectText(img_path):
    client = vision_v1.ImageAnnotatorClient()

    with open(img_path, 'rb') as image_file:
        content = image_file.read()

    image = vision_v1.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    df = pd.DataFrame(columns=['word','xmin', 'xmax', 'ymin', 'ymax'])

    # Get image dimensions
    image = Image.open(img_path)
    width, height = image.size

    rows = []

    for text in texts[1:]:  # Start from 1 to skip the full image annotation
        vertices = text.bounding_poly.vertices
        xmin = vertices[0].x
        ymin = vertices[0].y
        xmax = vertices[2].x
        ymax = vertices[2].y
        confidence = text.confidence
        text = text.description

        row = {
            'word': text,
            'xmin': xmin,
            'xmax': xmax,
            'ymin': ymin,
            'ymax': ymax,
            'label': 'others'  # Constant value "others" for the label
        }
        rows.append(row)
    new_df = pd.DataFrame(rows)
    df = pd.concat([df, new_df], ignore_index=True)
    return df

@annotatingTool_bp.route('/annotate/perform-ocr/<image_name>')
def perform_ocr(image_name):
    img_path = os.path.join(images_path, image_name)
    df = detectText(img_path)  # Your OCR function

    # Save to CSV in the annotations directory
    output_path = os.path.join(annotations_path, image_name.replace('.jpg', '.csv'))
    df.to_csv(output_path, index=False)

    return jsonify({'message': f'OCR results saved to {output_path}'})

@annotatingTool_bp.route('/annotate/predict-labels/<image_name>')
def predictLabels(image_name):
    new_csv_path = os.path.join(predictions_path, image_name[:-4] + '.csv')
    if not os.path.exists(new_csv_path):
        # OCR not performed, perform OCR first
        perform_ocr(image_name)

    global model
    global processor
    def unnormalize_box(bbox, width, height):
     #print('shape is: ', np.asarray(bbox).shape, ' and box has values: ', bbox)
     return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
     ]
    def getInference(processor, encoding, outputs, width, height):
        # get predictions
        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        token_boxes = encoding.bbox.squeeze().tolist()

        # only keep non-subword predictions
        preds = []
        l_words = []
        bboxes = []
        token_section_num = []

        if (len(token_boxes) == 512):
            predictions = [predictions]
            token_boxes = [token_boxes]

        box_token_dict = {}
        for i in range(0, len(token_boxes)):
            #print(len(token_boxes[i]))
            # skip first 128 tokens from second list to last one (128 is the stride value) except the first window
            initial_j = 0 if i == 0 else 129
            for j in range(initial_j, len(token_boxes[i])):
                unnormal_box = unnormalize_box(token_boxes[i][j], width, height)
                if (np.asarray(token_boxes[i][j]).shape != (4,)):
                    continue
                elif (token_boxes[i][j] == [0, 0, 0, 0] or token_boxes[i][j] == 0):
                    #print(processor.tokenizer.decode(encoding["input_ids"][i][j]))
                    continue
                else:
                    bbox = tuple(unnormal_box)  # Convert the list to a tuple
                    token = processor.tokenizer.decode(encoding["input_ids"][i][j])
                    if bbox not in box_token_dict:
                        box_token_dict[bbox] = [token] # token = word
                    else:
                        box_token_dict[bbox].append(token)

        box_token_dict = {bbox: [''.join(words)] for bbox, words in box_token_dict.items()}
        boxes = list(box_token_dict.keys())
        words = list(box_token_dict.values())
        preds = []

        # here we wil find all the predictions foreach boundingbox.
        # for example: [120, 80, 70, 30] = [7, 7, 1] (label 7 and 1 referring to a specific label in id2label
        box_prediction_dict = {}
        for i in range(0, len(token_boxes)):
            for j in range(0, len(token_boxes[i])):
                if (np.asarray(token_boxes[i][j]).shape != (4,)):
                    continue
                elif (token_boxes[i][j] == [0, 0, 0, 0] or token_boxes[i][j] == 0):
                    continue
                else:
                    bbox = tuple(token_boxes[i][j])  # Convert the list to a tuple
                    prediction = predictions[i][j]
                    if bbox not in box_prediction_dict:
                        box_prediction_dict[bbox] = [prediction]
                    else:
                        box_prediction_dict[bbox].append(prediction)

        # now foreach bbox we have a list of it's predictions, then we are going to count them and
        # select the most repeated one as the final prediction for our bbox
        for i, (bbox, predictions) in enumerate(box_prediction_dict.items()):
            count_dict = {}  # Dictionary to store the count of each prediction label
            for prediction in predictions:
                if prediction in count_dict:
                    count_dict[prediction] += 1
                else:
                    count_dict[prediction] = 1

            max_count = max(count_dict.values())  # Find the maximum count of a prediction label
            max_predictions = [key for key, value in count_dict.items() if
                            value == max_count]  # Find the predictions with the maximum count
            # If there is only one prediction with the maximum count so we can add it to the preds, but
            # if there are more than one, so we don't know which one should be label for the box
            if len(max_predictions) == 1:
                preds.append(max_predictions[0])
            else:
                others_id = next((id_ for id_, label in id2label.items() if label == 'others'), None)
                #print(f'others_id:{others_id}')
                if others_id in count_dict:
                    #print('yess it is here')
                    count_dict.pop(others_id)
                    new_max_count = max(count_dict.values())
                    new_max_predictions = [key for key, value in count_dict.items() if
                                    value == new_max_count]  # Find the predictions with the maximum count
                    if len(new_max_predictions) == 1:
                        preds.append(new_max_predictions[0])
                        continue
                # the idea is to look at next and previous items in the dictionary, if they have a label like the labels
                # that we have for this box, so we can select it for label
                # ex: ANKORIG SRL - ANKORIG = [0, 6] SRL = [6] so--> ANKORIG could be considered as 6
                max_next_prev_item = []
                # Check the previous item (if it exists)
                if i - 1 >= 0:
                    prev_predictions = box_prediction_dict[
                        list(box_prediction_dict.keys())[i - 1]]  # Get the predictions of the previous item
                    prev_count_dict = {key: prev_predictions.count(key) for key in
                                    set(prev_predictions)}  # Count the occurrences of each prediction
                    max_prev_count = max(prev_count_dict.values())  # Find the maximum count of a prediction label
                    max_prev_predictions = [key for key, value in prev_count_dict.items() if
                                            value == max_prev_count]  # Find the predictions with the maximum count
                    max_next_prev_item.extend(max_prev_predictions)  # Add the predictions to the max_next_prev_item list

                # Check the next item (if it exists)
                if i + 1 < len(box_prediction_dict):
                    next_predictions = box_prediction_dict[
                        list(box_prediction_dict.keys())[i + 1]]  # Get the predictions of the next item
                    next_count_dict = {key: next_predictions.count(key) for key in
                                    set(next_predictions)}  # Count the occurrences of each prediction
                    max_next_count = max(next_count_dict.values())  # Find the maximum count of a prediction label
                    max_next_predictions = [key for key, value in next_count_dict.items() if
                                            value == max_next_count]  # Find the predictions with the maximum count
                    max_next_prev_item.extend(max_next_predictions)  # Add the predictions to the max_next_prev_item list

                if any(prediction in max_next_prev_item for prediction in
                    max_predictions):  # If there are common predictions between max_next_prev_item and max_predictions
                    common_predictions = list(
                        set(max_predictions).intersection(max_next_prev_item))  # Find the common predictions
                    # Add a randomly chosen common prediction to the preds, random would work if we have again more than one choice
                    preds.append(random.choice(common_predictions))
                else:
                    preds.append(random.choice(
                        max_predictions))  # Add a randomly chosen prediction with the maximum count to the preds

        flattened_words = [word[0].strip() for word in words] # because words are list of lists
        return boxes, preds, flattened_words
    def create_adjusted_bounding_box(row, width_scale, height_scale):
            # Calculate scaled coordinates directly
            x1, y1 = int(row['xmin'] * width_scale), int(row['ymin'] * height_scale)
            x2, y2 = int(row['xmax'] * width_scale), int(row['ymax'] * height_scale)
            # Ensure coordinates are ordered correctly
            return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    def iob_to_label(label):
        return id2label.get(label, 'others')
    
    try:
          #new_csv_path = os.path.join(predictions_path, image_name[:-4] + '.csv')
          image = Image.open(os.path.join(images_path, image_name))
          width, height = image.size
          doc_csv_name = image_name[:-4] + '.csv'
          df = pd.read_csv(os.path.join(annotations_path, doc_csv_name)).fillna(' ')
          width_scale, height_scale = 1000 / width, 1000 / height
          original_boxes = [[row['xmin'], row['ymin'], row['xmax'], row['ymax']] for _, row in df.iterrows()]
          boxes = [create_adjusted_bounding_box(row, width_scale, height_scale) for _, row in df.iterrows()]
          words = list(df['word'])
          labels = list(df['label'])
          if len(words)>1000:
              print(f'large document ({image_name}) with more than 1000 words. due to risk of crash on memory document will skip')
          inference_image = [image.convert("RGB")]

          encoding = processor(inference_image, words, boxes=boxes, truncation=True, return_offsets_mapping=True,
                              return_tensors="pt", padding="max_length", stride =128, max_length=512,
                              return_overflowing_tokens=True)

          offset_mapping = encoding.pop('offset_mapping')
          overflow_to_sample_mapping = encoding.pop('overflow_to_sample_mapping')

          x = []
          for i in range(0, len(encoding['pixel_values'])):
            x.append(encoding['pixel_values'][i])
          x = torch.stack(x)
          encoding['pixel_values'] = x
          # forward pass
          outputs = model(**encoding)

          detectedBoxes, detectedLabelsID, detectedWords = getInference(processor, encoding, outputs, width, height)

          detectedLabels = []
          for id in detectedLabelsID:
              detectedLabels.append(iob_to_label(id))


          xmin_list, ymin_list, xmax_list, ymax_list = [], [], [], []

          if len(original_boxes) != len(detectedBoxes):
              print(f'document {image_name} has different boxes, so it skipped in saving data!')


          for box in original_boxes: # replaced with detectedBoxes
              xmin_list.append(box[0])
              ymin_list.append(box[1])
              xmax_list.append(box[2])
              ymax_list.append(box[3])

          prediction_df = pd.DataFrame({
              'word': words,
              'xmin': xmin_list,
              'xmax': xmax_list,
              'ymin': ymin_list,
              'ymax': ymax_list,
              'label': detectedLabels})

          prediction_df.to_csv(new_csv_path, index=False)
          return jsonify({'message': f'prediction results saved to {new_csv_path}'})

    except TypeError as e:
        print(f"Skipping {image_name} due to error: {e}")
        return jsonify({'message': f'error happenned in getting predictions for {new_csv_path}'})

def read_csv(annotation_path, isDataFromReviewed = False):
    annotation_df = pd.read_csv(annotation_path).fillna(' ')

    image_data = []
    for index, ann_row in annotation_df.iterrows():
        word = ann_row['word']
        xmin, xmax, ymin, ymax, label = ann_row[['xmin', 'xmax', 'ymin', 'ymax', 'label']]
        
        image_data.append({
            'index': index,
            'word': word,
            'coords': [xmin, ymin, xmax, ymax],
            'label': label
        })

    return image_data

@annotatingTool_bp.route('/annotate/get-image-data/<image_name>')
def get_image_data(image_name):
    annotation_csv = os.path.join(annotations_path, image_name[:-4]+'.csv')
    prediction_csv = os.path.join(predictions_path, image_name[:-4]+'.csv')
    reviewed_csv = os.path.join(reviewed_path, image_name[:-4]+'.csv')
    if os.path.exists(reviewed_csv):
            data[image_name] = read_csv(reviewed_csv)
    elif os.path.exists(prediction_csv):
            data[image_name] = read_csv(prediction_csv)
    elif os.path.exists(annotation_csv):
            data[image_name] = read_csv(annotation_csv)
    return data[image_name]

def populateLabel2Color(id2label):
    #np.random.seed(42) 
    colors = ['darkblue', 'dodgerblue', 'aqua', 'turquoise','violet','DeepPink', 'darkviolet', 'MediumSlateBlue', 'maroon', 'Peru', 'lightcoral', 'darkorange', 'OrangeRed', 'khaki', 'gold', 'olivedrab', 'YellowGreen', 'lime']
    label2color = {}
    for label in id2label.values():
        if label == 'others':
            label2color[label] = 'black'  # Always assign 'black' to 'others'
        else:
            label2color[label] = np.random.choice(colors)
    return label2color
    
@annotatingTool_bp.route('/annotate/save-image-data', methods=['POST'])
def save_image_data():
    data = request.json
    image_name = data['imageName']
    retrievedData = data['imageData']
    # Prepare DataFrame
    columns = ['word', 'xmin', 'xmax', 'ymin', 'ymax', 'label']
    rows = []

    for item in retrievedData:
        coords = item['coords']
        word = item['word']
        label = item['label']
        rows.append([word, coords[0], coords[2], coords[1], coords[3], label])

    df = pd.DataFrame(rows, columns=columns)

    # Save to CSV
    csv_path = os.path.join(reviewed_path, f"{image_name[:-4]}.csv")
    df.to_csv(csv_path, index=False)

    return jsonify({"message": "Data saved successfully", "file_path": csv_path})

