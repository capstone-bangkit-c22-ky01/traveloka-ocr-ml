import os

from numpy import percentile

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json

import tensorflow as tf
import tensorflow_addons as tfa
from flask import Flask, jsonify, request
from PIL import Image
from tensorflow import keras

from dataset import AlignCollate, SingleDataset, tensorflow_dataloader
from utils import CTCLabelConverter

arial_model = keras.models.load_model(
    "saved_models/Arial_Model/best_accuracy",
    custom_objects={"AAP": tfa.layers.AdaptiveAveragePooling2D},
)
nik_model = keras.models.load_model(
    "saved_models/NIK_model/best_accuracy",
    custom_objects={"AAP": tfa.layers.AdaptiveAveragePooling2D},
)

app = Flask(__name__)


def getObject(file_json, label):
    objects = []
    objects.append(file_json["image"])
    objects.append(file_json["class"][label]["Xmin"])
    objects.append(file_json["class"][label]["Ymin"])
    objects.append(file_json["class"][label]["Xmax"])
    objects.append(file_json["class"][label]["Ymax"])
    return objects  # [image, Xmin, Ymin, Xmax, Ymax]

#def getObject(file_json, label):
#    objects = []
#    objects.append(file_json["image"])
#    image = Image.open(file_json["image"])
#    w, h = image.size
#    objects.append(int(file_json["class"][label]["Xmin"]/300*w))
#    objects.append(int(file_json["class"][label]["Ymin"]/300*h))
#    objects.append(int(file_json["class"][label]["Xmax"]/300*w))
#    objects.append(int(file_json["class"][label]["Ymax"]/300*h))
#    return objects  # [image, Xmin, Ymin, Xmax, Ymax]


def predict_arial(json_input):
    converter = CTCLabelConverter("abcdefghijklmnopqrstuvwxyz,. -")
    model = arial_model

    AlignCollate_demo = AlignCollate(imgH=32, imgW=100, keep_ratio_with_pad=False)

    labels = ["name", "sex", "married", "nationality"]
    objs = []
    for label in labels:
        objs.append(getObject(json_input, label))

    demo_datas = []
    for obj in objs:
        demo_data = SingleDataset(
            image=Image.open(obj[0]),
            # opt=opt,
            left=obj[1],
            top=obj[2],
            right=obj[3],
            bottom=obj[4],
            collate_fn=AlignCollate_demo,
        )
        demo_datas.append(demo_data)

    arials = []
    for demo_data in demo_datas:
        demo_loader = tensorflow_dataloader(
            demo_data,
            batch_size=1,
            shuffle=True,
            num_workers=4,
            collate_fn=AlignCollate_demo,
        )
        image, text_for_pred = next(demo_loader.as_numpy_iterator())
        batch_size = image.shape[0]
        text_for_pred = tf.zeros(shape=(batch_size, 25), dtype=tf.float64)
        # length_for_pred = tf.constant([25] * batch_size, dtype=tf.int32)

        preds = model(image, text_for_pred)

        # Select max probabilty (greedy decoding) then decode index to character
        preds_size = tf.constant([preds.shape[1]] * batch_size)
        preds_index = tf.math.argmax(preds, axis=2)
        # preds_index = preds_index.view(-1)
        preds_str = converter.decode(preds_index, preds_size)

        arials.append(preds_str[0].upper())

    return arials  # [name, sex, married, nationality]


def predict_nik(json_input):
    converter = CTCLabelConverter("0123456789")

    model = nik_model

    AlignCollate_demo = AlignCollate(imgH=32, imgW=100, keep_ratio_with_pad=False)
    obj = getObject(json_input, "NIK")
    demo_data = SingleDataset(
        image=Image.open(obj[0]),
        # opt=opt,
        left=obj[1],
        top=obj[2],
        right=obj[3],
        bottom=obj[4],
        collate_fn=AlignCollate_demo,
    )
    demo_loader = tensorflow_dataloader(
        demo_data,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        collate_fn=AlignCollate_demo,
    )
    image, text_for_pred = next(demo_loader.as_numpy_iterator())
    batch_size = image.shape[0]
    text_for_pred = tf.zeros(shape=(batch_size, 25), dtype=tf.float64)
    # length_for_pred = tf.constant([25] * batch_size, dtype=tf.int32)

    preds = model(image, text_for_pred)

    preds_size = tf.constant([preds.shape[1]] * batch_size)
    preds_index = tf.math.argmax(preds, axis=2)
    preds_str = converter.decode(preds_index, preds_size)
    nik = preds_str[0]
    return nik  # String NIK

def process_sex(input):
    male = ['K', 'L']
    female = ['P', 'R']
    sex = 0
    for ch in input:
        if ch in male:
            sex += 1
        elif ch in female:
            sex -= 1

    return "LAKI-LAKI" if sex >= 0 else "PEREMPUAN"

def process_gender(input):
    gender = re.sub(r'[abcdefghijoqstuvwxyz,. -]', '', input) + ' '
    if 'l' in gender or 'k' in gender:
        final = 'laki-laki'
    elif 'p' in gender or 'r' in gender or 'm' in gender or 'n' in gender:
        final = 'perempuan'
    elif ' ' in gender:
        final = 'default'
    return final

def process_married(input):
    kawin = re.sub(r'[abcdefghijlmnopqrstuvxyz,. -]', '', input)
    kawin = re.sub(r'[kw]+', 'kawin', kawin)
    cerai = re.sub(r'[abdefghijklmnopqstuvwxyz,. -]', '', input)
    cerai = re.sub(r'[cr]+', 'cerai', cerai)

    if kawin == 'kawin':
        belum = re.sub(r'[acdfghijnopqrstvwxyz,. -]', '', input)
        belum = re.sub(r'[belum]+', 'belum', input)
        if belum == 'belum':
            final = 'belum kawin'
        else:
            final = 'kawin'
    elif cerai == 'cerai':
        cerai = re.sub(r'[abcefgijklnoqrsuvwxyz,. -]', '', input) + ' '
        if 'h' in cerai or 'd' in cerai or 'p' in cerai:
            final = 'cerai hidup'
        elif 'm' in cerai or 't' in cerai:
            final = 'cerai mati'
        elif " " in cerai:
            final = 'default'
    else:
        final = 'default'
    return final

def process_national(input):
    national = re.sub(r'[abcdefghjklmopqrstuvxyz,. -]', '', input) + ' '

    if 'wni ' in national or 'w' in national or 'ni' in national:
        final = 'Indonesia'
    elif ' ' in national:
        final = 'default'
    return final

def predict(json_input):
    nik = predict_nik(json_input)
    arials = predict_arial(json_input)
    dict = {
        "nik": nik,
        "name": arials[0],
        "sex": process_sex(arials[1]),
        "married": process_married(arials[2]),
        "nationality": process_national(arials[3]),
    }
    return json.dumps(dict)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # content = request.json

        # print(content)
        if request.json is None:
            return jsonify({"error": "no data"})

        try:
            prediction = predict(request.get_json())
            return prediction
        except Exception as e:
            return jsonify({"error": str(e)})

        # return predict(request.json)

        # file = request.files.get('file')
        # if file is None or file.filename == "":
        #     return jsonify({"error": "no file"})

        # try:
        #     image_bytes = file.read()
        #     print(image_bytes)
        #     pillow_img = Image.open(io.BytesIO(image_bytes)).convert('L')
        #     tensor = transform_image(pillow_img)
        #     prediction = predict(tensor)
        #     data = {"prediction": int(prediction)}
        #     return jsonify(data)

    return "OK"


if __name__ == "__main__":
    app.run(debug=True)
