import argparse
import string

import tensorflow as tf
from PIL import Image
from tensorflow import keras

from dataset import AlignCollate, RawDataset, SingleDataset, tensorflow_dataloader
from model import Model
from preprocess_image import crop_image
from utils import CTCLabelConverter


def demo(opt):
    # load model
    model = keras.models.load_model(opt.saved_model)
    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(
        imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD
    )
    left_image, top_image, right_image, bottom_image = 115, 86, 390, 131
    demo_data = SingleDataset(
        image=Image.open(opt.image_path),
        opt=opt,
        left=left_image,
        top=top_image,
        right=right_image,
        bottom=bottom_image,
        collate_fn=AlignCollate_demo,
    )
    demo_loader = tensorflow_dataloader(
        demo_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo,
    )
    image, text_for_pred = next(demo_loader.as_numpy_iterator())
    print(image.shape)
    batch_size = image.shape[0]
    text_for_pred = tf.zeros(shape=(batch_size, opt.batch_max_length), dtype=tf.float64)
    length_for_pred = tf.constant([opt.batch_max_length] * batch_size, dtype=tf.int32)
    if "CTC" in opt.Prediction:
        preds = model(image, text_for_pred)

        # Select max probabilty (greedy decoding) then decode index to character
        preds_size = tf.constant([preds.size(1)] * batch_size)
        _, preds_index = preds.max(2)
        # preds_index = preds_index.view(-1)
        preds_str = converter.decode(preds_index, preds_size)

    else:
        preds = model(image, text_for_pred, is_train=False)

        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)

    preds_prob = F.softmax(preds, dim=2)
    preds_max_prob, _ = preds_prob.max(dim=2)

    print(f"{preds_str:25s}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path", type=str, help="Image path of the data", required=True
    )
    parser.add_argument(
        "--workers", type=int, help="number of data loading workers", default=4
    )
    parser.add_argument("--batch_size", type=int, default=192, help="input batch size")
    parser.add_argument(
        "--saved_model", required=True, help="path to saved_model to evaluation"
    )
    """ Data processing """
    parser.add_argument(
        "--batch_max_length", type=int, default=25, help="maximum-label-length"
    )
    parser.add_argument(
        "--imgH", type=int, default=32, help="the height of the input image"
    )
    parser.add_argument(
        "--imgW", type=int, default=100, help="the width of the input image"
    )
    parser.add_argument("--rgb", action="store_true", help="use rgb input")
    parser.add_argument(
        "--character",
        type=str,
        default="0123456789",
        help="character label",
    )
    parser.add_argument(
        "--sensitive", action="store_true", help="for sensitive character mode"
    )
    parser.add_argument(
        "--PAD",
        action="store_true",
        help="whether to keep ratio then pad for image resize",
    )
    """ Model Architecture """
    parser.add_argument(
        "--Transformation",
        type=str,
        default="None",
        help="Transformation stage. None|TPS",
    )
    parser.add_argument(
        "--FeatureExtraction",
        type=str,
        default="VGG",
        help="FeatureExtraction stage. VGG|RCNN|ResNet",
    )
    parser.add_argument(
        "--SequenceModeling",
        type=str,
        default="None",
        help="SequenceModeling stage. None|BiLSTM",
    )
    parser.add_argument(
        "--Prediction", type=str, default="CTC", help="Prediction stage. CTC|Attn"
    )
    parser.add_argument(
        "--num_fiducial",
        type=int,
        default=20,
        help="number of fiducial points of TPS-STN",
    )
    parser.add_argument(
        "--input_channel",
        type=int,
        default=1,
        help="the number of input channel of Feature extractor",
    )
    parser.add_argument(
        "--output_channel",
        type=int,
        default=512,
        help="the number of output channel of Feature extractor",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=256, help="the size of the LSTM hidden state"
    )

    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    opt.num_gpu = len(tf.config.list_physical_devices("GPU"))

    demo(opt)
