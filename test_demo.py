import argparse
import os
import random
import string
import sys
import time
from test import validation

import numpy as np
import tensorflow as tf
from tensorflow import keras

from dataset import (
    AlignCollate,
    ApplyCollate,
    Batch_Balanced_Dataset,
    hierarchical_dataset,
    tensorflow_dataloader,
)
from model import Model
from modules.custom import custom_sparse_categorical_crossentropy
from utils import Averager, CTCLabelConverter, CTCLabelConverterForBaiduWarpctc


def main(opt):
    opt.select_data = opt.select_data.split("-")
    opt.batch_ratio = opt.batch_ratio.split("-")
    train_dataset = Batch_Balanced_Dataset(opt)
    a = train_dataset.get_batch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", help="Where to store logs and models")
    parser.add_argument("--train_data", required=True, help="path to training dataset")
    parser.add_argument(
        "--valid_data", required=True, help="path to validation dataset"
    )
    parser.add_argument(
        "--manualSeed", type=int, default=1111, help="for random seed setting"
    )
    parser.add_argument(
        "--workers", type=int, help="number of data loading workers", default=4
    )
    parser.add_argument("--batch_size", type=int, default=192, help="input batch size")
    parser.add_argument(
        "--num_iter", type=int, default=300000, help="number of iterations to train for"
    )
    parser.add_argument(
        "--valInterval", type=int, default=2000, help="Interval between each validation"
    )
    parser.add_argument(
        "--saved_model", default="", help="path to model to continue training"
    )
    parser.add_argument("--FT", action="store_true", help="whether to do fine-tuning")
    parser.add_argument(
        "--adam", action="store_true", help="Whether to use adam (default is Adadelta)"
    )
    parser.add_argument(
        "--lr", type=float, default=1, help="learning rate, default=1.0 for Adadelta"
    )
    parser.add_argument(
        "--beta1", type=float, default=0.9, help="beta1 for adam. default=0.9"
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.95,
        help="decay rate rho for Adadelta. default=0.95",
    )
    parser.add_argument(
        "--eps", type=float, default=1e-8, help="eps for Adadelta. default=1e-8"
    )
    parser.add_argument(
        "--grad_clip", type=float, default=5, help="gradient clipping value. default=5"
    )
    parser.add_argument(
        "--baiduCTC", action="store_true", help="for data_filtering_off mode"
    )
    """ Data processing """
    parser.add_argument(
        "--select_data",
        type=str,
        default="MJ-ST",
        help="select training data (default is MJ-ST, which means MJ and ST used as training data)",
    )
    parser.add_argument(
        "--batch_ratio",
        type=str,
        default="0.5-0.5",
        help="assign ratio for each selected data in the batch",
    )
    parser.add_argument(
        "--total_data_usage_ratio",
        type=str,
        default="1.0",
        help="total data usage ratio, this ratio is multiplied to total number of data.",
    )
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
        default="0123456789abcdefghijklmnopqrstuvwxyz",
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
    parser.add_argument(
        "--data_filtering_off", action="store_true", help="for data_filtering_off mode"
    )
    """ Model Architecture """
    parser.add_argument(
        "--Transformation",
        type=str,
        required=True,
        help="Transformation stage. None|TPS",
    )
    parser.add_argument(
        "--FeatureExtraction",
        type=str,
        required=True,
        help="FeatureExtraction stage. VGG|RCNN|ResNet",
    )
    parser.add_argument(
        "--SequenceModeling",
        type=str,
        required=True,
        help="SequenceModeling stage. None|BiLSTM",
    )
    parser.add_argument(
        "--Prediction", type=str, required=True, help="Prediction stage. CTC|Attn"
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

    if not opt.exp_name:
        opt.exp_name = f"{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}"
        opt.exp_name += f"-Seed{opt.manualSeed}"
        # print(opt.exp_name)

    os.makedirs(f"./saved_models/{opt.exp_name}", exist_ok=True)

    """ vocab / character number configuration """
    if opt.sensitive:
        # opt.character += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    tf.random.set_seed(opt.manualSeed)

    opt.num_gpu = len(tf.config.list_physical_devices("GPU"))
    # print('device count', opt.num_gpu)
    if opt.num_gpu > 1:
        print("------ Use multi-GPU setting ------")
        print(
            "if you stuck too long time with multi-GPU setting, try to set --workers 0"
        )
        # check multi-GPU issue https://github.com/clovaai/deep-text-recognition-benchmark/issues/1
        opt.workers = opt.workers * opt.num_gpu
        opt.batch_size = opt.batch_size * opt.num_gpu

        """ previous version
        print('To equlize batch stats to 1-GPU setting, the batch_size is multiplied with num_gpu and multiplied batch_size is ', opt.batch_size)
        opt.batch_size = opt.batch_size * opt.num_gpu
        print('To equalize the number of epochs to 1-GPU setting, num_iter is divided with num_gpu by default.')
        If you dont care about it, just commnet out these line.)
        opt.num_iter = int(opt.num_iter / opt.num_gpu)
        """

    main(opt)
