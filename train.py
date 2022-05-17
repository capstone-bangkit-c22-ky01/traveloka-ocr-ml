import argparse
import os
import random
import string
import sys
import time
import tensorflow as tf
import numpy as np
from tensorflow import keras

from utils import CTCLabelConverter, CTCLabelConverterForBaiduWarpctc
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset, tensorflow_dataloader
from model import Model

def ignore_index(tensor, ignored_index=0):
    return tf.cast(tf.not_equal(tensor, ignored_index), tf.float32)

def train(opt):
    """ dataset preparation """
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')
    
    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    
    train_dataset = Batch_Balanced_Dataset(opt)
    
    log = open(f"./saved_models/{opt.exp_name}/log_dataset.txt", "a")
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)
    valid_loader = tensorflow_dataloader(valid_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.workers), collate_fn=AlignCollate_valid)
    log.write(valid_dataset_log)
    print("-" * 80)
    log.write("-" * 80 + "\n")
    log.close()
    
    """ model configuration """
    if opt.baiduCTC:
        converter = CTCLabelConverterForBaiduWarpctc(opt.character)
    else:
        converter = CTCLabelConverter(opt.character)
            
    opt.num_class = len(converter.character)
    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    
    # weight initialization
    # TODO if model behave bad
    
    if opt.saved_model != "":
        print(f"loading pretrained model from {opt.saved_model}")
        model = keras.models.load_model(opt.saved_model)
    print("Model → ")
    model.summary()
    
    if "CTC" in opt.Prediction:
        criterion = tf.nn.ctc_loss
    else:
        # kalo gak yang sparse ya yang biasa
        # selalu gunakan ignored_index=0 disini
        criterion = tf.nn.sparse_softmax_cross_entropy_with_logits
    
    loss_avg = Averager()
    
        


