import argparse
import os
import random
import string
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

from dataset import (AlignCollate, Batch_Balanced_Dataset,
                     hierarchical_dataset, tensorflow_dataloader)
from model import Model
from modules.custom import custom_sparse_categorical_crossentropy
from utils import Averager, CTCLabelConverter, CTCLabelConverterForBaiduWarpctc


def ignore_index(tensor, ignored_index=0):
    return tf.cast(tf.not_equal(tensor, ignored_index), tf.float32)


def train(opt):
    """dataset preparation"""
    if not opt.data_filtering_off:
        print(
            "Filtering the images containing characters which are not in opt.character"
        )
        print("Filtering the images whose label is longer than opt.batch_max_length")

    opt.select_data = opt.select_data.split("-")
    opt.batch_ratio = opt.batch_ratio.split("-")

    train_dataset = Batch_Balanced_Dataset(opt)

    log = open(f"./saved_models/{opt.exp_name}/log_dataset.txt", "a")
    AlignCollate_valid = AlignCollate(
        imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD
    )
    valid_dataset, valid_dataset_log = hierarchical_dataset(
        root=opt.valid_data, opt=opt
    )
    valid_loader = tensorflow_dataloader(
        valid_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid,
    )
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
        criterion = custom_sparse_categorical_crossentropy

    loss_avg = Averager()
    
    if opt.adam:
        optimizer = keras.optimizers.Adam(learning_rate=opt.lr, beta_1=opt.beta1, beta_2=0.999)
    else:
        optimizer = keras.optimizers.Adadelta(learning_rate=opt.lr, rho=opt.rho, epsilon=opt.eps)
        
    print("Optimizer:")
    print(optimizer)
    
    """ final options """
    # print(opt)
    with open(f'./saved_models/{opt.exp_name}/opt.txt', 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)
        
    """ start training """
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass
        
    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    iteration = start_iter
    
    strategy = tf.distribute.MirroredStrategy()
    
    while(True):
        image_tensors, labels = train_dataset.get_batch()
        image = image_tensors
        text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
        batch_size = image.shape[0]
        
        if "CTC" in opt.Prediction:
            with tf.GradientTape() as tape:
                preds = model(image, text)
                preds_size = tf.constant([preds.shape[1]] * batch_size)
            if opt.baiduCTC:
                preds = tf.transpose(preds, perm=[1, 0, 2])
                cost = criterion(labels=preds, logits=text, label_length=preds_size, logit_length=length) / batch_size
            else:
                preds = tf.nn.log_softmax(preds, axis=2)
                preds = tf.transpose(preds, perm=[1, 0, 2])
                cost = criterion(labels=preds, logits=text, label_length=preds_size, logit_length=length)
        
        # this could be total mess
        variables = model.trainable_variables
        gradients = tape.gradient(cost, variables)
        gradients, _ = tf.clip_by_global_norm(gradients, opt.grad_clip)
        optimizer.apply_gradients(zip(gradients, variables))
        
        loss_avg.add(cost)
        # continue to validation
