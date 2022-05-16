import math
import os
import re
import sys
from typing import Sequence

import lmdb
import six
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import preprocessing


# taken from python 3.5 docs
def _accumulate(iterable, fn=lambda x, y: x + y):
    "Return running totals"
    # _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total


def tensorflow_dataloader(
    dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
    collate_fn=None,
    prefetch_factor=2,
):
    data = tf.data.Dataset.from_generator(dataset)
    if shuffle:
        data = data.shuffle(len(data))  # rawan error
    if collate_fn:
        data = data.map(collate_fn, num_parallel_calls=num_workers)
    data = data.batch(batch_size)
    data = data.prefetch(prefetch_factor)
    return data


# rawan error
class Subset(keras.utils.Sequence):
    def __init__(self, dataset, indices: Sequence):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        else:
            self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class ResizeNormalize(object):
    def __init__(self, size, interpolation=Image.BICUBIC) -> None:
        self.size = size
        self.interpolation = interpolation
        self.toTensor = preprocessing.image.img_to_array

    def __call__(self, img) -> tf.Tensor:
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img = tf.math.subtract(img)
        img = tf.math.divide(img)


class NormalizePAD(object):
    def __init__(self, max_size, PAD_type="right") -> None:
        self.toTensor = preprocessing.image.img_to_array
        self.max_size = max_size
        self.max_width_half = tf.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, image: Image) -> tf.Tensor:
        img = self.toTensor(image)
        img = tf.math.subtract(img)
        img = tf.math.divide(img)

        c, h, w = img.shape
        Pad_img = tf.zeros(shape=self.max_size)
        Pad_img[:, :, :w] = img
        if self.max_size[2] != w:
            Pad_img[:, :, w:] = tf.broadcast_to(
                tf.expand_dims(img, axis=2), shape=(c, h, self.max_size[2] - w)
            )

        return Pad_img


class AlignCollate(object):
    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False) -> None:
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        if self.keep_ratio_with_pad:
            resized_max_w = self.imgW
            input_channel = 3 if images[0].mode == "RGB" else 1
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.shape

                ratio = w / float(h)

                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))
                # resized_image.save('./image_test/%d_test.jpg' % w)

            image_tensors = tf.concat(
                [tf.expand_dims(t, axis=0) for t in resized_image], 0
            )

        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = tf.concat(
                [tf.expand_dims(t, axis=0) for t in resized_image], 0
            )

        return image_tensors, labels


class Batch_Balanced_Dataset(object):
    def __init__(self, opt) -> None:
        """
        Modulate the data ratio in the batch.
        For example, when select_data is "MJ-ST" and batch_ratio is "0.5-0.5",
        the 50% of the batch is filled with MJ and the other 50% of the batch is filled with ST.
        """
        log = open(f"./saved_models/{opt.exp_name}/log_dataset.txt", "a")
        dashed_line = "-" * 80
        print(dashed_line)
        log.write(dashed_line + "\n")
        print(
            f"dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}"
        )
        log.write(
            f"dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}\n"
        )
        assert len(opt.select_data) == len(opt.batch_ratio)

        _AlignCollate = AlignCollate(
            imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD
        )
        self.data_loader_list = []
        self.dataloader_iter_list = []
        batch_size_list = []
        Total_batch_size = 0
        for selected_d, batch_ratio_d in zip(opt.select_data, opt.batch_ratio):
            _batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)
            print(dashed_line)
            log.write(dashed_line + "\\n")
            _dataset, _dataset_log = hierarchical_dataset(
                root=opt.train_data, opt=opt, select_data=[selected_d]
            )
            total_number_dataset = len(_dataset)
            log.write(_dataset_log)

            """
            The total number of data can be modified with opt.total_data_usage_ratio.
            ex) opt.total_data_usage_ratio = 1 indicates 100% usage, and 0.2 indicates 20% usage.
            See 4.2 section in our paper.
            """
            number_dataset = int(
                total_number_dataset * float(opt.total_data_usage_ratio)
            )
            dataset_split = [number_dataset, total_number_dataset - number_dataset]
            indices = range(total_number_dataset)
            _dataset, _ = [
                Subset(_dataset, indices[offset - length : offset])
                for offset, length in zip(_accumulate(dataset_split), dataset_split)
            ]
            selected_d_log = f"num total samples of {selected_d}: {total_number_dataset} x {opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n"
            selected_d_log += f"num samples of {selected_d} per batch: {opt.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}"
            print(selected_d_log)
            log.write(selected_d_log + "\\n")
            batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            _data_loader = tensorflow_dataloader(
                _dataset,
                batch_size=_batch_size,
                shuffle=True,
                collate_fn=_AlignCollate,
                num_workers=int(opt.workers),
            )
            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))

        Total_batch_size_log = f"{dashed_line}\n"
        batch_size_sum = "+".join(batch_size_list)
        Total_batch_size_log += (
            f"Total_batch_size: {batch_size_sum} = {Total_batch_size}\n"
        )
        Total_batch_size_log += f"{dashed_line}"
        opt.batch_size = Total_batch_size

        print(Total_batch_size_log)
        log.write(Total_batch_size_log + "\n")
        log.close()

    def get_batch(self):
        balanced_batch_images = []
        balanced_batch_texts = []

        for i, data_loader_iter in enumerate(self.data_loader_list):
            try:
                image, text = data_loader_iter.next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, text = self.dataloader_iter_list[i].next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except ValueError:
                pass

        balanced_batch_images = tf.concat(balanced_batch_images, axis=0)

        return balanced_batch_images, balanced_batch_texts
