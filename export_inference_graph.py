r"""Saves out a GraphDef containing the architecture of the model."""

from __future__ import print_function

import argparse
import os
import sys
import time

import tensorflow as tf
from tensorflow.python.platform import gfile

from models.BiSeNet import *
#from tensorlayer_nets import *

# from tools import decode_labels, prepare_label, inv_preprocess
# from image_reader import ImageReader
# from inference import preprocess, check_input

# from hyperparams import *
from builders import model_builder

#! ATTENTION
IMG_MEAN = np.array((126.63854281333334, 123.24824418666667, 113.14331923666667), dtype = np.float32)
INPUT_SIZE = '512,512'

label_names = ['unlabeled', 'ground', 'road', 'sidewalk', 'rail track', 'building',
               'wall', 'fence', 'bridge', 'tunnel', 'pole', 'traffic light',
               'traffic sign', 'vegetation', 'terrain', 'person', 'car', 'motorcycle']
label_colours = [(0,  0, 0), (128, 64,128), (244, 35,232), (250,170,160), (70, 70, 70), (102, 102,156),
                 (190,153,153), (180,165,180), (150,120, 90), (153,153,153), (250,170, 30),
                 (220,220,  0), (107,142, 35), (152,251,152), (80, 150, 250), (255,  0,  0),
                 (0, 60,100), (0,  0,  255)]
#! ATTENTION

tf.app.flags.DEFINE_boolean(
    'is_training', False,
    'Whether to save out a training-focused version of the model.')

#tf.app.flags.DEFINE_integer(
#    'image_size', None,
#    'The image size to use, otherwise use the model default_image_size.')

tf.app.flags.DEFINE_integer(
    'batch_size', None,
    'Batch size for the exported model. Defaulted to "None" so batch size can '
    'be specified at model runtime.')

tf.app.flags.DEFINE_string(
    'output_file', '', 'Where to save the resulting file to.')

FLAGS = tf.app.flags.FLAGS


def main(_):
    if not FLAGS.output_file:
        raise ValueError('You must supply the path to save to with --output_file')

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default() as graph:
        shape = INPUT_SIZE.split(',')
        shape = (int(shape[0]), int(shape[1]), 3)

        x = tf.placeholder(name='input', dtype = tf.float32, shape = (None, shape[0], shape[1], 3))

        img_tf = tf.cast(x, dtype = tf.float32)
        # Extract mean.
        img_tf -= IMG_MEAN

        print(img_tf)
        # Create network.

        net_input = tf.placeholder(name='input', dtype=tf.float32, shape=[None, None, None, 3])
        net_output = tf.placeholder(tf.float32, shape=[None, None, None, 32])

        net, _ = model_builder.build_model(model_name="BiSeNet", frontend="ResNet50", net_input=net_input,
                                                     num_classes=32, crop_width=512,
                                                     crop_height=512, is_training=False)
        # net =
        #net = psp_net({'inputs': img_tf}, is_training = False, num_classes = NUM_CLASSES)
        # net = unext(img_tf, is_train = False, n_out = NUM_CLASSES)
        #net = ICNet_BN({'data': img_tf}, is_training = False, num_classes = NUM_CLASSES)

        raw_output = graph.get_tensor_by_name('logits/BiasAdd:0')
        # raw_output = net.outputs
        #raw_output = net.layers['conv6']
        # output = tf.image.resize_bilinear(raw_output, tf.shape(img_tf)[1:3,], name = 'raw_output')
        #output = tf.nn.softmax(raw_output)
        output = tf.argmax(raw_output, dimension = 3)
        pred = tf.expand_dims(output, dim = 3, name = 'indices')

        # Adding additional params to graph. It is necessary also to point them as outputs in graph freeze conversation, otherwise they will be cuted
        tf.constant(label_colours, name = 'label_colours')
        tf.constant(label_names, name = 'label_names')

        shape = INPUT_SIZE.split(',')
        shape = (int(shape[0]), int(shape[1]), 3)
        tf.constant(shape, name = 'input_size')
        tf.constant(["indices"], name = "output_name")

        graph_def = graph.as_graph_def()
        with gfile.GFile(FLAGS.output_file, 'wb') as f:
            f.write(graph_def.SerializeToString())
            print('Successfull written to', FLAGS.output_file)


if __name__ == '__main__':
    tf.app.run()