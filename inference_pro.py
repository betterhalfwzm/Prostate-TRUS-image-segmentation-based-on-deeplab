# -*- coding: UTF-8 -*-  
"""Run inference a DeepLab v3 model using tf.estimator API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import os
import sys
import cv2
import tensorflow as tf

import deeplab_model
from utils import preprocessing
from utils import dataset_util

from PIL import Image

from scipy import misc  
import matplotlib.pyplot as plt
from skimage import measure,draw
from tensorflow.python import debug as tf_debug

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='E:/tensorflow-deeplab-v3-plus/paper/ori/',
                    help='The directory containing the image data.')

parser.add_argument('--output_dir', type=str, default='E:/tensorflow-deeplab-v3-plus/model/30225/deeplab_model_superlow2',
                    help='Path to the directory to generate the inference results')

parser.add_argument('--infer_data_list', type=str, default='E:/tensorflow-deeplab-v3-plus/paper/test.txt',
                    help='Path to the file listing the inferring images.')

parser.add_argument('--model_dir', type=str, default='E:/tensorflow-deeplab-v3-plus/model/30225/deeplab_model_superlow2',
                    help="Base directory for the model. "
                         "Make sure 'model_checkpoint_path' given in 'checkpoint' file matches "
                         "with checkpoint name.")

parser.add_argument('--base_architecture', type=str, default='resnet_v2_101',
                    choices=['resnet_v2_50', 'resnet_v2_101'],
                    help='The architecture of base Resnet building block.')

parser.add_argument('--output_stride', type=int, default=16,
                    choices=[8, 16],
                    help='Output stride for DeepLab v3. Currently 8 or 16 is supported.')

parser.add_argument('--debug', action='store_true',
                    help='Whether to use debugger to track down bad values during training.')

_NUM_CLASSES = 2


def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  pred_hooks = None
  if FLAGS.debug:
    debug_hook = tf_debug.LocalCLIDebugHook()
    pred_hooks = [debug_hook]

  model = tf.estimator.Estimator(
      model_fn=deeplab_model.deeplabv3_plus_model_fn,
      model_dir=FLAGS.model_dir,
      params={
          'output_stride': FLAGS.output_stride,
          'batch_size': 1,  # Batch size must be 1 because the images' size may differ
          'base_architecture': FLAGS.base_architecture,
          'pre_trained_model': None,
          'batch_norm_decay': None,
          'num_classes': _NUM_CLASSES,
      })

  examples = dataset_util.read_examples_list(FLAGS.infer_data_list)
  image_files = [os.path.join(FLAGS.data_dir, filename) for filename in examples]
  #for i in range(20):
      #img = plt.imread(image_files[i])

  #for i in range(20):
      # img = plt.imread(image_files[i])
       #plt.imshow(img)
       #plt.show()
  predictions = model.predict(
        input_fn=lambda: preprocessing.eval_input_fn(image_files),
        hooks=pred_hooks)

  output_dir = FLAGS.output_dir
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  for pred_dict, image_path in zip(predictions, image_files):

    image_basename = os.path.splitext(os.path.basename(image_path))[0]

    output_filename = image_basename + '_mask.png'
    path_to_output = os.path.join(output_dir, output_filename)


    print("generating:", path_to_output)
    mask = pred_dict['decoded_labels']

    mask = Image.fromarray(mask)
    #plt.axis('off')
   #plt.savefig(path_to_output, bbox_inches='tight')

    #image = cv2.imread("D:\\picture\\%d.jpg" % (i))  
    #i = int(input('Enter image number: '))
    #for i in range(20):

    misc.imsave(path_to_output,mask)
    img = plt.imread(image_path)

    #plt.figure(figsize=(14,10))
    plt.axis('off')
    #去掉四周空白部分
    fig = plt.gcf()
    fig.set_size_inches(5.48,4.56) #dpi = 300, output = 700*700 pixels
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)

    plt.imshow(img, 'gray', interpolation='none')

    #plt.axis('off')

    plt.imshow(mask, 'jet', interpolation='none', alpha=0.5)


    
    plt.savefig(path_to_output)
    #misc.imsave(path_to_output,mask - img)
    plt.pause(1.001)
    plt.close()
    plt.show()
    plt.pause(1.1)

'''
    plt.subplot(121)
    plt.title('CAR_image')
    plt.imshow(img)
    plt.subplot(122)
    plt.title('Result')
    img1 = cv2.imread(path_to_output)
    plt.imshow(img1)
    #plt.imshow(mask, 'jet', interpolation='none', alpha=0.7)
    plt.pause(10.1)  
'''

    
    

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
