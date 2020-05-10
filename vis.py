# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import mxnet as mx
from mxnet import gluon

import argparse
import os
import numpy as np
import cv2

from gradcam_utils import visualize_gradcam
from smoothgrad_utils import visualize_smoothgrad

# Receive image path from command line
parser = argparse.ArgumentParser(description='visualization demo')
parser.add_argument('--image-path', metavar='image_path', type=str, help='path to the image file')
parser.add_argument('--vis_method', type=str, help='path to the image file')
args = parser.parse_args()

# Define the network you want to visualize first.
network = gluon.model_zoo.vision.resnet18_v1(pretrained=True, ctx=mx.cpu())
# Define the image size
image_sz = (224, 224)

def run_inference(net, data):
    """Run the input image through the network and return the predicted category as integer"""
    out = net(data)
    return out.argmax(axis=1).asnumpy()[0]

def get_input(img_path):
    """Preprocess the image before running it through the network"""
    with open(img_path, 'rb') as fp:
        img_bytes = fp.read()
    data = mx.img.imdecode(img_bytes)
    data = mx.image.imresize(data, image_sz[0], image_sz[1])
    data = data.astype(np.float32)
    data = data/255
    # These mean values were obtained from
    # https://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html
    data = mx.image.color_normalize(data,
                                    mean=mx.nd.array([0.485, 0.456, 0.406]),
                                    std=mx.nd.array([0.229, 0.224, 0.225]))
    data = mx.nd.transpose(data, (2,0,1)) # Channel first
    input_x = data.expand_dims(axis=0)
    return input_x

def read_image_cv(path):
    return cv2.resize(cv2.imread(path), image_sz)

if __name__=="__main__":
    if args.vis_method == 'gradcam':
        # Define the layer you wat to visualize
        last_conv_layer_name = 'resnetv10_pool1'
        input_x = get_input(args.image_path)
        origin_img = read_image_cv(args.image_path)
        viz_img = visualize_gradcam(network, input_x, origin_img, last_conv_layer_name)
        print("Predicted category {}".format(run_inference(network, input_x)))
        img_name = os.path.split(args.image_path)[1].split('.')[0]
        out_file_name = img_name+"_gradcam"+'.jpg'
        cv2.imwrite(os.path.join(os.path.dirname(args.image_path), out_file_name), viz_img)
        print(out_file_name + "...vis done")

    if args.vis_method == "smoothgrad":
        input_x = get_input(args.image_path)
        origin_img = read_image_cv(args.image_path)
        viz_img = visualize_smoothgrad(network, input_x, origin_img)
        print("Predicted category {}".format(run_inference(network, input_x)))
        img_name = os.path.split(args.image_path)[1].split('.')[0]
        out_file_name = img_name+"_smoothgrd"+'.jpg'
        cv2.imwrite(os.path.join(os.path.dirname(args.image_path), out_file_name), viz_img)
        print(out_file_name + "...vis done")






