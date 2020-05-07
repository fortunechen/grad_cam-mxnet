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

from __future__ import print_function

import mxnet as mx
import mxnet.ndarray as nd

from mxnet import gluon
from mxnet import autograd
from mxnet.gluon import nn

import numpy as np
import cv2

hooks = []

target_feature = 0

def get_conv_out_and_grad(net, image, class_id=None, conv_layer_name=None):
    """Get the output and gradients of output of a convolutional layer.

    Parameters:
    ----------
    net: Block
        Network to use for visualization.
    image: NDArray
        Preprocessed image to use for visualization.
    class_id: int
        Category ID this image belongs to. If not provided,
        network's prediction will be used.
    conv_layer_name: str
        Name of the convolutional layer whose output and output's gradients need to be acptured."""
    # def attach_target_layer_gradient(block):
    #     def hook_func(block, _, outputs):
    #         print(block.prefix[:-1], block.name, conv_layer_name)
    #         if block.prefix[:-1] == conv_layer_name:
    #             outputs.attach_grad()
    #             global target_feature
    #             target_feature = outputs
    #             print(outputs.shape)
    #             print(target_feature.shape)
    #     hooks.append(block.register_forward_hook(hook_func))

    def attach_target_layer_gradient(block):
        def hook_func(block, inputs):
            print(block.name)
            if block.name == conv_layer_name:
                inputs[0].attach_grad()
                global target_feature
                target_feature = inputs[0]
                print(inputs[0].shape)
                print(target_feature.shape)
        hooks.append(block.register_forward_pre_hook(hook_func))

    net.apply(attach_target_layer_gradient)
    with autograd.record(train_mode=False):
        out = net(image)
    if class_id is None:
        model_output = out.asnumpy()
        class_id = np.argmax(model_output)
    # Create a one-hot target with class_id and backprop with the created target
    one_hot_target = mx.nd.one_hot(mx.nd.array([class_id]), 1000)
    out.backward(one_hot_target, train_mode=False)
    return target_feature[0].asnumpy(), target_feature.grad[0].asnumpy()


def get_cam(conv_out, conv_out_grad):
    """Compute CAM. Refer section 3 of https://arxiv.org/abs/1610.02391 for details"""
    weights = np.mean(conv_out_grad, axis=(1, 2))
    cam = np.zeros(conv_out.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_out[i, :, :]
    cam = np.maximum(cam, 0)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam)) 
    cam = np.uint8(cam * 255)
    return cam


def get_img_heatmap(orig_img, activation_map):
    """Draw a heatmap on top of the original image using intensities from activation_map"""
    heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)
    img_heatmap = np.float32(heatmap) + np.float32(orig_img)
    img_heatmap = img_heatmap / np.max(img_heatmap)
    img_heatmap *= 255
    return heatmap, img_heatmap.astype(np.uint8)

def get_heatmap(net, preprocessed_img, orig_img, conv_layer_name):
    # Returns grad-cam heatmap, guided grad-cam, guided grad-cam saliency
    conv_out, conv_out_grad = get_conv_out_and_grad(net, preprocessed_img, conv_layer_name=conv_layer_name)
    cam = get_cam(conv_out,conv_out_grad)
    cam = cv2.resize(cam, preprocessed_img.shape[2:])
    heatmap, img_heatmap = get_img_heatmap(orig_img, cam)
    return heatmap, img_heatmap

