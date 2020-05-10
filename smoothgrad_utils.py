from mxnet import ndarray as nd
from mxnet import autograd
import numpy as np
import mxnet as mx
import cv2

def get_img_grad(net, input_x):
	input_x.attach_grad()
	with autograd.record(train_mode=False):
		out = net(input_x)
	model_output = out.asnumpy()
	class_id = np.argmax(model_output)
	one_hot_target = mx.nd.one_hot(mx.nd.array([class_id]), 1000)
	out.backward(one_hot_target, train_mode=False)
	return input_x.grad[0].asnumpy()

def to_gray_image(avg_gradients, percentile=99):
	img_2d = np.sum(avg_gradients, axis=0)
	span = abs(np.percentile(img_2d, percentile))
	img_2d[img_2d > span] = span
	img_2d -= img_2d.min()
	img_2d /= img_2d.max()
	return (img_2d * 255).astype(np.uint8)

def visualize_smoothgrad(net, input_x, origin_img, stdev_spread=0.15, n_samples=40):
	total_gradients = np.zeros(input_x[0].shape)
	stdev = stdev_spread * (np.max(input_x.asnumpy()) - np.min(input_x.asnumpy()))
	for i in range(n_samples):
		noise = np.random.normal(0, stdev, input_x.shape)
		input_x_with_noise = input_x + nd.array(noise)
		grad = get_img_grad(net, input_x_with_noise)
		total_gradients += (grad * grad)
	avg_gradients = total_gradients / n_samples
	gray_gradient = to_gray_image(avg_gradients)
	return np.hstack((origin_img, cv2.cvtColor(gray_gradient, cv2.COLOR_GRAY2BGR)))
