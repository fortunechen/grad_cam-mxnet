## Grad CAM implementation by MXNet

### Usage
```python gradcam.py --image-path imgs/bird.jpg```

Current official implementation of grad-cam only supports the visualization of conv2d and need recode the network architecture. Ours supports any layer in the network and can use official network API. 

you can change the pretrained nwtwork and layers to be visulized in the code ```gradcam.py```

### Other implementation
- [Pytorch](https://github.com/jacobgil/pytorch-grad-cam)
- [MXNet official](https://github.com/apache/incubator-mxnet/blob/8ac7fb930fdfa6ef3ac61be7569a17eb95f1ad4c/docs/tutorial_utils/vision/cnn_visualization/gradcam.py)