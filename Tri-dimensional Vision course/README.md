
# Autonomous robot navigation

#### Project description :

The goal of this project is to program a robot to randomly move in a room, and detect and avoid obstacles entirely based on a camera stream.  
Therefore, this study was tackled as a binary image classification problem (of course coupled with robot navigation). We built a dataset composed of 298 photos (157 "obstacles" and 141 "non-obstacles").  
We trained a ResNet-18 model using PyTorch (ResNet-18 is a convolutional neural network developed by Microsoft Research - [scientific paper](https://www.google.com)).  
The mobile robot was a JetBot from Nvidia, equipped with a Jetson Nano board, which are specifically developed for AI.
<br/>
#### Interpretation

With such a small dataset, the results were very satisfying, the robot moving multiple minutes before running into a non-recognized obstacle.  
We also ran some additional tests by computing the RGB and grayscale level of the images for each of the two classes, and we saw that, in our test environment, it was possible to obtain good results only by looking at these features, therefore developing a much simpler model than having to train a CNN.

<br/>
<br/>

## Reference
© Sébastien Roy  
University of Montreal, Canada  
IFT 6145: Tri-dimensional Vision

