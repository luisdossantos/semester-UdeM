#!/usr/bin/python3

import Robot
import torchvision.transforms as transforms
from dataset import ImageClassificationDataset
from utils import preprocess
import torch.nn.functional as F
import time
import uuid
import random
from jetcam.csi_camera import CSICamera
#from jetbot import bgr8_to_jpeg


import enum
import cv2
def bgr8_to_jpeg(value, quality=75):
    return bytes(cv2.imencode('.jpg', value)[1])


#-_________________________________________________


TASK = 'Photos'

CATEGORIES = ['non_obstacle', 'obstacle']

DATASETS = ['A']

TRANSFORMS = transforms.Compose([
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

datasets = {}
for name in DATASETS:
    datasets[name] = ImageClassificationDataset('datasets/'+TASK + '_' + name, CATEGORIES, TRANSFORMS)

print("{} task with {} categories defined".format(TASK, CATEGORIES))


print("init torch...")
import torch
import torchvision

print("init cuda...")
device = torch.device('cuda')

# premier dataset seulement
dataset = datasets[DATASETS[0]]

dataset_0=[]
dataset_1=[]

for im in dataset :
	if (im[1] == 0):
		dataset_0.append(im)
	else :
		dataset_1.append(im)

taille_train_0 = int(len(dataset_0)*0.80)
taille_train_1 = int(len(dataset_1)*0.80)
taille_valid_0 = len(dataset_0) - int(len(dataset_0)*0.80)
taille_valid_1 = len(dataset_1) - int(len(dataset_1)*0.80)

dataset_train = []
dataset_valid = []

for i in range (taille_train_0) :
	dataset_train.append(dataset_0[i])
for i in range (taille_train_1) :
	dataset_train.append(dataset_1[i])
for i in range (taille_valid_0) :
	dataset_valid.append(dataset_0[i])
for i in range (taille_valid_1) :
	dataset_valid.append(dataset_1[i])

#print(len(dataset_train))
#print(len(dataset_valid))


model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(512, len(dataset.categories))

    
model = model.to(device)

#-_________________________________________________


model_fname="models/obstacle.pth"
print("loading model..." , model_fname)
model.load_state_dict(torch.load(model_fname))


LEFT_TRIM   = 0
RIGHT_TRIM  = 0
robot = Robot.Robot(left_trim=LEFT_TRIM, right_trim=RIGHT_TRIM)



#
# 3280x2464  fps=21, 28
# 1920x1080  fps=30
# 1280x720   fps=60, 120
# 
camera = CSICamera(width=205, height=154, capture_width=3280, capture_height=2464, capture_fps=10)
#camera = CSICamera(width=3280, height=2464, capture_width=3280, capture_height=2464, capture_fps=10)

numA=0
numB=0


print("warming up...")
for x in range(0, 10):
    camera.read()

vLeft=0
vRight=0

print("GO!")
while True:
    im=camera.read()

    preprocessed = preprocess(im)
    output = model(preprocessed)
    output = F.softmax(output, dim=1).detach().cpu().numpy().flatten()
    category_index = output.argmax()
    prediction=dataset.categories[category_index]
    yo=list(output)
    print("predict %15s  %10.3f %10.3f" % (prediction,yo[0],yo[1]))

    threshold=0.70
    if (prediction == 'obstacle' and yo[1]>threshold) :
        vLeft=-40
        vRight=40
        robot.leftSpeed(vLeft)
        robot.rightSpeed(vRight)

    else :
            vLeft=-40
            vRight=-40
            robot.leftSpeed(vLeft)
            robot.rightSpeed(vRight)

