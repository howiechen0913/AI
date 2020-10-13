# -*- coding: utf-8 -*-
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F 
from torchvision import datasets, transforms, models
import os
import time
import math
import pandas as pd
import  cv2 
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import itertools

classes_types=['Neutral','Happy','Sad','Surprise','Fear','Disgust','Anger','Contempt','Non-Face']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
size = (256,256)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('/home/howchen' +  '/' + "_confusion_matrix_925_cnnplusrnn.jpg", dpi=400)
    plt.savefig('cm_9_25_cnnplusrnn.jpg',dpi=400)

class AlexNet(nn.Module):

    def __init__(self, num_classes=9,lstm_hidden_size=256, num_lstm_layers=1, bidirectional=True):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
        )
        self.num_directions = 2 if bidirectional else 1
        self.num_lstm_layers = num_lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm1 = nn.LSTM(input_size=256 * 6 * 6,
                             hidden_size=lstm_hidden_size,
                             num_layers=num_lstm_layers,
                             batch_first=True,
                             dropout=0.5,
                             bidirectional=bidirectional)
        self.linear1 = nn.Sequential(nn.Linear(lstm_hidden_size * self.num_directions * num_lstm_layers, 64),nn.ReLU(inplace=True))
        self.output_layer = nn.Linear(64, 9)

    def init_hidden(self, x):
        batch_size = x.size(0)
        h = x.data.new(
                self.num_directions * self.num_lstm_layers, batch_size, self.lstm_hidden_size).zero_()
        c = x.data.new(
                self.num_directions * self.num_lstm_layers, batch_size, self.lstm_hidden_size).zero_()
        return Variable(h).cuda(), Variable(c).cuda()

    def forward(self, x):
        B = x.size(0)
        x = self.features(x)
        x = x.view(B , -1).transpose(0, 1).contiguous().view(256 * 6 * 6, B, 1)
        #x = self.classifier(x)
        x = x.permute(1,2,0)
        h, c = self.init_hidden(x)
        x = x.permute(2,0,1)
        x, (h, c) = self.lstm1(x, (h, c))
        h = h.transpose_(0, 1).contiguous().view(B, -1)
        x = self.linear1(h)
        x = self.output_layer(x)
        return x


data_transforms = {
	'AffectNet_Image_class_train2': transforms.Compose([
		transforms.Resize(256),
        transforms.FiveCrop(224),
		transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
	]),
	'AffectNet_Image_class_val2': transforms.Compose([
		transforms.Resize(224),
		transforms.ToTensor()
	])
}

batch_size = 512
data_dir = r'/home/shared_dataset/AffectNet_reorg'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['AffectNet_Image_class_train2', 'AffectNet_Image_class_val2']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['AffectNet_Image_class_train2', 'AffectNet_Image_class_val2']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['AffectNet_Image_class_train2', 'AffectNet_Image_class_val2']}
class_name = image_datasets['AffectNet_Image_class_train2'].classes
print(class_name)

train_batch_num = math.ceil(dataset_sizes['AffectNet_Image_class_train2']/batch_size)
val_batch_num = math.ceil(dataset_sizes['AffectNet_Image_class_val2']/batch_size)


train_batch_num = math.ceil(dataset_sizes['AffectNet_Image_class_train2']/batch_size)
val_batch_num = math.ceil(dataset_sizes['AffectNet_Image_class_val2']/batch_size)

use_gpu = torch.cuda.is_available()
print(use_gpu)
def test_model():
	model = models.alexnet(num_classes=11)
	print(model)
	input, target = next(iter(dataloaders['AffectNet_Image_class_train2']))
	bs, ncrops, c, h, w = input.size()
	print(bs, ncrops, c, h, w)
	input = input.view(-1, c, h, w)
	print(input.size())
	input = Variable(input)
	result = model(input)
	result_avg = result.view(bs, ncrops, -1).mean(1)
	print(result_avg)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
	since = time.time()

	best_model_wts = model.state_dict()
	best_acc = 0.0

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		for phase in ['AffectNet_Image_class_train2', 'AffectNet_Image_class_val2']:
			if phase == 'AffectNet_Image_class_train2':
				scheduler.step()
				model.train(True)
			else:
				model.train(False)

			running_loss = 0.0
			running_corrects = 0

			for i, data in enumerate(dataloaders[phase]):
				inputs, labels = data
				# np.reshape(inputs, (50, 5 ,224 , 224 , 3))


                                                   		
				if phase == 'AffectNet_Image_class_train2':
					bs, ncrops, c, h, w = inputs.size()
					inputs = inputs.view(-1, c, h, w)
					if use_gpu == True :
						inputs, labels = Variable(inputs.cuda(device)), Variable(labels.cuda(device))
                       
					else:
						inputs, labels = Variable(inputs), Variable(labels)
					optimizer.zero_grad()
					result = model(inputs)
					outputs = result.view(bs, ncrops, -1).mean(1)
					
				elif phase == 'AffectNet_Image_class_val2':
					if use_gpu == True:
						inputs, labels = Variable(inputs.cuda(device)), Variable(labels.cuda(device))
					else:
						inputs, labels = Variable(inputs), Variable(labels)
					outputs = model(inputs)

				_, preds = torch.max(outputs, 1)
				loss = criterion(outputs, labels)

				if phase == 'AffectNet_Image_class_train2':
					loss.backward()
					optimizer.step()

				running_loss += loss.data
				running_corrects += (preds == labels.data).sum().item()

			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_acc = running_corrects / dataset_sizes[phase]

			print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

			if phase == 'AffectNet_Image_class_val2' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = model.state_dict()

			print()

		time_elapsed = time.time() - since
		print('Training complete in {:0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
		print('Best val Acc: {:4f}'.format(best_acc))

		model.load_state_dict(best_model_wts)
		torch.save(model, 'best_model.pkl')
		torch.save(model.state_dict(), 'model_params.pkl')

if __name__ == '__main__':
	# test_model()
	h_n = None
	model = AlexNet()
	if use_gpu == True:
		model = model.cuda(device)
	tarage = torch.FloatTensor([0.050,0.0279,0.147,0.266,0.588,0.986,0.35,1,0.046]).float()
	criterion = nn.CrossEntropyLoss(weight=tarage).cuda(device)
	#optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
	train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=20)

nb_classes = 9

confusion_matrix = torch.zeros(nb_classes, nb_classes)
with torch.no_grad():
    for i, (inputs, classes) in enumerate(dataloaders['AffectNet_Image_class_val2']):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

print(confusion_matrix)
confusion_matrix2 = confusion_matrix.numpy()
plot_confusion_matrix(confusion_matrix2, classes=classes_types, normalize=True, title='Normalized confusion matrix')
