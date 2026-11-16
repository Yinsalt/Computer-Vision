import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms



BATCH_SIZE = 128
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = os.path.join(os.getcwd(),"data")

CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822,0.4465), (0.2470,0.2435,0.2616)

cifar_base_train = transforms.Compose([
	transforms.ToTensor(),
	transforms.RandomCrop(32,padding=4),
	transforms.RandomHorizontalFlip(0.3),
	transforms.Normalize(CIFAR_MEAN,CIFAR_STD)
	])

cifar_base_test = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(CIFAR_MEAN,CIFAR_STD)
	])


cifar_train = datasets.CIFAR10(ROOT,train=True,download=True,transform=cifar_base_train)
cifar_test = datasets.CIFAR10(ROOT,train=False,download=True,transform=cifar_base_test)

CIFAR_CLASSES = cifar_train.classes

cifar_dataloader_train = DataLoader(cifar_train,batch_size=BATCH_SIZE, shuffle=True)
cifar_dataloader_test = DataLoader(cifar_test,batch_size=BATCH_SIZE, shuffle=False)



def block(c_in,c_out):
	return nn.Sequential(
		nn.Conv2d(c_in,c_out,3,1,1,bias=False),
		nn.BatchNorm2d(c_out),
		nn.ReLU()
	)

class CIFAR_CNN(nn.Module):
	def __init__(self,output_classes=10):
		super().__init__()
		self.features = nn.Sequential(
			block(3,64), block(64,64), nn.MaxPool2d(2),
			block(64,128), block(128,128), nn.MaxPool2d(2),
			block(128,256), block(256,256), nn.MaxPool2d(2),
			)

		self.head = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Flatten(),
			nn.Dropout(0.3),
			nn.Linear(256,output_classes)
		)

	def forward(self,x):
		return self.head(self.features(x))



def cifar_epoch_trainer(model,optimizer,loss_fn,train_dataloader):
	model.train()
	total,right,total_loss = 0,0,0.0
	for images, labels in train_dataloader:
		images,labels = images.to(DEVICE),labels.to(DEVICE)
		optimizer.zero_grad()
		y_hat = model(images)
		loss = loss_fn(y_hat,labels)
		loss.backward()
		optimizer.step()
		y = y_hat.argmax(dim=1)
		total += len(labels)
		right += (y==labels).sum().item()
		total_loss += loss.item()
	acc = right/total
	total_loss /= len(train_dataloader)
	print(f"Total Training Loss: {total_loss:.2f}, Accuracy:{acc:.2f}")


def cifar_epoch_evaluator(model,loss_fn,test_dataloader):
	model.eval()
	total, right, total_loss = 0,0,0.0
	with torch.no_grad():
		for images, labels in test_dataloader:
			images,labels = images.to(DEVICE),labels.to(DEVICE)
			y_hat = model(images)
			loss = loss_fn(y_hat,labels)
			y = y_hat.argmax(dim=1)
			total += len(labels)
			right += (y==labels).sum().item()
			total_loss += loss.item()
	acc = right/total
	total_loss /= len(test_dataloader)
	print(f"Total Test Loss: {total_loss:.2f},Accuracy: {acc:.2f}")


cifar_net = CIFAR_CNN().to(DEVICE)
epochs = 30
optimizer = optim.AdamW(cifar_net.parameters(), lr=LR,weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = epochs)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(epochs):
	lr = optimizer.param_groups[0]["lr"]
	print(f"Epoch {epoch+1}: lr = {lr:.6f}")
	cifar_epoch_trainer(cifar_net, optimizer, loss_fn,cifar_dataloader_train)
	cifar_epoch_evaluator(cifar_net,loss_fn,cifar_dataloader_test)
	scheduler.step()
