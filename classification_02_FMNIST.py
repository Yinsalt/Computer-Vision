import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms



BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



fmnist_tf = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.2860,),(0.3530,))
	])




root = os.path.join(os.getcwd(),"data")
fmnist_train = datasets.FashionMNIST(root=root, train=True,download=True,transform=fmnist_tf)
fmnist_test = datasets.FashionMNIST(root=root, train=False,download=True,transform=fmnist_tf)

FMNIST_CLASSES = fmnist_train.classes

fmnist_train_dataloader = DataLoader(fmnist_train,batch_size=BATCH_SIZE, shuffle=True)
fmnist_test_dataloader = DataLoader(fmnist_test,batch_size=BATCH_SIZE, shuffle=False)


class FMNIST_CNN(nn.Module):
	def __init__(self, OUTPUT_SIZE=10):
		super().__init__()
		self.conv_net = nn.Sequential(
			nn.Conv2d(1,32,3,1,1),
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Dropout(0.1),
			nn.ReLU(),
			nn.Conv2d(32,64,3,1,1),
			nn.MaxPool2d(2)
		)
		self.MLP = nn.Sequential(
			nn.Flatten(),
			nn.Linear(64*7*7,128),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(128,OUTPUT_SIZE)
		)

	def forward(self,x):
		return self.MLP(self.conv_net(x))


def train_network(model,optimizer,loss_fn,train_dataloader):
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


def eval_network(model,loss_fn,test_dataloader):
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


CNN = FMNIST_CNN(10).to(DEVICE)

LR = 1e-4
EPOCHS = 10
optimizer_CNN = optim.AdamW(CNN.parameters(),lr=LR)
scheduler_CNN = optim.lr_scheduler.CosineAnnealingLR(optimizer_CNN,T_max=EPOCHS)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
	lr = optimizer_CNN.param_groups[0]["lr"]
	print(f"Epoch {epoch+1}: lr = {lr:.6f}")
	train_network(CNN,optimizer_CNN,loss_fn,fmnist_train_dataloader)
	eval_network(CNN,loss_fn,fmnist_test_dataloader)
	scheduler_CNN.step()
