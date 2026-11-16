import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = os.path.join(os.getcwd(),"data")

SVHN_MEAN, SVHN_STD = (0.4377,0.4438,0.4728), (0.1980,0.2010,0.1970)

svhn_tf = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(SVHN_MEAN,SVHN_STD)
])

svhn_train = datasets.SVHN(ROOT,split="train",download=True,transform=svhn_tf)
svhn_test = datasets.SVHN(ROOT,split="test",download=True,transform=svhn_tf)

train_dataloader = DataLoader(svhn_train,batch_size=64,shuffle=True)
test_dataloader = DataLoader(svhn_test,batch_size=64,shuffle=False)



def block(I,O,dropout=0.0):
	return nn.Sequential(
		nn.Conv2d(I,O,3,1,1),
		nn.BatchNorm2d(O),
		nn.Dropout2d(dropout),
		nn.ReLU()
	)

class SBHN_NET(nn.Module):
	def __init__(self,output_dim):
		super().__init__()
		self.conv_net = nn.Sequential(
			block(3,16,0.1),block(16,16,0.1),nn.MaxPool2d(2),
			block(16,32,0.1),block(32,32,0.1),nn.MaxPool2d(2),
			block(32,64,0.05),block(64,64,0.05),nn.MaxPool2d(2),
			nn.Flatten(),
			nn.Linear(1024,256),
			nn.ReLU(),
			nn.Linear(256,output_dim)
			)

	def forward(self,x):
		return self.conv_net(x)


sbhn_net = SBHN_NET(10).to(DEVICE)
lr=1e-3
epochs = 20
total_steps = len(train_dataloader)*epochs
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(sbhn_net.parameters(),lr=lr)
epoch_scheduler = lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)
batch_scheduler = lr_scheduler.OneCycleLR(optimizer,max_lr=3e-3,total_steps=total_steps)

def train_sbhn(model,optimizer,loss_fn,train_dataloader,batch_scheduler,epoch_scheduler):
	right, total, total_loss = 0,0,0.0
	model.train()
	for images, labels in train_dataloader:
		images, labels = images.to(DEVICE), labels.to(DEVICE)
		optimizer.zero_grad()
		y_hat =model(images)
		loss = loss_fn(y_hat,labels)
		loss.backward()
		optimizer.step()
		batch_scheduler.step()

		right += (labels==y_hat.argmax(dim=1)).sum().item()
		total += len(labels)
		total_loss += loss.item()
	acc = right / total
	total_loss /= len(train_dataloader)
	print(f"Train-Loss: {total_loss},Train-Accuracy:{acc}")
	epoch_scheduler.step()
	return total_loss, acc


def test_sbhn(model, loss_fn, test_dataloader):
	right,total,total_loss= 0,0,0.0
	model.eval()
	with torch.no_grad():
		for images, labels in test_dataloader:
			images, labels = images.to(DEVICE), labels.to(DEVICE)
			y_hat = model(images)
			right += (y_hat.argmax(dim=1)==labels).sum().item()
			total_loss +=loss_fn(y_hat,labels).item()
			total +=len(labels)
	acc = right/total
	total_loss/=len(test_dataloader)
	print(f"Test-Loss: {total_loss}, Test-Accuracy: {acc}")



for epoch in range(epochs):
	print(f"Epoch: {epoch}")
	train_sbhn(sbhn_net,optimizer,loss_fn, train_dataloader,batch_scheduler,epoch_scheduler)
	test_sbhn(sbhn_net,loss_fn,test_dataloader)
