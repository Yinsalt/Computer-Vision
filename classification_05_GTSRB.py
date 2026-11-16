import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from collections import Counter


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = os.path.join(os.getcwd(),"data")


GTSRB_MEAN, GTSRB_STD = (0.3403, 0.3121, 0.3214), (0.2724, 0.2608,0.2669)

gtsrb_tf = transforms.Compose([
	transforms.Resize((48,48)),
	transforms.ToTensor(),
	transforms.RandomCrop(48,padding=4),
	transforms.Normalize(GTSRB_MEAN,GTSRB_STD)
	])

gtsrb_tf2 = transforms.Compose([
	transforms.Resize((48,48)),
	transforms.ToTensor(),
	transforms.Normalize(GTSRB_MEAN,GTSRB_STD)
	])


gtsrb_train = datasets.GTSRB(ROOT,split="train",download=True,transform=gtsrb_tf)
gtsrb_test = datasets.GTSRB(ROOT,split="test",download=True,transform=gtsrb_tf2)

dataloader_train = DataLoader(gtsrb_train,batch_size=64,shuffle=True)
dataloader_test = DataLoader(gtsrb_test,batch_size=64,shuffle=False)

gtsrb_train_labels = [label for _, label in gtsrb_train._samples]


LR = 1e-3
counter = Counter(label for label in gtsrb_train_labels)
mean = len(gtsrb_train)/len(set(gtsrb_train_labels))
rescale_dict = {}
for i in range(43):
	rescale_dict[i]=1/(counter[i]/mean)

class_w = torch.tensor([rescale_dict[i] for i in range(43)],dtype=torch.float32)
loss_fn = nn.CrossEntropyLoss(weight=class_w.to(DEVICE))
loss_eval=nn.CrossEntropyLoss()



def block(I,O,dropout=0.0):
	return nn.Sequential(
		nn.Conv2d(I,O,3,1,1),
		nn.BatchNorm2d(O),
		nn.Dropout2d(dropout),
		nn.ReLU()
	)

class GTSRB_NET(nn.Module):
	def __init__(self):
		super().__init__()
		self.net = nn.Sequential(
			block(3,16,0.1),block(16,16,0.1),nn.MaxPool2d(2),
			block(16,32,0.1),block(32,32,0.1),nn.MaxPool2d(2),
			block(32,128,0.1),block(128,128,0.1),nn.MaxPool2d(2),
			nn.Flatten(),nn.Linear(6*6*128,256),nn.ReLU(),nn.Linear(256,43)
		)
	def forward(self,x):
		return self.net(x)


gtsrb_net = GTSRB_NET().to(DEVICE)

epochs = 10
optimizer = optim.AdamW(gtsrb_net.parameters(), lr=LR)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)

def train_gtsrb(model, optimizer, loss_fn, train_dataloader):
	right, total, total_loss = 0,0,0.0
	model.train()
	for images, labels in train_dataloader:
		images,labels = images.to(DEVICE),labels.to(DEVICE)
		optimizer.zero_grad()
		y_hat = model(images)
		loss = loss_fn(y_hat,labels)
		loss.backward()
		optimizer.step()
		right += (y_hat.argmax(dim=1)==labels).sum().item()
		total += len(labels)
		total_loss += loss.item()
	acc = right/total
	total_loss /= len(train_dataloader)
	print(f"Total Training Loss: {total_loss},Accuracy: {acc}")

def test_gtsrb(model,loss_fn,test_dataloader):
	right,total,total_loss = 0,0,0.0
	model.eval()
	with torch.no_grad():
		for images, labels in test_dataloader:
			images, labels = images.to(DEVICE), labels.to(DEVICE)
			y_hat = model(images)
			total_loss += loss_fn(y_hat,labels).item()
			right +=(y_hat.argmax(dim=1)==labels).sum().item()
			total+=len(labels)
	acc = right/total
	total_loss /=len(test_dataloader)
	print(f"Total Test Loss: {total_loss},Accuracy: {acc}")


for _ in range(epochs):
	print(f"Epoch: {epoch}")
	train_gtsrb(gtsrb_net,optimizer,loss_fn,dataloader_train)
	test_gtsrb(gtsrb_net,loss_eval,dataloader_test)
	scheduler.step()
