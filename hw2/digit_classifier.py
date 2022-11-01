import os
import argparse

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

MEAN =[0.5, 0.5, 0.5]     #[1, 1, 1]
STD = [0.5, 0.5, 0.5]     #[1, 1, 1]

def load_checkpoint(checkpoint_path, model):
	state = torch.load(checkpoint_path, map_location = "cuda")
	model.load_state_dict(state['state_dict'])
	print('model loaded from %s' % checkpoint_path)

class DATA(Dataset):
	def __init__(self, path):
		self.img_dir = path

		self.data = []
		self.labels = []
		for filename in os.listdir(self.img_dir):
			self.data.append(os.path.join(self.img_dir, filename))
			self.labels.append(int(filename[0]))

		self.transform = transforms.Compose([
							transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
							transforms.Normalize(MEAN, STD)
						])

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		img_path = self.data[idx]
		img = Image.open(img_path).convert('RGB')

		return self.transform(img), self.labels[idx]

class Classifier(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 4 * 4, 128)
		self.fc2 = nn.Linear(128, 64)
		self.fc3 = nn.Linear(64, 10)

	def forward(self, x):
		x = self.pool(nn.functional.relu(self.conv1(x)))
		x = self.pool(nn.functional.relu(self.conv2(x)))
		x = torch.flatten(x, 1) # flatten all dimensions except batch
		x = nn.functional.relu(self.fc1(x))
		x = nn.functional.relu(self.fc2(x))
		x = self.fc3(x)
		return x

def classify_dir(prefix, model_path='./Classifier.pth'):
	net = Classifier()
	load_checkpoint(model_path, net)

	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	net = net.to(device)

	data_loader = torch.utils.data.DataLoader(
		DATA(prefix),
		batch_size=32, 
		num_workers=4,
		shuffle=False
	)

	correct = 0
	total = 0
	net.eval()
	with torch.no_grad():
		for idx, (imgs, labels) in enumerate(data_loader):
			imgs, labels = imgs.to(device), labels.to(device)
			output = net(imgs)
			_, pred = torch.max(output, 1)
			correct += (pred == labels).detach().sum().item()
			total += len(pred)
	print('acc = {} (correct/total = {}/{})'.format(float(correct)/total, correct, total))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--folder", type=str)
	parser.add_argument('--model_path', type=str, default='./Classifier.pth')
	args = parser.parse_args()

	classify_dir(args.folder, args.model_path)
