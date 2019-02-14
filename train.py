from model import Texture_model
from dataset import Texture_dataset_train, Texture_dataset_val
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sys import stdout
import time


class WCE_loss(nn.Module):
	def __init__(self):
		super(WCE_loss, self).__init__()

	def sum_ij(self, x):
		return torch.sum(torch.sum(x, dim=3), dim=2)

	def forward(self, pred, gt):
		N_fg = self.sum_ij(gt)
		N_bg = self.sum_ij(1 - gt)
		L_fg = -1 * self.sum_ij(torch.log(pred + 1e-16) * gt) / N_fg
		L_bg = -1 * self.sum_ij(torch.log(1 - pred + 1e-16) * (1 - gt)) / N_bg
		L = L_fg + L_bg
		return torch.mean(L)

def train():
	num_epoch = 20000
	cur_lr = initial_lr = 1e-5
	steps_to_decay_lr = 500
	num_max_model = 5
	saved_model = []
	best_loss = np.inf
	save_model_path = '/fast_data/one_shot_texture_models/'

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = Texture_model()
	model.to(device)

	train_dataset = Texture_dataset_train(200, 'train_texture.npy')
	val_dataset = Texture_dataset_val(240, 'val_texture.npy')
	train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
	val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=4)

	loss_func = WCE_loss()
	opt = torch.optim.Adam(model.parameters(), lr=initial_lr)
	for epoch in range(num_epoch):
		tic = time.time()
		training_loss = 0
		testing_loss = 0
		for i, batch in enumerate(train_dataloader):
			x, y, x_ref = batch
			x, y, x_ref = x.to(device), y.to(device), x_ref.to(device)

			opt.zero_grad()

			loss = loss_func(model(x, x_ref), y)
			loss.backward()
			opt.step()

			training_loss += loss.item()
			stdout.write('\r%d' % i)
			stdout.flush()
			# print(i)
		training_loss /= i
		# print(training_loss)

		with torch.no_grad():
			for i, batch in enumerate(val_dataloader):
				x, y, x_ref = batch
				x, y, x_ref = x.to(device), y.to(device), x_ref.to(device)

				loss = loss_func(model(x, x_ref), y)

				testing_loss += loss.item()
			testing_loss /= i
		toc = time.time()
		print('\r%5d/%5d training loss: %.5f, validation loss: %.5f (%d sec)' % (epoch + 1, num_epoch, training_loss, testing_loss, toc - tic))

		if testing_loss < best_loss:
			best_loss = testing_loss
			model_name = "model_%.5f.pt" % best_loss
			saved_model.append(model_name)
			torch.save(model.state_dict(), save_model_path + model_name)
			if len(saved_model) > 5:
				saved_model.pop(0)


		if (epoch + 1) % steps_to_decay_lr == 0:
			cur_lr /= 2
			for g in opt.param_groups:
				g['lr'] = cur_lr
			print("Reducing learning rate to", cur_lr)

if __name__ == '__main__':
	train()