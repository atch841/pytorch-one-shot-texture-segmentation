from torch.utils.data import Dataset
from data_loader import Data_loader
import numpy as np
'''
Dynamic generate data when training. 
Load generated data when validating.
'''

class Texture_dataset_train(Dataset):
	def __init__(self, data_size, textures_path, max_region=10):
		self.data_size = data_size
		self.data = Data_loader(textures_path, 1, max_region)

	def __len__(self):
		return self.data_size

	def __getitem__(self, idx):
		x, y, x_ref = self.data.get_batch_data()
		x = x[0]
		y = y[0]
		x_ref = x_ref[0]
		x = np.swapaxes(x, 1, 2)
		x = np.swapaxes(x, 0, 1)
		y = np.swapaxes(y, 1, 2)
		y = np.swapaxes(y, 0, 1)
		x_ref = np.swapaxes(x_ref, 1, 2)
		x_ref = np.swapaxes(x_ref, 0, 1)
		x, y, x_ref = x.astype('float32'), y.astype('float32'), x_ref.astype('float32')
		return x, y, x_ref


class Texture_dataset_val(Dataset):
	def __init__(self, data_size, textures_path, max_region=10):
		self.data_size = data_size
		self.data = Data_loader(textures_path, 1, max_region)
		self.preload = []
		for i in range(self.data_size):
			x, y, x_ref = self.data.get_batch_data()
			x = x[0]
			y = y[0]
			x_ref = x_ref[0]
			x = np.swapaxes(x, 1, 2)
			x = np.swapaxes(x, 0, 1)
			y = np.swapaxes(y, 1, 2)
			y = np.swapaxes(y, 0, 1)
			x_ref = np.swapaxes(x_ref, 1, 2)
			x_ref = np.swapaxes(x_ref, 0, 1)
			x, y, x_ref = x.astype('float32'), y.astype('float32'), x_ref.astype('float32')
			self.preload.append((x, y, x_ref))

	def __len__(self):
		return self.data_size

	def __getitem__(self, idx):
		return self.preload[idx]