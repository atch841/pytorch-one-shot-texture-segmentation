from torch.utils.data import Dataset
from data_loader import Data_loader

class Texture_dataset(Dataset):
	def __init__(self, data_size, textures_path, batch_size, max_region=10):
		self.data_size = data_size
		self.data = Data_loader(textures_path, batch_size, max_region)

	def __len__(self):
		return self.data_size

	def __getitem__(self, idx):
		return self.data.get_batch_data()