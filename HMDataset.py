import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import h5py

class HMDataset(Dataset):
  def __init__(self, jsonl_file, root_dir, transform = None):
    self.annotations = pd.read_json(jsonl_file, lines=True)
    self.root_dir = root_dir
    self.transform = transform

  def __len__(self):
    return self.annotations.shape[0]

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    id = self.annotations["id"][idx]

    # img_name = os.path.join(self.root_dir, self.annotations['img'][idx])
    # image = io.imread(img_name)

    img_path = self.root_dir + self.annotations['img'][idx]
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    label = self.annotations["label"][idx]

    text = self.annotations["text"][idx]

    sample = {"id": id, "image":image, "label":label, "text":text}

    if self.transform:
      sample["image"] = self.transform(sample["image"])

    return sample


# Alternate Dataset using the HDF5 File (Slightly Faster)
class HMDataset_H5(Dataset):
  def __init__(self, h5_file, transform = None):
    self.h5_file = h5py.File(h5_file, 'r')
    self.transform = transform

  def __len__(self):
    return len(self.h5_file["label"])

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    id = self.h5_file["id"][idx]

    image = self.h5_file["image"][idx]

    text = str(self.h5_file["text"][idx])[2:-1]

    label = self.h5_file["label"][idx]

    sample = {"id": id, "image":image, "label":label, "text":text}

    if self.transform:
      sample["image"] = self.transform(sample["image"])

    return sample