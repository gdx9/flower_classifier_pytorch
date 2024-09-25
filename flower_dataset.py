import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import random

class FlowersDataset(Dataset):
    def __init__(self, image_label_pair_list, image_size, is_train):
        self.data = image_label_pair_list

        size_before_crop = (image_size+30,image_size+30)
        size_random_crop = (image_size+10,image_size+10)

        if is_train:
            self.transforms = transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.RandomApply(
                    transforms=[
                        transforms.Resize(size_before_crop),
                        transforms.RandomCrop(size=size_random_crop)
                    ],
                    p=0.5),
                transforms.RandomRotation((-25,25)),
                transforms.Resize((image_size,image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Resize((image_size,image_size)),
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data[idx][0]).convert('RGB')
        image = self.transforms(image)
        y = torch.nn.functional.one_hot(torch.tensor(self.data[idx][1]),5).type(torch.float32)
        return image, y

def prepare_train_valid_pairs():
    data_dir  = 'flower_photos'

    class_folders = [os.path.join(data_dir, subdir)
                   for subdir in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, subdir))]

    train_pairs = []
    valid_pairs = []
    for class_label, class_path in enumerate(class_folders):
        class_files = [os.path.join(class_path, file)
                for file in os.listdir(class_path) if file.endswith('.jpg')]

        class_pairs = [(img_path, class_label) for img_path in class_files]
        random.shuffle(class_pairs)

        num_pairs_total = len(class_pairs)
        num_pairs_train = int(num_pairs_total * 0.8)

        train_pairs.extend(class_pairs[:num_pairs_train])
        valid_pairs.extend(class_pairs[num_pairs_train:])

    return train_pairs,valid_pairs
