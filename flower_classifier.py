from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import os
import glob
import random
import copy
import cv2 as cv
import numpy as np
#from matplotlib.pyplot import imshow

class FlowersDataset(Dataset):
    def __init__(self, image_label_pair_list, image_transforms):
        self.data = image_label_pair_list
        self.transforms = image_transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data[idx][0]).convert('RGB')
        image = self.transforms(image)
        y = torch.nn.functional.one_hot(torch.tensor(self.data[idx][1]),5).type(torch.float32)
        return image, y

def train_and_export_flower_model():
    ########################
    # prepare image datasets
    ########################
    IMAGE_SIZE = 180
    data_dir  = "flower_photos"

    def get_img_label_list(files_list, class_label):
        class_data = [(img_path, class_label) for img_path in files_list]
        return class_data

    class_folders = [os.path.join(data_dir, subdir)
                   for subdir in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, subdir))]

    data = []
    for label, class_path in enumerate(class_folders):
        class_files = [os.path.join(class_path, file)
                for file in os.listdir(class_path) if file.endswith(".jpg")]
        data += get_img_label_list(class_files, label)

    # shuffle data
    random.shuffle(data)

    # limit
    #data = data[:200]

    all_files_num = len(data)
    train_num = int(all_files_num * 0.8)

    # datasets
    train_dataset = FlowersDataset(data[:train_num], transforms.Compose([
                transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation((-5,5)),
                #transforms.RandomResizedCrop
                transforms.ToTensor()
            ]))
    valid_dataset = FlowersDataset(data[train_num:], transforms.Compose([
                transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
                transforms.ToTensor()
            ]))

    # dataset loaders
    train_loader  = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=8)
    valid_loader  = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=32, num_workers=8)


    ###############
    # prepare model
    ###############
    # model
    model_flowers = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1, stride=1),
                nn.MaxPool2d(2),
                nn.ReLU(),

                nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=1),
                nn.MaxPool2d(2),
                nn.ReLU(),

                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1),
                nn.MaxPool2d(2),
                nn.ReLU(),

                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
                nn.MaxPool2d(2),
                nn.ReLU(),

                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(5*5*128, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 5),
                nn.Softmax()
            )
    for layer in model_flowers:
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
    print(model_flowers)

    ######################
    # get gpu if available
    ######################

    if torch.cuda.is_available():
        print("cuda available:", torch.cuda.get_device_name(0))
    cuda = torch.device('cuda:0')
    cpu = torch.device('cpu')

    #################
    # training params
    #################

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_flowers.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                 mode='min', factor=0.1, patience=10, threshold=1e-4,
                 verbose=True)
    
    ###########################
    # get optimal learning rate
    ###########################
    """
    for learning_rate in [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4,
               5e-5, 1e-5, 5e-6, 1e-6]:
        model_copy = copy.deepcopy(model_flowers)
        model_copy.to(cuda)
        optimizer = torch.optim.Adam(model_copy.parameters(), lr=learning_rate)

        loss_list = []
        accuracy_list_train = []

        for epoch in range(5):
            epoch_losses = []
            epoch_correct = 0
            model_copy.train()
            for x,y in train_loader:
                x,y = x.to(cuda), y.to(cuda)
                optimizer.zero_grad()
                z = model_copy(x)
                loss = criterion(z,y)

                epoch_correct += (torch.argmax(z, 1) == torch.argmax(y, 1)).sum().item()
                epoch_losses.append(loss.data.item())
                loss.backward()
                optimizer.step()

            loss_list.append(torch.mean(torch.tensor(epoch_losses)).item())
            accuracy_list_train.append(epoch_correct / len(train_loader.dataset))

        # after 5 epochs
        print("lr {:.5f}. loss {:.5f}, accuracy {:.5f}".format(learning_rate, loss_list[-1], accuracy_list_train[-1]))

        del model_copy
    """

    ##########################
    # training and fine tuning
    ##########################

    model_flowers.to(cuda)

    loss_list_train = []
    accuracy_list_train = []
    accuracy_list_valid = []

    epochs = 400

    for epoch in range(epochs):

        # training set
        epoch_losses = []
        epoch_correct = 0
        model_flowers.train()
        
        for x,y in train_loader:
            x,y = x.to(cuda), y.to(cuda)
            z = model_flowers(x)
            loss = criterion(z,y)

            epoch_correct += (torch.argmax(z, 1) == torch.argmax(y, 1)).sum().item()
            epoch_losses.append(loss.data.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_list_train.append(torch.mean(torch.tensor(epoch_losses)).item())
        accuracy_list_train.append(epoch_correct / len(train_loader.dataset))

        # validation set
        epoch_correct = 0
        model_flowers.eval()
        for x, y in valid_loader:
            x,y = x.to(cuda), y.to(cuda)
            z = model_flowers(x)
            epoch_correct += (torch.argmax(z, 1) == torch.argmax(y, 1)).sum().item()

        accuracy_list_valid.append(epoch_correct / len(valid_loader.dataset))

        lr_scheduler.step(loss_list_train[-1])

        print("epoch {}, loss: {:.5f}, accuracy: {:.5f}, val_accuracy: {:.5f}".format(
            epoch+1, loss_list_train[-1], accuracy_list_train[-1], accuracy_list_valid[-1]))
    print("training done")

    #############
    # check image
    #############

    for image_check, y_check in train_loader:
        break

    image_check = torch.reshape(image_check[0], (1,3,IMAGE_SIZE,IMAGE_SIZE))
    y_check = y_check[0]

    # show only R-channel
    #imshow(np.reshape(image_check[0].numpy()[0], (IMAGE_SIZE,IMAGE_SIZE)))

    image_check = image_check.to(cuda)
    z = model_flowers(image_check)
    print(torch.argmax(z).item(), torch.argmax(y_check))

    ############
    # save model
    ############
    model_flowers.to(cpu)
    model_flowers.eval()

    model_name = "model_flowers.onnx"
    torch.onnx.export(model_flowers,
                        torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE),# input example with size
                        model_name,
                        verbose=False,
                        input_names=["actual_input"],
                        output_names=["output"],
                        export_params=True)

    ###################
    # check saved model
    ###################

    network = cv.dnn.readNetFromONNX(model_name)

    NUMBER_CHECK = 1001
    input_image = cv.imread(train_loader.dataset.data[NUMBER_CHECK][0], cv.IMREAD_COLOR)

    input_blob = cv.dnn.blobFromImage(
        image=input_image,
        scalefactor=1./255.,
        size=(IMAGE_SIZE,IMAGE_SIZE))

    network.setInput(input_blob)
    out = network.forward()

    print(torch.argmax(torch.tensor(out)).item(), train_loader.dataset.data[NUMBER_CHECK][1])

if __name__ == '__main__':
    train_and_export_flower_model()
