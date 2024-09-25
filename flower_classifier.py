import torch
import torch.nn as nn
import torch.nn.functional as F

import time

from flower_model import FlowerModel
from flower_dataset import prepare_train_valid_pairs,FlowersDataset
from flower_model_export_onnx import add_softmax_and_export_onnx

def train_for_device(model_flowers, train_loader,valid_loader, device,epochs=300):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model_flowers.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                 mode='min', factor=0.5, patience=10, threshold=1e-4)

    model_flowers.to(device)

    loss_list_train = []
    accuracy_list_train = []
    accuracy_list_valid = []

    for epoch in range(epochs):
        st = time.time()

        # training
        epoch_losses = []
        epoch_correct = 0
        model_flowers.train()

        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            z = model_flowers(x)
            loss = criterion(z,y)

            epoch_correct += (torch.argmax(F.softmax(z,dim=1), 1) == torch.argmax(y, 1)).sum().item()
            epoch_losses.append(loss.data.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_flowers.parameters(), 1.0)
            optimizer.step()

        loss_list_train.append(torch.mean(torch.tensor(epoch_losses)).item())
        accuracy_list_train.append(epoch_correct / len(train_loader.dataset))

        # validation
        epoch_correct = 0
        model_flowers.eval()

        with torch.no_grad():
            for x, y in valid_loader:
                x,y = x.to(device), y.to(device)
                z = model_flowers(x)
                epoch_correct += (torch.argmax(F.softmax(z,dim=1), 1) == torch.argmax(y, 1)).sum().item()

        accuracy_list_valid.append(epoch_correct / len(valid_loader.dataset))

        lr_scheduler.step(loss_list_train[-1])
        lr = lr_scheduler.get_last_lr()[0]

        dur = time.time() - st
        print('epoch {}, tr loss: {:.5f}, accuracy: train {:.5f}, val {:.5f}. lr: {:.8f} | {:.1f}sec'.format(
            epoch+1, loss_list_train[-1], accuracy_list_train[-1], accuracy_list_valid[-1], lr,dur))

        if lr < 1e-7:
            print('lr is too small. break')
            break

if '__main__' == __name__:
    # prepare dataset
    IMAGE_SIZE = 180
    train_pairs,valid_pairs = prepare_train_valid_pairs()
    train_dataset = FlowersDataset(train_pairs, IMAGE_SIZE, True)
    valid_dataset = FlowersDataset(valid_pairs, IMAGE_SIZE, False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=8)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=32, num_workers=8)

    # prepare model
    model_flowers = FlowerModel()

    # device
    device = torch.device('cpu')
    if torch.cuda.is_available():
        print('cuda available:', torch.cuda.get_device_name(0))
        device = torch.device('cuda:0')

    # train
    train_for_device(model_flowers,train_loader,valid_loader,device,epochs=300)
    print('training done')

    add_softmax_and_export_onnx(model_flowers,IMAGE_SIZE)
    print('completed')
