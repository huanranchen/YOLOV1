import torch
import torch.nn as nn
from data_utils import get_loader
from YOLO import YOLOV1
import os
from tqdm import tqdm


def train(
        num_epoch=100,
        lr=1e-3,
        batch_size=100,
        weight_decay=1e-5
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLOV1().to(device)
    print(model)
    if os.path.exists('model.pth'):
        model.load_state_dict(torch.load('model.pth'))
    loader = get_loader(batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epoch + 1):
        epoch_loss = 0
        for data in tqdm(loader):
            optimizer.zero_grad()
            x, y = data
            x = x.to(device)
            y = y.to(device)
            pre = model(x)


            loss = model.loss(pre, y)

            epoch_loss += loss.item()
            loss.backward()

            optimizer.step()


        print(f'epoch {epoch}, loss = {epoch_loss}')
        torch.save(model.state_dict(),'model.pth')