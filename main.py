import torch
from YOLO import YOLOV1
from data_utils import get_loader
from visualization import visualize_one_picture
import torch

loader = get_loader(batch_size=1)
model = YOLOV1()
model.load_state_dict(torch.load('model.pth', map_location='cpu'), )
for i in loader:
    x, y = i
    pre = model.predict(x)

    print(pre[:,:,0])
    print(torch.sum(pre-y))
    visualize_one_picture(x, y)
    assert False
