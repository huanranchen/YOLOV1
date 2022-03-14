import sys
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from BBox import convert_tensor_to_a_list_of_bbox
import numpy as np


def draw_line(im, bbox):
    draw = ImageDraw.Draw(im)
    draw.rectangle(bbox.convert_to_4_corner_coordinate(), outline='red', width=5)


def visualize_one_picture(x, y):
    '''
    :param x: a tensor, CxHxW
    :param y: tensor, label
    :return:nothing. print on the screen
    '''
    if len(x.shape) == 4:
        x = x.squeeze(0)
    # inverse operate of torchvision.transform.ToTensor()
    x *= 255

    bbox = convert_tensor_to_a_list_of_bbox(y)
    x = x.permute(1, 2, 0)


    x = x.numpy()
    x = Image.fromarray(np.uint8(x))
    for i in bbox:
        draw_line(x, i)
    x.show()
