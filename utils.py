import torch
from BBox import BBox


def IOU(x, y):
    '''
    :param x:a Bbox instance
    :param y:same with x
    :return:IOU score
    '''
    assert isinstance(x, BBox)
    assert isinstance(y, BBox)

    ax0 = x.x - x.w / 2
    ax1 = x.x + x.w / 2
    ay0 = x.y - x.h / 2
    ay1 = x.y + x.h / 2

    bx0 = y.x - y.w / 2
    bx1 = y.x + y.w / 2
    by0 = y.y - y.h / 2
    by1 = y.y + y.h / 2

    Sx = x.w * x.h
    Sy = y.w * y.h

    w = min(bx1, ax1) - max(bx0, ax0)
    h = min(by1, ay1) - max(by0, ay0)

    if w <= 0 or h <= 0:
        return 0

    intersect = w * h

    return intersect / (Sx + Sy - intersect)
