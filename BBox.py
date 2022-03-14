import torch


class BBox:
    def __init__(self, x, y, w, h):
        '''
        :param x: center coordinate
        :param y: center coordinate
        :param w: lenth_horizontal
        :param h: lenth_vertical

        '''
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def convert_to_4_corner_coordinate(self):
        x1 = self.x - self.w / 2
        x2 = self.x + self.w / 2
        y1 = self.y - self.h / 2
        y2 = self.y + self.h / 2

        return [x1, y1, x2, y2]


def convert_tensor_to_a_list_of_bbox(tensor, img_size=(448, 448), prob_threshold = 0.33):
    '''
    :param tensor: label type, should not contain batch, shape should be 49x25 or 1x49x25, so return less than 49 bbox
    :param img_size: 448x448 should.
    :param prob_threshold:
    :return: a list of bbox
    '''
    result = []
    if len(tensor.shape) == 3:
        tensor = tensor.squeeze(0)

    for i in range(tensor.shape[0]):
        #print(tensor.shape,tensor[i,5:].shape)
        max_prob, _ = torch.max(tensor[i, 5:],dim = 0)
        if tensor[i][0]*max_prob >= prob_threshold:
            #print(tensor[i][3],tensor[i][4])

            grid_x = i % 7
            grid_y = i // 7
            center_x = tensor[i][1] * 64 + grid_x * 64
            center_y = tensor[i][2] * 64 + grid_y * 64
            w = tensor[i][3] * 448
            h = tensor[i][4] * 448
            result.append(BBox(center_x, center_y, w, h))

    return result

def convert_slender_tensor_to_a_list_of_bbox(tensor, img_size=(448, 448)):
    '''
    :param tensor: label type, should not contain batch, shape should be 49x4 or 1x49x4, so return less than 49 bbox
    :param img_size: 448x448 should.
    :return: a list of bbox
    '''
    result = []
    if len(tensor.shape) == 3:
        tensor = tensor.squeeze(0)

    for i in range(tensor.shape[0]):
        grid_x = i % 7
        grid_y = i // 7
        center_x = tensor[i][0] * 64 + grid_x * 64
        center_y = tensor[i][1] * 64 + grid_y * 64
        w = tensor[i][2] * 448
        h = tensor[i][3] * 448
        result.append(BBox(center_x, center_y, w, h))

    return result