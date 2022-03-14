import torch
from torchvision import models as models
import torch.nn as nn
from utils import IOU
from BBox import convert_slender_tensor_to_a_list_of_bbox, BBox


class YOLOV1(nn.Module):
    def __init__(self):
        '''
        这里我一个grid只有一个预测值，并且一共只有20类
        '''
        super(YOLOV1, self).__init__()
        self.cnn = models.regnet_y_400mf(pretrained=True)
        self.fc = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(1000, 7 * 7 * (5 + 20)),  # 5 is confidence + coordinate, 20 is class probabilities
            nn.Sigmoid(),
        )
        self.Loss = nn.MSELoss(reduction='mean')

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        x = x.reshape(x.shape[0], 49, 25)

        return x

    def loss(self, x, target, lambda1=5, lambda2=0.1, lambda3=0.5):
        '''
        :param x: pred
        :param target: data utils will process that
        :param lambda1: for is_instance class
        :param lambda2: for coordinate prob
        :param lambda3: don't care about this....
        :return: loss
        '''
        is_instance_index = target[:, :, 0] == 1
        is_not_instance_index = target[:, :, 0] != 1
        loss1 = lambda1 * self.Loss(x[is_instance_index], target[is_instance_index])
        loss2 = lambda2 * self.Loss(x[is_not_instance_index], target[is_not_instance_index])
        return loss1 + loss2

    def predict(self, x):
        x = self.forward(x)

        for batch in range(x.shape[0]):
            x[batch, :, :] = self.predict_one(x[batch, :, :])

        return x

    def predict_one(self, x, IOU_threshold=0.4, prob_threshold=0.7):
        '''
        :param x: without batch_size, 49x25
        :param IOU_threshold:
        :param prob_threshold:
        :return: same shape, confidence all be 0 or 1
        '''
        true_prob = x[:, 0].view(-1, 1) * x[:, 5:]

        # delete the prob which less than prob_threshold
        zero = torch.zeros_like(true_prob[:, 5:])
        # print(torch.where(true_prob[:, 5:] < prob_threshold, zero, true_prob[:, 5:]).shape)
        true_prob[:, 5:] = torch.where(true_prob[:, 5:] < prob_threshold, zero, true_prob[:, 5:])
        del zero

        # iterate through all classes, dim 5 to dim 25
        for i in range(20):
            a = x[:, 5 + i]
            x[:, 5 + i] = NMS(x[:, 1:5], x[:, 5 + i], IOU_threshold)
            #print(torch.sum(a - x[:, 5 + i]))

        return x


def NMS(coordinate, scores, IOU_threshold=0.1):
    '''
    :param coordinate:(49,4)
    :param scores: (49,)
    :param IOU_threshold:
    :return:new_scores, same shape with scores
    '''
    bboxs = convert_slender_tensor_to_a_list_of_bbox(coordinate)

    _, order = scores.sort(0, descending=True)
    # print(order)
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            if IOU(bboxs[order[i]], bboxs[order[j]]) > IOU_threshold:
                scores[order[j]] = 0
                bboxs[order[j]] = BBox(0, 0, 0, 0)

    return scores
