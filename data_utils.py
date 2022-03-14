from torch.utils.data import Dataset, DataLoader
import torch
import os
from PIL import Image
from torchvision import transforms
import xml.etree.ElementTree as ET
import numpy as np




def get_absolute_coordinate_to_relevant_coordinate(absolute_coordinate, img_size):
    '''
    :param absolute_coordinate: [xmin, ymin, xmax, ymax]
    :param img_size:[width, height]
    :return:[(grid_x,grid_y),x_relative_center,y_relative_center, relative_lenth, relative_height]
    grid_x & grid_y: 0, 1, 2, 3, 4, 5, 6
    '''
    #compute patches
    width, height = tuple(img_size)
    len_each_patch = width/7
    height_each_patch = height/7


    #relative lenth and height
    xmin, ymin, xmax, ymax = tuple(absolute_coordinate)
    relative_height = (ymax-ymin)/height
    relative_lenth = (xmax-xmin)/width


    #grid x and y
    x_center_coordinate = (xmin+xmax)*0.5
    y_center_coordinate = (ymin+ymax)*0.5
    grid_x = int(x_center_coordinate / len_each_patch)
    grid_y = int(y_center_coordinate / height_each_patch)

    #relative center
    x_relative_center = (x_center_coordinate - len_each_patch*grid_x)/len_each_patch
    y_relative_center = (y_center_coordinate - height_each_patch*grid_y)/height_each_patch

    return [(grid_x,grid_y),x_relative_center,y_relative_center,relative_lenth,relative_height]



def convert_label_to_true_label(labels):
    '''
    :param labels: #label is a lot of this_one, each is [1 grid_coordinate relative_coordinate 20_class_probability]
    :return: True label, 49xD, D is [1 or 0, grid_coordinate relative_coordinate 20_class_probability]
    '''
    true_label = []
    for i in range(7):
        for j in range(7):
            is_finded = False
            for one_label in labels:
                #could find
                if (j,i) == one_label[1]:
                    one_label.pop(1)
                    true_label.append(torch.tensor(one_label))
                    labels.remove(one_label)
                    is_finded = True
                    break

            if is_finded is False:
                # can't find
                one_label = [0] * 25
                true_label.append(torch.tensor(one_label))


    # this line is because a weiry problem. I solve this problem by
    # https://blog.csdn.net/Petersburg/article/details/120179077

    true_label = torch.tensor(np.array([item.detach().numpy() for item in true_label]), dtype=torch.float32)

    return true_label




class MyDataset(Dataset):
    def __init__(self, data_path = './data/JPEGImages', label_path = './data/Annotations'):
        self.data_path = data_path
        self.label_path = label_path
        self.x = sorted([os.path.join(data_path, x) for x in os.listdir(data_path) if x.endswith(".jpg")])
        self.y = sorted([os.path.join(label_path, x) for x in os.listdir(label_path) if x.endswith(".xml")])
        self.transform = transforms.Compose([
            transforms.Resize([448, 448]),
            transforms.ToTensor(),
        ])
        # dictionary from class name to number, vary from 0-19
        self.dic = {}

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        # process image
        data_name = self.x[item]
        im = Image.open(data_name)


        im = self.transform(im)


        # process label
        label = []
        label_name = self.y[item]
        tree = ET.parse(label_name)
        root = tree.getroot()
        for child in root:
            if child.tag == 'size':
                img_size = []
                for size in child:
                    if size.tag == 'width':
                        img_size.append(int(size.text))
                    if size.tag == 'height':
                        img_size.append(int(size.text))


            if child.tag == 'object':
                for attribute in child:
                    # recognize this time information
                    this_one = [1]
                    if attribute.tag == 'name':
                        # add label
                        name = attribute.text
                        if name in self.dic:
                            pass
                        else:
                            self.dic[name] = len(self.dic)
                        this_one.append(self.dic[name])

                    if attribute.tag == 'bndbox':

                        absolute_coordinate = []
                        for coordinate in attribute:
                            absolute_coordinate.append(int(coordinate.text))



                        relative_coordinate = get_absolute_coordinate_to_relevant_coordinate(absolute_coordinate,img_size)

                        #置信概率、relative coordinate、20 class probability
                        this_one = this_one + relative_coordinate
                        classes = [0]*20
                        classes[self.dic[name]] = 1
                        this_one = this_one + classes
                        label.append(this_one)

                        break

        #label is a lot of this_one, each is [1 grid_coordinate relative_coordinate 20_class_probability]
        true_label = convert_label_to_true_label(label)
        return im, true_label

def get_loader(batch_size = 100):
    dataset = MyDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader