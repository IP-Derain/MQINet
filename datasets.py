import os
import random
import torch
import torchvision.transforms.functional as ttf
from torch.utils.data import Dataset
from PIL import Image


class MyTrainDataSet(Dataset):
    def __init__(self, inputPathTrainLeft, targetPathTrainLeft, inputPathTrainRight, targetPathTrainRight, patch_size=64):
        super(MyTrainDataSet, self).__init__()

        self.inputPathLeft = inputPathTrainLeft
        self.inputImagesLeft = os.listdir(inputPathTrainLeft)
        self.inputImagesLeft.sort(key=lambda x: int(x.split('.')[0]))
        self.inputPathRight = inputPathTrainRight
        self.inputImagesRight = os.listdir(inputPathTrainRight)
        self.inputImagesRight.sort(key=lambda x: int(x.split('.')[0]))

        self.targetPathLeft = targetPathTrainLeft
        self.targetImagesLeft = os.listdir(targetPathTrainLeft)
        self.targetImagesLeft.sort(key=lambda x: int(x.split('.')[0]))
        self.targetPathRight = targetPathTrainRight
        self.targetImagesRight = os.listdir(targetPathTrainRight)
        self.targetImagesRight.sort(key=lambda x: int(x.split('.')[0]))

        self.ps = patch_size

    def __len__(self):
        return len(self.targetImagesLeft)

    def __getitem__(self, index):

        ps = self.ps
        index = index % len(self.targetImagesLeft)

        inputImagePathLeft = os.path.join(self.inputPathLeft, self.inputImagesLeft[index])
        inputImageLeft = Image.open(inputImagePathLeft).convert('RGB')
        inputImagePathRight = os.path.join(self.inputPathRight, self.inputImagesLeft[index])
        inputImageRight = Image.open(inputImagePathRight).convert('RGB')

        targetImagePathLeft = os.path.join(self.targetPathLeft, self.targetImagesLeft[index])
        targetImageLeft = Image.open(targetImagePathLeft).convert('RGB')
        targetImagePathRight = os.path.join(self.targetPathRight, self.targetImagesLeft[index])
        targetImageRight = Image.open(targetImagePathRight).convert('RGB')

        inputImageLeft = ttf.to_tensor(inputImageLeft)
        targetImageLeft = ttf.to_tensor(targetImageLeft)
        inputImageRight = ttf.to_tensor(inputImageRight)
        targetImageRight = ttf.to_tensor(targetImageRight)

        hh, ww = targetImageLeft.shape[1], targetImageLeft.shape[2]

        rr = random.randint(0, hh-ps)
        cc = random.randint(0, ww-ps)
        #
        inputLeft = inputImageLeft[:, rr:rr+ps, cc:cc+ps]
        targetLeft = targetImageLeft[:, rr:rr+ps, cc:cc+ps]
        inputRight = inputImageRight[:, rr:rr+ps, cc:cc+ps]
        targetRight = targetImageRight[:, rr:rr+ps, cc:cc+ps]

        input_ = torch.cat((inputLeft, inputRight), 0)
        target = torch.cat((targetLeft, targetRight), 0)

        return input_, target


class MyValueDataSet(Dataset):
    def __init__(self, inputPathTrainLeft, targetPathTrainLeft, inputPathTrainRight, targetPathTrainRight,
                 patch_size=64):
        super(MyValueDataSet, self).__init__()

        self.inputPathLeft = inputPathTrainLeft
        self.inputImagesLeft = os.listdir(inputPathTrainLeft)
        self.inputImagesLeft.sort(key=lambda x: int(x.split('.')[0]))
        self.inputPathRight = inputPathTrainRight
        self.inputImagesRight = os.listdir(inputPathTrainRight)

        self.targetPathLeft = targetPathTrainLeft
        self.targetImagesLeft = os.listdir(targetPathTrainLeft)
        self.targetImagesLeft.sort(key=lambda x: int(x.split('.')[0]))
        self.targetPathRight = targetPathTrainRight
        self.targetImagesRight = os.listdir(targetPathTrainRight)

        self.ps = patch_size

    def __len__(self):
        return len(self.targetImagesLeft)

    def __getitem__(self, index):
        ps = self.ps
        index = index % len(self.targetImagesLeft)

        inputImagePathLeft = os.path.join(self.inputPathLeft, self.inputImagesLeft[index])
        inputImageLeft = Image.open(inputImagePathLeft).convert('RGB')
        inputImagePathRight = os.path.join(self.inputPathRight, self.inputImagesLeft[index])
        inputImageRight = Image.open(inputImagePathRight).convert('RGB')

        targetImagePathLeft = os.path.join(self.targetPathLeft, self.targetImagesLeft[index])
        targetImageLeft = Image.open(targetImagePathLeft).convert('RGB')
        targetImagePathRight = os.path.join(self.targetPathRight, self.targetImagesLeft[index])
        targetImageRight = Image.open(targetImagePathRight).convert('RGB')

        inputImageLeft = ttf.to_tensor(inputImageLeft)
        targetImageLeft = ttf.to_tensor(targetImageLeft)
        inputImageRight = ttf.to_tensor(inputImageRight)
        targetImageRight = ttf.to_tensor(targetImageRight)

        inputLeft = ttf.center_crop(inputImageLeft, (ps, ps))
        targetLeft = ttf.center_crop(targetImageLeft, (ps, ps))
        inputRight = ttf.center_crop(inputImageRight, (ps, ps))
        targetRight = ttf.center_crop(targetImageRight, (ps, ps))

        input_ = torch.cat((inputLeft, inputRight), 0)
        target = torch.cat((targetLeft, targetRight), 0)

        return input_, target


class MyTestDataSet(Dataset):
    def __init__(self, inputPathTrainLeft, targetPathTrainLeft, inputPathTrainRight, targetPathTrainRight):
        super(MyTestDataSet, self).__init__()

        self.inputPathLeft = inputPathTrainLeft
        self.inputImagesLeft = os.listdir(inputPathTrainLeft)
        self.inputImagesLeft.sort(key=lambda x: int(x.split('.')[0]))
        self.inputPathRight = inputPathTrainRight
        self.inputImagesRight = os.listdir(inputPathTrainRight)
        self.inputImagesRight.sort(key=lambda x: int(x.split('.')[0]))

        self.targetPathLeft = targetPathTrainLeft
        self.targetImagesLeft = os.listdir(targetPathTrainLeft)
        self.targetImagesLeft.sort(key=lambda x: int(x.split('.')[0]))
        self.targetPathRight = targetPathTrainRight
        self.targetImagesRight = os.listdir(targetPathTrainRight)
        self.targetImagesRight.sort(key=lambda x: int(x.split('.')[0]))

    def __len__(self):
        return len(self.inputImagesLeft)

    def __getitem__(self, index):
        index = index % len(self.inputImagesLeft)

        inputImagePathLeft = os.path.join(self.inputPathLeft, self.inputImagesLeft[index])
        inputImageLeft = Image.open(inputImagePathLeft).convert('RGB')
        inputImagePathRight = os.path.join(self.inputPathRight, self.inputImagesLeft[index])
        inputImageRight = Image.open(inputImagePathRight).convert('RGB')

        inputLeft = ttf.to_tensor(inputImageLeft)
        inputRight = ttf.to_tensor(inputImageRight)

        targetImagePathLeft = os.path.join(self.targetPathLeft, self.targetImagesLeft[index])
        targetImageLeft = Image.open(targetImagePathLeft).convert('RGB')
        targetImagePathRight = os.path.join(self.targetPathRight, self.targetImagesLeft[index])
        targetImageRight = Image.open(targetImagePathRight).convert('RGB')

        targetLeft = ttf.to_tensor(targetImageLeft)
        targetRight = ttf.to_tensor(targetImageRight)

        input_ = torch.cat((inputLeft, inputRight), 0)
        target = torch.cat((targetLeft, targetRight), 0)

        return input_, target, self.inputImagesLeft[index]
