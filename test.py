import sys
import time
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from MQINet import MQINet
from datasets import *
from options import Options
from utils import pad, unpad


def test():

    opt = Options()

    myNet = MQINet()
    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
        print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
    myNet = nn.DataParallel(myNet, device_ids)
    if opt.CUDA_USE:
        myNet = myNet.cuda()

    # testing dataset
    datasetTest = MyTestDataSet(opt.Input_Path_Test_Left,
                                opt.Target_Path_Test_Left,
                                opt.Input_Path_Test_Right,
                                opt.Target_Path_Test_Right)
    testLoader = DataLoader(dataset=datasetTest,
                            batch_size=1,
                            shuffle=False,
                            drop_last=False,
                            num_workers=opt.Num_Works,
                            pin_memory=True)

    print('--------------------------------------------------------------')
    if opt.CUDA_USE:
        myNet.load_state_dict(torch.load(opt.MODEL_SAVE_PATH))
    else:
        myNet.load_state_dict(torch.load(opt.MODEL_SAVE_PATH, map_location=torch.device('cpu')))
    myNet.eval()

    with torch.no_grad():
        timeStart = time.time()
        for index, (x, y, name) in enumerate(tqdm(testLoader, desc='Testing !!! ', file=sys.stdout), 0):
            torch.cuda.empty_cache()

            input_test = x.cuda() if opt.CUDA_USE else x
            target = y.cuda() if opt.CUDA_USE else y

            input_test, pad_size = pad(input_test, factor=16)  # 将输入补成 16 的倍数
            output_test = myNet(input_test).clamp_(-1, 1)
            output_test = unpad(output_test, pad_size)  # 将补上的像素去掉，保持输出输出大小一致

            save_image(output_test[:, :3, :, :], opt.Result_Path_Test_Left + name[0])
            save_image(output_test[:, 3:, :, :], opt.Result_Path_Test_Right + name[0])
        timeEnd = time.time()
        print('---------------------------------------------------------')
        print("Testing Process Finished !!! Time: {:.4f} s".format(timeEnd - timeStart))


if __name__ == '__main__':
    test()
