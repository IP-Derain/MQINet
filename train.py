import sys
import time
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from utils import torchPSNR
from MQINet import MQINet
from datasets import *
from options import Options
import torch.nn.functional as F


def train():

    opt = Options()
    cudnn.benchmark = True
    best_psnr = 0
    best_epoch = 0

    random.seed(opt.Seed)
    torch.manual_seed(opt.Seed)
    torch.cuda.manual_seed(opt.Seed)
    torch.manual_seed(opt.Seed)

    myNet = MQINet()
    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
        print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
    myNet = nn.DataParallel(myNet, device_ids)
    if opt.CUDA_USE:
        myNet = myNet.cuda()

    # optimizer
    optimizer = optim.Adam(myNet.parameters(), lr=opt.Learning_Rate)
    # schedule
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.Epoch, eta_min=1e-7)

    # training dataset
    datasetTrain = MyTrainDataSet(opt.Input_Path_Train_Left,
                                  opt.Target_Path_Train_Left,
                                  opt.Input_Path_Train_Right,
                                  opt.Target_Path_Train_Right,
                                  patch_size=opt.Patch_Size_Train)
    trainLoader = DataLoader(dataset=datasetTrain,
                             batch_size=opt.Batch_Size_Train,
                             shuffle=True,
                             drop_last=True,
                             num_workers=opt.Num_Works,
                             pin_memory=True)

    # validation dataset
    datasetValue = MyValueDataSet(opt.Input_Path_Val_Left,
                                  opt.Target_Path_Val_Left,
                                  opt.Input_Path_Val_Right,
                                  opt.Target_Path_Val_Right,
                                  patch_size=opt.Patch_Size_Val)
    valueLoader = DataLoader(dataset=datasetValue,
                             batch_size=opt.Batch_Size_Val,
                             shuffle=True,
                             drop_last=True,
                             num_workers=opt.Num_Works,
                             pin_memory=True)

    # begin training
    print('-------------------------------------------------------------------------------------------------------')
    if os.path.exists(opt.MODEL_RESUME_PATH):
        if opt.CUDA_USE:  # CUDA
            myNet.load_state_dict(torch.load(opt.MODEL_RESUME_PATH))
        else:  # CPU
            myNet.load_state_dict(torch.load(opt.MODEL_RESUME_PATH, map_location=torch.device('cpu')))

    for epoch in range(opt.Epoch):
        myNet.train()
        iters = tqdm(trainLoader, file=sys.stdout)
        epochLoss = 0
        timeStart = time.time()
        for index, (x, y) in enumerate(iters, 0):

            myNet.zero_grad()
            optimizer.zero_grad()

            if opt.CUDA_USE:
                input_train, target = Variable(x).cuda(), Variable(y).cuda()
            else:
                input_train, target = Variable(x), Variable(y)

            output_train = myNet(input_train)
            loss = F.l1_loss(output_train, target)

            loss.backward()
            optimizer.step()
            epochLoss += loss.item()

            iters.set_description('Training !!!  Epoch %d / %d,  Batch Loss %.6f' % (epoch+1, opt.Epoch, loss.item()))

        if epoch % 3 == 0:
            myNet.eval()
            psnr_val_rgb = []
            for index, (x, y) in enumerate(valueLoader, 0):
                input_, target_value = (x.cuda(), y.cuda()) if opt.CUDA_USE else (x, y)
                with torch.no_grad():
                    output_value = myNet(input_)
                for output_value, target_value in zip(output_value, target_value):
                    psnr_val_rgb.append(torchPSNR(output_value, target_value))

            psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()

            if psnr_val_rgb >= best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch = epoch
                torch.save(myNet.state_dict(), opt.MODEL_SAVE_PATH)
        scheduler.step(epoch)
        timeEnd = time.time()
        print("------------------------------------------------------------")
        print("Epoch:  {}  Finished,  Time:  {:.4f} s,  Loss:  {:.6f}, current psnr:  {:.3f}, best psnr:  {:.3f}.".format(epoch+1, timeEnd-timeStart, epochLoss, psnr_val_rgb, best_psnr))
        print('-------------------------------------------------------------------------------------------------------')
    print("Training Process Finished ! Best Epoch : {} , Best PSNR : {:.2f}".format(best_epoch, best_psnr))


if __name__ == '__main__':
    train()