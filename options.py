
class Options():
    def __init__(self):
        super().__init__()
        self.Seed = 1234
        self.Epoch = 200
        self.Learning_Rate = 1e-3
        self.Batch_Size_Train = 32
        self.Batch_Size_Val = 8
        self.Patch_Size_Train = 128
        self.Patch_Size_Val = 256

        flag = 0
        dataset_list = ['k12', 'k15', 'StereoCityScapes']
        dataset = dataset_list[flag]

        # train
        self.Input_Path_Train_Left = './'+dataset+'/testing/left_input/'
        self.Target_Path_Train_Left = './'+dataset+'/testing/left_target/'
        self.Input_Path_Train_Right = './'+dataset+'/testing/right_input/'
        self.Target_Path_Train_Right = './'+dataset+'/testing/right_target/'

        # validation
        self.Input_Path_Val_Left = './'+dataset+'/val/left_input/'
        self.Target_Path_Val_Left = './'+dataset+'/val/left_target/'
        self.Input_Path_Val_Right = './'+dataset+'/val/right_input/'
        self.Target_Path_Val_Right = './'+dataset+'/val/right_target/'

        # test
        self.Input_Path_Test_Left = './'+dataset+'/testing/left_input/'
        self.Target_Path_Test_Left = './'+dataset+'/testing/left_target/'
        self.Result_Path_Test_Left = './'+dataset+'/testing/left_result/'
        self.Input_Path_Test_Right = './'+dataset+'/testing/right_input/'
        self.Target_Path_Test_Right = './'+dataset+'/testing/right_target/'
        self.Result_Path_Test_Right = './'+dataset+'/testing/right_result/'


        # models
        self.MODEL_RESUME_PATH = './model_k12.pth'
        self.MODEL_SAVE_PATH = './model_k12.pth'

        self.Num_Works = 4
        self.CUDA_USE = True