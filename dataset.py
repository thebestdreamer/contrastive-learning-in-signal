import torch
from torch.utils.data import Dataset
from augmentation import augment,pwvd
import numpy as np

class SigData(Dataset):
    def __init__(self, snr):
        data_path = 'D:\\360安全浏览器下载\\资料文件\\0-待完成任务\\RML2016.10a\\save_dataset\\4class_data_snr'+str(snr)+'.npy'
        super(SigData, self).__init__()
        self.data = np.load(data_path)

    def __getitem__(self, index):
        sig_wake,sig_str = augment(self.data[index][np.newaxis,:,:])
        img_str = pwvd(sig_str[0])
        img_wake = pwvd(sig_wake[0])
        return torch.from_numpy(img_wake),torch.from_numpy(img_str)
        # return torch.tensor(sig_wake),torch.from_numpy(sig_str)

    def __len__(self):
        return len(self.data)


def main():
    # 数据集加载
    sig_data = SigData(snr=18)
    print(sig_data[0])

if __name__ == '__main__':
    main()
