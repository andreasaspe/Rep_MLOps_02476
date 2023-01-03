import torch
import numpy as np

#There are three labels: images, labels and allow_pickle

class YourOwnDataset(torch.utils.data.Dataset):
    def __init__(self, input_file_path, transformations) :
        super().__init__()
        self.path = input_file_path

        with np.load(self.path) as fh:
            # I assume fh['data_x'] is a list you get the idea  
            self.data = fh['data_x']
            self.labels = fh['data_y']

    # in getitem, we retrieve one item based on the input index
    def __getitem__(self, index):
        data = self.data[index]
        # based on the loss you chose and what you have in mind, 
        # you can transform you label, here I assume they are 
        # integer numbers (like, 1, 3, etc as labels used for classification)
        label = self.labels[index]
        img = self.transforms(img)
        img = img.reshape(img.shape[0],-1)
        return img, label

    def __len__(self):
        return len(self.data)

def mnist():
    # exchange with the corrupted mnist dataset
    train0 = np.load('/Users/andreasaspe/iCloud/DTU/11.semester/MLOps_02476/dtu_mlops/data/corruptmnist/train_0.npz')
    train1 = np.load('/Users/andreasaspe/iCloud/DTU/11.semester/MLOps_02476/dtu_mlops/data/corruptmnist/train_1.npz')
    train2 = np.load('/Users/andreasaspe/iCloud/DTU/11.semester/MLOps_02476/dtu_mlops/data/corruptmnist/train_2.npz')
    train3 = np.load('/Users/andreasaspe/iCloud/DTU/11.semester/MLOps_02476/dtu_mlops/data/corruptmnist/train_3.npz')
    train4 = np.load('/Users/andreasaspe/iCloud/DTU/11.semester/MLOps_02476/dtu_mlops/data/corruptmnist/train_4.npz')
    test = np.load('/Users/andreasaspe/iCloud/DTU/11.semester/MLOps_02476/dtu_mlops/data/corruptmnist/test.npz')
    print(train['images'].shape)
    #train = torch.randn(50000, 784)
    #test = torch.randn(10000, 784) 
    return train, test

mnist()

torch.vision.

#https://pytorch.org/tutorials/beginner/basics/data_tutorial.html