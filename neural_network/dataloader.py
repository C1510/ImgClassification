import os,sys, random  # when loading file paths
import pandas as pd  # for lookup in annotation file
import torch
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Load img
import torchvision.transforms as transforms
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None,split=0.9,train=True):
        self.split = split
        self.train=train
        self.root_dir = root_dir
        self.transform = transform

        self.dirs = [os.path.join(root_dir, o) for o in os.listdir(root_dir) if
                     os.path.isdir(os.path.join(root_dir,o))]
        self.class_names = [o for o in os.listdir(root_dir) if
                     os.path.isdir(os.path.join(root_dir, o))]


        self.img_list = [self.return_files(dir=d) for d in self.dirs]
        self.class_balance = [len(l) for l in self.img_list]
        print('Classes',self.class_balance)
        self.classes = len(self.img_list)
        self.img_list = self.train_split(self.img_list)
        self.img_dict = {}
        self.flat_img=[]

        for i,img_list in enumerate(self.img_list):
            for j in img_list:
                label = np.zeros(self.classes)
                label[i]=1
                self.img_dict[j]=label
                self.flat_img.append(j)

        self.df = pd.DataFrame.from_dict(self.img_dict,orient='index',columns=self.class_names)
        print('Dataset size:',len(self.df))


    def train_split(self,img_list):
        min_class=min(self.class_balance)
        train = int(np.ceil(min_class *self.split))
        img_list_train=[]
        img_list_test=[]
        for i in img_list:
            random.shuffle(i)
            img_list_train.append(i[:train])
            img_list_test.append(i[train:])
        return img_list_train if self.train else img_list_test

    def return_files(self,dir):
        print(dir)
        return [os.path.join(dir, o) for o in os.listdir(dir)
                if os.path.isfile(os.path.join(dir,o))]

    def __len__(self):
        return len(self.img_dict)

    def __getitem__(self, id):
        img_name = self.df.index[id]
        img_class = self.df.loc[img_name,:]
        img_class = list(img_class)
        img = Image.open(img_name)

        if self.transform is not None:
            img=img.convert('RGB')
            img = self.transform(img)
        img_class = img_class.index(1)
        return img, img_class


class MyCollate:
    def __init__(self,classes):
        self.classes=classes
        return

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        for count,i in enumerate(imgs):
            if i.shape[1]!=1:
                imgs[count]= transforms.Grayscale(i) #torch.mean(i,dim=1,keepdim=True)
        # print(imgs[0].shape)
        maxdimx = max([i.shape[2] for i in imgs])
        maxdimy = max([i.shape[3] for i in imgs])
        new_imgs = [torch.zeros(1,1,maxdimx, maxdimy) for _ in range(len(imgs))]
        for count, (new, old) in enumerate(zip(new_imgs,imgs)):
            new_imgs[count][:,:,:old.shape[2],:old.shape[3]] = old
        imgs = torch.cat(new_imgs, dim=0)
        targets = torch.tensor([item[1] for item in batch])
        return imgs, targets


def get_loader(
    root_folder,
    transform,
    batch_size=128,
    num_workers=0,
    shuffle=True,
    pin_memory=False,
    train=True,
    split=0.9
):
    dataset = ImageDataset(root_folder, transform=transform,split=split,train=train)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(classes=dataset.classes),
    )

    return loader, dataset


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.ColorJitter(brightness=np.random.uniform(0,0.5), contrast=np.random.uniform(0,0.5)),
         transforms.Grayscale(),]
    )

    loader, dataset = get_loader(
        "data", transform=transform
    )
    print([os.path.basename(os.path.normpath(i)) for i in dataset.dirs])
    tran1 = transforms.ToPILImage()

    from matplotlib.pyplot import imshow
    import matplotlib.pyplot as plt
    w = 10
    h = 10
    fig = plt.figure(figsize=(8, 8))
    columns = 4
    rows = 5
    # for i in range(1, columns * rows + 1):
    #     img = np.random.randint(10, size=(h, w))
    #
    #     plt.imshow(img)
    # plt.show()

    for idx, (imgs, y) in enumerate(loader):
        # print(imgs[0,:,:,:])
        # print(y)
        for j in range(imgs.shape[0]):
            iii=imgs[j,0,:,:].numpy()
            fig.add_subplot(rows, columns, j+1)
            plt.imshow(iii,cmap='gray')
        plt.show()

