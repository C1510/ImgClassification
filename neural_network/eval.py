import torch
import torchvision
import torchvision.transforms as transforms
from dataloader import get_loader
from dataloader import *
from custom_resnet import resnet18
from torch.optim.lr_scheduler import ReduceLROnPlateau
import shutil

###########################################

#Specifies image to test with
img_name = 'ex1.tif'

#Location of models
model_path = './models'

#Specify either these parameters to point to the correct model
model_no = 0
batch_name = 'test'
username = 'usr'
train_image_name = 'train1.tif'
#Or specify model name exactly (leave like model_name = '' to use above parameters)
model_name = ''

num_classes = 3

###########################################

img_name_ = img_name.split('.')[0]
train_image_name_ = train_image_name.split('.')[0]

data_folder = f'../imgs_classified_png/{batch_name}_{img_name_}_noclass/'
data_folder_out = f'classified_png/{batch_name}_{img_name_}_{username}_{model_no}/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if model_name == '':
    model_name = f'{batch_name}_{train_image_name_}_{username}_{model_no}'
PATH = f'{model_path}/{model_name}.pth'
loss = 'ce'

transform = transforms.Compose(
        [transforms.Resize((100, 100)),
         transforms.ToTensor(),
         #transforms.ColorJitter(brightness=0.5, contrast=0.5),
         #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         transforms.Grayscale(),
         transforms.Normalize(0.5,0.5),
         #transforms.RandomPerspective(distortion_scale=0.6, p=1.0)
        ]
    )

# transform = transforms.Compose(
#         [transforms.ToTensor(),
#          transforms.Grayscale()]
#     )

test_loader, test_dataset = get_loader(
        data_folder, transform=transform,
    batch_size=128,
    num_workers=0,
    shuffle=False,
    pin_memory=False,
    train=False,
    split=0.0,
    )

classes_list = [os.path.basename(os.path.normpath(i)) for i in test_dataset.dirs]
classes=dict(zip(classes_list,[i for i in range(len(classes_list))]))

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 48, (5, 5), (1, 1))
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(48, 36, (3, 3))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(36, 12, (3, 3))
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(12*6*6, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.bn1 = nn.BatchNorm2d(48)
        self.bn2 = nn.BatchNorm2d(36)
        self.bn3 = nn.BatchNorm2d(12)
        self.bn4 = nn.BatchNorm1d(12*6*6)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(84)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.bn1(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.bn2(x)
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.bn3(x)
        x = x.view(-1, 12*6*6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
net.load_state_dict(torch.load(PATH))
print('loading success')
net = net.to(device)
net.eval()

if loss != 'BCE':
    criterion = nn.CrossEntropyLoss()
else:
    criterion = nn.BCEWithLogitsLoss()

def test_it(loader, set = 'train'):
    labels_out = []
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            labels = labels.to(device)
            images = images.to(device)
            outputs = net(images)
            outputs = torch.argmax(outputs,dim=1).cpu()
            labels_out.append(outputs)
    return labels_out

labels_out = test_it(test_loader, set = 'test')
labels = torch.cat(labels_out,dim = 0).reshape(-1)
df = test_dataset.df
print(df)
df['labelled']=list([i.item() for i in labels.reshape(-1)])
df.reset_index(inplace=True)
print(df)

def move_files(stats):
    # This function takes the stats file and an image, and saves your classifications to the
    # imgs_classified and imgs_classified_png folders.
    for c, col in stats.iterrows():
        # Saves file according to classification
        print('heh',col['index'], col['labelled'])
        if not os.path.exists(data_folder_out):
            os.makedirs(data_folder_out)
        class_ = col['labelled']
        if not os.path.exists(data_folder_out+str(class_)):
            os.makedirs(data_folder_out+str(class_))
        print(data_folder_out+str(class_)+'/'+str(c)+'.png')
        shutil.copy(col['index'], data_folder_out+str(class_)+'/'+str(c)+'.png')

move_files(df)
