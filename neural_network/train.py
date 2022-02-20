import torch
import torchvision
import torchvision.transforms as transforms
from dataloader import get_loader
from dataloader import *
from custom_resnet import resnet18
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim

###########################################

epochs=100
model_path = './models'
model_no = 0
loss = 'CE'
img_name = 'Original_Halved.tif'
batch_name = 'test'
username = 'mark'

###########################################

img_name_ = img_name.split('.')[0]
data_folder = f'../imgs_classified_png/{batch_name}_{img_name_}_{username}/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATH = model_path+str(model_no)+'.pth'
PATH = f'{model_path}/{batch_name}_{img_name_}_{username}_{model_no}.pth'
loss = 'ce'

transform = transforms.Compose(
        [transforms.Resize((100, 100)),
         transforms.ToTensor(),
         #transforms.ColorJitter(brightness=0.5, contrast=0.5),
         #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         transforms.Grayscale(),
         # transforms.Normalize(0.5,0.5),
         transforms.RandomRotation(180),
         #transforms.RandomPerspective(distortion_scale=0.6, p=1.0)
        ]
    )

train_loader, train_dataset = get_loader(
        data_folder, transform=transform,
    batch_size=64,
    num_workers=0,
    shuffle=True,
    pin_memory=False,
    train=True,
    split=0.9,
    )

test_loader, test_dataset = get_loader(
        data_folder, transform=transform,
    batch_size=64,
    num_workers=0,
    shuffle=True,
    pin_memory=False,
    train=False,
    split=0.9,
    )

num_classes = train_dataset.classes
classes_list = [os.path.basename(os.path.normpath(i)) for i in train_dataset.dirs]
classes=dict(zip(classes_list,[i for i in range(len(classes_list))]))

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 96, (9, 9), (2, 2))
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(96, 12, (3, 3))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(12*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 12*6*6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
net.to(device)

if loss != 'BCE':
    criterion = nn.CrossEntropyLoss()
else:
    criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(net.parameters(), lr=0.01)

def test_it(loader, set = 'train'):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            labels = labels.to(device)
            images = images.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the %s images: %d %%' % (set,
            100 * correct / total))
    return correct/total

scheduler = ReduceLROnPlateau(optimizer, 'min')
for epoch in range(epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        labels = labels.reshape(-1).to(device)
        # get the inputs; data is a list of [inputs, labels]
        # zero the parameter gradients
        optimizer.zero_grad()
        # if loss !='BCE':
        #     labels=torch.argmax(labels,dim=1)
        outputs = net(inputs.to(device))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    scheduler.step(loss)
    if epoch % 1 == 0:    # print every 2000 mini-batches
        with torch.no_grad():
            train_acc = test_it(train_loader, set='train')
            test_acc = test_it(test_loader, set = 'test')
            if train_acc == 1:
                break
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss))
        running_loss = 0.0

print('Finished Training')


torch.save(net.state_dict(), PATH)

# correct = 0
# total = 0
# with torch.no_grad():
#     for data in test_loader:
#         images, labels = data
#         outputs = net(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# print('Accuracy of the network on the 10000 test images: %d %%' % (
#     100 * correct / total))
#
# class_correct = list(0. for i in range(num_classes))
# class_total = list(0. for i in range(num_classes))
# with torch.no_grad():
#     for (images, labels) in test_loader:
#         outputs = net(images)
#         _, predicted = torch.max(outputs, 1)
#         c = (predicted == labels).squeeze()
#         for i in range(4):
#             label = labels[i]
#             class_correct[label] += c[i].item()
#             class_total[label] += 1
#
#
# for i in range(10):
#     print('Accuracy of %5s : %2d %%' % (
#         classes[i], 100 * class_correct[i] / class_total[i]))
