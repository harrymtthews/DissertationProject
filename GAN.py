import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import os
import pandas as pd

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Is GPU connected?
# assert torch.zeros(1).to(device).device.type=='cuda'

BATCH_SIZE = 16
IMG_SIZE = 32
CHANNELS = 1
IMG_PATH = 'ann-docset-main/datasets/IAMonDo-Images/categories/'
LABELS = ('Bracket', 'Encircling', 'Sideline', 'Underline')

# Find a way to filter out annotations of too high resolution, i.e. max of 64x64
#   - This should help reduce the amount of crap in the dataset.

class AnnotationDataset(torch.utils.data.Dataset):

    IMG_PATH = 'ann-docset-main/datasets/IAMonDo-Images/categories/'
    LABELS = ('Bracket', 'Encircling', 'Sideline', 'Underline')

    def __init__(self, transform=None, target_transform=None):

        self.LABEL_PATHS = (os.path.join(IMG_PATH, f'Marking_{LABELS[i]}') for i in range(len(LABELS)))
        files = []
        for l, lp in zip(LABELS, self.LABEL_PATHS):
            files += [(l, os.path.join(f'Marking_{l}', p.name)) for p in os.scandir(lp)] 
        self.img_labels = pd.DataFrame(files, columns=('label', 'path'))
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(IMG_PATH, self.img_labels.iloc[idx]['path'])
        image = torchvision.io.read_image(img_path)
        label = self.img_labels.iloc[idx]['label']
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

dataset_loader = torch.utils.data.DataLoader(
    AnnotationDataset(transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize([IMG_SIZE, IMG_SIZE]),
        # torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
    ])),
    shuffle=True, batch_size=BATCH_SIZE, drop_last=True
)

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

train_iterator = iter(cycle(dataset_loader))


class Generator(nn.Module):
    def __init__(self, latent_size=100, label_size=4):
        super(Generator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(latent_size+label_size, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, CHANNELS*IMG_SIZE*IMG_SIZE),
            nn.Tanh()
        )
        
    def forward(self, x, c):
        x, c = x.view(x.size(0), -1), c.view(c.size(0), -1).float()
        x = torch.cat((x, c), 1) # [input, label] concatenated
        x = self.layer(x)
        return x.view(x.size(0), CHANNELS, IMG_SIZE, IMG_SIZE)

# define the discriminator
class Discriminator(nn.Module):
    def __init__(self, label_size=4):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(CHANNELS*IMG_SIZE*IMG_SIZE+label_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x, c):        
        x, c = x.view(x.size(0), -1), c.view(c.size(0), -1).float()
        x = torch.cat((x, c), 1) # [input, label] concatenated
        return self.layer(x)
        
G = Generator().to(device)
D = Discriminator().to(device)

print(f'Generator has {len(torch.nn.utils.parameters_to_vector(G.parameters()))} parameters.')
print(f'Discriminator has {len(torch.nn.utils.parameters_to_vector(D.parameters()))} parameters')

# initialise the optimiser
optimiser_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimiser_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = nn.BCELoss()

epoch = 0
# training loop
while (epoch < 20000):
    
    # arrays for metrics
    logs = {}
    gen_loss_arr = np.zeros(0)
    dis_loss_arr = np.zeros(0)

    # iterate over the train dateset
    for i, batch in enumerate(dataset_loader):

        x, t = batch
        x = x.to(device)
        t = torch.tensor(tuple((LABELS.index(t[i]) for i in range(BATCH_SIZE)))).to(device)
        y = torch.zeros(x.size(0), len(LABELS)).long().to(device).scatter(1, t.view(x.size(0),1), 1)

        z = torch.randn(x.size(0), 100).to(device)
        l_r = criterion(D(x,y), torch.ones([BATCH_SIZE,1]).to(device)) # real -> 1
        l_f = criterion(D(G(z,y),y), torch.zeros([BATCH_SIZE,1]).to(device)) # fake -> 0
        loss_d = (l_r + l_f)/2.0
        optimiser_D.zero_grad()
        loss_d.backward()
        optimiser_D.step()
        
        # train generator
        z = torch.randn(x.size(0), 100).to(device)
        loss_g = criterion(D(G(z,y),y), torch.ones([BATCH_SIZE,1]).to(device)) # fake -> 1
        optimiser_G.zero_grad()
        loss_g.backward()
        optimiser_G.step()

        gen_loss_arr = np.append(gen_loss_arr, loss_g.item())
        dis_loss_arr = np.append(dis_loss_arr, loss_d.item())

    # conditional sample of 10x10
    G.eval()
    print('loss d: {:.3f}, loss g: {:.3f}'.format(gen_loss_arr.mean(), dis_loss_arr.mean()))
    grid = np.zeros([IMG_SIZE*10, IMG_SIZE*10])
    for j in range(10):
        c = torch.zeros([10, 10]).to(device)
        c[:, j] = 1
        z = torch.randn(10, 100).to(device)
        y_hat = G(z,c).view(10, IMG_SIZE, IMG_SIZE)
        result = y_hat.cpu().data.numpy()
        grid[j*IMG_SIZE:(j+1)*IMG_SIZE] = np.concatenate([x for x in result], axis=-1)
    plt.grid(False)
    plt.imshow(grid, cmap='gray')
    plt.show()
    plt.pause(0.0001)
    G.train()

    epoch = epoch+1
