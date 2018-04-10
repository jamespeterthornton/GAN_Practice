from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

batchSize = 64
imageSize = 64
num_epochs = 200

transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

dataset = dset.CIFAR10(root='./data', download=True, transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batchSize, num_workers=2)

def weights_init(m):
    classname=m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.main(x)
        return x 

netG = G()
netG.cuda()
netG.apply(weights_init)

class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        outputs = self.main(inputs)
        return outputs.view(-1)

netD = D()
netD.cuda()
netD.apply(weights_init)

criterion = nn.BCELoss()
optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas = (0.5, 0.999))

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        netD.zero_grad()
        real, _ = data
        input = Variable(real.cuda())
        target = Variable(torch.ones(input.size()[0]).cuda())
        output = netD(input)
        errD_real = criterion(output, target)

        noise = Variable(torch.randn(input.size()[0], 100, 1, 1).cuda())
        fake = netG(noise)
        target = Variable(torch.zeros(input.size()[0]).cuda())
        output = netD(fake.detach())
        errD_fake = criterion(output, target)
       
        errD = errD_fake + errD_real 
        errD.backward()
        optimizerD.step()

        netG.zero_grad()
        target = Variable(torch.ones(input.size()[0]).cuda())
        output = netD(fake)
        errG = criterion(output, target)
        errG.backward()
        optimizerG.step()

        if i % 100 == 0:
            print("[%d/%d][%d/%d] Loss_D: %.4f, Loss_G: %.4f" % (epoch, num_epochs, i, len(dataloader), errD.data[0], errG.data[0]))
            vutils.save_image(real, "%s/real_samples.png" % "./results", normalize = True)
            fake = netG(noise)
            vutils.save_image(fake.data, "%s/fake_samples_epoch%03d.png" % ("./results", epoch), normalize = True)










