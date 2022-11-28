import torch
import torch.nn as nn
from torch import optim
from torch.autograd.variable import Variable

from torchvision import transforms
import torchvision.datasets as datasets

from Generator_Discriminator import Discriminator
from Generator_Discriminator import Generator

import matplotlib.pyplot as plt

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# load dataset
# normalizing images between -1 to 1
mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([.5], [.5])
        ]))

batch_size = 100

# creating dataloader
data_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)

# number of batches
no_batches = len(data_loader)

# defining the discriminator
discriminator = Discriminator()

# defining the generator
generator = Generator()

# loss
criterion = nn.BCELoss()

# optimizers
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

ones_target = Variable(torch.ones(batch_size, 1))
zeros_target = Variable(torch.zeros(batch_size, 1))

if device.type == 'cuda':
    discriminator.cuda()
    generator.cuda()
    ones_target = ones_target.to(device)
    zeros_target = zeros_target.to(device)

# training
no_epochs = 1000
# Create logger instance
for epoch in range(no_epochs):
    for itr, (real_img, _) in enumerate(data_loader):

        ### Train Discriminator

        # generating a 1-d vector of gaussian sampled random values
        rand_noise = Variable(torch.randn(batch_size, 128))

        if device.type == 'cuda':
            rand_noise = rand_noise.to(device)
            real_img = real_img.to(device)

        fake_img = generator(rand_noise).detach()

        # Reset gradients
        d_optimizer.zero_grad()

        # 1.1 Train on Real Data
        prediction_real = discriminator(real_img)
        # Calculate error and backpropagate
        error_real = criterion(prediction_real, ones_target)
        error_real.backward()

        # 1.2 Train on Fake Data
        prediction_fake = discriminator(fake_img.view(batch_size, 1, 28, 28))
        # Calculate error and backpropagate
        error_fake = criterion(prediction_fake, zeros_target)
        error_fake.backward()

        # 1.3 Update weights with gradients
        d_optimizer.step()

        d_error = error_real + error_fake


        ### Train Generator
        # Generate fake data
        fake_img = generator(rand_noise)

        # Reset gradients
        g_optimizer.zero_grad()

        # Sample noise and generate fake data
        prediction = discriminator(fake_img.view(batch_size, 1, 28, 28))

        # Calculate error and backpropagate
        g_error = criterion(prediction, ones_target)
        g_error.backward()  # Update weights with gradients
        g_optimizer.step()  # Return error

        print("Epoch: {}, Itr: {}, Discriminator error: {}, Generator error: {}"
              .format(epoch, itr, d_error, g_error))

        if itr % 100 == 99:
            rand_noise = Variable(torch.randn(10, 128))

            if device.type == 'cuda':
                rand_noise = rand_noise.to(device)

            test_images = generator(rand_noise).view(10, 1, 28, 28)
            test_images = test_images.data

            # visualize results
            fig = plt.figure(figsize=(20, 10))
            for i in range(0, 10):
                img = transforms.ToPILImage(mode='L')(test_images[i].squeeze(0).detach().cpu())
                fig.add_subplot(2, 5, i+1)
                plt.imshow(img)
            plt.savefig('gen_images/gen_'+str(epoch)+'_'+str(itr)+'.png')
