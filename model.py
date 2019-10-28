import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import csv
import cv2
import numpy as np

class SdcSimDataset(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path
        self.lines = []
        with open(root_path + 'driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                self.lines.append(line)

    def __len__(self):
        return len(self.lines)

    def get_image(self, path, base_path='../data/'):
        # load image and conver to RGB
        filename = path.split('/')[-1]
        path = base_path + 'IMG/' + filename
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def preprocess(self, image):
        image = image[70:135, :]
        # cv2.imshow("cropped", image)
        # cv2.waitKey(0)
        # image = cv2.resize(image, (32, 32))
        image = image / 255.0 - 0.5
        return image

    def __getitem__(self, idx):
        images = []
        angles = []
        line = self.lines[idx]
        angle = float(line[3])
        correction = 0.2

        # CENTER
        image = self.get_image(line[0], self.root_path)
        image = self.preprocess(image)
        images.append(image.transpose(2,0,1))
        angles.append(angle)

        # Augmenting images by flipping across y-axis
        images.append(cv2.flip(image, 1).transpose(2,0,1))
        angles.append(-angle)

        # LEFT
        image = self.get_image(line[1], self.root_path)
        image = self.preprocess(image)
        images.append(image.transpose(2,0,1))
        angles.append(angle + correction)

        # RIGHT
        image = self.get_image(line[2], self.root_path)
        image = self.preprocess(image)
        images.append(image.transpose(2,0,1))
        angles.append(angle - correction)

        # X_train = torch.Tensor(np.stack(images))
        # y_train = torch.Tensor(angles)
        # breakpoint()
        # sample = {'image': torch.from_numpy(np.stack(images)),
        #         'angles': torch.from_numpy(np.stack(angles))}
        sample = (images, angles)

        return sample

# TODO
# create the network
class Net(nn.Module):
   def __init__(self):
        super(Net, self).__init__()
        self.conv_layers = nn.Sequential(
            # input is batch_size x 3 x 66 x 200
            nn.Conv2d(3, 24, 5, stride=2, bias=False),
            #nn.ELU(0.2, inplace=True),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2, bias=False),
            nn.ELU(),
            nn.BatchNorm2d(36),
            
            nn.Conv2d(36, 48, 5, stride=2, bias=False),
            nn.ELU(),
            nn.BatchNorm2d(48),
            
            nn.Conv2d(48, 64, 3, stride=1, bias=False),
            nn.ELU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 64, 3, stride=1, bias=False),
            nn.ELU(),
            nn.Dropout(p=0.4)
        )
        self.linear_layers = nn.Sequential(
            #input from sequential conv layers
            nn.Linear(in_features=2112, out_features=100, bias=False),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50, bias=False),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10, bias=False),
            nn.ELU(),
            nn.Linear(in_features=10, out_features=1, bias=False))
        # self._initialize_weights()

        
   # def _initialize_weights(self):
   #      for m in self.modules():
   #          if isinstance(m, nn.Conv2d):
   #              n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
   #              init.normal(m.weight, mean=0, std=0.02)
   #          elif isinstance(m, nn.BatchNorm2d):
   #              init.normal(m.weight, mean=1, std=0.02)
   #              init.constant(m.bias, 0)

   def forward(self, input):
        print(input.size())
        output = self.conv_layers(input)
        output = output.view(output.size(0), 2112)
        output = self.linear_layers(output)
        return output



def collate_fn(batch):
    images = []
    angles = []
    for each in batch:
        images.extend(each[0])
        angles.extend(each[1])
    return torch.from_numpy(np.stack(images)).float(), torch.from_numpy(np.stack(angles)).float().unsqueeze(-1)

if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True
    dataset = SdcSimDataset('/home/shreyansh/Documents/CarND-Behavioral-Cloning-P3/recording data/')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_fn)

    net = Net()
    if torch.cuda.is_available():
        net = net.cuda()
    # TODO
    # Create loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = 0.0001)
    for epoch in range(15):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, (images, angles) in enumerate(dataloader):
            if torch.cuda.is_available():
                images = images.cuda()
                angles = angles.cuda()

            optimizer.zero_grad()
            outputs = net (images)
            loss = criterion(outputs,angles)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            print_every = 10
            if i % print_every == print_every - 1:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / print_every))
                running_loss = 0.0

    torch.save(net, 'model.h5')
    print('modelCreated')
