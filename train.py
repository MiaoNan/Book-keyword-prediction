import torch
import time
import torch.nn.init as init
import torch.nn as nn
import functions as fc
from torch.autograd import Variable
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Training the FC to predict word vector
def train_slide(pth_path = None):
    print('Initializing networks')
    net = torch.nn.Transformer(d_model=700, nhead=7, num_encoder_layers=6,
                               num_decoder_layers=6, dim_feedforward=2100, dropout=0.1,
                               activation='relu', custom_encoder=None, custom_decoder=None)


    if not pth_path is None:
        net.load_state_dict(torch.load(pth_path))
    net = net.cuda()
    for param in net.parameters():
        init.normal_(param, mean=0, std=0.01)
    # loss
    criterion = nn.MSELoss(size_average=False).cuda()
    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
    batch_size = 100  # backward each 100 data
    epochs = 25
    divided_part = 10
    print('Loading test set...')
    test_x = fc.slide_window_dataset(type='test', part=None, window_len=7, divided_part=None)
    for epoch in range(epochs):
        net.train()
        start = time.time()
        for part in range(divided_part):
            print("Training...part {} of {}".format(part + 1, divided_part))
            train_x = fc.slide_window_dataset(type='train', part=part, window_len=7, divided_part=divided_part)
            train_x = torch.FloatTensor(train_x).cuda()
            # training
            for i in range(train_x.shape[0] - 1):
                x = train_x[i]
                x = Variable(x).reshape([1, 1, -1])
                y = train_x[i + 1]
                y = Variable(y).reshape([1, 1, -1])
                output = net(x, y)
                loss = criterion(output, y)  # set im to y label
                # back propagation
                running_loss = 0.0
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if (i % 100 == 0):
                    print('training epoch {}, part {} of {}, data {} of {}, loss: {:.4f}'.format(epoch, part,
                                                                                                 divided_part,
                                                                                                 int(i / batch_size),
                                                                                                 train_x.shape[0] + 1,
                                                                                                 running_loss /
                                                                                                 train_x.shape[0] - 1))
                i += 1
        # Validate
        net.eval()
        for i in range(test_x.shape[0] - 1):
            x = test_x[i]
            x = Variable(x)
            output = net(x)
            y = test_x[i + 1]
            y = Variable(y)
            loss = criterion(output, y)  # set im to y label
            running_loss = 0.0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('testing epoch {}, loss: {:.4f}'.format(epoch, running_loss / 2000))
        end = time.time()
        print("time cost: %f" % (end - start))