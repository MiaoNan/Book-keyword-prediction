import torch
import pandas as pd
import functions as fc
import numpy as np
import os

from torch.autograd import Variable

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_predict_array(pth_path):  # Predict
    net = torch.nn.Transformer(d_model=100, nhead=2, num_encoder_layers=3,
                               num_decoder_layers=3, dim_feedforward=300, dropout=0.1,
                               activation='relu', custom_encoder=None, custom_decoder=None)
    net.load_state_dict(torch.load(pth_path))
    net = net.cuda()
    print('Loading dataset...')
    gen_x = fc.slide_window_dataset(type='gen', part=0)
    gen_x = torch.FloatTensor(gen_x).cuda()
    batch_size = 100
    predicted_array = []
    print('Predicting...')
    sum_count = 0
    for i in range(gen_x.shape[0] - 1):
        if (i + int(batch_size * 1.75)) > (gen_x.shape[0] - 1):
            break
        x = gen_x[i:i + batch_size]
        x = Variable(x).reshape([batch_size, 1, -1])
        y = gen_x[i + int(batch_size * 0.75): i + int(batch_size * 1.75)]
        y = Variable(y).reshape([batch_size, 1, -1])
        output = net(x, y)
        output = torch.squeeze(output.detach().to('cpu'), dim=1)
        predicted_array.append(output.numpy()[-25:, :])
        if i % int(gen_x.shape[0] / 100) == 0:
            print('Predicting, {}% finished...'.format(sum_count))
            sum_count += 1
    predicted_array = np.array(predicted_array)
    word_top, word_vector = top_50_word_vec()
    word_top_count = np.zeros(word_top.__len__())

    # Calculate nearest word
    print('Calculating nearest word')
    sum_count = 0
    for i in range(predicted_array.__len__()):
        array_slice = predicted_array[i]
        for s in range(array_slice.__len__()):
            min_dist = -1
            min_index = -1
            for j in range(word_top.__len__()):
                t = fc.cal_dist1(word_vector[j], array_slice[s])
                if min_dist == -1 or t < min_dist:
                    min_dist = t
                    min_index = j
            word_top_count[min_index] += 1
        if i != 0 and i % int(predicted_array.__len__() / 100) == 0:
            print('Calculating, {}% finished...'.format(sum_count))
            sum_count += 1
    word_top = np.array(word_top)
    word_top_count = np.array(word_top_count)
    result = np.vstack((word_top, word_top_count))
    save_path = data_path + 'predict_result.csv'
    output = pd.DataFrame(result)
    output.to_csv(save_path, header=0, encoding='utf-8')
    print('finished')