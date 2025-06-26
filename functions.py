import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cfg = config.Cfgs()
base_path = cfg.base_path
data_path = cfg.data_path


def data_saver(data, save_path):
    data.to_csv(save_path, encoding='utf-8_sig')
    print("data have been saved to {}".format(save_path))


def clear_spilted_title(set_list):
    for i in set_list:
        while '"' in i:
            i.remove('"')
        while '?' in i:
            i.remove('?')
        while '/' in i:
            i.remove('/')
        while '\\' in i:
            i.remove('\\')
        while '\'' in i:
            i.remove('\'')
        while ',' in i:
            i.remove(',')
        while '.' in i:
            i.remove('.')
        while ')' in i:
            i.remove(')')
        while '(' in i:
            i.remove('(')
        while '-' in i:
            i.remove('-')
        while '，' in i:
            i.remove('，')
        while '。' in i:
            i.remove('。')
        while '？' in i:
            i.remove('？')
        while '！' in i:
            i.remove('！')
        while '：' in i:
            i.remove('：')
        while ':' in i:
            i.remove(':')
        while ' ' in i:
            i.remove(' ')
        while '<' in i:
            i.remove('<')
        while '>' in i:
            i.remove('>')
        while '《' in i:
            i.remove('《')
        while '》' in i:
            i.remove('》')
        return set_list


def cal_dist1(x, y):
    sum = 0
    for i in range(x.shape[0]):
        t = x[i] - y[i]
        if t < 0:
            t = -t
        sum += t
    return sum


def cal_word_fre(base_path):
    file_path = data_path + 'merged-utf-8.csv'
    data = pd.read_csv(file_path, header=0, encoding='utf-8')
    book_title = data['图书题名']  # 1687835
    book_title = set(book_title)  # 113186
    # book_title = np.array(book_title)
    # book_title.remove(None)

    set_list = []
    for i in book_title:
        if (i != i):
            continue
        set_list.extend(jieba.lcut(i, cut_all=False))

    set_list = clear_spilted_title(set_list)

    dict = {}

    for i in set_list:
        if i in dict.keys():
            dict[i] += 1
        else:
            dict[i] = 1

    with open(data_path + 'word_frequency.csv', 'w', encoding='utf-8_sig') as f:
        [f.write('{0},{1}\n'.format(key, value)) for key, value in dict.items()]


def press_fre(base_path):  # Calculate publisher frequency when book name match the top60 keywords
    file_path = data_path + 'merged-utf-8.csv'
    data = pd.read_csv(file_path, index_col=0, encoding='utf-8')
    file_path = data_path + 'word_frequency_top_50.csv'
    word_top = pd.read_csv(file_path, header=0, encoding='utf-8')
    word_top_key = word_top['name']
    word_top_key[0] = '中国'

    press = data['图书出版社']  # Press name
    for i in range(press.__len__()):
        if press[i] != press[i]:  # to decide it is nan or not.
            press[i] = 0
    press = np.array(press)

    book_title = data['图书题名'].astype(str)
    # book_title = np.array(book_title)  # 489809
    # book_title = np.unique(book_title)  # 187498

    set_list = []
    for i in book_title:
        set_list.append(jieba.lcut(i, cut_all=False))

    pub_dict = {}
    for i in range(set_list.__len__()):
        for name in set_list[i]:
            for key_word in word_top_key:
                if name == key_word:
                    pub_name = press[i]
                    if pub_name in pub_dict.keys():
                        pub_dict[pub_name] += 1
                    else:
                        pub_dict[pub_name] = 1

    keyword_pub_count = np.zeros((1085, 52))
    keyword_pub_count = keyword_pub_count.tolist()
    pub_dict = list(pub_dict)
    j = 1
    for i in pub_dict:
        keyword_pub_count[j][0] = i
        j += 1
    j = 1
    for i in word_top_key:
        keyword_pub_count[0][j] = i
        j += 1
    for i in range(set_list.__len__()):
        print(i)
        for name in set_list[i]:
            for a in range(word_top_key.__len__()):
                if name == word_top_key[a]:
                    pub_name = press[i]
                    for b in range(pub_dict.__len__()):
                        if pub_name == pub_dict[b]:
                            keyword_pub_count[b + 1][a + 1] += 1
    # keyword_pub_count = keyword_pub_count.tolist()
    # for i in range(word_top_key.__len__()):
    #     keyword_pub_count[0][i + 1] = word_top_key[i]
    # for i in range(pub_dict.__len__()):
    #     keyword_pub_count[i + 1][0] = pub_dict[i]
    # keyword_pub_count = pd.DataFrame(keyword_pub_count)

    save_path = data_path + 'word_frequency_count_press_50.csv'
    keyword_pub_count.to_csv(save_path, encoding='utf-8_sig')


def merge_to_one(base_path):  # Merge 2013-2020 8 files into one file.
    i = 0
    save_path = data_path + 'merged-utf-8.csv'
    data_name = ['1_2017', '2_2018', '3_2019', '4_2020', '5_2021']
    for csv_name in data_name:
        print(csv_name)
        if i == 0:
            file_path = data_path + csv_name + '.csv'
            data = pd.read_csv(file_path, header=0, encoding='utf-8')
            column = data.columns
            data = np.array(data)
            merge = data
            i += 1
            continue
        data = pd.read_csv(file_path, header=0, encoding='utf-8')
        input = np.array(data)
        merge = np.vstack((merge, input))
    output = pd.DataFrame(merge, columns=column)
    output.to_csv(save_path)


def slide_window_dataset(type, part, window_len=7, divided_part=10):  # 1687835
    file_path = data_path + 'book_title_vector.csv'
    data = pd.read_csv(file_path, header=0, encoding='utf-8')
    data = np.array(data)
    data = np.delete(data, 0, 1)  # delete first column
    slide_window_data = []
    if type == 'origin':
        return data[-370794:-70794, :], data[-70794:-60794, :]
    elif type == 'train':
        count = 0
        data_len = data.shape[0] - 10000
        for i in range(int(data_len / divided_part) * part, (int(data_len / divided_part) * (part+1) - window_len)):
            mat = data[i:i + window_len, ]
            vec = mat.flatten()
            slide_window_data.append(vec)
            if i != 0 and i % int(data_len / divided_part / 10) == 0:
                count += 10
                print('loading {}% finished'.format(count))
        print('loading part {} finished'.format(part + 1))
        # slide_window_data = torch.Tensor(slide_window_data).cuda()
    elif type == 'test':
        for i in range(1677835, 1687835):
            mat = data[i:i + window_len, ]
            vec = mat.flatten()
            slide_window_data.append(vec)
    elif type == 'gen':
        return data[-150000:, :]
    else:
        exit(0)
    slide_window_data = np.array(slide_window_data)
    return slide_window_data


    '''
    while i <= 20:
        if i == 13:
            file_path = data_path + str(i) + '-utf-8.csv'
            data = pd.read_csv(file_path, header=0, encoding='utf-8')
            column = data.columns
            data = np.array(data)
            merge = data
            print(i)
            i += 1
            continue
        data = pd.read_csv(file_path, header=0, encoding='utf-8')
        input = np.array(data)
        merge = np.vstack((merge, input))
        print(i)
        i += 1
    output = pd.DataFrame(merge, columns=column)
    output.to_csv(save_path)
    '''

def draw_bar():
    file_path = data_path + 'word_frequency_count_press_50_sort.csv'
    data = pd.read_csv(file_path, header=0, encoding='utf-8')
    data = np.array(data)
    data = np.delete(data, 0, axis=0)  # delete first row
    # pub_num = data[:,0]
    pub_count = data[:,2]
    pub_num = np.arange(1, 1085, 1)
    pub_count = pub_count.astype(int)

    fig = plt.figure()
    # 设置坐标轴名称
    plt.xlabel('Publisher')
    plt.ylabel('Count')
    my_x_ticks = np.arange(0, 1100, 100)
    my_y_ticks = np.arange(0, 50000, 5000)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    plt.bar(pub_num, pub_count, align='edge', alpha=0.7)
    plt.show()
    plt.savefig('./test2.jpg')




