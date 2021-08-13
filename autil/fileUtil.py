import pickle

def savepickle(path,data):
    f = open(path, 'wb')
    pickle.dump(data, f)
    f.close()

def loadpickle(path):
    with open(path, "rb+") as f:
        data = pickle.load(f)
    return data

def load_triples_list(file_path):
    '''
    :param file_path:
    :return:
    '''
    new_list = []
    with open(file_path, encoding='utf-8', mode='r') as f:
        for line in f:
            th = line[:-1].split('\t')
            th = [int(i) for i in th]
            new_list.append(th)

    return new_list


def load_link_list(file_path):
    '''
    :param file_path:
    :return:
    '''
    new_list = []
    with open(file_path, encoding='utf-8', mode='r') as f:
        for line in f:
            th = line[:-1].split('\t')  # '-' 改为\t
            new_list.append((int(th[0]), int(th[1])))

    return new_list
