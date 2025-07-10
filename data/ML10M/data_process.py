
import json
import os
import pickle
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sp


def generate_dict(path, file):
    user_interaction = {}
    with open(os.path.join(path, file)) as f:
        data = f.readlines()
        for row in data:
            user, item = row.strip().split()
            user, item = int(user), int(item)

            if user not in user_interaction:
                user_interaction[user] = [item]
            elif item not in user_interaction[user]:
                user_interaction[user].append(item)
    return user_interaction


def generate_interact(path):
    buy_dict = generate_dict(path, 'pos.txt')
    with open(os.path.join(path, 'pos_dict.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(buy_dict))

    cart_dict = generate_dict(path, 'neutral.txt')
    with open(os.path.join(path, 'neutral_dict.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(cart_dict))

    collect_dict = generate_dict(path, 'neg.txt')
    with open(os.path.join(path, 'neg_dict.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(collect_dict))

    for dic in [buy_dict, cart_dict]:
        for k, v in dic.items():
            if k not in collect_dict:
                collect_dict[k] = v
            item = collect_dict[k]
            item.extend(v)
    for k, v in collect_dict.items():
        item = collect_dict[k]
        item = list(set(item))
        collect_dict[k] = sorted(item)


    # shutil.copyfile('buy_dict.txt', 'train_dict.txt')

    validation_dict = generate_dict(path, 'validation.txt')
    with open(os.path.join(path, 'validation_dict.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(validation_dict))

    test_dict = generate_dict(path, 'test.txt')
    with open(os.path.join(path, 'test_dict.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(test_dict))


def generate_all_interact(path):
    all_dict = {}
    files = ['neg', 'pos', 'neutral']
    for file in files:
        with open(os.path.join(path, file+'_dict.txt')) as r:
            data = json.load(r)
            for k, v in data.items():
                if all_dict.get(k, None) is None:
                    all_dict[k] = v
                else:
                    total = all_dict[k]
                    total.extend(v)
                    all_dict[k] = sorted(list(set(total)))
        with open(os.path.join(path, 'all.txt'), 'w') as w1, open(os.path.join(path, 'all_dict.txt'), 'w') as w2:
            for k, v in all_dict.items():
                for i in v:
                    w1.write('{} {}\n'.format(int(k), i))
            w2.write(json.dumps(all_dict))


def pos_sampling(path):
    behaviors = ['neg', 'pos', 'neutral']
    with open(os.path.join(path, 'pos_sampling.txt'), 'w') as f:
        for index, file in enumerate(behaviors):
            with open(os.path.join(path, file + '_dict.txt'), encoding='utf-8') as r:
                tmp_dict = json.load(r)
                for k in tmp_dict:
                    for v in tmp_dict[k]:
                        f.write('{} {} {} 1\n'.format(k, v, index))

def item_inter(path, behaviors):
    for behavior in behaviors:
        all_inter = set()
        with open(os.path.join(path, behavior + '_dict.txt')) as f:
            data = json.load(f)
            for v in data.values():
                i = len(v)
                m = 0
                while m < i:
                    n = 0
                    while n < i:
                        all_inter.add((v[m], v[n]))
                        n += 1
                    m += 1
        row = []
        col = []
        for item in all_inter:
            row.append(item[0])
            col.append(item[1])
        indict = len(row)
        item_graph = sp.coo_matrix((np.ones(indict), (row, col)), shape=[8705, 8705])
        item_graph_degree = item_graph.toarray().sum(axis=0).reshape(-1, 1)
        info = {'row': row, 'col': col, 'degree': item_graph_degree.tolist()}
        with open(os.path.join(path, behavior+'_item_graph.txt'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(info))

if __name__ == '__main__':
    # read_data()
    # read_test()
    # split_items()

    generate_interact('.')
    generate_all_interact('.')
    pos_sampling('.')
    # files = ['neg', 'pos', 'neutral']
    # item_inter('.', files)