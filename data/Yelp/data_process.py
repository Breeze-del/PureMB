
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

    click_dict = generate_dict(path, 'tip.txt')
    with open(os.path.join(path, 'tip_dict.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(click_dict))

    collect_dict = generate_dict(path, 'neg.txt')
    with open(os.path.join(path, 'neg_dict.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(collect_dict))

    for dic in [buy_dict, cart_dict, collect_dict]:
        for k, v in dic.items():
            if k not in click_dict:
                click_dict[k] = v
            item = click_dict[k]
            item.extend(v)
    for k, v in click_dict.items():
        item = click_dict[k]
        item = list(set(item))
        click_dict[k] = sorted(item)

    with open(os.path.join(path, 'all_train_interact_dict.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(click_dict))

    # shutil.copyfile('buy_dict.txt', 'train_dict.txt')

    validation_dict = generate_dict(path, 'validation.txt')
    with open(os.path.join(path, 'validation_dict.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(validation_dict))

    test_dict = generate_dict(path, 'test.txt')
    with open(os.path.join(path, 'test_dict.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(test_dict))


def generate_all_interact(path):
    all_dict = {}
    files = ['neg', 'pos', 'tip', 'neutral']
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
    behaviors = ['neg', 'pos', 'tip', 'neutral']
    with open(os.path.join(path, 'pos_sampling.txt'), 'w') as f:
        for index, file in enumerate(behaviors):
            with open(os.path.join(path, file + '_dict.txt'), encoding='utf-8') as r:
                tmp_dict = json.load(r)
                for k in tmp_dict:
                    for v in tmp_dict[k]:
                        f.write('{} {} {} 1\n'.format(k, v, index))

if __name__ == '__main__':
    # read_data()
    # read_test()
    # split_items()
    files = ['neg', 'pos', 'tip', 'neutral']
    generate_interact('.')
    generate_all_interact('.')
    pos_sampling('.')