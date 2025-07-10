
import json
import os
import random
import shutil

import numpy as np
from loguru import logger
import scipy.sparse as sp

seed = 2021
np.random.seed(seed)
random.seed(seed)

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


@logger.catch()
def generate_interact(path):
    buy_dict = generate_dict(path, 'buy.txt')
    with open(os.path.join(path, 'buy_dict.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(buy_dict))

    cart_dict = generate_dict(path, 'cart.txt')
    with open(os.path.join(path, 'cart_dict.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(cart_dict))

    click_dict = generate_dict(path, 'click.txt')
    with open(os.path.join(path, 'click_dict.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(click_dict))

    collect_dict = generate_dict(path, 'collect.txt')
    with open(os.path.join(path, 'collect_dict.txt'), 'w', encoding='utf-8') as f:
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


    validation_dict = generate_dict(path, 'validation.txt')
    with open(os.path.join(path, 'validation_dict.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(validation_dict))

    test_dict = generate_dict(path, 'test.txt')
    with open(os.path.join(path, 'test_dict.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(test_dict))

def generate_all_interact(path):
    all_dict = {}
    files = ['click', 'cart', 'collect', 'buy']
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


def remove_train_interact():
    origin = './origin'
    dest = './clean'
    files = ['click', 'cart', 'collect', 'buy']
    with open(os.path.join(origin, 'test_dict.txt'), encoding='utf-8') as t:
        test_data = json.load(t)
    shutil.copyfile(os.path.join(origin, 'test_dict.txt'), os.path.join(dest, 'test_dict.txt'))
    shutil.copyfile(os.path.join(origin, 'validation_dict.txt'), os.path.join(dest, 'validation_dict.txt'))
    for file in files:
        with open(os.path.join(origin, file + '_dict.txt'), encoding='utf-8') as r:
            tmp_dict = json.load(r)
            for key, val in test_data.items():
                tmp_val = tmp_dict.get(key, None)
                if tmp_val is not None:
                    tmp_val = np.setdiff1d(tmp_val, val)
                    if len(tmp_val) < 1:
                        tmp_dict.pop(key)
                    else:
                        tmp_dict[key] = tmp_val.tolist()
            with open(os.path.join(dest, file + '_dict.txt'), 'w') as w1, open(os.path.join(dest, file + '.txt'), 'w') as w2:
                w1.write(json.dumps(tmp_dict))
                for k, v in tmp_dict.items():
                    for i in v:
                        w2.write("{} {} \n".format(int(k), i))


def pos_sampling(path):
    behaviors = ['click', 'collect', 'cart', 'buy']
    with open(os.path.join(path, 'pos_sampling.txt'), 'w') as f:
        for index, file in enumerate(behaviors):
            with open(os.path.join(path, file + '_dict.txt'), encoding='utf-8') as r:
                tmp_dict = json.load(r)
                for k in tmp_dict:
                    for v in tmp_dict[k]:
                        f.write('{} {} {} 1\n'.format(k, v, index))

def generate_potential_negative_samples(path, total_item=11953, sample_size=20):
    buy_dict_path = os.path.join(path, 'buy_dict.txt')
    output_path = os.path.join(path, 'potential_negative_list.txt')

    with open(buy_dict_path, 'r') as f:
        buy_dict = json.load(f)

    all_users = set(map(int, buy_dict.keys()))
    max_user = max(all_users)

    potential_neg_list = [[] for _ in range(max_user + 1)]
    all_items = set(range(1, total_item + 1))

    for user in range(1, max_user + 1):
        buy_items = set(buy_dict.get(str(user), []))
        candidate_pool = list(all_items - buy_items)

        if len(candidate_pool) == 0:
            negative_candidates = []
        else:
            negative_candidates = random.sample(candidate_pool, min(sample_size, len(candidate_pool)))

        potential_neg_list[user] = sorted(negative_candidates)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(potential_neg_list, f)


if __name__ == '__main__':
    path = '.'
    generate_interact(path)
    generate_all_interact(path)
    pos_sampling(path)
    generate_potential_negative_samples(path)

