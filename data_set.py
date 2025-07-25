
import argparse
import os
import random
import json
import torch
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from torch.utils.data import Dataset, DataLoader
import numpy as np

SEED = 2025
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


class TestDate(Dataset):
    def __init__(self, user_count, item_count, samples=None):
        self.user_count = user_count
        self.item_count = item_count
        self.samples = samples

    def __getitem__(self, idx):
        return int(self.samples[idx])

    def __len__(self):
        return len(self.samples)


class BehaviorDate(Dataset):
    def __init__(self, user_count, item_count, pos_sampling, neg_count, behavior_dict=None, behaviors=None):
        self.user_count = user_count
        self.item_count = item_count
        self.pos_sampling = pos_sampling
        self.behavior_dict = behavior_dict
        self.behaviors = behaviors
        self.neg_count = neg_count

    def __getitem__(self, idx):
        # generate positive and negative samples pairs under each behavior
        total = []

        pos = self.pos_sampling[idx]
        u_id = pos[0]
        total.append(pos)

        all_inter = self.behavior_dict['all'].get(str(u_id), None)
        for i in range(self.neg_count):
            item = random.randint(1, self.item_count)
            while np.isin(item, all_inter):
                item = random.randint(1, self.item_count)
            neg = list(pos)
            neg[1] = item
            neg[-1] = 0
            total.append(neg)

        buy_inter = self.behavior_dict[self.behaviors[-1]].get(str(u_id), None)
        if buy_inter is None:
            signal = [0, 0, 0, 0]
        else:
            p_item = random.choice(buy_inter)
            n_item = random.randint(1, self.item_count)
            while np.isin(n_item, all_inter):
                n_item = random.randint(1, self.item_count)
            signal = [pos[0], p_item, n_item, 0]

        total.append(signal)

        return np.array(total)

    def __len__(self):
        return len(self.pos_sampling)


class DataSet(object):

    def __init__(self, args):

        self.behaviors = args.behaviors
        self.path = args.data_path
        self.loss_type = args.loss_type
        self.neg_count = args.neg_count

        self.__get_count()
        self.__get_pos_sampling()
        self.__get_behavior_items()
        self.__get_validation_dict()
        self.__get_test_dict()
        self.__get_sparse_interact_dict()

        self.validation_gt_length = np.array([len(x) for _, x in self.validation_interacts.items()])
        self.test_gt_length = np.array([len(x) for _, x in self.test_interacts.items()])

    def __get_count(self):
        with open(os.path.join(self.path, 'count.txt'), encoding='utf-8') as f:
            count = json.load(f)
            self.user_count = count['user']
            self.item_count = count['item']

    def __get_pos_sampling(self):
        with open(os.path.join(self.path, 'pos_sampling.txt'), encoding='utf-8') as f:
            data = f.readlines()
            arr = []
            for line in data:
                line = line.strip('\n').strip().split()
                arr.append([int(x) for x in line])
            self.pos_sampling = arr

    def __get_behavior_items(self):
        """
        load the list of items corresponding to the user under each behavior
        :return:
        """
        self.train_behavior_dict = {}
        for behavior in self.behaviors:
            with open(os.path.join(self.path, behavior + '_dict.txt'), encoding='utf-8') as f:
                b_dict = json.load(f)
                self.train_behavior_dict[behavior] = b_dict
        with open(os.path.join(self.path, 'all_dict.txt'), encoding='utf-8') as f:
            b_dict = json.load(f)
            self.train_behavior_dict['all'] = b_dict

    # def __get_all_item_users(self):
    #     with open(os.path.join(self.path, 'all.txt'), encoding='utf-8') as f:

    def __get_test_dict(self):
        """
        load the list of items that the user has interacted with in the test set
        :return:
        """
        with open(os.path.join(self.path, 'test_dict.txt'), encoding='utf-8') as f:
            b_dict = json.load(f)
            self.test_interacts = b_dict

    def __get_validation_dict(self):
        """
        load the list of items that the user has interacted with in the validation set
        :return:
        """
        with open(os.path.join(self.path, 'validation_dict.txt'), encoding='utf-8') as f:
            b_dict = json.load(f)
            self.validation_interacts = b_dict

    def __get_sparse_interact_dict(self):
        """
        load graphs

        :return:
        """

        self.inter_matrix = []
        self.user_item_inter_set = []
        all_row = []
        all_col = []
        for behavior in self.behaviors:
            # tmp_dict = {}
            with open(os.path.join(self.path, behavior + '.txt'), encoding='utf-8') as f:
                data = f.readlines()
                row = []
                col = []
                for line in data:
                    line = line.strip('\n').strip().split()
                    row.append(int(line[0]))
                    col.append(int(line[1]))

                values = torch.ones(len(row), dtype=torch.float32)
                inter_matrix = sp.coo_matrix((values, (row, col)), [self.user_count + 1, self.item_count + 1])
                user_item_set = [list(row.nonzero()[1]) for row in inter_matrix.tocsr()]
                self.inter_matrix.append(inter_matrix)
                self.user_item_inter_set.append(user_item_set)
                all_row.extend(row)
                all_col.extend(col)
        all_edge_index = list(set(zip(all_row, all_col)))
        all_row = [sub[0] for sub in all_edge_index]
        all_col = [sub[1] for sub in all_edge_index]
        values = torch.ones(len(all_row), dtype=torch.float32)
        self.all_inter_matrix = sp.coo_matrix((values, (all_row, all_col)), [self.user_count + 1, self.item_count + 1])

    def behavior_dataset(self):
        return BehaviorDate(self.user_count, self.item_count, self.pos_sampling, self.neg_count, self.train_behavior_dict, self.behaviors)


    def validate_dataset(self):
        return TestDate(self.user_count, self.item_count, samples=list(self.validation_interacts.keys()))

    def test_dataset(self):
        return TestDate(self.user_count, self.item_count, samples=list(self.test_interacts.keys()))



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Set args', add_help=False)
    parser.add_argument('--behaviors', type=list, default=['tip', 'neutral', 'neg', 'pos'], help='')
    parser.add_argument('--data_path', type=str, default='./data/Yelp', help='')
    parser.add_argument('--loss_type', type=str, default='bpr', help='')
    parser.add_argument('--neg_count', type=int, default=4)
    args = parser.parse_args()
    dataset = DataSet(args)

    loader = DataLoader(dataset=dataset.behavior_dataset(), batch_size=5, shuffle=True)
    for index, item in enumerate(loader):
        print(index, '-----', item)
