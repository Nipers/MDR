import numpy as np
import torch  
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *
import pdb, os, time
import pickle
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import math, random
from collections import Counter


domain_indicator = '301'
columns = ['301', '205', '206', '207', '216', '101', '121', '122', '124', '125', '126', '127', '128', '129']
spec_cols = ['210_0', '210_1', '210_2', '210_3', '210_4']
item_cols = ['205', '206', '207', '216']

device = 'cuda:{}'.format(int(time.time())%8)
domain_list = [1, 0, 2]
LR = 1e-4#  / 100

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embed_size = 10
        self.domain_list = domain_list
        self.backbone_shape = [self.embed_size * 14, 256, 128, 64]
        self.keep_prob = 0.9
        self.lr = LR
        self.pertubation = 1e-1
        self._lambda = 1e-4
        self.conf_orth = 1e-1
        self.conf_backbone = 1
        self.conf_bias = 1e-3
        self.conf_orth_loss = 1e-5
        self.conf_concat = 1

        self.pertubation, self.conf_orth, self.conf_backbone, self.conf_bias, self.conf_orth_loss, self.conf_concat = 1e-1, 1e-1, 1, 1e-5, 1e-3, 1

        self.pertubation = torch.tensor(1e-1, requires_grad = True)
        self.conf_orth = torch.tensor(1e-1, requires_grad = True)
        self.conf_backbone = torch.tensor(1.0, requires_grad = True)
        self.conf_bias = torch.tensor(1e-5, requires_grad = True)
        self.conf_orth_loss = torch.tensor(1e-3, requires_grad = True)
        self.conf_concat = torch.tensor(1.0, requires_grad = True)

        self._lambda = torch.tensor(1e-4, requires_grad = True)

        self.embedding_layer = nn.Embedding(921445, self.embed_size)

        self.backbone = self.mlp(self.backbone_shape)
        self.bias = self.mlp(self.backbone_shape)

        self.backbone_cls = nn.Linear(self.backbone_shape[-1], 1)# .uniform_(-0.001,0.001)
        self.bias_cls = nn.Linear(self.backbone_shape[-1], len(self.domain_list))
        self.concat_cls = nn.Linear(self.backbone_shape[-1]*2, 1)

        self.adj = self.mlp([self.backbone_shape[-1], 16])
        
        self.expert_domain = nn.ParameterList()
        self.meta = nn.ModuleList()
        self.domain_cls = nn.ModuleList()
        self.bn = nn.ModuleList()
        for idx in self.domain_list:
            self.expert_domain.append(torch.FloatTensor(self.embed_size * 5, len(self.domain_list), 1).uniform_(-0.001, 0.001))
            self.meta.append(self.mlp([len(self.domain_list), 256]))
            self.domain_cls.append(self.mlp([16,1], True))

            self.bn.append(nn.BatchNorm1d(len(self.domain_list), eps=1e-3, momentum=0.01))
        
        self.sigmoid_loss = nn.BCEWithLogitsLoss()
        self.softmax_loss = nn.CrossEntropyLoss()

        self.dropout = nn.Dropout(1-self.keep_prob)


    def mlp(self, net_shape, is_out = False):
        tmp = nn.ModuleList()
        for idx in range(len(net_shape)-1):
            tmp.append(nn.LeakyReLU())

            temp_param = nn.Linear(net_shape[idx], net_shape[idx+1])
            torch.nn.init.uniform_(temp_param.weight, a=-0.001, b=0.001)
            torch.nn.init.constant_(temp_param.bias, 0)

            tmp.append(temp_param)

            tmp.append(nn.BatchNorm1d(net_shape[idx+1], eps=1e-3, momentum=0.01))
            if not is_out:
                tmp.append(nn.Dropout(1-self.keep_prob))
        return nn.Sequential(*tmp)

    def get_embed(self, features, mode):
        if mode == 'domain':
            feat_embed = self.embedding_layer(features['301'])
        elif mode == 'feature':
            feat_embed = []
            for key in columns:
                if key == domain_indicator:
                    continue
                feat_embed.append(self.embedding_layer(features[key]))
            multi_embed = torch.zeros(features['301'].shape[0], 10).to(device)
            for feat in spec_cols:
                multi_embed += self.embedding_layer(features[feat])
            multi_embed /= 5
            feat_embed.append(multi_embed)
            feat_embed = torch.cat(feat_embed, 1)
        elif mode == 'item':
            feat_embed = []
            for key in item_cols:
                feat_embed.append(self.embedding_layer(features[key]))
            feat_embed = torch.cat(feat_embed, 1)
        return feat_embed
    
    def get_expert(self, embed, idx):
        m = nn.LeakyReLU()
        hidden_output = torch.einsum('ij,jkl->ikl',[m(embed), self.expert_domain[idx]])
        if hidden_output.shape[0] > 1:
            hidden_output = self.bn[idx](hidden_output)
        if self.training:
            hidden_output = self.dropout(hidden_output)
        return hidden_output

    def calc_orth(self, grads):
        vi = grads[0]
        orth_list = [vi]
        for i in range(1, len(grads)):
            vi = grads[i] - torch.sum(torch.mul(grads[i], vi)) / torch.sum(torch.mul(vi, vi)) * vi
            orth_list.append(vi)
        return orth_list

    def forward(self, x, click):

        feature_embed = self.get_embed(x, 'feature')
        domain_rel_embed = self.get_embed(x, 'domain')
        item_embed = self.get_embed(x, 'item')

        domain_rel_embed_re = domain_rel_embed.unsqueeze(1)

        domain_rel_embed_re = domain_rel_embed_re.repeat(1,4,1).reshape([-1, 40])

        inter_embed = torch.multiply(item_embed, domain_rel_embed_re)
        backbone_out = self.backbone(feature_embed)
        
        cls_id = [25, 27]

        backbone_cls_weight, bias_cls_weight = None, None
        for idx,param in enumerate(self.parameters()):
            if idx == cls_id[0]:
                backbone_cls_weight = param
            elif idx == cls_id[1]:
                bias_cls_weight = param

        if self.training:
            backbone_logit = self.backbone_cls(backbone_out)
            with torch.no_grad():
                backbone_pred = torch.sigmoid(backbone_logit).view(-1)
            backbone_grad = torch.matmul(torch.reshape(backbone_pred - click, [-1,1]), backbone_cls_weight)
            _backbone_grad = backbone_out - self.lr * backbone_grad


            adj_backbone = self.adj(_backbone_grad)
        else:
            adj_backbone = self.adj(backbone_out)

        bias_out = self.bias(feature_embed)
        # pdb.set_trace()
        bias_weight = F.softmax(self.bias_cls(bias_out), dim=1)

        concat_embed = torch.concat([domain_rel_embed, inter_embed], axis=1)

        # adj_backbone = adj_backbone.unsqueeze(-1)


        domain_out = {}

        for idx in self.domain_list:
            idx_tensor = torch.tensor(idx).to(device)
            domain_mask = torch.eq(x[domain_indicator], idx_tensor)

            domain_mask = domain_mask.unsqueeze(1)
            # 当前batch下，样本领域为idx的样本中，领域分类情况 和 domain的embed
            domain_bias_weight = torch.masked_select(bias_weight, domain_mask).view(-1, len(self.domain_list))

            domain_embed = torch.masked_select(concat_embed, domain_mask).view(-1, concat_embed.shape[-1])


            expert_list = []
            for _idx in self.domain_list:
                # 将idx领域的embed，用_idx领域的特征提取器来提取，f^_idx(S^idx)，未激活
                if idx == _idx:
                    # [part_batch, len(domain), 1] = [part_batch, domain_rel_embed] * [embed_size, len(domain), 1]
                    domain_expert = self.get_expert(domain_embed, _idx)
                else:
                    with torch.no_grad():
                        domain_expert = self.get_expert(domain_embed, _idx)
                expert_list.append(domain_expert)

            # [part_batch, len(domain), len(domain)]
            domain_all_expert = torch.concat(expert_list, axis=2)

            # 当前batch下，样本领域为idx的样本中，领域分类情况，[part_batch, 1, len(domain)]
            domain_gate = domain_bias_weight.unsqueeze(axis=1)

            # [part_batch, len(domain), len(domain)]
            domain_gate = [domain_gate for _ in range(len(self.domain_list))]
            domain_gate = torch.concat(domain_gate, axis=1)

            # TransNet，[part_batch, len(self.domain_list)]
            domain_expert_out = torch.sum(torch.mul(domain_gate, domain_all_expert), axis=2)

            # 当前batch下，idx领域的样本中，形变后的XIR，[part_batch, 16, 1]
            # pdb.set_trace()
            adj_backbone_fltr = torch.masked_select(adj_backbone, domain_mask).view(-1, adj_backbone.shape[-1])
            adj_backbone_fltr = adj_backbone_fltr.unsqueeze(-1)

            # 形变后的TransNet，[part_batch, 256] -> [part_batch, 16, 16]
            domain_meta = self.meta[idx](domain_expert_out)
            domain_meta = domain_meta.view([-1, 16, 16])

            # TransNet * backbone_out = [part_batch, 16, 16] -> [part_batch, 1, 16]
            domain_meta_out = torch.sum(torch.mul(adj_backbone_fltr, domain_meta), axis=1)
            
            # 用idx领域的分类器分类
            domain_out[idx] = self.domain_cls[idx](domain_meta_out).view(-1)
        
        # argumentation，算loss3
        # [batch, len(domain)]


        backbone_logit = self.backbone_cls(backbone_out)
        bias_logit = self.bias_cls(bias_out)
        bias_label = F.one_hot(x[domain_indicator], len(self.domain_list)).to(device)

        with torch.no_grad():
            # [batch, 128] = [batch, len(domain)] * [len(domain), 128]
            grad_aug = torch.matmul(F.softmax(bias_logit, dim=1) - bias_label, bias_cls_weight)
        # [batch, 128]
        grad_aug_norm = self.pertubation * F.normalize(grad_aug, p=2, dim=1)
        # [batch, 1]
        # pdb.set_trace()
        ratio = torch.Tensor(grad_aug_norm.shape[0], 1).uniform_(-1,1).to(device)
        aug_bias = ratio * grad_aug_norm

        concat_aug_embed = torch.concat([backbone_out, bias_out + aug_bias], axis=1)
        concat_aug_logit = self.concat_cls(concat_aug_embed).view(-1)

        
        # concat_aug_logit = self.concat_cls(concat_embed).view(-1)

        # loss part
        '''
        _w = self.state_dict()['backbone.1.weight'].T
        for i in range(1, len(self.backbone_shape)-1):
            _w = torch.matmul(_w, self.state_dict()['backbone.{}.weight'.format(4*i+1)].T)
        _w = torch.matmul(_w, self.state_dict()['backbone_cls.weight'].T)

        _w_bias = self.state_dict()['bias.1.weight'].T
        for j in range(1, len(self.backbone_shape)-1):
            _w_bias = torch.matmul(_w_bias, self.state_dict()['bias.{}.weight'.format(4*j+1)].T)
        _w_bias = torch.matmul(_w_bias, self.state_dict()['bias_cls.weight'].T)
        '''

        backbone_weight_id = [1,5,9, 25]
        bias_weight_id = [13,17,21, 27]
        cls_id = [25, 27]

        _w_backbone, _w_bias = None, None
        # for idx, param in enumerate(self.parameters()):
        #     print(idx, param.shape)
        # pdb.set_trace()
        for idx,param in enumerate(self.parameters()):
            if idx in backbone_weight_id:
                if _w_backbone == None:
                    _w_backbone = param.T
                else:
                    _w_backbone = torch.matmul(_w_backbone, param.T)
            if idx in bias_weight_id:
                if _w_bias == None:
                    _w_bias = param.T
                else:
                    _w_bias = torch.matmul(_w_bias, param.T)

        with torch.no_grad():
            yd_pred = torch.sigmoid(backbone_logit).view(-1)
        backbone_grad = torch.matmul(torch.reshape(yd_pred-click, [-1,1]), _w_backbone.T)
        backbone_grad_norm = F.normalize(backbone_grad, p=2, dim=1)

        with torch.no_grad():
            alpha_pred = F.softmax(bias_logit, dim=1)
        bias_grad = torch.matmul(alpha_pred-bias_label, _w_bias.T)
        bias_grad_norm = F.normalize(bias_grad, p=2, dim=1)

        grad_inner = torch.sum(torch.mul(backbone_grad_norm, bias_grad_norm), axis=1)

        loss_orth = self.conf_orth * torch.mean(torch.square(grad_inner))
        # pdb.set_trace()
        
        backbone_loss = self.conf_backbone * torch.mean(self.sigmoid_loss(backbone_logit.view(-1), click.float()).view(-1))
        # pdb.set_trace()
        bias_loss = self.conf_bias * torch.mean(self.softmax_loss(bias_logit, bias_label.float()))
        concat_aug_loss = self.conf_concat * torch.mean(self.sigmoid_loss(concat_aug_logit, click.float()))

        loss = 0.
        grads_list = []
        batch_label = {}
        batch_pred = {}
        print_loss = []
        loss_d_list = []

        for idx in self.domain_list:
            idx_tensor = torch.tensor(idx).to(device)
            domain_mask = torch.eq(x[domain_indicator], idx_tensor)
            click_label = torch.masked_select(click, domain_mask)
            # print(click_label.shape[0], click.shape[0], torch.mean(self.sigmoid_loss(domain_out[idx], click_label.float())))
            loss_d = click_label.shape[0] / click.shape[0] * torch.mean(self.sigmoid_loss(domain_out[idx], click_label.float()))

            # pdb.set_trace()
            
            
            if self.training:
                grad = Variable(torch.autograd.grad(loss_d, backbone_out, grad_outputs=torch.ones(loss_d.size()), retain_graph=True)[0], requires_grad = True)
                # pdb.set_trace()
                grads_list.append(torch.sum(grad, axis=0))
            
            loss_d_list.append(loss_d)
            loss += loss_d
            with torch.no_grad():
                print_loss.append(loss_d.item())

            batch_label[idx] = click_label.cpu().tolist()
            batch_pred[idx] = torch.sigmoid(domain_out[idx]).detach().cpu().tolist()

        if self.training:
            orth_grads = self.calc_orth(grads_list)
            for i in range(len(grads_list)):
                loss_orth_d = self.conf_orth_loss * torch.norm(orth_grads[i] - grads_list[i]) ** 2 / 2
                # pdb.set_trace()
                loss += loss_orth_d
                with torch.no_grad():
                    print_loss.append(loss_orth_d.item())
        
        loss += backbone_loss + bias_loss + loss_orth + concat_aug_loss
        with torch.no_grad():
            print_loss.append(backbone_loss.item())
            print_loss.append(bias_loss.item())
            print_loss.append(loss_orth.item())
            print_loss.append(concat_aug_loss.item())

        '''
        for key in self.state_dict():
            if 'embed' in key:
                loss = self._lambda * torch.norm(self.state_dict()[key]) ** 2 / 2
                with torch.no_grad():
                    print_loss.append((self._lambda * torch.norm(self.state_dict()[key]) ** 2 / 2).item())
                pdb.set_trace()
        '''
        
        for param in self.parameters():
            loss += self._lambda * torch.norm(param) ** 2 / 2
            with torch.no_grad():
                print_loss.append((self._lambda * torch.norm(param) ** 2 / 2).item())
            # pdb.set_trace()
            break

        return loss, batch_label, batch_pred, print_loss


data_path = '.'
batch_size = 6000

train_dir = './h5/train_data/'
f_list = os.listdir(train_dir)
f_list.sort()
print(f_list)
train_data = []
mini_data_block = 500
for idx, fn in enumerate(f_list):
    if os.path.splitext(fn)[1] == '.h5':
        data = pd.read_hdf(train_dir+fn, key='data')
        train_data.append(data)
    if idx >= mini_data_block:
        break

test_dir = './h5/test_data/'
f_list = os.listdir(test_dir)
test_data = []
for idx, fn in enumerate(f_list):
    if os.path.splitext(fn)[1] == '.h5':
        data = pd.read_hdf(test_dir+fn, key='data')
        test_data.append(data)
    if idx >= mini_data_block:
        break
# train_data = pd.concat(train_data)
# test_data = pd.concat(test_data)
# pdb.set_trace()

# train_dataloader = get_dataloader(train_data, batch_size, shuffle=True)
# dev_dataloader = get_dataloader(test_data, batch_size, shuffle=False)
# test_dataloader = get_dataloader(test_data, batch_size, shuffle=False)

def train(model, optimizer, scheduler, debug = False):
    best_auc = 0
    min_dis_auc = 1
    early_stop = 10
    # first_batch = True


    for idx in range(2):
        
        train_samples = 0
        # random.shuffle(train_data)
        for block_idx, cur_data in tqdm(enumerate(train_data)):
            
            dataloader = get_dataloader(cur_data, batch_size, shuffle=True)
            # if train_samples >= 8462000:
            #     break
            model.train()
            for step, batch in enumerate(dataloader):
                click, features = batch
                # pdb.set_trace()
                
                skip = False
                # pdb.set_trace()
                info = Counter(features['301'].numpy())
                if len(info) != 3:
                    skip = True
                for k in info:
                    if info[k]<=1:
                        skip = True
                if skip:
                    continue
                
                # train_samples += info[0] + info[1] + info[2]
                # pdb.set_trace()
                for key in features.keys():
                    features[key] = torch.from_numpy(np.array(features[key])).to(device)
                click = click.to(device)
                # pdb.set_trace()
                loss, _, _, print_loss = model(features, click)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                # break
                # if first_batch:

            print(block_idx)
            if (block_idx+1) % 3 == 0 or (block_idx+1)==len(train_data):#  or True:
                print('loss: ', loss.item())
                print('domain_loss: ', print_loss[:3])
                print('domain_orth_loss: ', print_loss[3:6])
                print('backbone_loss: ', print_loss[6])
                print('bias_loss: ', print_loss[7])
                print('orth_loss: ', print_loss[8])
                print('concat_loss: ', print_loss[9])
                print('embed_loss: ', print_loss[10])
                # first_batch = False
        
                scheduler.step()
                # first_batch = True

                # print('eval...')
                model.eval()
                print('---- testing epoch {} ----'.format(idx))
                total_pred = {i:[] for i in domain_list}
                total_label = {i:[] for i in domain_list}

                for cur_data in tqdm(test_data):
                    dataloader = get_dataloader(cur_data, batch_size*5, shuffle=False)
                    for step, batch in enumerate(dataloader):
                        click, features = batch

                        for key in features.keys():
                            features[key] = torch.from_numpy(np.array(features[key])).to(device)
                        click = click.to(device)
                        with torch.no_grad():
                            _, label, pred, _ = model(features, click)
                        
                        for domain in domain_list:
                            total_pred[domain].extend(pred[domain])
                            total_label[domain].extend(label[domain])
                    # pdb.set_trace()
                    # break
                auc = [cal_auc(total_label[d], total_pred[d]) for d in domain_list]

                mix_auc = cal_auc(total_label[0]+total_label[1]+total_label[2], total_pred[0]+total_pred[1]+total_pred[2])

                dis_auc = abs(auc[0]-0.6179)+abs(auc[1]-0.6231)+abs(auc[2]-0.5996)

                print('epoch {}\nmix_auc: {}\ndomain auc: {} {} {}'.format(idx, mix_auc, auc[0], auc[1], auc[2]))

                # if best_auc < mix_auc:
                if min_dis_auc > dis_auc:
                    best_model = model.state_dict()
                    best_auc = mix_auc
                    min_dis_auc = dis_auc
                    # torch.save(best_model, './log/casualint_{}.bin'.format(int(best_auc*10000)))
                    torch.save(best_model, './model/casualint_{}.bin'.format(int(min_dis_auc*10000)))
                    cnt = 0
                elif cnt < early_stop:
                    cnt += 1
                else:
                    '''
                    model.load_state_dict(best_model)
                    
                    total_pred = []
                    total_label = []
                    model.eval()
                    for step, batch in enumerate(test_dataloader):
                        click, features = batch

                        # for key in features.keys():
                        #     features[key] = features[key].to(device)
                        click = click.to(device)
                        with torch.no_grad():
                            _, label, pred = model(features, click)
                        total_pred.extend(pred)
                        total_label.extend(label)
                    auc = cal_auc(total_label, total_pred)
                    print('Test AUC is {}'.format(auc))
                    '''
                    torch.save(model.state_dict(), 'pytorch{}.bin'.format(best_auc))
                    print('early_stop best auc: ', best_auc)
                    break
    return best_auc

def init_param(model):
    torch.nn.init.uniform_(model.embedding_layer.weight, a=-0.001, b=0.001)

    torch.nn.init.uniform_(model.backbone_cls.weight, a=-0.001, b=0.001)
    torch.nn.init.constant_(model.backbone_cls.bias, 0)

    torch.nn.init.uniform_(model.bias_cls.weight, a=-0.001, b=0.001)
    torch.nn.init.constant_(model.bias_cls.bias, 0)

    torch.nn.init.uniform_(model.concat_cls.weight, a=-0.001, b=0.001)
    torch.nn.init.constant_(model.concat_cls.bias, 0)

def eval():
    model = Net()
    f_list = os.listdir('./model/')
    f_list.sort()
    print(f_list)
    for m in f_list:
        print(m)
        model.load_state_dict(torch.load('./model/casualint_21.bin', map_location = 'cpu'))
        model.to(device)
        model.eval()
        total_pred = {i:[] for i in domain_list}
        total_label = {i:[] for i in domain_list}
        for cur_data in tqdm(test_data):
            dataloader = get_dataloader(cur_data, batch_size, shuffle=False)
            for step, batch in enumerate(dataloader):
                click, features = batch
                for key in features.keys():
                    features[key] = torch.from_numpy(np.array(features[key])).to(device)
                click = click.to(device)
                with torch.no_grad():
                    _, label, pred, _ = model(features, click)
        
                for domain in domain_list:
                    total_pred[domain].extend(pred[domain])
                    total_label[domain].extend(label[domain])

        auc = [cal_auc(total_label[d], total_pred[d]) for d in domain_list]
        mix_auc = cal_auc(total_label[0]+total_label[1]+total_label[2], total_pred[0]+total_pred[1]+total_pred[2])

        dis_auc = abs(auc[0]-0.6179)+abs(auc[1]-0.6231)+abs(auc[2]-0.5996)

        print('epoch {}\nmix_auc: {}\ndomain auc: {} {} {}\ndis_auc: {}'.format(idx, mix_auc, auc[0], auc[1], auc[2], dis_auc))

def test_param():
    record_auc = 0
    record_para = 0

    for para in [1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]:
        print('now trying ', para)
        model = Net()
        model.to(device)
        model.conf_orth_loss = para#  * 1e-6
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, eps = 5e-8)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.6)
        auc = train(model, optimizer, scheduler, False)
        if auc > record_auc:
            record_auc = auc
            record_para = para
    print('conf_orth_loss: {}, auc: {}'.format(record_para, record_auc))

def test_base():
    model = Net()
    init_param(model)
    for param in model.parameters():
        param.requires_grad = True

    # os.system('rm ./log/*.bin')
    # torch.save(model.state_dict(), './log/init.bin')
    # model.load_state_dict(torch.load('./log/init.bin', map_location = 'cpu'))
    # for param in model.parameters():
        # print(param)
        # pdb.set_trace()

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, eps = 5e-8)#, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.6)
    auc = train(model, optimizer, scheduler, True)
    print('auc: ', auc)
    print('-'*20)


for _ in range(1):
    eval()
    # test_base()
    
'''
train set: 42300135
test_set: 43016840

step 1
domain_loss: 0.39671, 0.29054, 0.005891
domain_orth_loss: 0, 1.04e-32, 1.99e-35
backbone_loss: 0.69314
bias_loss: 0
orth_loss: 3.04e-14
concat_loss: 0.69315
embed_loss: 0.00015

epoch 1
domain_loss: 0.37987, 0.24782, 0.00353
domain_orth_loss: 0, 1.77e-17, 6.22e-20
backbone_loss: 0.21098
bias_loss: 0
orth_loss: 2.47e-7
concat_loss: 0.16330
embed_loss: 0.00326

epoch 2
domain_loss: 0.33842415, 0.23957498, 0.0041640345
domain_orth_loss: 0.0, 6.1305565e-17, 1.5287814e-21
backbone_loss: 0.17152019
bias_loss: 0.0
orth_loss: 2.0290763e-07
concat_loss: 0.17005184
embed_loss: 0.0067766644


odict_keys(['embedding_layer.weight', 'backbone.1.weight', 'backbone.1.bias', 'backbone.2.weight', 'backbone.2.bias', 'backbone.2.running_mean', 'backbone.2.running_var', 'backbone.2.num_batches_tracked', 'backbone.5.weight', 'backbone.5.bias', 'backbone.6.weight', 'backbone.6.bias', 'backbone.6.running_mean', 'backbone.6.running_var', 'backbone.6.num_batches_tracked', 'backbone.9.weight', 'backbone.9.bias', 'backbone.10.weight', 'backbone.10.bias', 'backbone.10.running_mean', 'backbone.10.running_var', 'backbone.10.num_batches_tracked', 'bias.1.weight', 'bias.1.bias', 'bias.2.weight', 'bias.2.bias', 'bias.2.running_mean', 'bias.2.running_var', 'bias.2.num_batches_tracked', 'bias.5.weight', 'bias.5.bias', 'bias.6.weight', 'bias.6.bias', 'bias.6.running_mean', 'bias.6.running_var', 'bias.6.num_batches_tracked', 'bias.9.weight', 'bias.9.bias', 'bias.10.weight', 'bias.10.bias', 'bias.10.running_mean', 'bias.10.running_var', 'bias.10.num_batches_tracked', 'backbone_cls.weight', 'backbone_cls.bias', 'bias_cls.weight', 'bias_cls.bias', 'concat_cls.weight', 'concat_cls.bias', 'adj.1.weight', 'adj.1.bias', 'adj.2.weight', 'adj.2.bias', 'adj.2.running_mean', 'adj.2.running_var', 'adj.2.num_batches_tracked', 'expert_domain.0', 'expert_domain.1', 'expert_domain.2', 'meta.0.1.weight', 'meta.0.1.bias', 'meta.0.2.weight', 'meta.0.2.bias', 'meta.0.2.running_mean', 'meta.0.2.running_var', 'meta.0.2.num_batches_tracked', 'meta.1.1.weight', 'meta.1.1.bias', 'meta.1.2.weight', 'meta.1.2.bias', 'meta.1.2.running_mean', 'meta.1.2.running_var', 'meta.1.2.num_batches_tracked', 'meta.2.1.weight', 'meta.2.1.bias', 'meta.2.2.weight', 'meta.2.2.bias', 'meta.2.2.running_mean', 'meta.2.2.running_var', 'meta.2.2.num_batches_tracked', 'domain_cls.0.1.weight', 'domain_cls.0.1.bias', 'domain_cls.0.2.weight', 'domain_cls.0.2.bias', 'domain_cls.0.2.running_mean', 'domain_cls.0.2.running_var', 'domain_cls.0.2.num_batches_tracked', 'domain_cls.1.1.weight', 'domain_cls.1.1.bias', 'domain_cls.1.2.weight', 'domain_cls.1.2.bias', 'domain_cls.1.2.running_mean', 'domain_cls.1.2.running_var', 'domain_cls.1.2.num_batches_tracked', 'domain_cls.2.1.weight', 'domain_cls.2.1.bias', 'domain_cls.2.2.weight', 'domain_cls.2.2.bias', 'domain_cls.2.2.running_mean', 'domain_cls.2.2.running_var', 'domain_cls.2.2.num_batches_tracked', 'bn.0.weight', 'bn.0.bias', 'bn.0.running_mean', 'bn.0.running_var', 'bn.0.num_batches_tracked', 'bn.1.weight', 'bn.1.bias', 'bn.1.running_mean', 'bn.1.running_var', 'bn.1.num_batches_tracked', 'bn.2.weight', 'bn.2.bias', 'bn.2.running_mean', 'bn.2.running_var', 'bn.2.num_batches_tracked'])
'''