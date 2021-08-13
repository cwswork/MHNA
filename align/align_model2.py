import random
import torch
import torch.nn as nn

from autil.sparse_tensor import SpecialSpmm
from autil import alignment2


class HET_align2(nn.Module):  # herited classes: nn.Module
    def __init__(self, input_data, config):
        super(HET_align2, self).__init__()
        # Super Parameter
        self.leakyrelu = nn.LeakyReLU(config.LeakyReLU_alpha)  # LeakyReLU_alpha:
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)  #
        self.sigmoid = nn.Sigmoid()
        self.special_spmm = SpecialSpmm()  # sparse matrix multiplication

        self.dropout = nn.Dropout(config.dropout)
        self.top_k = config.top_k
        self.metric = config.metric
        self.neg_k = config.neg_k
        self.alpha1 = config.alpha1  # = 0.1
        self.alpha2 = config.alpha2  # = 0.3
        self.gamma = config.gamma
        # datasets
        self.kg_E = input_data.kg_E
        self.kg_R = input_data.kg_R

        # name embedding
        self.primal_e_0 = input_data.kg_entity_embed  # (E,300)
        # dimension
        if '100K' in config.datasetPath:
            self.is_100K = True
            self.e_dim = 100
            self.r_dim = 200
            # 名称嵌入，权重
            self.kg_name_w = nn.Parameter(torch.zeros(size=(300, self.e_dim)))
            self.kg_name_b = nn.Parameter(torch.zeros(size=(self.e_dim, 1)))
        else:
            self.is_100K = False
            self.e_dim = 300
            self.r_dim = 600

        # weight######################################
        if config.cuda:
            self.tensor_zero = torch.tensor(0.).cuda()
        else:
            self.tensor_zero = torch.tensor(0.)
        # Relation triples
        self.r_head = input_data.r_head
        r_head_sum = torch.unsqueeze(torch.sum(self.r_head, dim=-1), -1)  # (R,E)
        self.r_head_sum = torch.where(r_head_sum == 0, self.tensor_zero, 1. / r_head_sum)  # Instead of countdown
        self.r_tail = input_data.r_tail
        r_tail_sum = torch.unsqueeze(torch.sum(self.r_tail, dim=-1), -1)  # (R,E)
        self.r_tail_sum = torch.where(r_tail_sum == 0, self.tensor_zero, 1. / r_tail_sum) # Instead of countdown

        self.e_adj_index = input_data.e_adj_index
        self.e_adj_data = input_data.e_adj_data
        self.eer_adj_index = input_data.eer_adj_index
        self.eer_adj_data = input_data.eer_adj_data

        # Define the relation coefficient loading on E
        if self.is_100K:
            self.w_R_Left = nn.Parameter(torch.zeros(size=(self.e_dim, self.e_dim)))  # W_r
            self.w_R_Right = nn.Parameter(torch.zeros(size=(self.e_dim, self.e_dim)))
        else:
            self.be_L = nn.Parameter(torch.zeros(size=(self.kg_E, 1)))
            self.be_R = nn.Parameter(torch.zeros(size=(self.kg_E, 1)))
        self.atten_r = nn.Parameter(torch.zeros(size=(self.r_dim, 1))) # R attention

        # GCN+highway
        self.gcnW1 = nn.Parameter(torch.zeros(size=(self.e_dim, self.e_dim)))
        self.highwayWr = nn.Parameter(torch.zeros(size=(self.e_dim, self.e_dim)))
        self.highwaybr = nn.Parameter(torch.zeros(size=(self.e_dim, 1)))


    def init_weights(self):
        for m in self.parameters():
            if isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m.data)  # Xavier

    # align model
    def forward(self):
        if self.is_100K:
            name_embed = torch.mm(self.primal_e_0, self.kg_name_w) + self.kg_name_b.squeeze(1)
        else:
            name_embed = self.primal_e_0

        # 1
        r_embed_1 = self.add_r_layer(name_embed)  # (R,r_dim)
        e_embed_1 = self.add_e_att_layer(name_embed, r_embed_1)
        e_embed_1 = name_embed + self.alpha1 * e_embed_1   # (E,dim)
        # Two layers
        r_embed_2 = self.add_r_layer(e_embed_1)
        e_embed_2 = self.add_e_att_layer(e_embed_1, r_embed_2)
        e_embed_2 = name_embed + self.alpha2 * e_embed_2

        # Equation 12~14
        gcn_e_1 = self.add_diag_layer(e_embed_2)  # (E,dim)
        gcn_e_2 = self.highway(e_embed_2, gcn_e_1)
        gcn_e_3 = self.add_diag_layer(gcn_e_2)
        output_layer = self.highway(gcn_e_2, gcn_e_3)  # (E,dim)

        return output_layer


    # add a highway layer
    def highway(self, e_layer1, e_layer2):
        # (E,dim) * (dim,dim)
        transform_gate = torch.mm(e_layer1, self.highwayWr) + self.highwaybr.squeeze(1)
        transform_gate = self.sigmoid(transform_gate)
        e_layer = transform_gate * e_layer2 + (1.0 - transform_gate) * e_layer1

        return e_layer

    # add a gcn layer
    def add_diag_layer(self, e_inlayer):
        e_inlayer = self.dropout(e_inlayer)
        e_inlayer = torch.mm(e_inlayer, self.gcnW1)  # (E,dim)*(dim,dim) =>(E,dim)
        # e_adj 生成e邻居矩阵，稀疏  e_adj: (E,E)* (E,dim) =>(E,dim)
        e_out = self.special_spmm(self.e_adj_index, self.e_adj_data, torch.Size([self.kg_E, self.kg_E]), e_inlayer)
        return self.relu(e_out)


    def add_e_att_layer(self, e_inlayer, r_layer):
        '''
         According to relational embedding, new entity embedding is calculated
        '''
        e_i_layer = e_inlayer[self.eer_adj_index[0, :], :]
        e_j_layer = e_inlayer[self.eer_adj_index[1, :], :]
        e_ij_embed = torch.cat((e_i_layer, e_j_layer), dim=1)

        #  # [ei||ej]*rij        # (D,r_dim)  D = 176396
        r_qtr = r_layer[self.eer_adj_data]
        eer_embed = e_ij_embed * r_qtr  # (D,r_dim)

        # ee_atten = leakyrelu(a*eer_embed) ，eer_embed: D x rdi
        ee_atten = torch.exp(-self.leakyrelu(torch.matmul(eer_embed, self.atten_r).squeeze()))  # (D,r_dim)*(r_dim,1) => D

        # e_rowsum => (E,E)*(E,1) = (E,1)
        dv = 'cuda' if e_inlayer.is_cuda else 'cpu'
        e_rowsum = self.special_spmm(self.eer_adj_index, ee_atten, torch.Size([self.kg_E, self.kg_E]),
                                  torch.ones(size=(self.kg_E, 1), device=dv))  # (E,E,rdim)(E,1) => (E,dim)
        e_rowsum = torch.where(e_rowsum == 0, self.tensor_zero, 1. / e_rowsum)

        # e_out: attention*h = ee_atten * e_embed => (E,E)*(E,dim) = (E,dim)
        e_out = self.special_spmm(self.eer_adj_index, ee_atten, torch.Size([self.kg_E, self.kg_E]), e_inlayer)  # (E,dim)
        e_out = e_out * e_rowsum
        return self.relu(e_out)  # (E,dim)


    # add relation layer
    def add_r_layer(self, e_inlayer):
        '''
        According to entity embedding, relational embedding is calculated
        '''
        if self.is_100K:
            L_e_inlayer = torch.mm(e_inlayer, self.w_R_Left)
        else:
            L_e_inlayer = e_inlayer * self.be_L  # (E,d).(E,1) => (E,d)
        L_r_embed = torch.matmul(self.r_head, L_e_inlayer)  # (R,E)*(E,d) => (R,d)
        L_r_embed = L_r_embed * self.r_head_sum

        if self.is_100K:
            R_e_inlayer = torch.mm(e_inlayer, self.w_R_Right)
        else:
            R_e_inlayer = e_inlayer * self.be_R  # (E,d).(E,1) => (E,d)
        R_r_embed = torch.matmul(self.r_tail, R_e_inlayer)  # (R,E)*(E,d) => (R,d)
        R_r_embed = R_r_embed * self.r_tail_sum

        r_embed = torch.cat([L_r_embed, R_r_embed], dim=-1)  # (r,600)
        return self.relu(r_embed)   # shape=# (R,2*d)


    def get_loss(self, e_out_embed, tt_neg_pairs):
        #t = len(tt_neg_pairs)
        left_x = e_out_embed[tt_neg_pairs[:, 0]]
        right_x = e_out_embed[tt_neg_pairs[:, 1]]
        A = alignment2.mypair_distance_min(left_x, right_x, distance_type=self.metric)
        D = (A + self.gamma)

        # neg 1
        neg_left_x = e_out_embed[tt_neg_pairs[:, 2]]
        B = alignment2.mypair_distance_min(left_x, neg_left_x, distance_type=self.metric)
        loss1 = self.relu(D - B)
        # neg 2
        neg_right_x = e_out_embed[tt_neg_pairs[:, 3]]
        C = alignment2.mypair_distance_min(right_x, neg_right_x, distance_type=self.metric)
        loss2 = self.relu(D - C)

        #loss = torch.mean(loss1 + loss2)
        return (torch.sum(loss1) + torch.sum(loss2))  # / (2.0 * self.neg_k * t)
