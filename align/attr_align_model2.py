import torch
import torch.nn as nn

from autil.sparse_tensor import SpecialSpmm
from autil import alignment2


class HET_attr_align2(nn.Module):
    def __init__(self, input_data, config):
        super(HET_attr_align2, self).__init__()
        #
        self.leakyrelu = nn.LeakyReLU(config.LeakyReLU_alpha)  # alpha: leakyrelu
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.special_spmm = SpecialSpmm()

        self.dropout = nn.Dropout(config.dropout)
        self.top_k = config.top_k
        self.neg_k = config.neg_k
        self.metric = config.metric
        self.alpha3 = config.alpha3  # alpha = 0.1
        self.gamma = config.gamma  # gamma = 1.0  margin based loss
        self.l_bata = config.l_bata
        # datasets
        self.kg_E = input_data.kg_E
        self.kg_R = input_data.kg_R
        self.kg_M = input_data.kg_M
        self.kg_V = input_data.kg_V

        self.set_data(input_data, config)
        #####################################################################

        # dimension
        if '100K' in config.datasetPath:
            self.is_100K = True
            self.e_dim = self.v_dim = 100
            # Name embedding
            self.kg_name_w = nn.Parameter(torch.zeros(size=(300, self.e_dim)))
            self.kg_name_b = nn.Parameter(torch.zeros(size=(self.e_dim, 1)))
            # Value embedding
            self.kg_value_w = nn.Parameter(torch.zeros(size=(300, self.v_dim)))
            self.kg_value_b = nn.Parameter(torch.zeros(size=(self.v_dim, 1)))

            # relations
            self.be_L = nn.Parameter(torch.zeros(size=(self.e_dim, self.e_dim)))  # W_r
            self.be_R = nn.Parameter(torch.zeros(size=(self.e_dim, self.e_dim)))
            # attribute
            self.bm_LE = nn.Parameter(torch.zeros(size=(self.e_dim, self.e_dim)))  # W_r
            self.bm_LV = nn.Parameter(torch.zeros(size=(self.v_dim, self.v_dim)))
        else:
            self.is_100K = False
            self.e_dim = self.v_dim = 300
            # relations
            self.be_L = nn.Parameter(torch.zeros(size=(self.kg_E, 1)))
            self.be_R = nn.Parameter(torch.zeros(size=(self.kg_E, 1)))
            # attribute
            self.bm_LE = nn.Parameter(torch.zeros(size=(self.kg_E, 1)))
            self.bm_LV = nn.Parameter(torch.zeros(size=(self.kg_V, 1)))

        self.r_dim = self.m_dim = self.e_dim * 2
        self.atten_r = nn.Parameter(torch.zeros(size=(self.r_dim, 1)))  # R attention
        self.atten_m = nn.Parameter(torch.zeros(size=(self.m_dim, 1)))  # m attention

        # GCN+highway
        self.gcnW1 = nn.Parameter(torch.zeros(size=(self.e_dim, self.e_dim)))
        self.highwayWr1 = nn.Parameter(torch.zeros(size=(self.e_dim, self.e_dim)))
        self.highwaybr1 = nn.Parameter(torch.zeros(size=(self.e_dim, 1)))

        # GCN+highway
        self.gcnW2 = nn.Parameter(torch.zeros(size=(self.e_dim, self.e_dim)))
        self.highwayWr2 = nn.Parameter(torch.zeros(size=(self.e_dim, self.e_dim)))
        self.highwaybr2 = nn.Parameter(torch.zeros(size=(self.e_dim, 1)))


    def set_data(self, input_data, config):
        if config.cuda:
            self.tensor_zero = torch.tensor(0.).cuda()
        else:
            self.tensor_zero = torch.tensor(0.)
        ## Relations
        self.primal_e_0 = input_data.kg_entity_embed
        self.r_head = input_data.r_head
        self.r_tail = input_data.r_tail
        r_head_sum = torch.unsqueeze(torch.sum(self.r_head, dim=-1), -1)  # (R,E)
        self.r_head_sum = torch.where(r_head_sum == 0, self.tensor_zero, 1. / r_head_sum)
        r_tail_sum = torch.unsqueeze(torch.sum(self.r_tail, dim=-1), -1)  # (R,E)
        self.r_tail_sum = torch.where(r_tail_sum == 0, self.tensor_zero, 1. / r_tail_sum)

        self.e_adj_index = input_data.e_adj_index
        self.e_adj_data = input_data.e_adj_data
        self.eer_adj_index = input_data.eer_adj_index
        self.eer_adj_data = input_data.eer_adj_data

        ## Attributes
        self.primal_v_0 = input_data.kg_value_embed  # (V,300)
        self.m_head2e = input_data.m_head2e
        self.m_tail2v = input_data.m_tail2v
        m_head2e_sum = torch.unsqueeze(torch.sum(self.m_head2e, dim=-1), -1)  # (M,E)
        self.m_head2e_sum = torch.where(m_head2e_sum == 0, self.tensor_zero, 1. / m_head2e_sum)
        m_tail2v_sum = torch.unsqueeze(torch.sum(self.m_tail2v, dim=-1), -1)  # (M,V)
        self.m_tail2v_sum = torch.where(m_tail2v_sum == 0, self.tensor_zero, 1. / m_tail2v_sum)
        # (E,V)
        self.emv_adj_index = input_data.emv_adj_index
        self.emv_adj_data = input_data.emv_adj_data


    def init_weights(self):
        for m in self.parameters():
            if isinstance(m, nn.Parameter):
                if len(m.shape) == 2:
                    nn.init.xavier_normal_(m.data)  # Xavier


    # align model
    def forward(self):
        if self.is_100K:
            name_embed = torch.mm(self.primal_e_0, self.kg_name_w) + self.kg_name_b.squeeze(1)
            value_embed = torch.mm(self.primal_v_0, self.kg_value_w) + self.kg_value_b.squeeze(1)
        else:
            name_embed = self.primal_e_0   #(V,300)
            value_embed = self.primal_v_0

        # 1
        r_embed_1 = self.add_r_layer(name_embed)  # (R,r_dim)
        se_embed_1 = self.add_se_att_layer(name_embed, r_embed_1)
        se_embed = name_embed + self.alpha3 * se_embed_1   # (E,dim)
        # GCN+highway
        gcn_se_1 = self.add_diag_layer(se_embed, self.gcnW1)  # (E,2*dim)
        gcn_se_1 = self.highway(se_embed, gcn_se_1, self.highwayWr1, self.highwaybr1)
        gcn_se_2 = self.add_diag_layer(gcn_se_1, self.gcnW1)
        se_layer = self.highway(gcn_se_1, gcn_se_2, self.highwayWr1, self.highwaybr1)  # (E,dim)

        # 2
        m_embed_1 = self.add_m_layer(name_embed, value_embed)  # (M,dim)
        #m_embed_1 = self.primal_m_0 + self.alpha * m_embed_1
        ce_embed = self.add_ce_att_layer(name_embed, m_embed_1, value_embed)   # (E,dim)
        #ce_embed = name_embed + self.alpha * ce_embed_1

        # GCN+highway
        gcn_ce_1 = self.add_diag_layer(ce_embed, self.gcnW2)  # (E,2*dim)
        gcn_ce_1 = self.highway(ce_embed, gcn_ce_1, self.highwayWr2, self.highwaybr2)
        gcn_ce_2 = self.add_diag_layer(gcn_ce_1, self.gcnW2)
        ce_layer = self.highway(gcn_ce_1, gcn_ce_2, self.highwayWr2, self.highwaybr2)  # (E,dim)

        return se_layer, ce_layer


    # add a highway layer
    def highway(self, e_layer1, e_layer2, highwayWr, highwaybr):
        # (E,dim) * (dim,dim)
        transform_gate = torch.mm(e_layer1, highwayWr) + highwaybr.squeeze(1)
        transform_gate = self.sigmoid(transform_gate)
        e_layer = transform_gate * e_layer2 + (1.0 - transform_gate) * e_layer1
        return e_layer


    # add a gcn layer
    def add_diag_layer(self, e_inlayer, gcnW1):
        e_inlayer = self.dropout(e_inlayer)
        e_inlayer = torch.mm(e_inlayer, gcnW1)  #(E,dim)*(dim,dim) =>(E,dim)
        # e_adj (E,E)*(E,e_dim) =>(E,e_dim)
        e_out = self.special_spmm(self.e_adj_index, self.e_adj_data, torch.Size([self.kg_E, self.kg_E]), e_inlayer)
        return self.relu(e_out)

    # se attention
    def add_se_att_layer(self, e_inlayer, r_layer):
        e_i_layer = e_inlayer[self.eer_adj_index[0, :], :]
        e_j_layer = e_inlayer[self.eer_adj_index[1, :], :]
        e_ij_embed = torch.cat((e_i_layer, e_j_layer), dim=1)

        # [ei||ej]*rij   (D,r_dim)
        r_qtr = r_layer[self.eer_adj_data]
        eer_embed = e_ij_embed * r_qtr  # (D,r_dim)

        # ee_atten = leakyrelu(a*eer_embed) ，eer_embed: D x rdi
        eer_atten = torch.exp(-self.leakyrelu(torch.mm(eer_embed, self.atten_r).squeeze()))  # (D,r_dim)*(r_dim,1) => D

        # e_rowsum => (E,E)*(E,1) = (E,1)
        dv = 'cuda' if e_inlayer.is_cuda else 'cpu'
        e_rowsum = self.special_spmm(self.eer_adj_index, eer_atten, torch.Size([self.kg_E, self.kg_E]),
                                  torch.ones(size=(self.kg_E, 1), device=dv))  # (E,E,rdim)(E,1) => (E,dim)
        e_rowsum = torch.where(e_rowsum == 0, self.tensor_zero, 1. / e_rowsum)

        # e_out: attention*h = eer_atten * e_embed => (E,M)*(M,dim) = (E,dim)
        e_out = self.special_spmm(self.eer_adj_index, eer_atten, torch.Size([self.kg_E, self.kg_E]), e_inlayer)  # (E,dim)
        e_out = e_out * e_rowsum

        return self.relu(e_out)  # (E,dim)


    def add_r_layer(self, e_inlayer):
        if self.is_100K:
            L_e_inlayer = torch.mm(e_inlayer, self.be_L)
        else:
            L_e_inlayer = e_inlayer * self.be_L  # (E,d).(E,1) => (E,d)
        L_r_embed = torch.matmul(self.r_head, L_e_inlayer)  # (R,E)*(E,d) => (R,d)
        L_r_embed = L_r_embed * self.r_head_sum    # / r_head_sum

        if self.is_100K:
            R_e_inlayer = torch.mm(e_inlayer, self.be_R)
        else:
            R_e_inlayer = e_inlayer * self.be_R  # (E,d).(E,1) => (E,d)
        R_r_embed = torch.matmul(self.r_tail, R_e_inlayer)  # (R,E)*(E,d) => (R,d)
        R_r_embed = R_r_embed * self.r_tail_sum   # / r_tail_sum

        r_embed = torch.cat([L_r_embed, R_r_embed], dim=-1)  # (r,600)
        return self.relu(r_embed)   # shape=# (R,2*d)


    def add_m_layer(self, e_inlayer, v_inlayer):
        if self.is_100K:
            L_e_inlayer = torch.mm(e_inlayer, self.bm_LE)
        else:
            L_e_inlayer = e_inlayer * self.bm_LE  # (E,d).(E,1) => (E,d)
        L_m_embed = torch.matmul(self.m_head2e, L_e_inlayer)  # (M,E)*(E,d) => (M,e_dim)
        L_m_embed = L_m_embed * self.m_head2e_sum

        if self.is_100K:
            R_v_inlayer = torch.mm(v_inlayer, self.bm_LV)
        else:
            R_v_inlayer = v_inlayer * self.bm_LV  # (E,d).(E,1) => (E,d)
        R_m_embed = torch.matmul(self.m_tail2v, R_v_inlayer)  # (M,V)*(V,d) => (M,v_dim)
        R_m_embed = R_m_embed * self.m_tail2v_sum

        m_embed = torch.cat([L_m_embed, R_m_embed], dim=-1)  # (M, e_dim+v_dim)
        return self.relu(m_embed)   # shape=# (M, m_dim)


    # ce attention
    def add_ce_att_layer(self, e_inlayer, m_layer, v_layer):
        #  ei * mj  (D,dim)  D = 239903   (E,M+E-M)
        e_embed = e_inlayer[self.emv_adj_index[0], :]
        v_embed = v_layer[self.emv_adj_index[1], :]
        # em_embed = e_embed || v_embed
        ev_embed = torch.cat((e_embed, v_embed), dim=1)

        m_embed = m_layer[self.emv_adj_data, :]
        # [xei||xvj]* mij
        emv_embed = ev_embed * m_embed  # (D,m_dim)
        em_atten = torch.exp(-self.leakyrelu(torch.mm(emv_embed, self.atten_m).squeeze()))  # (D,m_dim)*(m_dim,1) => D

        # em_rowsum => (E,V+E-V)*(V+E-V,1) = (E,1)
        dv = 'cuda' if e_inlayer.is_cuda else 'cpu'
        em_rowsum = self.special_spmm(self.emv_adj_index, em_atten,
                      torch.Size([self.kg_V, self.kg_V]), torch.ones(size=(self.kg_V, 1), device=dv))
        em_rowsum = torch.where(em_rowsum == 0, self.tensor_zero, 1./em_rowsum)

        # e_out: attention*h = em_atten * e_embed
        em_out = self.special_spmm(self.emv_adj_index, em_atten,
                                   torch.Size([self.kg_V, self.kg_V]), v_layer)
        em_out = em_out * em_rowsum
        return self.relu(em_out[:self.kg_E, :])  # (V,dim) => (E,dim)


#############################################
    def get_loss(self, es_embed, ec_embed, tt_neg_pairs_es, tt_neg_pairs_ec):
        loss1 = self.get_loss_each(es_embed, tt_neg_pairs_es)
        loss2 = self.get_loss_each(ec_embed, tt_neg_pairs_ec)

        output_layer = torch.cat((es_embed, ec_embed), dim=1)
        return loss1 + self.l_bata * loss2, output_layer

    def get_loss_each(self, e_out_embed, tt_neg_pairs):
        left_x = e_out_embed[tt_neg_pairs[:, 0]]
        right_x = e_out_embed[tt_neg_pairs[:, 1]]
        A = alignment2.mypair_distance_min(left_x, right_x, distance_type=self.metric)
        D = (A + self.gamma)

        # net 1
        neg_left_x = e_out_embed[tt_neg_pairs[:, 2]]
        B = alignment2.mypair_distance_min(left_x, neg_left_x, distance_type=self.metric)
        loss1 = self.relu(D - B)
        # net 2
        neg_right_x = e_out_embed[tt_neg_pairs[:, 3]]
        C = alignment2.mypair_distance_min(right_x, neg_right_x, distance_type=self.metric)
        loss2 = self.relu(D - C)

        # loss = torch.mean(loss1 + loss2)
        return (torch.sum(loss1) + torch.sum(loss2))  # / (2.0 * self.neg_k * t)
