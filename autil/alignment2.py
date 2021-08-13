import gc
import numpy as np
import torch
from torch.nn import functional


def mypair_distance_min(a, b, distance_type="L1"):
    ''' Find the distance of the entity pair. The similarity is high, the value is low '''
    if distance_type == "L1":
        return functional.pairwise_distance(a, b, p=1)
    elif distance_type == "L2":
        return functional.pairwise_distance(a, b, p=2)
    elif distance_type == "L2squared":
        return torch.pow(functional.pairwise_distance(a, b, p=2), 2)
    elif distance_type == "cosine":
        return 1 - torch.cosine_similarity(a, b)

##neg#################
def gen_neg(ent_embed, tt_links_new, metric, neg_k):
    es1 = [e1 for e1,e2 in tt_links_new]
    es2 = [e1 for e1,e2 in tt_links_new]
    neg1_array = gen_neg_each(ent_embed, es1, metric, neg_k)
    neg2_array = gen_neg_each(ent_embed, es2, metric, neg_k)

    neg_pair = []
    for i in range(len(es2)):
        e1, e2 = tt_links_new[i]
        for j in range(neg_k):
            neg_pair.append((e1, e2, neg1_array[i][j], neg2_array[i][j]))

    neg_pair = torch.LongTensor(np.array(neg_pair))  # eer_adj_data
    if ent_embed.is_cuda:
        neg_pair = neg_pair.cuda()

    return neg_pair


def gen_neg_each(ent_embed, left_ents, metric, neg_k):
    max_index = torch_sim_max_topk(ent_embed[left_ents, :], ent_embed, top_num=neg_k + 1, metric=metric)

    e_t = len(left_ents)
    neg = []
    for i in range(e_t):
        rank = max_index[i, :].tolist()
        if left_ents[i] == rank[0]:
            rank = rank[1:]
        else:
            if left_ents[i] in rank:
                rank.remove(left_ents[i])
            else:
                rank = rank[:neg_k]
        neg.append(rank)

    neg = np.array(neg) # neg.reshape((e_t * self.neg_k,))
    return neg  # (n*k,)


######################
# accuracy
def my_accuracy(e_out_embed, train_ILL_tensor, metric, top_k, fromLeft=True):
    with torch.no_grad():
        kg1_embed = e_out_embed[train_ILL_tensor[:, 0], :]
        kg2_embed = e_out_embed[train_ILL_tensor[:, 1], :]

        if fromLeft:  # From left
            all_hits, mr, mrr, hits1_list, noHits1_list = my_alignment(kg1_embed, kg2_embed, top_k=top_k, metric=metric)
        else:  # From right
            all_hits, mr, mrr, hits1_list, noHits1_list = my_alignment(kg2_embed, kg1_embed, top_k=top_k, metric=metric)

        return all_hits, mr, mrr


def my_alignment(kg1_entity_embed, kg2_entity_embed, top_k, metric='L1'):

    max_index = torch_sim_max_topk(kg1_entity_embed, kg2_entity_embed, top_num=top_k[-1], metric=metric)
    # left
    mr = 0
    mrr = 0
    notin_candi = 0
    tt_num = len(kg1_entity_embed)
    all_hits = [0] * len(top_k)
    hits1_list = list()
    noHits1_list = list()
    for row_i in range(tt_num):
        e2_ranks_index = max_index[row_i, :].tolist()
        e1_index_gold, e2_index_gold = row_i, row_i
        hits1_list.append((e1_index_gold, e2_ranks_index[0]))

        if e2_index_gold != e2_ranks_index[0]:
            noHits1_list.append((e1_index_gold, e2_index_gold, e2_ranks_index[0]))
        if e2_index_gold not in e2_ranks_index:
            notin_candi += 1
        else:
            rank_index = e2_ranks_index.index(e2_index_gold)
            mr += (rank_index + 1)
            mrr += 1 / (rank_index + 1)
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    all_hits[j] += 1

    assert len(hits1_list) == tt_num
    all_hits = [round(hh / tt_num * 100, 4) for hh in all_hits]
    mr /= tt_num  # The average of all levels in the comparison
    mrr /= tt_num  # Mean of all countdown rankings

    return all_hits, mr, mrr, hits1_list, noHits1_list #, notin_candi


def torch_sim_max_topk(embed1, embed2, top_num, metric='manhattan', is_hseg=True):
    if is_hseg:
        if embed1.is_cuda:
            left_batch_size = 1000
        else:
            left_batch_size = 50000
        links_len = embed1.shape[0]
        max_index_list = []
        for i in np.arange(0, links_len, left_batch_size):
            end = min(i + left_batch_size, links_len)
            max_index_batch = torch_sim_max_vseg(embed1[i:end, :], embed2, top_num, metric=metric)
            max_index_list.append(max_index_batch)

        max_index = torch.cat(max_index_list, 0)
        # max_index = np.concatenate(max_index_list, axis=0)

        del max_index_list, max_index_batch
        gc.collect()
    else:
        max_index = torch_sim_max_vseg(embed1, embed2, top_num, metric=metric)

    if max_index.is_cuda:
        max_index = max_index.detach().cpu().numpy()
    else:
        max_index = max_index.detach().numpy()

    return max_index


### Vertical split
def torch_sim_max_vseg(embed1, embed2, top_num, metric):
    right_len = embed2.shape[0]
    batch_size = 50000

    #max_index_merge, max_scoce_merge = None, None
    max_index_list, max_scoce_list = [], []
    for beg_index in np.arange(0, right_len, batch_size):
        end = min(beg_index + batch_size, right_len)
        max_scoce_batch, max_index_batch = torch_sim_max_batch(embed1, embed2[beg_index:end, :], top_num, metric=metric)
        if beg_index != 0:
            max_index_batch += beg_index
        max_index_list.append(max_index_batch)
        max_scoce_list.append(max_scoce_batch)

    max_scoce_merge = torch.cat(max_scoce_list, 1)
    max_index_merge = torch.cat(max_index_list, 1)

    #top_index = np.argsort(-max_scoce_merge, axis=1)
    top_index = max_scoce_merge.argsort(dim=-1, descending=True)
    top_index = top_index[:, :top_num]
    #
    row_count = embed1.shape[0]
    max_index_merge = max_index_merge.int()
    max_index = torch.zeros((row_count, top_num), )
    for i in range(row_count):
        #max_scoce[i] = max_scoce_merge[i, top_index[i]]
        max_index[i] = max_index_merge[i, top_index[i]]
    max_index = max_index.int()

    return max_index


def torch_sim_max_batch(embed1, embed2, top_num, metric='manhattan'):
    '''  '''
    if metric == 'L1' or metric == 'manhattan':  # L1 Manhattan
        sim_mat = - torch.cdist(embed1, embed2, p=1.0)
    elif metric == 'L2' or metric == 'euclidean':  # L2 euclidean
        sim_mat = - torch.cdist(embed1, embed2, p=2.0)
    elif metric == 'cosine': # cosine
        sim_mat = cosine_similarity3(embed1, embed2)  # [batch, net1, net1]

    if len(embed2) > top_num:
        max_scoce_batch, max_index_batch = sim_mat.topk(k=top_num, dim=-1, largest=True)  # get top num
    else:
        max_scoce_batch, max_index_batch = sim_mat.topk(k=len(embed2), dim=-1, largest=True)  # get top num

    del sim_mat
    gc.collect()
    return max_scoce_batch, max_index_batch



##similarity########
def cosine_similarity3(a, b):
    '''
    a shape: [num_item_1, embedding_dim]
    b shape: [num_item_2, embedding_dim]
    return sim_matrix: [num_item_1, num_item_2]
    '''
    a = a / torch.clamp(a.norm(dim=-1, keepdim=True, p=2), min=1e-10)
    b = b / torch.clamp(b.norm(dim=-1, keepdim=True, p=2), min=1e-10)
    #sim = torch.mm(a, b.t())
    if len(a.shape) == 3:
        sim = torch.bmm(a, torch.transpose(b, 1, 2))
    else:
        sim = torch.mm(a, b.t())
    return sim



def divide_batch(idx_list, batch_size):
    ''' Divide into N tasks '''
    total = len(idx_list)
    if batch_size <= 0 or total <= batch_size:
        return [idx_list]
    else:
        n = total // batch_size
        batchs_list = []
        for i in range(n):
            beg = i * batch_size
            batchs_list.append(idx_list[beg:beg + batch_size])

        if beg + batch_size < total:
            batchs_list.append(idx_list[beg + batch_size:])
        return batchs_list

