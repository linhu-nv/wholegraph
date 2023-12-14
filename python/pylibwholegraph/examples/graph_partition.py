import argparse
import os
import numpy as np
from scipy.sparse import coo_matrix
import pickle
from ogb.nodeproppred import NodePropPredDataset
from ogb.nodeproppred import DglNodePropPredDataset
import dgl
import time
import torch as th


def save_array(np_array, save_path, array_file_name):
    array_full_path = os.path.join(save_path, array_file_name)
    with open(array_full_path, 'wb') as f:
        np_array.tofile(f)


def load_data(args):
    data = DglNodePropPredDataset(name=args.dataset, root=args.ogb_root_dir)
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]
    labels = labels[:, 0]

    graph.ndata["features"] = graph.ndata.pop("feat")
    graph.ndata["labels"] = labels
    num_labels = len(th.unique(labels[th.logical_not(th.isnan(labels))]))

    # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )
    train_mask = th.zeros((graph.num_nodes(),), dtype=th.bool)
    train_mask[train_nid] = True
    val_mask = th.zeros((graph.num_nodes(),), dtype=th.bool)
    val_mask[val_nid] = True
    test_mask = th.zeros((graph.num_nodes(),), dtype=th.bool)
    test_mask[test_nid] = True
    graph.ndata["train_mask"] = train_mask
    graph.ndata["val_mask"] = val_mask
    graph.ndata["test_mask"] = test_mask
    return graph, num_labels


def graph_partition(args, graph):
    if args.balance_train:
        balance_ntypes = graph.ndata["train_mask"]
        print("we aim to get a balanced train idx")
    else:
        balance_ntypes = None
    node_map, _ = dgl.distributed.partition_graph(
        graph,
        args.dataset,
        args.num_parts,
        out_path=args.ogb_root_dir,
        part_method=args.part_method,
        return_mapping=True,
        balance_ntypes=balance_ntypes,
        balance_edges=args.balance_edges,
        num_trainers_per_machine=args.num_trainers_per_machine,
    )
    return node_map


'''
def new_id2dgl_part_id(id):
    if id < 622443:
        return 0
    if id < 1241335:
        return 1
    if id < 1820248:
        return 2
    return 3
'''


def build_partitioned_subgraphs(args, new2origin_node_map):
    # load data
    ogb_root = args.ogb_root_dir
    dataset = NodePropPredDataset(name=args.dataset, root=ogb_root)
    graph, label = dataset[0]
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = (
        split_idx["train"],
        split_idx["valid"],
        split_idx["test"],
    )
    print("train idx ", len(train_idx), "valid_idx", len(valid_idx), "test index", len(test_idx))
    assert len(train_idx) + len(valid_idx) + len(test_idx) == len(new2origin_node_map)
    num_part = args.num_parts
    node_num = len(new2origin_node_map)
    rank_node_num = int((node_num + num_part - 1) / num_part)
    rank_train_node_num = int((len(train_idx) + num_part - 1) / num_part)
    convert_dir = args.dataset + '/' + args.convert_dir

    originid2labeltype = np.zeros_like(node_map)
    originid2labeltype[train_idx] = 1
    originid2labeltype[valid_idx] = 2
    originid2labeltype[test_idx] = 3

    # calculate the dgl partition boundary
    dgl_part_off = [0]
    for i in range(len(node_map)):
        if i == 0:
            continue
        if node_map[i] < node_map[i - 1]:
            dgl_part_off.append(i)
    dgl_part_off.append(node_num)
    print("the dgl_part_off is ", dgl_part_off)

    # balance train indices and node num among partitions
    rank_valid_test_node_num = rank_node_num - rank_train_node_num
    real_ranks_train_nodes = [0] * num_part
    real_ranks_other_nodes = [0] * num_part
    extra_train_nodes = []
    extra_other_nodes = []
    ranks_node_id = [list() for i in range(num_part)]
    for i in range(num_part):  # scan all nodes, collect already-in-right-rank nodes and extra nodes
        if i == num_part - 1:
            std_train_node_num = len(train_idx) - rank_train_node_num * (num_part - 1)
            std_other_node_num = node_num - rank_node_num * (num_part - 1)
            std_other_node_num -= std_train_node_num
        else:
            std_train_node_num = rank_train_node_num
            std_other_node_num = rank_valid_test_node_num
        my_rank_start = dgl_part_off[i]
        my_rank_end = dgl_part_off[i + 1]
        for j in range(my_rank_start, my_rank_end):
            origin_id = new2origin_node_map[j]
            label_type = originid2labeltype[origin_id]
            if label_type == 1:  # this is a train node
                if real_ranks_train_nodes[i] >= std_train_node_num:  # push back to extra
                    extra_train_nodes.append(origin_id)
                else:
                    ranks_node_id[i].append(origin_id)
                    real_ranks_train_nodes[i] += 1
            else:  # valid node or test node
                if real_ranks_other_nodes[i] >= std_other_node_num:
                    extra_other_nodes.append(origin_id)
                else:
                    ranks_node_id[i].append(origin_id)
                    real_ranks_other_nodes[i] += 1
    for i in range(num_part):  # fullfill those unfilled ranks with extra nodes
        if i == num_part - 1:
            std_train_node_num = len(train_idx) - rank_train_node_num * (num_part - 1)
            std_other_node_num = node_num - rank_node_num * (num_part - 1)
            std_other_node_num -= std_train_node_num
        else:
            std_train_node_num = rank_train_node_num
            std_other_node_num = rank_valid_test_node_num
        needed_train_nodes = std_train_node_num - real_ranks_train_nodes[i]
        needed_other_nodes = std_other_node_num - real_ranks_other_nodes[i]
        if needed_train_nodes > 0:
            assert len(extra_train_nodes) >= needed_train_nodes
            ranks_node_id[i] += extra_train_nodes[:needed_train_nodes]
            extra_train_nodes = extra_train_nodes[needed_train_nodes:]
        if needed_other_nodes > 0:
            assert len(extra_other_nodes) >= needed_other_nodes
            ranks_node_id[i] += extra_other_nodes[:needed_other_nodes]
            extra_other_nodes = extra_other_nodes[needed_other_nodes:]
    assert len(extra_other_nodes) == 0
    assert len(extra_train_nodes) == 0

    for i in range(num_part - 1):
        assert len(ranks_node_id[i]) == rank_node_num

    new2origin_node_map = []
    for i in range(num_part):
        new2origin_node_map += ranks_node_id[i]
    # build reverse node map
    origin2new_node_map = [None] * node_num
    for i in range(node_num):
        origin2new_node_map[new2origin_node_map[i]] = i

    if not os.path.exists(convert_dir):
        print(f"creating directory {convert_dir}...")
        os.makedirs(convert_dir)

    # label reordering and partition
    new_train_idx = [origin2new_node_map[i] for i in train_idx]
    new_valid_idx = [origin2new_node_map[i] for i in valid_idx]
    new_test_idx = [origin2new_node_map[i] for i in test_idx]
    train_idx_set = [list() for i in range(num_part)]
    valid_idx_set = [list() for i in range(num_part)]
    test_idx_set = [list() for i in range(num_part)]
    train_label_set = [list() for i in range(num_part)]
    valid_label_set = [list() for i in range(num_part)]
    test_label_set = [list() for i in range(num_part)]

    # no order is guaranteed between the indices and labels
    # if needed, we can add a sort here to seperated indices and labels
    for i in range(len(new_train_idx)):
        idx = new_train_idx[i]
        part_id = int(idx / rank_node_num)
        train_idx_set[part_id].append(idx)
        train_label_set[part_id].append(label[new2origin_node_map[idx]])
    for i in range(len(new_test_idx)):
        idx = new_test_idx[i]
        part_id = int(idx / rank_node_num)
        test_idx_set[part_id].append(idx)
        test_label_set[part_id].append(label[new2origin_node_map[idx]])
    for i in range(len(new_valid_idx)):
        idx = new_valid_idx[i]
        part_id = int(idx / rank_node_num)
        valid_idx_set[part_id].append(idx)
        valid_label_set[part_id].append(label[new2origin_node_map[idx]])

    for i in range(num_part):
        print("the training node of part ", i, " is ", len(train_idx_set[i]))

    # dump pickle data
    print("saving pickle data...")
    for i in range(num_part):
        data_and_label = {
            "train_idx": train_idx_set[i],
            "valid_idx": valid_idx_set[i],
            "test_idx": test_idx_set[i],
            "train_label": train_label_set[i],
            "valid_label": valid_label_set[i],
            "test_label": test_label_set[i],
        }
        dump_file_name = 'ogbn_products_data_and_label_' + str(i) + '.pkl'
        with open(os.path.join(convert_dir, dump_file_name), "wb") as f:
            pickle.dump(data_and_label, f)

    # reorder features
    node_feat = graph["node_feat"].astype(np.dtype(args.node_feat_format))
    node_feat_new = np.empty_like(node_feat)
    for i in range(len(node_feat)):
        new_idx = origin2new_node_map[i]
        node_feat_new[new_idx] = node_feat[i]
    # start_node_off: int = 0
    print("saving features...")
    # dump a single feature file
    with open(os.path.join(convert_dir, 'node_feat.bin'), "wb") as f:
        node_feat_new.tofile(f)
    '''
    for i in range(num_part):
        end_node_off = start_node_off + rank_node_num
        file_name = 'node_feat_' + str(i) + '.bin'
        end_node_off = int(end_node_off)
        if i == num_part - 1:
            dumped_feat = node_feat_new[start_node_off:]
        else:
            dumped_feat = node_feat_new[start_node_off:end_node_off]
        with open(os.path.join(args.convert_dir, file_name), "wb") as f:
            dumped_feat.tofile(f)
        start_node_off += rank_node_num
    '''
    # build csr of reordered graphs
    # Currently, graph structures are in every nodes? so the csr is stored in one file
    edge_index = graph["edge_index"]
    coo_src_ids = edge_index[0, :].astype(np.int32)
    coo_dst_ids = edge_index[1, :].astype(np.int32)
    if args.add_reverse_edges:
        arg_graph_src = np.concatenate([coo_src_ids, coo_dst_ids])
        arg_graph_dst = np.concatenate([coo_dst_ids, coo_src_ids])
    else:
        arg_graph_src = coo_src_ids
        arg_graph_dst = coo_dst_ids
    values = np.arange(len(arg_graph_src), dtype='int64')
    new_arg_graph_src = np.empty_like(arg_graph_src)
    new_arg_graph_dst = np.empty_like(arg_graph_dst)
    for i in range(len(new_arg_graph_src)):
        new_arg_graph_src[i] = origin2new_node_map[arg_graph_src[i]]
        new_arg_graph_dst[i] = origin2new_node_map[arg_graph_dst[i]]
    coo_graph = coo_matrix((values, (arg_graph_src, arg_graph_dst)), shape=(node_num, node_num))
    csr_graph = coo_graph.tocsr()
    csr_row_ptr = csr_graph.indptr.astype(dtype='int64')
    csr_col_ind = csr_graph.indices.astype(dtype='int32')
    print("saving csr graph...")
    save_array(csr_row_ptr, convert_dir, 'homograph_csr_row_ptr')
    save_array(csr_col_ind, convert_dir, 'homograph_csr_col_idx')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # partition-related arguments
    parser.add_argument("--num_parts", type=int, default=4,
                        help="number of partitions")
    parser.add_argument("--part_method", type=str, default="metis",
                        help="the partition method")
    parser.add_argument("--balance_train", action="store_true",
                        help="balance the training size in each partition.")
    parser.add_argument("--undirected", action="store_true",
                        help="turn the graph into an undirected graph.")
    parser.add_argument("--balance_edges", action="store_true",
                        help="balance the number of edges in each partition.")
    parser.add_argument("--num_trainers_per_machine", type=int, default=1,
                        help="the number of trainers per machine. The trainer ids are stored\
                                in the node feature 'trainer_id'")
    parser.add_argument("--load_node_map", type=bool, default=True,
                        help="determine build or load partitioned graph node map")
    # graph-related arguments
    parser.add_argument("--dataset", type=str, default="ogbn-products",
                        help="datasets: ogbn-products, ogbn-papers100M")
    parser.add_argument('--ogb_root_dir', type=str, default='dataset',
                        help='root dir of containing ogb datasets')
    parser.add_argument('--convert_dir', type=str, default='converted',
                        help='output dir containing converted datasets')
    parser.add_argument('--node_feat_format', type=str, default='float32',
                        choices=['float32', 'float16'],
                        help='save format of node feature')
    parser.add_argument('--add_reverse_edges', type=bool, default=True,
                        help='whether to add reverse edges')
    args = parser.parse_args()
    # load data for dgl graph partition
    start = time.time()
    if args.load_node_map is False:
        graph, _ = load_data(args)
        print("load {} takes {:.3f} seconds".format(args.dataset, time.time() - start))
        print(
            "train: {}, valid: {}, test: {}".format(
                th.sum(graph.ndata["train_mask"]),
                th.sum(graph.ndata["val_mask"]),
                th.sum(graph.ndata["test_mask"]),
            )
        )
        # graph partition
        node_map = graph_partition(args, graph).numpy()
        np.save("node_map_ogbn_products.npy", node_map)
    else:
        node_map = np.load('node_map_' + str(args.dataset) + '.npy')
    gp_start = time.time()
    print("graph partition {} takes {:.3f} seconds".format(args.dataset, time.time() - gp_start))
    bps_start = time.time()
    # build partitioned subgraphs
    build_partitioned_subgraphs(args, node_map)
    print("build partition subgraphs {} takes {:.3f} seconds".format(args.dataset, time.time() - bps_start))
