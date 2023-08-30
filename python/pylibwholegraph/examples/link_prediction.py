# Copyright (c) 2019-2023, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import os
import time
from optparse import OptionParser

import apex
import torch

import pylibwholegraph.torch as wgth
#this is from old link prediction
import torch.nn.functional as F
import torchmetrics.functional as MF
from apex.parallel import DistributedDataParallel as DDP
from mpi4py import MPI
from ogb.linkproppred import Evaluator


import pylibwholegraph.torch as wgth
from pylibwholegraph.torch.embedding import WholeMemoryEmbedding, WholeMemoryEmbeddingModule
#from wholegraph.torch import wholegraph_pytorch as wg

parser = OptionParser(conflict_handler="resolve")

wgth.add_distributed_launch_options(parser)
#TODO -lr may be 0.001 in default, -e as 1, 
wgth.add_training_options(parser)
wgth.add_common_graph_options(parser)
wgth.add_common_model_options(parser)
wgth.add_common_sampler_options(parser)
#TODO:-w may be 8 in default
wgth.add_dataloader_options(parser)
#TODO:-n may be 30,30,30 in default
wgth.add_common_sampler_options(parser)

(options, args) = parser.parse_args()

#parser.add_option(
#    "--use_nccl",
#    action="store_true",
#    dest="use_nccl",
#    default=False,
#    help="whether use nccl for embeddings, default False",
#)

use_chunked = True
use_host_memory = False

def parse_max_neighbors(num_layer, neighbor_str):
    neighbor_str_vec = neighbor_str.split(",")
    max_neighbors = []
    for ns in neighbor_str_vec:
        max_neighbors.append(int(ns))
    assert len(max_neighbors) == 1 or len(max_neighbors) == num_layer
    if len(max_neighbors) != num_layer:
        for i in range(1, num_layer):
            max_neighbors.append(max_neighbors[0])
    # max_neighbors.reverse()
    return max_neighbors


class EdgePredictionGNNModel(torch.nn.Module):
    def __init__(
        self, graph: wgth.GraphStructure, node_feat, options
    ):
        super().__init__()
        self.graph = graph
        self.num_layer = options.layernum
        self.hidden_feat_dim = options.hiddensize
        self.max_neighbors = parse_max_neighbors(self.num_layer, options.neighbors)
        num_head = options.heads if (options.model == "gat") else 1
        assert self.hidden_feat_dim % num_head == 0
        in_feat_dim = options.feat_dim
        self.gnn_layers = wgth.gnn_model.create_gnn_layers(
            in_feat_dim, self.hidden_feat_dim, self.hidden_feat_dim//num_head, 
            self.num_layer, num_head, options.model
        )
        self.mean_output = True if options.model == "gat" else False
        self.add_self_loop = True if options.model == "gat" else False
        self.gather_fn = WholeMemoryEmbeddingModule(node_feat)
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_feat_dim, self.hidden_feat_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_feat_dim, self.hidden_feat_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_feat_dim, 1),
        )

    def gnn_forward(self, ids):
        ids = ids.to(self.graph.csr_col_ind.dtype).cuda()
        (
            target_gids,
            edge_indice,
            csr_row_ptrs,
            csr_col_inds,
        ) = self.graph.multilayer_sample_without_replacement(
            ids, self.max_neighbors
        )
        x_feat = self.gather_fn(target_gids[0])
        # x_feat = self.graph.gather(target_gids[0])
        # num_nodes = [target_gid.shape[0] for target_gid in target_gids]
        # print('num_nodes %s' % (num_nodes, ))
        for i in range(self.num_layer):
            x_target_feat = x_feat[: target_gids[i + 1].numel()]
            sub_graph = wgth.create_sub_graph(
                target_gids[i],
                target_gids[i + 1],
                edge_indice[i],
                csr_row_ptrs[i],
                csr_col_inds[i],
                self.max_neighbors[i],
                self.add_self_loop,
            )
            x_feat = wgth.gnn_model.layer_forward(self.gnn_layers[i], x_feat, x_target_feat, sub_graph)
            if i != self.num_layer - 1:
                if options.framework == "dgl":
                    x_feat = x_feat.flatten(1)
                x_feat = F.relu(x_feat)
                # x_feat = F.dropout(x_feat, options.dropout, training=self.training)
        if options.framework == "dgl" and self.mean_output:
            out_feat = x_feat.mean(1)
        else:
            out_feat = x_feat
        return out_feat

    def predict(self, h_src, h_dst):
        return self.predictor(h_src * h_dst)

    def fullbatch_single_layer_forward(
        self, graph_structure, i, input_feat:WholeMemoryEmbedding, output_feat, batch_size
    ):
        start_node_id = (
            graph_structure.node_count 
            * wgth.get_local_rank()
            // wgth.get_local_size()
        )
        end_node_id = (
            graph_structure.node_count
            * (wgth.get_local_rank() + 1)
            // wgth.get_local_size()
        )
        min_node_count = graph_structure.node_count // wgth.get_local_size()
        total_node_count = end_node_id - start_node_id
        batch_count = max((min_node_count + batch_size - 1) // batch_size, 1)
        last_batchsize = total_node_count - (batch_count - 1) * batch_size
        embedding_lookup_fn = wgth.embedding.EmbeddingLookupFn.apply
        for batch_id in range(batch_count):
            current_batchsize = (
                last_batchsize if batch_id == batch_count - 1 else batch_size
            )
            batch_start_node_id = start_node_id + batch_id * batch_size
            target_ids = torch.arange(
                batch_start_node_id,
                batch_start_node_id + current_batchsize,
                dtype=graph_structure.csr_col_ind.dtype,
                device="cuda",
            )
            #TODO we have the following assumption between unweight.. and unweight..single layer
            #output_sample_offset_tensor = neighboor_gids_offset
            #output_dest_context = neighboor_gids_vdata
            #output_center_localid_context = neighboor_src_lids
            (
                neighboor_gids_offset,
                neighboor_gids_vdata,
                neighboor_src_lids,
            ) = wgth.wholegraph_ops.unweighted_sample_without_replacement(
                graph_structure.csr_row_ptr.wmb_tensor,
                graph_structure.csr_col_ind.wmb_tensor,
                target_ids,
                max_sample_count = -1,
                need_center_local_output = True
            )
            (
                unique_gids,
                neighbor_raw_to_unique_mapping,
                #unique_output_neighbor_count,
            ) = wgth.graph_ops.append_unique(target_ids, neighboor_gids_vdata, True)
            csr_row_ptr = neighboor_gids_offset
            csr_col_ind = neighbor_raw_to_unique_mapping
            assert csr_row_ptr.size(0) == current_batchsize + 1
            #sample_dup_count = unique_output_neighbor_count
            neighboor_count = neighboor_gids_vdata.size()[0]
            edge_indice_i = torch.cat(
                [
                    torch.reshape(neighbor_raw_to_unique_mapping, (1, neighboor_count)),
                    torch.reshape(neighboor_src_lids, (1, neighboor_count)),
                ]
            )
            target_ids_i = unique_gids
            #TODO what is dummy_input
            x_feat = embedding_lookup_fn(target_ids_i, torch.tensor([]).cuda(), input_feat)
            sub_graph = wgth.create_sub_graph(
                target_ids_i,
                target_ids,
                edge_indice_i,
                csr_row_ptr,
                csr_col_ind,
                self.max_neighbors[i],
                self.add_self_loop,
            )
            x_target_feat = x_feat[: target_ids.numel()]
            #TODO this crash when use cugraph
            #print("the batch is ", batch_id, "layer is", i, x_feat.shape, x_target_feat.shape,
            #     "subgraph info", sub_graph[0].shape, sub_graph[1].shape, sub_graph[2],
            #     "data structure location", x_feat.is_cuda, x_target_feat.is_cuda, sub_graph[0].is_cuda)
            x_feat = wgth.gnn_model.layer_forward(self.gnn_layers[i], x_feat, x_target_feat, sub_graph)
            #TODO anything todo with cugraph?
            if i != self.num_layer - 1:
                if options.framework == "dgl":
                    x_feat = x_feat.flatten(1)
                x_feat = F.relu(x_feat)
            else:
                if options.framework == "dgl" and self.mean_output:
                    x_feat = x_feat.mean(1)
            #TODO is this right ?
            scatter_idx = torch.arange(batch_start_node_id, 
                                       batch_start_node_id + x_feat.shape[0],
                                       device="cuda")     
            wgth.wholememory_scatter_functor(x_feat, scatter_idx, output_feat.get_embedding_tensor().wmb_tensor)
            assert output_feat.shape[0] == graph_structure.node_count and output_feat.dim() == 2

    def forward(self, src_ids, pos_dst_ids, neg_dst_ids):
        assert src_ids.shape == pos_dst_ids.shape and src_ids.shape == neg_dst_ids.shape
        id_count = src_ids.size(0)
        ids = torch.cat([src_ids, pos_dst_ids, neg_dst_ids])
        # add both forward and reverse edge into hashset
        #TODO remove this structure, is this ok?
        #exclude_edge_hashset = torch.ops.wholegraph.create_edge_hashset(
        #    torch.cat([src_ids, pos_dst_ids]), torch.cat([pos_dst_ids, src_ids])
        #)
        ids_unique, reverse_map = torch.unique(ids, return_inverse=True)
        out_feat_unique = self.gnn_forward(ids_unique)
        out_feat = torch.nn.functional.embedding(reverse_map, out_feat_unique)
        src_feat, pos_dst_feat, neg_dst_feat = torch.split(out_feat, id_count)
        scores = self.predict(
            torch.cat([src_feat, src_feat]), torch.cat([pos_dst_feat, neg_dst_feat])
        )
        return scores[:id_count], scores[id_count:]


def compute_mrr(model, node_emb, src, dst, neg_dst, batch_size=1024):
    rr = torch.zeros(src.shape[0])
    embedding_lookup_fn = wgth.embedding.EmbeddingLookupFn.apply
    evaluator = Evaluator(name="ogbl-citation2")
    preds = []
    for start in range(0, src.shape[0], batch_size):
        end = min(start + batch_size, src.shape[0])
        all_dst = torch.cat([dst[start:end, None], neg_dst[start:end]], 1)
        #TODO what is dummy_input
        h_src = embedding_lookup_fn(src[start:end], torch.tensor([]).cuda(), node_emb)[:, None, :]
        h_dst = embedding_lookup_fn(all_dst.view(-1), torch.tensor([]).cuda(), node_emb).view(*all_dst.shape, -1)
        pred = model.predict(h_src, h_dst).squeeze(-1)
        relevance = torch.zeros(*pred.shape, dtype=torch.bool, device='cuda')#TODO why need explict 'cuda'
        relevance[:, 0] = True
        rr[start:end] = MF.retrieval_reciprocal_rank(pred, relevance)
        preds += [pred]
    all_pred = torch.cat(preds)
    pos_pred = all_pred[:, :1].squeeze(1)
    neg_pred = all_pred[:, 1:]
    ogb_mrr = (
        evaluator.eval(
            {
                "y_pred_pos": pos_pred,
                "y_pred_neg": neg_pred,
            }
        )["mrr_list"]
        .mean()
        .item()
    )
    return rr.mean().item(), ogb_mrr


@torch.no_grad()
def evaluate(model: EdgePredictionGNNModel, graph_structure, _embedding:WholeMemoryEmbedding, edge_split, local_comm):
    model.eval()
    node_feats = [None, None]
    embedding = _embedding
    #wm_tensor_type = get_intra_node_wm_tensor_type(use_chunked, use_host_memory)
    node_feats[0] = wgth.create_embedding(
        local_comm, #TODO is this right ?
        options.embedding_memory_type,#TODO is this right ?
        "cuda",
        torch.float,#TODO
        [embedding.shape[0], options.hiddensize],
    )
    if options.layernum > 1:
        node_feats[1] = wgth.create_embedding(
            local_comm, #TODO is this right ?
            options.embedding_memory_type,#TODO is this right ?
            "cuda",
            torch.float,
            [embedding.shape[0], options.hiddensize],
        )
    output_feat = node_feats[0]
    input_feat = embedding
    del embedding
    for i in range(options.layernum):
        model.fullbatch_single_layer_forward(
            graph_structure, i, input_feat, output_feat, 1024
        )
        local_comm.barrier()
        input_feat = output_feat
        output_feat = node_feats[(i + 1) % 2]
    del output_feat
    del node_feats[1]
    del node_feats[0]
    del node_feats
    dgl_mrr_results = []
    ogb_mrr_results = []
    for split in ["valid", "test"]:
        src = torch.from_numpy(edge_split[split]["source_node"]).cuda()
        dst = torch.from_numpy(edge_split[split]["target_node"]).cuda()
        neg_dst = torch.from_numpy(edge_split[split]["target_node_neg"]).cuda()
        dgl_mrr, ogb_mrr = compute_mrr(model, input_feat, src, dst, neg_dst)
        dgl_mrr_results.append(dgl_mrr)
        ogb_mrr_results.append(ogb_mrr)
    return dgl_mrr_results, ogb_mrr_results

def train(graph_structure, node_feat_wm_embedding:WholeMemoryEmbedding, model, optimizer, raw_model, edge_split, global_comm, local_comm):
#def train(dist_homo_graph, model, optimizer, raw_model, edge_split):
    print("start training...")
    train_step = 0
    epoch = 0
    train_start_time = time.time()
    dgl_mrr, ogb_mrr = evaluate(raw_model, graph_structure, node_feat_wm_embedding, edge_split, local_comm)
    if wgth.get_rank() == 0:
        print(
            "@Epoch",
            epoch,
            ", Validation DGL MRR:",
            dgl_mrr[0],
            "Test DGL MRR:",
            dgl_mrr[1],
            "Validation OGB MRR:",
            ogb_mrr[0],
            "Test OGB MRR:",
            ogb_mrr[1],
        )
    while epoch < options.epochs:
        epoch_iter_count = graph_structure.start_iter(options.batchsize)
        if wgth.get_rank() == 0:
            print("%d steps for epoch %d." % (epoch_iter_count, epoch))
        iter_id = 0
        while iter_id < epoch_iter_count:
            #TODO: correctness
            src_nid, pos_dst_nid = graph_structure.get_train_edge_batch(iter_id)
            #neg_dst_nid = torch.randint_like(src_nid, 0, graph_structure.node_count)
            neg_dst_nid = graph_structure.per_source_negative_sample(src_nid)
            optimizer.zero_grad()
            model.train()
            #print(iter_id, epoch_iter_count)
            #if (iter_id == 276):
            #    print(src_nid, pos_dst_nid, neg_dst_nid)
            pos_score, neg_score = model(src_nid, pos_dst_nid, neg_dst_nid)
            pos_label = torch.ones_like(pos_score)
            neg_label = torch.zeros_like(neg_score)
            score = torch.cat([pos_score, neg_score])
            labels = torch.cat([pos_label, neg_label])
            loss = F.binary_cross_entropy_with_logits(score, labels)
            loss.backward()
            optimizer.step()
            if wgth.get_rank() == 0 and train_step % 100 == 0:
                print(
                    "[%s] [LOSS] step=%d, loss=%f"
                    % (
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        train_step,
                        loss.cpu().item(),
                    )
                )
            train_step = train_step + 1
            iter_id = iter_id + 1
        epoch = epoch + 1
        dgl_mrr, ogb_mrr = evaluate(raw_model, graph_structure, node_feat_wm_embedding, edge_split, local_comm)
        if wgth.get_rank() == 0:
            print(
                "@Epoch",
                epoch,
                ", Validation DGL MRR:",
                dgl_mrr[0],
                "Test DGL MRR:",
                dgl_mrr[1],
                "Validation OGB MRR:",
                ogb_mrr[0],
                "Test OGB MRR:",
                ogb_mrr[1],
            )
    global_comm.barrier()
    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    if wgth.get_rank() == 0:
        print(
            "[%s] [TRAIN_TIME] train time is %.2f seconds"
            % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), train_time)
        )
        print("[EPOCH_TIME] %.2f seconds" % (train_time / options.epochs,))
#TODO move this function to better place
def load_pickle_link_pred_data(file_path: str):
    import pickle
    with open(file_path, "rb") as f:
        valid_and_test = pickle.load(f)
    return valid_and_test

def main_func():
    print("total rank num is", wgth.get_world_size())
    print(f"Rank={wgth.get_rank()}, local_rank={wgth.get_local_rank()}")
    #TODO now thread_num = 1, more threads to be supported. (device set accordingly)
    global_comm, local_comm = wgth.init_torch_env_and_create_wm_comm(
        wgth.get_rank(),
        wgth.get_world_size(),
        wgth.get_local_rank(),
        wgth.get_local_size(),
    )

    if options.use_cpp_ext:
        wgth.compile_cpp_extension()
    edge_split = load_pickle_link_pred_data(options.pickle_data_path)

    graph_structure = wgth.GraphStructure()
    graph_structure_wholememory_type = "chunked"
    graph_structure_wholememory_location = "cuda"
    csr_row_ptr_wm_tensor = wgth.create_wholememory_tensor_from_filelist(
        local_comm,
        graph_structure_wholememory_type,
        graph_structure_wholememory_location,
        os.path.join(options.root_dir, "homograph_csr_row_ptr"),
        torch.int64,
    )
    csr_col_ind_wm_tensor = wgth.create_wholememory_tensor_from_filelist(
        local_comm,
        graph_structure_wholememory_type,
        graph_structure_wholememory_location,
        os.path.join(options.root_dir, "homograph_csr_col_idx"),
        torch.int,
    )
    coo_row_wm_tensor = wgth.create_wholememory_tensor_from_filelist(
        local_comm,
        graph_structure_wholememory_type,
        graph_structure_wholememory_location,
        os.path.join(options.root_dir, "homograph_coo_row_ptr"),
        torch.int32,
    )
    graph_structure.set_graph(csr_row_ptr_wm_tensor, csr_col_ind_wm_tensor, coo_row_wm_tensor,
                              graph_structure_wholememory_type, graph_structure_wholememory_location)
    graph_structure.prepare_train_edges()

    feature_comm = global_comm if options.use_global_embedding else local_comm

    embedding_wholememory_type = options.embedding_memory_type
    embedding_wholememory_location = (
        "cpu" if options.cache_type != "none" or options.cache_ratio == 0.0 else "cuda"
    )
    if options.cache_ratio == 0.0:
        options.cache_type = "none"
    access_type = "readonly" if options.train_embedding is False else "readwrite"
    if wgth.get_rank() == 0:
        print(
            f"graph_structure: type={graph_structure_wholememory_type}, "
            f"location={graph_structure_wholememory_location}\n"
            f"embedding: type={embedding_wholememory_type}, location={embedding_wholememory_location}, "
            f"cache_type={options.cache_type}, cache_ratio={options.cache_ratio}, "
            f"trainable={options.train_embedding}"
        )
    cache_policy = wgth.create_builtin_cache_policy(
        options.cache_type,
        embedding_wholememory_type,
        embedding_wholememory_location,
        access_type,
        options.cache_ratio,
    )

    wm_optimizer = (
        None
        if options.train_embedding is False
        else wgth.create_wholememory_optimizer("adam", {})
    )

    if wm_optimizer is None:
        node_feat_wm_embedding = wgth.create_embedding_from_filelist(
            feature_comm,
            embedding_wholememory_type,
            embedding_wholememory_location,
            os.path.join(options.root_dir, "node_feat.bin"),
            torch.float,
            options.feat_dim,
            optimizer=wm_optimizer,
            cache_policy=cache_policy,
        )
    else:
        node_feat_wm_embedding = wgth.create_embedding(
            feature_comm,
            embedding_wholememory_type,
            embedding_wholememory_location,
            torch.float,
            [graph_structure.node_count, options.feat_dim],
            optimizer=wm_optimizer,
            cache_policy=cache_policy,
            random_init=True,
        )

    wgth.set_framework(options.framework)
    print("Rank=%d, Graph loaded." % (wgth.get_local_rank(),))
    raw_model = EdgePredictionGNNModel(graph_structure, node_feat_wm_embedding, options)
    print("Rank=%d, model created." % (wgth.get_local_rank(),))

    #model = wgth.HomoGNNModel(graph_structure, node_feat_wm_embedding, options)
    raw_model.cuda()
    print("Rank=%d, model movded to cuda." % (wgth.get_local_rank(),))
    model = DDP(raw_model, delay_allreduce=True)
    optimizer = apex.optimizers.FusedAdam(model.parameters(), lr=options.lr, weight_decay=5e-4)
    print("Rank=%d, optimizer created." % (wgth.get_local_rank(),))
    train(graph_structure, node_feat_wm_embedding, model, optimizer, raw_model, edge_split, global_comm, local_comm)
    #test(graph_structure, model)

    wgth.finalize()
if __name__ == "__main__":
    wgth.distributed_launch(options, main_func)