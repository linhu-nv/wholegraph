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

import torch
from typing import Union, List
from .tensor import WholeMemoryTensor
from . import graph_ops
from . import wholegraph_ops
import pylibwholegraph.torch as wgth
import numpy as np


class GraphStructure(object):
    r"""Graph structure storage
    Actually, it is the graph structure of one relation, represented in CSR format.
    It contains CSR representation of Graph structure, and also attributes associated with nodes and edges.
    """

    def __init__(self):
        super().__init__()
        self.node_count = 0
        self.edge_count = 0
        self.csr_row_ptr = None
        self.csr_col_ind = None
        self.coo_row = None
        self.memory_type = "chunked"
        self.memory_location = "cuda"
        self.id_type = torch.int32
        self.node_attributes = {}
        self.edge_attributes = {}

    def set_graph(
        self, csr_row_ptr: WholeMemoryTensor, csr_col_ind: WholeMemoryTensor, coo_row: WholeMemoryTensor = None,
        memory_type = "chunked", memory_location = "cuda", id_type = torch.int32
    ):
        """
        Set the CSR graph structure
        :param csr_row_ptr: CSR graph row pointer
        :param csr_col_ind: CSR graph column index
        :return: None
        """
        assert csr_row_ptr.dim() == 1
        assert csr_row_ptr.dtype == torch.int64
        assert csr_row_ptr.shape[0] > 1
        self.node_count = csr_row_ptr.shape[0] - 1
        self.edge_count = csr_col_ind.shape[0]
        assert csr_col_ind.dim() == 1
        assert csr_col_ind.dtype == torch.int32 or csr_col_ind.dtype == torch.int64
        self.csr_row_ptr = csr_row_ptr
        self.csr_col_ind = csr_col_ind
        if not coo_row == None:
            assert coo_row.dim() == 1
            assert coo_row.dtype == torch.int32 or coo_row.dtype == torch.int64
            self.coo_row = coo_row
        self.memory_type = memory_type
        self.memory_location = memory_location
        self.id_type = id_type

    def set_node_attribute(self, attr_name: str, attr_tensor: WholeMemoryTensor):
        """
        Set attribute for node
        :param attr_name: attribute name for node
        :param attr_tensor: attribute tensor
        :return: None
        """
        assert attr_name not in self.node_attributes
        assert attr_tensor.shape[0] == self.node_count
        self.node_attributes[attr_name] = attr_tensor

    def prepare_train_edges(self):#TODO what about rank num > 1?
        self.start_edge_idx = self.edge_count * wgth.get_rank() // wgth.get_world_size()
        self.end_edge_idx = (
            self.edge_count * (wgth.get_rank() + 1) // wgth.get_world_size()
        )
        self.truncate_count = self.edge_count // wgth.get_world_size()

    def start_iter(self, batch_size):
        self.batch_size = batch_size
        local_edge_count = self.end_edge_idx - self.start_edge_idx
        selected_count = self.truncate_count // batch_size * batch_size
        self.train_edge_idx_list = (
            torch.randperm(
                local_edge_count, dtype=torch.int64, device="cpu", pin_memory=True
            )
            + self.start_edge_idx
        )
        self.train_edge_idx_list = self.train_edge_idx_list[:selected_count]
        return selected_count // batch_size

    def get_train_edge_batch(self, iter_id):
        start_idx = iter_id * self.batch_size
        end_idx = (iter_id + 1) * self.batch_size
        if (self.memory_type == "chunked" and self.memory_location == "cuda"):
            src_nid_tensors = self.coo_row.get_all_chunked_tensor()
            dst_nid_tensors = self.csr_col_ind.get_all_chunked_tensor()
            src_nid_tensor = torch.cat(src_nid_tensors)
            dst_nid_tensor = torch.cat(dst_nid_tensors)
            assert src_nid_tensor.dim() == 1 
            assert dst_nid_tensor.dim() == 1
            #src_nid = src_nid_tensor[start_idx: end_idx]
            #dst_nid = dst_nid_tensor[start_idx: end_idx]
            idx_tensor = torch.tensor(self.train_edge_idx_list[start_idx:end_idx], device='cuda')
            src_nid = torch.gather(src_nid_tensor, 0, idx_tensor)
            dst_nid = torch.gather(dst_nid_tensor, 0, idx_tensor)
            return src_nid, dst_nid
        else:
            print ("not implemented yet")#TODO implement different memory types and locations
            src_nid = torch.tensor(1, end_idx - start_idx)
            dst_nid = torch.tensor(1, end_idx - start_idx)
            return src_nid, dst_nid
    
    def per_source_negative_sample(self, src_nid):
        return torch.tensor(np.random.randint(0, self.node_count, src_nid.shape[0]), device = 'cuda')

    def set_edge_attribute(self, attr_name: str, attr_tensor: WholeMemoryTensor):
        """
        Set attribute for edge
        :param attr_name: attribute name for edge
        :param attr_tensor: attribute tensor
        :return: None
        """
        assert attr_name not in self.edge_attributes
        assert attr_tensor.shape[0] == self.edge_count
        self.edge_attributes[attr_name] = attr_tensor

    def unweighted_sample_without_replacement_one_hop(
        self,
        center_nodes_tensor: torch.Tensor,
        max_sample_count: int,
        *,
        random_seed: Union[int, None] = None,
        need_center_local_output: bool = False,
        need_edge_output: bool = False
    ):
        """
        Unweighted Sample without replacement on CSR graph structure
        :param center_nodes_tensor: center node ids
        :param max_sample_count: max sample count for each center node
        :param random_seed: random seed for the sampler
        :param need_center_local_output: If True, output a tensor same length as sampled nodes but each element is the
            center node index in center_nodes_tensor.
        :param need_edge_output: If True, output the edge index of each sampled node
        :return: csr_row_ptr, sampled_nodes[, center_node_local_id, edge_index]
        """
        return wholegraph_ops.unweighted_sample_without_replacement(
            self.csr_row_ptr.wmb_tensor,
            self.csr_col_ind.wmb_tensor,
            center_nodes_tensor,
            max_sample_count,
            random_seed,
            need_center_local_output,
            need_edge_output,
        )

    def weighted_sample_without_replacement_one_hop(
        self,
        weight_name: str,
        center_nodes_tensor: torch.Tensor,
        max_sample_count: int,
        *,
        random_seed: Union[int, None] = None,
        need_center_local_output: bool = False,
        need_edge_output: bool = False
    ):
        """
        Weighted Sample without replacement on CSR graph structure with edge weights attribute
        :param weight_name: edge attribute name for weight
        :param center_nodes_tensor: center node ids
        :param max_sample_count: max sample count for each center node
        :param random_seed: random seed for the sampler
        :param need_center_local_output: If True, output a tensor same length as sampled nodes but each element is the
            center node index in center_nodes_tensor.
        :param need_edge_output: If True, output the edge index of each sampled node
        :return: csr_row_ptr, sampled_nodes[, center_node_local_id, edge_index]
        """
        assert weight_name in self.edge_attributes
        weight_tensor = self.edge_attributes[weight_name]
        return wholegraph_ops.weighted_sample_without_replacement(
            self.csr_row_ptr.wmb_tensor,
            self.csr_col_ind.wmb_tensor,
            weight_tensor.wmb_tensor,
            center_nodes_tensor,
            max_sample_count,
            random_seed,
            need_center_local_output,
            need_edge_output,
        )

    def multilayer_sample_without_replacement(
        self,
        node_ids: torch.Tensor,
        max_neighbors: List[int],
        weight_name: Union[str, None] = None,
    ):
        """
        Multilayer sample without replacement
        :param node_ids: initial node ids
        :param max_neighbors: maximum neighbor for each layer
        :param weight_name: edge attribute name for weight, if None, use unweighted sample
        :return: target_gids, edge_indice, csr_row_ptr, csr_col_ind
        """
        hops = len(max_neighbors)
        edge_indice = [None] * hops
        csr_row_ptr = [None] * hops
        csr_col_ind = [None] * hops
        target_gids = [None] * (hops + 1)
        target_gids[hops] = node_ids
        #print("this is the start of multilayer sample")
        for i in range(hops - 1, -1, -1):
            if weight_name is None:
                #if (node_ids.size(0) == 2039):
                #    print(len(target_gids[i + 1]), target_gids[i+1][0])
                #gid_copy = target_gids[i+1].clone().detach()
                #print("max neighbor ", max_neighbors[hops-i-1])
                smallTenor = torch.tensor([1,2,3,4,5], device='cuda')
                #print("Up: the gid size is ", gid_copy.shape[0], "first ele is ", gid_copy[0])
                (
                    neighbor_gids_offset,
                    neighbor_gids_vdata,
                    neighbor_src_lids,
                ) = self.unweighted_sample_without_replacement_one_hop(
                    target_gids[i + 1],
                    max_neighbors[hops - i - 1],
                    need_center_local_output=True,
                )
                if (neighbor_gids_offset == None
                    or neighbor_gids_vdata == None
                    or neighbor_src_lids == None):
                    #print("Down: the gid size is ", gid_copy.shape[0])
                    print(smallTenor)
                    #print("first ele is ", gid_copy[0])
                    #print("original id is ", target_gids[i+1])
                    #print(node_ids.size(0),len(target_gids[i + 1]))
                    #print("here is node_id:", target_gids[0])
                    #print("then is target_gid:", target_gids[0])
                    #print(target_gids[i+1])
            else:
                (
                    neighbor_gids_offset,
                    neighbor_gids_vdata,
                    neighbor_src_lids,
                ) = self.weighted_sample_without_replacement_one_hop(
                    weight_name,
                    target_gids[i + 1],
                    max_neighbors[hops - i - 1],
                    need_center_local_output=True,
                )
            (unique_gids, neighbor_raw_to_unique_mapping,) = graph_ops.append_unique(
                target_gids[i + 1],
                neighbor_gids_vdata,
                need_neighbor_raw_to_unique=True,
            )
            csr_row_ptr[i] = neighbor_gids_offset
            csr_col_ind[i] = neighbor_raw_to_unique_mapping
            neighbor_count = neighbor_gids_vdata.size()[0]
            edge_indice[i] = torch.cat(
                [
                    torch.reshape(neighbor_raw_to_unique_mapping, (1, neighbor_count)),
                    torch.reshape(neighbor_src_lids, (1, neighbor_count)),
                ]
            )
            target_gids[i] = unique_gids
        return target_gids, edge_indice, csr_row_ptr, csr_col_ind
