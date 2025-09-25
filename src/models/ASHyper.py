import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import degree, softmax
import torch_scatter
from torch_geometric.data import data as D
from torch.nn import Linear
from torch_geometric.utils import scatter

from src.models.pl_bases.default_module import DefaultPLModule


class SelfAttentionLayer(nn.Module):
    def __init__(self, configs):
        super(SelfAttentionLayer, self).__init__()
        self.query_weight = nn.Linear(configs.enc_in, configs.enc_in)
        self.key_weight = nn.Linear(configs.enc_in, configs.enc_in)
        self.value_weight = nn.Linear(configs.enc_in, configs.enc_in)

    def forward(self, x):
        q = self.query_weight(x)
        k = self.key_weight(x)
        v = self.value_weight(x)
        attention_scores = F.softmax(torch.matmul(q, k.transpose(1, 2)) / (k.shape[-1] ** 0.5), dim=-1)
        attended_values = torch.matmul(attention_scores, v)
        return attended_values


class ConvLayer(nn.Module):
    def __init__(self, c_in, window_size):
        super(ConvLayer, self).__init__()
        self.down_conv = nn.Conv1d(in_channels=c_in, out_channels=c_in,
                                  kernel_size=window_size, stride=window_size)
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.down_conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class ConvConstruct(nn.Module):
    def __init__(self, d_model, window_size, d_inner):
        super(ConvConstruct, self).__init__()
        if not isinstance(window_size, list):
            self.conv_layers = nn.ModuleList([ConvLayer(d_model, window_size)])
        else:
            self.conv_layers = nn.ModuleList()
            for ws in window_size:
                self.conv_layers.append(ConvLayer(d_model, ws))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        all_inputs = []
        enc_input = enc_input.permute(0, 2, 1)
        all_inputs.append(enc_input.permute(0, 2, 1))
        
        for conv_layer in self.conv_layers:
            enc_input = conv_layer(enc_input)
            all_inputs.append(enc_input.permute(0, 2, 1))
        
        return all_inputs


class HypergraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=False, heads=1, 
                 concat=True, negative_slope=0.2, dropout=0.1, bias=False):
        super(HypergraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        self.dropout = dropout
        
        # Linear transformation
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        
        # Simplified approach - disable attention to avoid dimension issues
        self.use_attention = False
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        if self.bias is not None:
            zeros(self.bias)


    
    def forward(self, x, hyperedge_index):
        # Apply linear transformation
        x_transformed = torch.matmul(x, self.weight)
        if self.bias is not None:
            x_transformed = x_transformed + self.bias
        
        # Simplified hypergraph convolution without complex message passing
        # This avoids the dimension mismatch issues
        
        # Basic hypergraph aggregation
        batch_size, seq_len, features = x_transformed.shape
        
        try:
            # Create a simple aggregation based on hyperedge structure
            node_indices = hyperedge_index[0]
            edge_indices = hyperedge_index[1]
            
            # Bounds checking
            valid_mask = (node_indices < seq_len) & (node_indices >= 0)
            node_indices = node_indices[valid_mask]
            edge_indices = edge_indices[valid_mask]
            
            if len(node_indices) == 0:
                return x_transformed, torch.tensor(0.0, device=x.device)
            
            # Simple aggregation: average pooling within hyperedges
            unique_edges = torch.unique(edge_indices)
            aggregated_features = x_transformed.clone()
            
            for edge_id in unique_edges:
                edge_mask = (edge_indices == edge_id)
                nodes_in_edge = node_indices[edge_mask]
                
                if len(nodes_in_edge) > 1:
                    # Average the features of nodes in this hyperedge
                    edge_features = x_transformed[:, nodes_in_edge, :].mean(dim=1, keepdim=True)
                    # Broadcast back to all nodes in the edge
                    aggregated_features[:, nodes_in_edge, :] = edge_features
            
            # Simple constraint loss
            constraint_loss = torch.mean(torch.abs(aggregated_features - x_transformed))
            
            return aggregated_features, constraint_loss
            
        except Exception as e:
            print(f"Warning: Hypergraph convolution failed: {e}")
            return x_transformed, torch.tensor(0.0, device=x.device)
    
    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels, self.out_channels)


class MultiAdaptiveHypergraph(nn.Module):
    def __init__(self, configs):
        super(MultiAdaptiveHypergraph, self).__init__()
        self.seq_len = configs.seq_len
        self.window_size = getattr(configs, 'window_size', [4, 4])
        self.inner_size = getattr(configs, 'inner_size', 4)
        self.dim = getattr(configs, 'd_model', 512)
        self.hyper_num = getattr(configs, 'hyper_num', [50, 30])
        self.alpha = 3
        self.k = getattr(configs, 'k', 10)
        
        self.embed_hy = nn.ModuleList()
        self.embed_nod = nn.ModuleList()
        self.lin_hy = nn.ModuleList()
        self.lin_nod = nn.ModuleList()
        
        for i in range(len(self.hyper_num)):
            self.embed_hy.append(nn.Embedding(self.hyper_num[i], self.dim))
            self.lin_hy.append(nn.Linear(self.dim, self.dim))
            self.lin_nod.append(nn.Linear(self.dim, self.dim))
            
            if i == 0:
                self.embed_nod.append(nn.Embedding(self.seq_len, self.dim))
            else:
                product = math.prod(self.window_size[:i])
                layer_size = math.floor(self.seq_len / product)
                self.embed_nod.append(nn.Embedding(int(layer_size), self.dim))
        
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        node_num = []
        node_num.append(self.seq_len)
        for i in range(len(self.window_size)):
            layer_size = math.floor(node_num[i] / self.window_size[i])
            node_num.append(layer_size)
        
        hyperedge_all = []
        
        for i in range(len(self.hyper_num)):
            # Ensure we have valid node numbers
            current_node_num = max(1, int(node_num[i]))
            hyp_idx = torch.arange(self.hyper_num[i]).to(x.device)
            node_idx = torch.arange(current_node_num).to(x.device)
            
            # Handle embedding dimension mismatch
            try:
                hyper_embed = self.embed_hy[i](hyp_idx)
                node_embed = self.embed_nod[i](node_idx)
            except Exception as e:
                print(f"Warning: Embedding error at layer {i}: {e}")
                # Create simple adjacency matrix as fallback
                simple_nodes = torch.arange(min(current_node_num, self.hyper_num[i])).tolist()
                simple_edges = torch.arange(len(simple_nodes)).tolist()
                hypergraph = np.vstack((simple_nodes, simple_edges))
                hyperedge_all.append(hypergraph)
                continue
            
            a = torch.mm(node_embed, hyper_embed.transpose(1, 0))
            adj = F.softmax(F.relu(self.alpha * a), dim=-1)
            mask = torch.zeros(node_embed.size(0), hyper_embed.size(0)).to(x.device)
            mask.fill_(float('0'))
            
            # Ensure k is not larger than available hyperedges
            k_actual = min(adj.size(1), self.k)
            if k_actual > 0:
                s1, t1 = adj.topk(k_actual, 1)
                mask.scatter_(1, t1, s1.fill_(1))
                adj = adj * mask
                adj = torch.where(adj > 0.5, torch.tensor(1).to(x.device), torch.tensor(0).to(x.device))
                adj = adj[:, (adj != 0).any(dim=0)]
                
                if adj.size(1) > 0:  # Ensure we have valid adjacency
                    matrix_array = torch.tensor(adj, dtype=torch.int)
                    result_list = [list(torch.nonzero(matrix_array[:, col]).flatten().tolist()) for col in
                                  range(matrix_array.shape[1])]
                    
                    # Handle empty result_list
                    if any(len(sublist) > 0 for sublist in result_list):
                        node_list = torch.cat([torch.tensor(sublist) for sublist in result_list if len(sublist) > 0]).tolist()
                        count_list = list(torch.sum(adj, dim=0).tolist())
                        hyperedge_list = torch.cat([torch.full((int(count),), idx) for idx, count in enumerate(count_list, start=0) if count > 0]).tolist()
                        hypergraph = np.vstack((node_list, hyperedge_list))
                    else:
                        # Fallback: create simple hypergraph
                        simple_nodes = list(range(min(current_node_num, 5)))
                        simple_edges = list(range(len(simple_nodes)))
                        hypergraph = np.vstack((simple_nodes, simple_edges))
                else:
                    # Fallback: create simple hypergraph
                    simple_nodes = list(range(min(current_node_num, 5)))
                    simple_edges = list(range(len(simple_nodes)))
                    hypergraph = np.vstack((simple_nodes, simple_edges))
            else:
                # Fallback: create simple hypergraph
                simple_nodes = list(range(min(current_node_num, 5)))
                simple_edges = list(range(len(simple_nodes)))
                hypergraph = np.vstack((simple_nodes, simple_edges))
                
            hyperedge_all.append(hypergraph)
        
        return hyperedge_all


def get_mask(input_size, window_size):
    all_size = [input_size]
    for ws in window_size:
        layer_size = max(1, all_size[-1] // ws)
        all_size.append(layer_size)
    return all_size


class ASHyper(DefaultPLModule):
    def __init__(self, configs, *args, **kwargs):
        super(ASHyper, self).__init__(configs)
        
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.individual = getattr(configs, 'individual', False)
        
        self.window_size = getattr(configs, 'window_size', [4, 4])
        self.hyper_num = getattr(configs, 'hyper_num', [50, 30])
        
        # Get all layer sizes
        self.all_size = get_mask(configs.seq_len, configs.window_size)
        self.ms_length = sum(self.all_size)
        
        # Linear layers
        if self.individual:
            self.linear = nn.ModuleList()
            for i in range(self.channels):
                self.linear.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.linear = nn.Linear(self.seq_len, self.pred_len)
            self.linear_tran = nn.Linear(self.pred_len, self.pred_len)

        # Convolution layers
        self.conv_layers = ConvConstruct(self.channels, self.window_size, self.channels)
        
        # Transformation layers
        self.out_tran = nn.Linear(self.ms_length, self.pred_len)
        self.out_tran.weight = nn.Parameter((1/self.ms_length) * torch.ones([self.pred_len, self.ms_length]))
        self.chan_tran = nn.Linear(getattr(configs, 'd_model', 512), configs.enc_in)
        self.inter_tran = nn.Linear(80, self.pred_len)
        self.concat_tran = nn.Linear(320, self.pred_len)
        
        # Model parameters
        self.dim = getattr(configs, 'd_model', 512)
        self.hyper_num_single = 50
        self.embed_hy = nn.Embedding(self.hyper_num_single, self.dim)
        self.embed_nod = nn.Embedding(self.ms_length, self.dim)
        
        self.idx = torch.arange(self.hyper_num_single)
        self.nod_idx = torch.arange(self.ms_length)
        self.alpha = 3
        self.k = 10

        # Hypergraph components
        self.multi_adp_hyper = MultiAdaptiveHypergraph(configs)
        self.hyper_num_list = self.hyper_num
        self.hy_conv = nn.ModuleList()
        self.hyperedge_atten = SelfAttentionLayer(configs)
        
        for i in range(len(self.hyper_num_list)):
            self.hy_conv.append(HypergraphConv(configs.enc_in, configs.enc_in))
        
        # Additional transformation layers
        self.slice_tran = nn.Linear(100, configs.pred_len)
        self.weight = nn.Parameter(torch.randn(self.pred_len, 76))
        
        self.argg = nn.ModuleList()
        for i in range(len(self.hyper_num_list)):
            self.argg.append(nn.Linear(self.all_size[i], self.pred_len))
        self.chan_tran = nn.Linear(configs.enc_in, configs.enc_in)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, *args, **kwargs):
        x = x_enc
        
        # Normalization
        mean_enc = x.mean(1, keepdim=True).detach()
        x = x - mean_enc
        std_enc = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / std_enc
        
        # Generate hypergraph adjacency matrices
        adj_matrix = self.multi_adp_hyper(x)
        
        # Multi-scale convolution
        seq_enc = self.conv_layers(x)
        
        # Calculate actual dimensions from conv output
        actual_sizes = [seq_enc[0].size(1)]  # Original input size
        for i in range(1, len(seq_enc)):
            actual_sizes.append(seq_enc[i].size(1))
        actual_ms_length = sum(actual_sizes)
        
        # Process hypergraph convolution for each scale
        sum_hyper_list = []
        result_con_loss = 0
        
        for i in range(len(self.hyper_num_list)):
            if i >= len(adj_matrix) or i >= len(seq_enc):
                continue
                
            mask = torch.tensor(adj_matrix[i]).to(x.device)
            
            # Check if mask is valid
            if mask.size(0) == 0 or mask.size(1) == 0:
                continue
            
            # Inter-scale processing
            node_value = seq_enc[i].permute(0, 2, 1)
            node_value = torch.tensor(node_value).to(x.device)
            edge_sums = {}
            
            # Safe processing of mask indices
            try:
                for edge_id, node_id in zip(mask[1], mask[0]):
                    edge_id_val = edge_id.item()
                    node_id_val = node_id.item()
                    
                    # Check bounds
                    if node_id_val < node_value.size(2):
                        if edge_id_val not in edge_sums:
                            edge_sums[edge_id_val] = node_value[:, :, node_id_val]
                        else:
                            edge_sums[edge_id_val] += node_value[:, :, node_id_val]
            except Exception as e:
                print(f"Warning: Error processing mask at layer {i}: {e}")
                continue
            
            # Add edge sums to list
            for edge_id, sum_value in edge_sums.items():
                sum_value = sum_value.unsqueeze(1)
                sum_hyper_list.append(sum_value)
            
            # Intra-scale processing
            try:
                output, constrain_loss = self.hy_conv[i](seq_enc[i], mask)
                
                if i == 0:
                    result_tensor = output
                    result_con_loss = constrain_loss
                else:
                    result_tensor = torch.cat((result_tensor, output), dim=1)
                    result_con_loss += constrain_loss
            except Exception as e:
                print(f"Warning: Hypergraph convolution failed at layer {i}: {e}")
                # Use original sequence as fallback
                if i == 0:
                    result_tensor = seq_enc[i]
                    result_con_loss = torch.tensor(0.0, device=x.device)
                else:
                    result_tensor = torch.cat((result_tensor, seq_enc[i]), dim=1)
                    result_con_loss += torch.tensor(0.0, device=x.device)
        
        # Store constraint loss for training
        self.constraint_loss = result_con_loss
        
        # Process hyperedge attention with safety checks
        if len(sum_hyper_list) > 0:
            sum_hyper_list = torch.cat(sum_hyper_list, dim=1)
            sum_hyper_list = sum_hyper_list.to(x.device)
            padding_need = max(0, 80 - sum_hyper_list.size(1))
            hyperedge_attention = self.hyperedge_atten(sum_hyper_list)
            pad = torch.nn.functional.pad(hyperedge_attention, (0, 0, 0, padding_need, 0, 0))
        else:
            # Create dummy pad when no hyperedge features are available
            batch_size = x.size(0)
            pad = torch.zeros(batch_size, 80, self.channels).to(x.device)
        
        # Linear transformation
        if self.individual:
            output = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.linear[i](x[:, :, i])
            x = output
        else:
            x = self.linear(x.permute(0, 2, 1))
            
            # Handle dimension mismatch for out_tran
            result_tensor_permuted = result_tensor.permute(0, 2, 1)
            actual_seq_len = result_tensor_permuted.size(-1)
            expected_seq_len = self.ms_length
            
            if actual_seq_len != expected_seq_len:
                # print(f"Warning: Dimension mismatch. Expected: {expected_seq_len}, Got: {actual_seq_len}")
                # Adjust out_tran to match actual dimensions
                if not hasattr(self, 'out_tran_adjusted') or self.out_tran_adjusted.in_features != actual_seq_len:
                    self.out_tran_adjusted = nn.Linear(actual_seq_len, self.pred_len).to(x.device)
                    # Initialize with similar weights as original
                    if actual_seq_len <= expected_seq_len:
                        self.out_tran_adjusted.weight.data = self.out_tran.weight.data[:, :actual_seq_len]
                    else:
                        # Pad with zeros or repeat
                        padded_weight = torch.zeros(self.pred_len, actual_seq_len, device=x.device)
                        padded_weight[:, :expected_seq_len] = self.out_tran.weight.data
                        self.out_tran_adjusted.weight.data = padded_weight
                x_out = self.out_tran_adjusted(result_tensor_permuted)
            else:
                x_out = self.out_tran(result_tensor_permuted)
            
            x_out_inter = self.inter_tran(pad.permute(0, 2, 1))
            x = x_out + x + x_out_inter
            x = self.linear_tran(x).permute(0, 2, 1)
        
        # Denormalization
        x = x * std_enc + mean_enc
        return x
    
    def training_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()

        dec_inp = self._build_decoder_input(batch_y)
        outputs = self(
            x_enc=batch_x,
            x_mark_enc=batch_x_mark,
            x_dec=dec_inp,
            x_mark_dec=batch_y_mark,
        )

        outputs = outputs[:, -self.args.dataset_args.pred_len :, :]
        batch_y = batch_y[:, -self.args.dataset_args.pred_len :, :]

        if self.target_var_index != -1:
            batch_y = batch_y[:, :, self.target_var_index].unsqueeze(-1)

        criterion = self._get_criterion(self.args.training_args.loss_func)
        prediction_loss = criterion(outputs, batch_y)
        
        # Add constraint loss if available
        total_loss = prediction_loss
        if hasattr(self, 'constraint_loss'):
            total_loss = prediction_loss + self.constraint_loss
            self.log("constraint_loss", self.constraint_loss, on_epoch=True)
        
        self.log("train_loss", total_loss, on_epoch=True)
        self.log("prediction_loss", prediction_loss, on_epoch=True)

        return total_loss