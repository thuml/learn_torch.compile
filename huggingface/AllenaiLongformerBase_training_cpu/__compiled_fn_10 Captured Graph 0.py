from __future__ import annotations



def forward(self, L_hidden_states_ : torch.Tensor, L_attention_mask_ : torch.Tensor, L_is_index_masked_ : torch.Tensor):
    l_hidden_states_ = L_hidden_states_
    l_attention_mask_ = L_attention_mask_
    l_is_index_masked_ = L_is_index_masked_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    hidden_states = l_hidden_states_.transpose(0, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    query_vectors = self.L__self___layer_0_attention_self_query(hidden_states)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    key_vectors = self.L__self___layer_0_attention_self_key(hidden_states)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    value_vectors = self.L__self___layer_0_attention_self_value(hidden_states);  hidden_states = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    query_vectors /= 8.0;  query_vectors_1 = query_vectors;  query_vectors = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:568, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view = query_vectors_1.view(1024, 1, 12, 64);  query_vectors_1 = None
    query_vectors_2 = view.transpose(0, 1);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:569, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_1 = key_vectors.view(1024, 1, 12, 64);  key_vectors = None
    key_vectors_1 = view_1.transpose(0, 1);  view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:830, code: chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
    div = torch.div(1024, 256, rounding_mode = 'trunc')
    chunks_count = div - 1;  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:833, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_3 = query_vectors_2.transpose(1, 2)
    query = transpose_3.reshape(12, 1024, 64);  transpose_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_4 = key_vectors_1.transpose(1, 2);  key_vectors_1 = None
    key = transpose_4.reshape(12, 1024, 64);  transpose_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_1 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_1 = query.view(12, div_1, 512, 64);  query = div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    query_1 = hidden_states_1.as_strided(size = [12, 3, 512, 64], stride = [64, 196608, 768, 1]);  hidden_states_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_2 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_2 = key.view(12, div_2, 512, 64);  key = div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    key_1 = hidden_states_2.as_strided(size = [12, 3, 512, 64], stride = [64, 196608, 768, 1]);  hidden_states_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    diagonal_chunked_attention_scores = torch.functional.einsum('bcxd,bcyd->bcxy', (query_1, key_1));  query_1 = key_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    hidden_states_padded = torch.nn.functional.pad(diagonal_chunked_attention_scores, (0, 0, 0, 1));  diagonal_chunked_attention_scores = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    diagonal_chunked_attention_scores_1 = hidden_states_padded.view(12, 3, 512, 513);  hidden_states_padded = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:856, code: (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
    add = chunks_count + 1;  chunks_count = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    diagonal_attention_scores = diagonal_chunked_attention_scores_1.new_zeros((12, add, 256, 513));  add = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem = diagonal_chunked_attention_scores_1[(slice(None, None, None), slice(None, None, None), slice(None, 256, None), slice(None, 257, None))]
    diagonal_attention_scores[(slice(None, None, None), slice(None, -1, None), slice(None, None, None), slice(256, None, None))] = getitem;  setitem = diagonal_attention_scores;  getitem = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_1 = diagonal_chunked_attention_scores_1[(slice(None, None, None), -1, slice(256, None, None), slice(None, 257, None))]
    diagonal_attention_scores[(slice(None, None, None), -1, slice(None, None, None), slice(256, None, None))] = getitem_1;  setitem_1 = diagonal_attention_scores;  getitem_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    getitem_2 = diagonal_chunked_attention_scores_1[(slice(None, None, None), slice(None, None, None), slice(-257, -1, None), slice(257, None, None))]
    diagonal_attention_scores[(slice(None, None, None), slice(1, None, None), slice(None, None, None), slice(None, 256, None))] = getitem_2;  setitem_2 = diagonal_attention_scores;  getitem_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    getitem_3 = diagonal_chunked_attention_scores_1[(slice(None, None, None), 0, slice(None, 255, None), slice(-255, None, None))];  diagonal_chunked_attention_scores_1 = None
    diagonal_attention_scores[(slice(None, None, None), 0, slice(1, 256, None), slice(1, 256, None))] = getitem_3;  setitem_3 = diagonal_attention_scores;  getitem_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    view_5 = diagonal_attention_scores.view(1, 12, 1024, 513);  diagonal_attention_scores = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:879, code: ).transpose(2, 1)
    attn_scores = view_5.transpose(2, 1);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    new_ones = attn_scores.new_ones(256, 257)
    tril = new_ones.tril();  new_ones = None
    beginning_mask_2d = tril.flip(dims = [0]);  tril = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    beginning_mask = beginning_mask_2d[(None, slice(None, None, None), None, slice(None, None, None))];  beginning_mask_2d = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    ending_mask = beginning_mask.flip(dims = (1, 3))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    beginning_input = attn_scores[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    beginning_mask_1 = beginning_mask.expand((1, 256, 12, 257));  beginning_mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    full_like = torch.full_like(beginning_input, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    bool_1 = beginning_mask_1.bool();  beginning_mask_1 = None
    where = full_like.where(bool_1, beginning_input);  full_like = bool_1 = beginning_input = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    attn_scores[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))] = where;  setitem_4 = attn_scores;  where = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    ending_input = attn_scores[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    ending_mask_1 = ending_mask.expand((1, 256, 12, 257));  ending_mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    full_like_1 = torch.full_like(ending_input, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    bool_2 = ending_mask_1.bool();  ending_mask_1 = None
    where_1 = full_like_1.where(bool_2, ending_input);  full_like_1 = bool_2 = ending_input = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    attn_scores[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))] = where_1;  setitem_5 = attn_scores;  where_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:576, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
    ne = l_attention_mask_ != 0
    remove_from_windowed_attention_mask = ne[(slice(None, None, None), slice(None, None, None), None, None)];  ne = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
    type_as = remove_from_windowed_attention_mask.type_as(query_vectors_2);  query_vectors_2 = None
    float_mask = type_as.masked_fill(remove_from_windowed_attention_mask, -3.4028234663852886e+38);  type_as = remove_from_windowed_attention_mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:584, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
    new_ones_1 = float_mask.new_ones(size = (1, 1024, 1, 1))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:830, code: chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
    div_3 = torch.div(1024, 256, rounding_mode = 'trunc')
    chunks_count_1 = div_3 - 1;  div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:833, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_6 = new_ones_1.transpose(1, 2);  new_ones_1 = None
    query_2 = transpose_6.reshape(1, 1024, 1);  transpose_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_7 = float_mask.transpose(1, 2);  float_mask = None
    key_2 = transpose_7.reshape(1, 1024, 1);  transpose_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_4 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_3 = query_2.view(1, div_4, 512, 1);  query_2 = div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    query_3 = hidden_states_3.as_strided(size = [1, 3, 512, 1], stride = [1024, 256, 1, 1]);  hidden_states_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_5 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_4 = key_2.view(1, div_5, 512, 1);  key_2 = div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    key_3 = hidden_states_4.as_strided(size = [1, 3, 512, 1], stride = [1024, 256, 1, 1]);  hidden_states_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    diagonal_chunked_attention_scores_2 = torch.functional.einsum('bcxd,bcyd->bcxy', (query_3, key_3));  query_3 = key_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    hidden_states_padded_2 = torch.nn.functional.pad(diagonal_chunked_attention_scores_2, (0, 0, 0, 1));  diagonal_chunked_attention_scores_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    diagonal_chunked_attention_scores_3 = hidden_states_padded_2.view(1, 3, 512, 513);  hidden_states_padded_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:856, code: (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
    add_1 = chunks_count_1 + 1;  chunks_count_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    diagonal_attention_scores_2 = diagonal_chunked_attention_scores_3.new_zeros((1, add_1, 256, 513));  add_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_8 = diagonal_chunked_attention_scores_3[(slice(None, None, None), slice(None, None, None), slice(None, 256, None), slice(None, 257, None))]
    diagonal_attention_scores_2[(slice(None, None, None), slice(None, -1, None), slice(None, None, None), slice(256, None, None))] = getitem_8;  setitem_6 = diagonal_attention_scores_2;  getitem_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_9 = diagonal_chunked_attention_scores_3[(slice(None, None, None), -1, slice(256, None, None), slice(None, 257, None))]
    diagonal_attention_scores_2[(slice(None, None, None), -1, slice(None, None, None), slice(256, None, None))] = getitem_9;  setitem_7 = diagonal_attention_scores_2;  getitem_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    getitem_10 = diagonal_chunked_attention_scores_3[(slice(None, None, None), slice(None, None, None), slice(-257, -1, None), slice(257, None, None))]
    diagonal_attention_scores_2[(slice(None, None, None), slice(1, None, None), slice(None, None, None), slice(None, 256, None))] = getitem_10;  setitem_8 = diagonal_attention_scores_2;  getitem_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    getitem_11 = diagonal_chunked_attention_scores_3[(slice(None, None, None), 0, slice(None, 255, None), slice(-255, None, None))];  diagonal_chunked_attention_scores_3 = None
    diagonal_attention_scores_2[(slice(None, None, None), 0, slice(1, 256, None), slice(1, 256, None))] = getitem_11;  setitem_9 = diagonal_attention_scores_2;  getitem_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    view_9 = diagonal_attention_scores_2.view(1, 1, 1024, 513);  diagonal_attention_scores_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:879, code: ).transpose(2, 1)
    diagonal_mask = view_9.transpose(2, 1);  view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    new_ones_2 = diagonal_mask.new_ones(256, 257)
    tril_1 = new_ones_2.tril();  new_ones_2 = None
    beginning_mask_2d_1 = tril_1.flip(dims = [0]);  tril_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    beginning_mask_2 = beginning_mask_2d_1[(None, slice(None, None, None), None, slice(None, None, None))];  beginning_mask_2d_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    ending_mask_2 = beginning_mask_2.flip(dims = (1, 3))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    beginning_input_1 = diagonal_mask[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    beginning_mask_3 = beginning_mask_2.expand((1, 256, 1, 257));  beginning_mask_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    full_like_2 = torch.full_like(beginning_input_1, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    bool_3 = beginning_mask_3.bool();  beginning_mask_3 = None
    where_2 = full_like_2.where(bool_3, beginning_input_1);  full_like_2 = bool_3 = beginning_input_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    diagonal_mask[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))] = where_2;  setitem_10 = diagonal_mask;  where_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    ending_input_1 = diagonal_mask[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    ending_mask_3 = ending_mask_2.expand((1, 256, 1, 257));  ending_mask_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    full_like_3 = torch.full_like(ending_input_1, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    bool_4 = ending_mask_3.bool();  ending_mask_3 = None
    where_3 = full_like_3.where(bool_4, ending_input_1);  full_like_3 = bool_4 = ending_input_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    diagonal_mask[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))] = where_3;  setitem_11 = diagonal_mask;  where_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    attn_scores += diagonal_mask;  attn_scores_1 = attn_scores;  attn_scores = diagonal_mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    attn_probs = torch.nn.functional.softmax(attn_scores_1, dim = -1, dtype = torch.float32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    getitem_15 = l_is_index_masked_[(slice(None, None, None), slice(None, None, None), None, None)]
    attn_probs_1 = torch.masked_fill(attn_probs, getitem_15, 0.0);  attn_probs = getitem_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:638, code: attn_probs = attn_probs.type_as(attn_scores)
    attn_probs_2 = attn_probs_1.type_as(attn_scores_1);  attn_probs_1 = attn_scores_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    attn_probs_3 = torch.nn.functional.dropout(attn_probs_2, p = 0.1, training = True);  attn_probs_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_10 = value_vectors.view(1024, 1, 12, 64);  value_vectors = None
    value_vectors_1 = view_10.transpose(0, 1);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:896, code: chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
    div_6 = torch.div(1024, 256, rounding_mode = 'trunc')
    chunks_count_2 = div_6 - 1;  div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    transpose_10 = attn_probs_3.transpose(1, 2);  attn_probs_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:901, code: torch.div(seq_len, window_overlap, rounding_mode="trunc"),
    div_7 = torch.div(1024, 256, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    chunked_attn_probs = transpose_10.reshape(12, div_7, 256, 513);  transpose_10 = div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_11 = value_vectors_1.transpose(1, 2);  value_vectors_1 = None
    value = transpose_11.reshape(12, 1024, 64);  transpose_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    padded_value = torch.nn.functional.pad(value, (0, 0, 256, 256), value = -1);  value = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:913, code: chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
    add_2 = chunks_count_2 + 1;  chunks_count_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    chunked_value = padded_value.as_strided(size = (12, add_2, 768, 64), stride = (98304, 16384, 64, 1));  padded_value = add_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    chunked_hidden_states = torch.nn.functional.pad(chunked_attn_probs, (0, 257));  chunked_attn_probs = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    chunked_hidden_states_1 = chunked_hidden_states.view(12, 4, -1);  chunked_hidden_states = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    chunked_hidden_states_2 = chunked_hidden_states_1[(slice(None, None, None), slice(None, None, None), slice(None, -256, None))];  chunked_hidden_states_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    chunked_hidden_states_3 = chunked_hidden_states_2.view(12, 4, 256, 769);  chunked_hidden_states_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    chunked_attn_probs_1 = chunked_hidden_states_3[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, -1, None))];  chunked_hidden_states_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    context = torch.functional.einsum('bcwd,bcdh->bcwh', (chunked_attn_probs_1, chunked_value));  chunked_attn_probs_1 = chunked_value = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    view_13 = context.view(1, 12, 1024, 64);  context = None
    attn_output = view_13.transpose(1, 2);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    transpose_13 = attn_output.transpose(0, 1);  attn_output = None
    reshape_6 = transpose_13.reshape(1024, 1, 768);  transpose_13 = None
    attn_output_1 = reshape_6.contiguous();  reshape_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    transpose_14 = attn_output_1.transpose(0, 1);  attn_output_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    hidden_states_5 = self.L__self___layer_0_attention_output_dense(transpose_14);  transpose_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    hidden_states_6 = self.L__self___layer_0_attention_output_dropout(hidden_states_5);  hidden_states_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_3 = hidden_states_6 + l_hidden_states_;  hidden_states_6 = l_hidden_states_ = None
    attn_output_3 = self.L__self___layer_0_attention_output_LayerNorm(add_3);  add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    hidden_states_8 = self.L__self___layer_0_intermediate_dense(attn_output_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output = torch._C._nn.gelu(hidden_states_8);  hidden_states_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    hidden_states_10 = self.L__self___layer_0_output_dense(intermediate_output);  intermediate_output = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    hidden_states_11 = self.L__self___layer_0_output_dropout(hidden_states_10);  hidden_states_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_4 = hidden_states_11 + attn_output_3;  hidden_states_11 = attn_output_3 = None
    hidden_states_13 = self.L__self___layer_0_output_LayerNorm(add_4);  add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    hidden_states_14 = hidden_states_13.transpose(0, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    query_vectors_3 = self.L__self___layer_1_attention_self_query(hidden_states_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    key_vectors_2 = self.L__self___layer_1_attention_self_key(hidden_states_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    value_vectors_2 = self.L__self___layer_1_attention_self_value(hidden_states_14);  hidden_states_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    query_vectors_3 /= 8.0;  query_vectors_4 = query_vectors_3;  query_vectors_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:568, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_14 = query_vectors_4.view(1024, 1, 12, 64);  query_vectors_4 = None
    query_vectors_5 = view_14.transpose(0, 1);  view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:569, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_15 = key_vectors_2.view(1024, 1, 12, 64);  key_vectors_2 = None
    key_vectors_3 = view_15.transpose(0, 1);  view_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:830, code: chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
    div_8 = torch.div(1024, 256, rounding_mode = 'trunc')
    chunks_count_3 = div_8 - 1;  div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:833, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_18 = query_vectors_5.transpose(1, 2)
    query_4 = transpose_18.reshape(12, 1024, 64);  transpose_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_19 = key_vectors_3.transpose(1, 2);  key_vectors_3 = None
    key_4 = transpose_19.reshape(12, 1024, 64);  transpose_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_9 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_15 = query_4.view(12, div_9, 512, 64);  query_4 = div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    query_5 = hidden_states_15.as_strided(size = [12, 3, 512, 64], stride = [64, 196608, 768, 1]);  hidden_states_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_10 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_16 = key_4.view(12, div_10, 512, 64);  key_4 = div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    key_5 = hidden_states_16.as_strided(size = [12, 3, 512, 64], stride = [64, 196608, 768, 1]);  hidden_states_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    diagonal_chunked_attention_scores_4 = torch.functional.einsum('bcxd,bcyd->bcxy', (query_5, key_5));  query_5 = key_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    hidden_states_padded_4 = torch.nn.functional.pad(diagonal_chunked_attention_scores_4, (0, 0, 0, 1));  diagonal_chunked_attention_scores_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    diagonal_chunked_attention_scores_5 = hidden_states_padded_4.view(12, 3, 512, 513);  hidden_states_padded_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:856, code: (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
    add_5 = chunks_count_3 + 1;  chunks_count_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    diagonal_attention_scores_4 = diagonal_chunked_attention_scores_5.new_zeros((12, add_5, 256, 513));  add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_18 = diagonal_chunked_attention_scores_5[(slice(None, None, None), slice(None, None, None), slice(None, 256, None), slice(None, 257, None))]
    diagonal_attention_scores_4[(slice(None, None, None), slice(None, -1, None), slice(None, None, None), slice(256, None, None))] = getitem_18;  setitem_12 = diagonal_attention_scores_4;  getitem_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_19 = diagonal_chunked_attention_scores_5[(slice(None, None, None), -1, slice(256, None, None), slice(None, 257, None))]
    diagonal_attention_scores_4[(slice(None, None, None), -1, slice(None, None, None), slice(256, None, None))] = getitem_19;  setitem_13 = diagonal_attention_scores_4;  getitem_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    getitem_20 = diagonal_chunked_attention_scores_5[(slice(None, None, None), slice(None, None, None), slice(-257, -1, None), slice(257, None, None))]
    diagonal_attention_scores_4[(slice(None, None, None), slice(1, None, None), slice(None, None, None), slice(None, 256, None))] = getitem_20;  setitem_14 = diagonal_attention_scores_4;  getitem_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    getitem_21 = diagonal_chunked_attention_scores_5[(slice(None, None, None), 0, slice(None, 255, None), slice(-255, None, None))];  diagonal_chunked_attention_scores_5 = None
    diagonal_attention_scores_4[(slice(None, None, None), 0, slice(1, 256, None), slice(1, 256, None))] = getitem_21;  setitem_15 = diagonal_attention_scores_4;  getitem_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    view_19 = diagonal_attention_scores_4.view(1, 12, 1024, 513);  diagonal_attention_scores_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:879, code: ).transpose(2, 1)
    attn_scores_2 = view_19.transpose(2, 1);  view_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    new_ones_3 = attn_scores_2.new_ones(256, 257)
    tril_2 = new_ones_3.tril();  new_ones_3 = None
    beginning_mask_2d_2 = tril_2.flip(dims = [0]);  tril_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    beginning_mask_4 = beginning_mask_2d_2[(None, slice(None, None, None), None, slice(None, None, None))];  beginning_mask_2d_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    ending_mask_4 = beginning_mask_4.flip(dims = (1, 3))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    beginning_input_2 = attn_scores_2[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    beginning_mask_5 = beginning_mask_4.expand((1, 256, 12, 257));  beginning_mask_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    full_like_4 = torch.full_like(beginning_input_2, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    bool_5 = beginning_mask_5.bool();  beginning_mask_5 = None
    where_4 = full_like_4.where(bool_5, beginning_input_2);  full_like_4 = bool_5 = beginning_input_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    attn_scores_2[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))] = where_4;  setitem_16 = attn_scores_2;  where_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    ending_input_2 = attn_scores_2[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    ending_mask_5 = ending_mask_4.expand((1, 256, 12, 257));  ending_mask_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    full_like_5 = torch.full_like(ending_input_2, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    bool_6 = ending_mask_5.bool();  ending_mask_5 = None
    where_5 = full_like_5.where(bool_6, ending_input_2);  full_like_5 = bool_6 = ending_input_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    attn_scores_2[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))] = where_5;  setitem_17 = attn_scores_2;  where_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:576, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
    ne_1 = l_attention_mask_ != 0
    remove_from_windowed_attention_mask_1 = ne_1[(slice(None, None, None), slice(None, None, None), None, None)];  ne_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
    type_as_2 = remove_from_windowed_attention_mask_1.type_as(query_vectors_5);  query_vectors_5 = None
    float_mask_1 = type_as_2.masked_fill(remove_from_windowed_attention_mask_1, -3.4028234663852886e+38);  type_as_2 = remove_from_windowed_attention_mask_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:584, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
    new_ones_4 = float_mask_1.new_ones(size = (1, 1024, 1, 1))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:830, code: chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
    div_11 = torch.div(1024, 256, rounding_mode = 'trunc')
    chunks_count_4 = div_11 - 1;  div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:833, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_21 = new_ones_4.transpose(1, 2);  new_ones_4 = None
    query_6 = transpose_21.reshape(1, 1024, 1);  transpose_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_22 = float_mask_1.transpose(1, 2);  float_mask_1 = None
    key_6 = transpose_22.reshape(1, 1024, 1);  transpose_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_12 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_17 = query_6.view(1, div_12, 512, 1);  query_6 = div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    query_7 = hidden_states_17.as_strided(size = [1, 3, 512, 1], stride = [1024, 256, 1, 1]);  hidden_states_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_13 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_18 = key_6.view(1, div_13, 512, 1);  key_6 = div_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    key_7 = hidden_states_18.as_strided(size = [1, 3, 512, 1], stride = [1024, 256, 1, 1]);  hidden_states_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    diagonal_chunked_attention_scores_6 = torch.functional.einsum('bcxd,bcyd->bcxy', (query_7, key_7));  query_7 = key_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    hidden_states_padded_6 = torch.nn.functional.pad(diagonal_chunked_attention_scores_6, (0, 0, 0, 1));  diagonal_chunked_attention_scores_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    diagonal_chunked_attention_scores_7 = hidden_states_padded_6.view(1, 3, 512, 513);  hidden_states_padded_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:856, code: (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
    add_6 = chunks_count_4 + 1;  chunks_count_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    diagonal_attention_scores_6 = diagonal_chunked_attention_scores_7.new_zeros((1, add_6, 256, 513));  add_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_26 = diagonal_chunked_attention_scores_7[(slice(None, None, None), slice(None, None, None), slice(None, 256, None), slice(None, 257, None))]
    diagonal_attention_scores_6[(slice(None, None, None), slice(None, -1, None), slice(None, None, None), slice(256, None, None))] = getitem_26;  setitem_18 = diagonal_attention_scores_6;  getitem_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_27 = diagonal_chunked_attention_scores_7[(slice(None, None, None), -1, slice(256, None, None), slice(None, 257, None))]
    diagonal_attention_scores_6[(slice(None, None, None), -1, slice(None, None, None), slice(256, None, None))] = getitem_27;  setitem_19 = diagonal_attention_scores_6;  getitem_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    getitem_28 = diagonal_chunked_attention_scores_7[(slice(None, None, None), slice(None, None, None), slice(-257, -1, None), slice(257, None, None))]
    diagonal_attention_scores_6[(slice(None, None, None), slice(1, None, None), slice(None, None, None), slice(None, 256, None))] = getitem_28;  setitem_20 = diagonal_attention_scores_6;  getitem_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    getitem_29 = diagonal_chunked_attention_scores_7[(slice(None, None, None), 0, slice(None, 255, None), slice(-255, None, None))];  diagonal_chunked_attention_scores_7 = None
    diagonal_attention_scores_6[(slice(None, None, None), 0, slice(1, 256, None), slice(1, 256, None))] = getitem_29;  setitem_21 = diagonal_attention_scores_6;  getitem_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    view_23 = diagonal_attention_scores_6.view(1, 1, 1024, 513);  diagonal_attention_scores_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:879, code: ).transpose(2, 1)
    diagonal_mask_1 = view_23.transpose(2, 1);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    new_ones_5 = diagonal_mask_1.new_ones(256, 257)
    tril_3 = new_ones_5.tril();  new_ones_5 = None
    beginning_mask_2d_3 = tril_3.flip(dims = [0]);  tril_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    beginning_mask_6 = beginning_mask_2d_3[(None, slice(None, None, None), None, slice(None, None, None))];  beginning_mask_2d_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    ending_mask_6 = beginning_mask_6.flip(dims = (1, 3))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    beginning_input_3 = diagonal_mask_1[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    beginning_mask_7 = beginning_mask_6.expand((1, 256, 1, 257));  beginning_mask_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    full_like_6 = torch.full_like(beginning_input_3, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    bool_7 = beginning_mask_7.bool();  beginning_mask_7 = None
    where_6 = full_like_6.where(bool_7, beginning_input_3);  full_like_6 = bool_7 = beginning_input_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    diagonal_mask_1[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))] = where_6;  setitem_22 = diagonal_mask_1;  where_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    ending_input_3 = diagonal_mask_1[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    ending_mask_7 = ending_mask_6.expand((1, 256, 1, 257));  ending_mask_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    full_like_7 = torch.full_like(ending_input_3, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    bool_8 = ending_mask_7.bool();  ending_mask_7 = None
    where_7 = full_like_7.where(bool_8, ending_input_3);  full_like_7 = bool_8 = ending_input_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    diagonal_mask_1[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))] = where_7;  setitem_23 = diagonal_mask_1;  where_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    attn_scores_2 += diagonal_mask_1;  attn_scores_3 = attn_scores_2;  attn_scores_2 = diagonal_mask_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    attn_probs_4 = torch.nn.functional.softmax(attn_scores_3, dim = -1, dtype = torch.float32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    getitem_33 = l_is_index_masked_[(slice(None, None, None), slice(None, None, None), None, None)]
    attn_probs_5 = torch.masked_fill(attn_probs_4, getitem_33, 0.0);  attn_probs_4 = getitem_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:638, code: attn_probs = attn_probs.type_as(attn_scores)
    attn_probs_6 = attn_probs_5.type_as(attn_scores_3);  attn_probs_5 = attn_scores_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    attn_probs_7 = torch.nn.functional.dropout(attn_probs_6, p = 0.1, training = True);  attn_probs_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_24 = value_vectors_2.view(1024, 1, 12, 64);  value_vectors_2 = None
    value_vectors_3 = view_24.transpose(0, 1);  view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:896, code: chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
    div_14 = torch.div(1024, 256, rounding_mode = 'trunc')
    chunks_count_5 = div_14 - 1;  div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    transpose_25 = attn_probs_7.transpose(1, 2);  attn_probs_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:901, code: torch.div(seq_len, window_overlap, rounding_mode="trunc"),
    div_15 = torch.div(1024, 256, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    chunked_attn_probs_2 = transpose_25.reshape(12, div_15, 256, 513);  transpose_25 = div_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_26 = value_vectors_3.transpose(1, 2);  value_vectors_3 = None
    value_1 = transpose_26.reshape(12, 1024, 64);  transpose_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    padded_value_1 = torch.nn.functional.pad(value_1, (0, 0, 256, 256), value = -1);  value_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:913, code: chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
    add_7 = chunks_count_5 + 1;  chunks_count_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    chunked_value_1 = padded_value_1.as_strided(size = (12, add_7, 768, 64), stride = (98304, 16384, 64, 1));  padded_value_1 = add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    chunked_hidden_states_5 = torch.nn.functional.pad(chunked_attn_probs_2, (0, 257));  chunked_attn_probs_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    chunked_hidden_states_6 = chunked_hidden_states_5.view(12, 4, -1);  chunked_hidden_states_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    chunked_hidden_states_7 = chunked_hidden_states_6[(slice(None, None, None), slice(None, None, None), slice(None, -256, None))];  chunked_hidden_states_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    chunked_hidden_states_8 = chunked_hidden_states_7.view(12, 4, 256, 769);  chunked_hidden_states_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    chunked_attn_probs_3 = chunked_hidden_states_8[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, -1, None))];  chunked_hidden_states_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    context_1 = torch.functional.einsum('bcwd,bcdh->bcwh', (chunked_attn_probs_3, chunked_value_1));  chunked_attn_probs_3 = chunked_value_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    view_27 = context_1.view(1, 12, 1024, 64);  context_1 = None
    attn_output_4 = view_27.transpose(1, 2);  view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    transpose_28 = attn_output_4.transpose(0, 1);  attn_output_4 = None
    reshape_13 = transpose_28.reshape(1024, 1, 768);  transpose_28 = None
    attn_output_5 = reshape_13.contiguous();  reshape_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    transpose_29 = attn_output_5.transpose(0, 1);  attn_output_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    hidden_states_19 = self.L__self___layer_1_attention_output_dense(transpose_29);  transpose_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    hidden_states_20 = self.L__self___layer_1_attention_output_dropout(hidden_states_19);  hidden_states_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_8 = hidden_states_20 + hidden_states_13;  hidden_states_20 = hidden_states_13 = None
    attn_output_7 = self.L__self___layer_1_attention_output_LayerNorm(add_8);  add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    hidden_states_22 = self.L__self___layer_1_intermediate_dense(attn_output_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_1 = torch._C._nn.gelu(hidden_states_22);  hidden_states_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    hidden_states_24 = self.L__self___layer_1_output_dense(intermediate_output_1);  intermediate_output_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    hidden_states_25 = self.L__self___layer_1_output_dropout(hidden_states_24);  hidden_states_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_9 = hidden_states_25 + attn_output_7;  hidden_states_25 = attn_output_7 = None
    hidden_states_27 = self.L__self___layer_1_output_LayerNorm(add_9);  add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    hidden_states_28 = hidden_states_27.transpose(0, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    query_vectors_6 = self.L__self___layer_2_attention_self_query(hidden_states_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    key_vectors_4 = self.L__self___layer_2_attention_self_key(hidden_states_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    value_vectors_4 = self.L__self___layer_2_attention_self_value(hidden_states_28);  hidden_states_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    query_vectors_6 /= 8.0;  query_vectors_7 = query_vectors_6;  query_vectors_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:568, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_28 = query_vectors_7.view(1024, 1, 12, 64);  query_vectors_7 = None
    query_vectors_8 = view_28.transpose(0, 1);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:569, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_29 = key_vectors_4.view(1024, 1, 12, 64);  key_vectors_4 = None
    key_vectors_5 = view_29.transpose(0, 1);  view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:830, code: chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
    div_16 = torch.div(1024, 256, rounding_mode = 'trunc')
    chunks_count_6 = div_16 - 1;  div_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:833, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_33 = query_vectors_8.transpose(1, 2)
    query_8 = transpose_33.reshape(12, 1024, 64);  transpose_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_34 = key_vectors_5.transpose(1, 2);  key_vectors_5 = None
    key_8 = transpose_34.reshape(12, 1024, 64);  transpose_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_17 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_29 = query_8.view(12, div_17, 512, 64);  query_8 = div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    query_9 = hidden_states_29.as_strided(size = [12, 3, 512, 64], stride = [64, 196608, 768, 1]);  hidden_states_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_18 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_30 = key_8.view(12, div_18, 512, 64);  key_8 = div_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    key_9 = hidden_states_30.as_strided(size = [12, 3, 512, 64], stride = [64, 196608, 768, 1]);  hidden_states_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    diagonal_chunked_attention_scores_8 = torch.functional.einsum('bcxd,bcyd->bcxy', (query_9, key_9));  query_9 = key_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    hidden_states_padded_8 = torch.nn.functional.pad(diagonal_chunked_attention_scores_8, (0, 0, 0, 1));  diagonal_chunked_attention_scores_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    diagonal_chunked_attention_scores_9 = hidden_states_padded_8.view(12, 3, 512, 513);  hidden_states_padded_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:856, code: (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
    add_10 = chunks_count_6 + 1;  chunks_count_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    diagonal_attention_scores_8 = diagonal_chunked_attention_scores_9.new_zeros((12, add_10, 256, 513));  add_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_36 = diagonal_chunked_attention_scores_9[(slice(None, None, None), slice(None, None, None), slice(None, 256, None), slice(None, 257, None))]
    diagonal_attention_scores_8[(slice(None, None, None), slice(None, -1, None), slice(None, None, None), slice(256, None, None))] = getitem_36;  setitem_24 = diagonal_attention_scores_8;  getitem_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_37 = diagonal_chunked_attention_scores_9[(slice(None, None, None), -1, slice(256, None, None), slice(None, 257, None))]
    diagonal_attention_scores_8[(slice(None, None, None), -1, slice(None, None, None), slice(256, None, None))] = getitem_37;  setitem_25 = diagonal_attention_scores_8;  getitem_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    getitem_38 = diagonal_chunked_attention_scores_9[(slice(None, None, None), slice(None, None, None), slice(-257, -1, None), slice(257, None, None))]
    diagonal_attention_scores_8[(slice(None, None, None), slice(1, None, None), slice(None, None, None), slice(None, 256, None))] = getitem_38;  setitem_26 = diagonal_attention_scores_8;  getitem_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    getitem_39 = diagonal_chunked_attention_scores_9[(slice(None, None, None), 0, slice(None, 255, None), slice(-255, None, None))];  diagonal_chunked_attention_scores_9 = None
    diagonal_attention_scores_8[(slice(None, None, None), 0, slice(1, 256, None), slice(1, 256, None))] = getitem_39;  setitem_27 = diagonal_attention_scores_8;  getitem_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    view_33 = diagonal_attention_scores_8.view(1, 12, 1024, 513);  diagonal_attention_scores_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:879, code: ).transpose(2, 1)
    attn_scores_4 = view_33.transpose(2, 1);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    new_ones_6 = attn_scores_4.new_ones(256, 257)
    tril_4 = new_ones_6.tril();  new_ones_6 = None
    beginning_mask_2d_4 = tril_4.flip(dims = [0]);  tril_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    beginning_mask_8 = beginning_mask_2d_4[(None, slice(None, None, None), None, slice(None, None, None))];  beginning_mask_2d_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    ending_mask_8 = beginning_mask_8.flip(dims = (1, 3))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    beginning_input_4 = attn_scores_4[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    beginning_mask_9 = beginning_mask_8.expand((1, 256, 12, 257));  beginning_mask_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    full_like_8 = torch.full_like(beginning_input_4, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    bool_9 = beginning_mask_9.bool();  beginning_mask_9 = None
    where_8 = full_like_8.where(bool_9, beginning_input_4);  full_like_8 = bool_9 = beginning_input_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    attn_scores_4[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))] = where_8;  setitem_28 = attn_scores_4;  where_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    ending_input_4 = attn_scores_4[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    ending_mask_9 = ending_mask_8.expand((1, 256, 12, 257));  ending_mask_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    full_like_9 = torch.full_like(ending_input_4, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    bool_10 = ending_mask_9.bool();  ending_mask_9 = None
    where_9 = full_like_9.where(bool_10, ending_input_4);  full_like_9 = bool_10 = ending_input_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    attn_scores_4[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))] = where_9;  setitem_29 = attn_scores_4;  where_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:576, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
    ne_2 = l_attention_mask_ != 0
    remove_from_windowed_attention_mask_2 = ne_2[(slice(None, None, None), slice(None, None, None), None, None)];  ne_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
    type_as_4 = remove_from_windowed_attention_mask_2.type_as(query_vectors_8);  query_vectors_8 = None
    float_mask_2 = type_as_4.masked_fill(remove_from_windowed_attention_mask_2, -3.4028234663852886e+38);  type_as_4 = remove_from_windowed_attention_mask_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:584, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
    new_ones_7 = float_mask_2.new_ones(size = (1, 1024, 1, 1))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:830, code: chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
    div_19 = torch.div(1024, 256, rounding_mode = 'trunc')
    chunks_count_7 = div_19 - 1;  div_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:833, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_36 = new_ones_7.transpose(1, 2);  new_ones_7 = None
    query_10 = transpose_36.reshape(1, 1024, 1);  transpose_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_37 = float_mask_2.transpose(1, 2);  float_mask_2 = None
    key_10 = transpose_37.reshape(1, 1024, 1);  transpose_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_20 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_31 = query_10.view(1, div_20, 512, 1);  query_10 = div_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    query_11 = hidden_states_31.as_strided(size = [1, 3, 512, 1], stride = [1024, 256, 1, 1]);  hidden_states_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_21 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_32 = key_10.view(1, div_21, 512, 1);  key_10 = div_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    key_11 = hidden_states_32.as_strided(size = [1, 3, 512, 1], stride = [1024, 256, 1, 1]);  hidden_states_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    diagonal_chunked_attention_scores_10 = torch.functional.einsum('bcxd,bcyd->bcxy', (query_11, key_11));  query_11 = key_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    hidden_states_padded_10 = torch.nn.functional.pad(diagonal_chunked_attention_scores_10, (0, 0, 0, 1));  diagonal_chunked_attention_scores_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    diagonal_chunked_attention_scores_11 = hidden_states_padded_10.view(1, 3, 512, 513);  hidden_states_padded_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:856, code: (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
    add_11 = chunks_count_7 + 1;  chunks_count_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    diagonal_attention_scores_10 = diagonal_chunked_attention_scores_11.new_zeros((1, add_11, 256, 513));  add_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_44 = diagonal_chunked_attention_scores_11[(slice(None, None, None), slice(None, None, None), slice(None, 256, None), slice(None, 257, None))]
    diagonal_attention_scores_10[(slice(None, None, None), slice(None, -1, None), slice(None, None, None), slice(256, None, None))] = getitem_44;  setitem_30 = diagonal_attention_scores_10;  getitem_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_45 = diagonal_chunked_attention_scores_11[(slice(None, None, None), -1, slice(256, None, None), slice(None, 257, None))]
    diagonal_attention_scores_10[(slice(None, None, None), -1, slice(None, None, None), slice(256, None, None))] = getitem_45;  setitem_31 = diagonal_attention_scores_10;  getitem_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    getitem_46 = diagonal_chunked_attention_scores_11[(slice(None, None, None), slice(None, None, None), slice(-257, -1, None), slice(257, None, None))]
    diagonal_attention_scores_10[(slice(None, None, None), slice(1, None, None), slice(None, None, None), slice(None, 256, None))] = getitem_46;  setitem_32 = diagonal_attention_scores_10;  getitem_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    getitem_47 = diagonal_chunked_attention_scores_11[(slice(None, None, None), 0, slice(None, 255, None), slice(-255, None, None))];  diagonal_chunked_attention_scores_11 = None
    diagonal_attention_scores_10[(slice(None, None, None), 0, slice(1, 256, None), slice(1, 256, None))] = getitem_47;  setitem_33 = diagonal_attention_scores_10;  getitem_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    view_37 = diagonal_attention_scores_10.view(1, 1, 1024, 513);  diagonal_attention_scores_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:879, code: ).transpose(2, 1)
    diagonal_mask_2 = view_37.transpose(2, 1);  view_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    new_ones_8 = diagonal_mask_2.new_ones(256, 257)
    tril_5 = new_ones_8.tril();  new_ones_8 = None
    beginning_mask_2d_5 = tril_5.flip(dims = [0]);  tril_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    beginning_mask_10 = beginning_mask_2d_5[(None, slice(None, None, None), None, slice(None, None, None))];  beginning_mask_2d_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    ending_mask_10 = beginning_mask_10.flip(dims = (1, 3))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    beginning_input_5 = diagonal_mask_2[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    beginning_mask_11 = beginning_mask_10.expand((1, 256, 1, 257));  beginning_mask_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    full_like_10 = torch.full_like(beginning_input_5, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    bool_11 = beginning_mask_11.bool();  beginning_mask_11 = None
    where_10 = full_like_10.where(bool_11, beginning_input_5);  full_like_10 = bool_11 = beginning_input_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    diagonal_mask_2[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))] = where_10;  setitem_34 = diagonal_mask_2;  where_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    ending_input_5 = diagonal_mask_2[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    ending_mask_11 = ending_mask_10.expand((1, 256, 1, 257));  ending_mask_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    full_like_11 = torch.full_like(ending_input_5, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    bool_12 = ending_mask_11.bool();  ending_mask_11 = None
    where_11 = full_like_11.where(bool_12, ending_input_5);  full_like_11 = bool_12 = ending_input_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    diagonal_mask_2[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))] = where_11;  setitem_35 = diagonal_mask_2;  where_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    attn_scores_4 += diagonal_mask_2;  attn_scores_5 = attn_scores_4;  attn_scores_4 = diagonal_mask_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    attn_probs_8 = torch.nn.functional.softmax(attn_scores_5, dim = -1, dtype = torch.float32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    getitem_51 = l_is_index_masked_[(slice(None, None, None), slice(None, None, None), None, None)]
    attn_probs_9 = torch.masked_fill(attn_probs_8, getitem_51, 0.0);  attn_probs_8 = getitem_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:638, code: attn_probs = attn_probs.type_as(attn_scores)
    attn_probs_10 = attn_probs_9.type_as(attn_scores_5);  attn_probs_9 = attn_scores_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    attn_probs_11 = torch.nn.functional.dropout(attn_probs_10, p = 0.1, training = True);  attn_probs_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_38 = value_vectors_4.view(1024, 1, 12, 64);  value_vectors_4 = None
    value_vectors_5 = view_38.transpose(0, 1);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:896, code: chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
    div_22 = torch.div(1024, 256, rounding_mode = 'trunc')
    chunks_count_8 = div_22 - 1;  div_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    transpose_40 = attn_probs_11.transpose(1, 2);  attn_probs_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:901, code: torch.div(seq_len, window_overlap, rounding_mode="trunc"),
    div_23 = torch.div(1024, 256, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    chunked_attn_probs_4 = transpose_40.reshape(12, div_23, 256, 513);  transpose_40 = div_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_41 = value_vectors_5.transpose(1, 2);  value_vectors_5 = None
    value_2 = transpose_41.reshape(12, 1024, 64);  transpose_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    padded_value_2 = torch.nn.functional.pad(value_2, (0, 0, 256, 256), value = -1);  value_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:913, code: chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
    add_12 = chunks_count_8 + 1;  chunks_count_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    chunked_value_2 = padded_value_2.as_strided(size = (12, add_12, 768, 64), stride = (98304, 16384, 64, 1));  padded_value_2 = add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    chunked_hidden_states_10 = torch.nn.functional.pad(chunked_attn_probs_4, (0, 257));  chunked_attn_probs_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    chunked_hidden_states_11 = chunked_hidden_states_10.view(12, 4, -1);  chunked_hidden_states_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    chunked_hidden_states_12 = chunked_hidden_states_11[(slice(None, None, None), slice(None, None, None), slice(None, -256, None))];  chunked_hidden_states_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    chunked_hidden_states_13 = chunked_hidden_states_12.view(12, 4, 256, 769);  chunked_hidden_states_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    chunked_attn_probs_5 = chunked_hidden_states_13[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, -1, None))];  chunked_hidden_states_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    context_2 = torch.functional.einsum('bcwd,bcdh->bcwh', (chunked_attn_probs_5, chunked_value_2));  chunked_attn_probs_5 = chunked_value_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    view_41 = context_2.view(1, 12, 1024, 64);  context_2 = None
    attn_output_8 = view_41.transpose(1, 2);  view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    transpose_43 = attn_output_8.transpose(0, 1);  attn_output_8 = None
    reshape_20 = transpose_43.reshape(1024, 1, 768);  transpose_43 = None
    attn_output_9 = reshape_20.contiguous();  reshape_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    transpose_44 = attn_output_9.transpose(0, 1);  attn_output_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    hidden_states_33 = self.L__self___layer_2_attention_output_dense(transpose_44);  transpose_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    hidden_states_34 = self.L__self___layer_2_attention_output_dropout(hidden_states_33);  hidden_states_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_13 = hidden_states_34 + hidden_states_27;  hidden_states_34 = hidden_states_27 = None
    attn_output_11 = self.L__self___layer_2_attention_output_LayerNorm(add_13);  add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    hidden_states_36 = self.L__self___layer_2_intermediate_dense(attn_output_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_2 = torch._C._nn.gelu(hidden_states_36);  hidden_states_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    hidden_states_38 = self.L__self___layer_2_output_dense(intermediate_output_2);  intermediate_output_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    hidden_states_39 = self.L__self___layer_2_output_dropout(hidden_states_38);  hidden_states_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_14 = hidden_states_39 + attn_output_11;  hidden_states_39 = attn_output_11 = None
    hidden_states_41 = self.L__self___layer_2_output_LayerNorm(add_14);  add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    hidden_states_42 = hidden_states_41.transpose(0, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    query_vectors_9 = self.L__self___layer_3_attention_self_query(hidden_states_42)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    key_vectors_6 = self.L__self___layer_3_attention_self_key(hidden_states_42)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    value_vectors_6 = self.L__self___layer_3_attention_self_value(hidden_states_42);  hidden_states_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    query_vectors_9 /= 8.0;  query_vectors_10 = query_vectors_9;  query_vectors_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:568, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_42 = query_vectors_10.view(1024, 1, 12, 64);  query_vectors_10 = None
    query_vectors_11 = view_42.transpose(0, 1);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:569, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_43 = key_vectors_6.view(1024, 1, 12, 64);  key_vectors_6 = None
    key_vectors_7 = view_43.transpose(0, 1);  view_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:830, code: chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
    div_24 = torch.div(1024, 256, rounding_mode = 'trunc')
    chunks_count_9 = div_24 - 1;  div_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:833, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_48 = query_vectors_11.transpose(1, 2)
    query_12 = transpose_48.reshape(12, 1024, 64);  transpose_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_49 = key_vectors_7.transpose(1, 2);  key_vectors_7 = None
    key_12 = transpose_49.reshape(12, 1024, 64);  transpose_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_25 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_43 = query_12.view(12, div_25, 512, 64);  query_12 = div_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    query_13 = hidden_states_43.as_strided(size = [12, 3, 512, 64], stride = [64, 196608, 768, 1]);  hidden_states_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_26 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_44 = key_12.view(12, div_26, 512, 64);  key_12 = div_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    key_13 = hidden_states_44.as_strided(size = [12, 3, 512, 64], stride = [64, 196608, 768, 1]);  hidden_states_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    diagonal_chunked_attention_scores_12 = torch.functional.einsum('bcxd,bcyd->bcxy', (query_13, key_13));  query_13 = key_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    hidden_states_padded_12 = torch.nn.functional.pad(diagonal_chunked_attention_scores_12, (0, 0, 0, 1));  diagonal_chunked_attention_scores_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    diagonal_chunked_attention_scores_13 = hidden_states_padded_12.view(12, 3, 512, 513);  hidden_states_padded_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:856, code: (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
    add_15 = chunks_count_9 + 1;  chunks_count_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    diagonal_attention_scores_12 = diagonal_chunked_attention_scores_13.new_zeros((12, add_15, 256, 513));  add_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_54 = diagonal_chunked_attention_scores_13[(slice(None, None, None), slice(None, None, None), slice(None, 256, None), slice(None, 257, None))]
    diagonal_attention_scores_12[(slice(None, None, None), slice(None, -1, None), slice(None, None, None), slice(256, None, None))] = getitem_54;  setitem_36 = diagonal_attention_scores_12;  getitem_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_55 = diagonal_chunked_attention_scores_13[(slice(None, None, None), -1, slice(256, None, None), slice(None, 257, None))]
    diagonal_attention_scores_12[(slice(None, None, None), -1, slice(None, None, None), slice(256, None, None))] = getitem_55;  setitem_37 = diagonal_attention_scores_12;  getitem_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    getitem_56 = diagonal_chunked_attention_scores_13[(slice(None, None, None), slice(None, None, None), slice(-257, -1, None), slice(257, None, None))]
    diagonal_attention_scores_12[(slice(None, None, None), slice(1, None, None), slice(None, None, None), slice(None, 256, None))] = getitem_56;  setitem_38 = diagonal_attention_scores_12;  getitem_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    getitem_57 = diagonal_chunked_attention_scores_13[(slice(None, None, None), 0, slice(None, 255, None), slice(-255, None, None))];  diagonal_chunked_attention_scores_13 = None
    diagonal_attention_scores_12[(slice(None, None, None), 0, slice(1, 256, None), slice(1, 256, None))] = getitem_57;  setitem_39 = diagonal_attention_scores_12;  getitem_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    view_47 = diagonal_attention_scores_12.view(1, 12, 1024, 513);  diagonal_attention_scores_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:879, code: ).transpose(2, 1)
    attn_scores_6 = view_47.transpose(2, 1);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    new_ones_9 = attn_scores_6.new_ones(256, 257)
    tril_6 = new_ones_9.tril();  new_ones_9 = None
    beginning_mask_2d_6 = tril_6.flip(dims = [0]);  tril_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    beginning_mask_12 = beginning_mask_2d_6[(None, slice(None, None, None), None, slice(None, None, None))];  beginning_mask_2d_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    ending_mask_12 = beginning_mask_12.flip(dims = (1, 3))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    beginning_input_6 = attn_scores_6[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    beginning_mask_13 = beginning_mask_12.expand((1, 256, 12, 257));  beginning_mask_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    full_like_12 = torch.full_like(beginning_input_6, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    bool_13 = beginning_mask_13.bool();  beginning_mask_13 = None
    where_12 = full_like_12.where(bool_13, beginning_input_6);  full_like_12 = bool_13 = beginning_input_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    attn_scores_6[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))] = where_12;  setitem_40 = attn_scores_6;  where_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    ending_input_6 = attn_scores_6[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    ending_mask_13 = ending_mask_12.expand((1, 256, 12, 257));  ending_mask_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    full_like_13 = torch.full_like(ending_input_6, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    bool_14 = ending_mask_13.bool();  ending_mask_13 = None
    where_13 = full_like_13.where(bool_14, ending_input_6);  full_like_13 = bool_14 = ending_input_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    attn_scores_6[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))] = where_13;  setitem_41 = attn_scores_6;  where_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:576, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
    ne_3 = l_attention_mask_ != 0
    remove_from_windowed_attention_mask_3 = ne_3[(slice(None, None, None), slice(None, None, None), None, None)];  ne_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
    type_as_6 = remove_from_windowed_attention_mask_3.type_as(query_vectors_11);  query_vectors_11 = None
    float_mask_3 = type_as_6.masked_fill(remove_from_windowed_attention_mask_3, -3.4028234663852886e+38);  type_as_6 = remove_from_windowed_attention_mask_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:584, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
    new_ones_10 = float_mask_3.new_ones(size = (1, 1024, 1, 1))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:830, code: chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
    div_27 = torch.div(1024, 256, rounding_mode = 'trunc')
    chunks_count_10 = div_27 - 1;  div_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:833, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_51 = new_ones_10.transpose(1, 2);  new_ones_10 = None
    query_14 = transpose_51.reshape(1, 1024, 1);  transpose_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_52 = float_mask_3.transpose(1, 2);  float_mask_3 = None
    key_14 = transpose_52.reshape(1, 1024, 1);  transpose_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_28 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_45 = query_14.view(1, div_28, 512, 1);  query_14 = div_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    query_15 = hidden_states_45.as_strided(size = [1, 3, 512, 1], stride = [1024, 256, 1, 1]);  hidden_states_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_29 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_46 = key_14.view(1, div_29, 512, 1);  key_14 = div_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    key_15 = hidden_states_46.as_strided(size = [1, 3, 512, 1], stride = [1024, 256, 1, 1]);  hidden_states_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    diagonal_chunked_attention_scores_14 = torch.functional.einsum('bcxd,bcyd->bcxy', (query_15, key_15));  query_15 = key_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    hidden_states_padded_14 = torch.nn.functional.pad(diagonal_chunked_attention_scores_14, (0, 0, 0, 1));  diagonal_chunked_attention_scores_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    diagonal_chunked_attention_scores_15 = hidden_states_padded_14.view(1, 3, 512, 513);  hidden_states_padded_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:856, code: (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
    add_16 = chunks_count_10 + 1;  chunks_count_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    diagonal_attention_scores_14 = diagonal_chunked_attention_scores_15.new_zeros((1, add_16, 256, 513));  add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_62 = diagonal_chunked_attention_scores_15[(slice(None, None, None), slice(None, None, None), slice(None, 256, None), slice(None, 257, None))]
    diagonal_attention_scores_14[(slice(None, None, None), slice(None, -1, None), slice(None, None, None), slice(256, None, None))] = getitem_62;  setitem_42 = diagonal_attention_scores_14;  getitem_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_63 = diagonal_chunked_attention_scores_15[(slice(None, None, None), -1, slice(256, None, None), slice(None, 257, None))]
    diagonal_attention_scores_14[(slice(None, None, None), -1, slice(None, None, None), slice(256, None, None))] = getitem_63;  setitem_43 = diagonal_attention_scores_14;  getitem_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    getitem_64 = diagonal_chunked_attention_scores_15[(slice(None, None, None), slice(None, None, None), slice(-257, -1, None), slice(257, None, None))]
    diagonal_attention_scores_14[(slice(None, None, None), slice(1, None, None), slice(None, None, None), slice(None, 256, None))] = getitem_64;  setitem_44 = diagonal_attention_scores_14;  getitem_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    getitem_65 = diagonal_chunked_attention_scores_15[(slice(None, None, None), 0, slice(None, 255, None), slice(-255, None, None))];  diagonal_chunked_attention_scores_15 = None
    diagonal_attention_scores_14[(slice(None, None, None), 0, slice(1, 256, None), slice(1, 256, None))] = getitem_65;  setitem_45 = diagonal_attention_scores_14;  getitem_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    view_51 = diagonal_attention_scores_14.view(1, 1, 1024, 513);  diagonal_attention_scores_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:879, code: ).transpose(2, 1)
    diagonal_mask_3 = view_51.transpose(2, 1);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    new_ones_11 = diagonal_mask_3.new_ones(256, 257)
    tril_7 = new_ones_11.tril();  new_ones_11 = None
    beginning_mask_2d_7 = tril_7.flip(dims = [0]);  tril_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    beginning_mask_14 = beginning_mask_2d_7[(None, slice(None, None, None), None, slice(None, None, None))];  beginning_mask_2d_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    ending_mask_14 = beginning_mask_14.flip(dims = (1, 3))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    beginning_input_7 = diagonal_mask_3[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    beginning_mask_15 = beginning_mask_14.expand((1, 256, 1, 257));  beginning_mask_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    full_like_14 = torch.full_like(beginning_input_7, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    bool_15 = beginning_mask_15.bool();  beginning_mask_15 = None
    where_14 = full_like_14.where(bool_15, beginning_input_7);  full_like_14 = bool_15 = beginning_input_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    diagonal_mask_3[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))] = where_14;  setitem_46 = diagonal_mask_3;  where_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    ending_input_7 = diagonal_mask_3[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    ending_mask_15 = ending_mask_14.expand((1, 256, 1, 257));  ending_mask_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    full_like_15 = torch.full_like(ending_input_7, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    bool_16 = ending_mask_15.bool();  ending_mask_15 = None
    where_15 = full_like_15.where(bool_16, ending_input_7);  full_like_15 = bool_16 = ending_input_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    diagonal_mask_3[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))] = where_15;  setitem_47 = diagonal_mask_3;  where_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    attn_scores_6 += diagonal_mask_3;  attn_scores_7 = attn_scores_6;  attn_scores_6 = diagonal_mask_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    attn_probs_12 = torch.nn.functional.softmax(attn_scores_7, dim = -1, dtype = torch.float32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    getitem_69 = l_is_index_masked_[(slice(None, None, None), slice(None, None, None), None, None)]
    attn_probs_13 = torch.masked_fill(attn_probs_12, getitem_69, 0.0);  attn_probs_12 = getitem_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:638, code: attn_probs = attn_probs.type_as(attn_scores)
    attn_probs_14 = attn_probs_13.type_as(attn_scores_7);  attn_probs_13 = attn_scores_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    attn_probs_15 = torch.nn.functional.dropout(attn_probs_14, p = 0.1, training = True);  attn_probs_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_52 = value_vectors_6.view(1024, 1, 12, 64);  value_vectors_6 = None
    value_vectors_7 = view_52.transpose(0, 1);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:896, code: chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
    div_30 = torch.div(1024, 256, rounding_mode = 'trunc')
    chunks_count_11 = div_30 - 1;  div_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    transpose_55 = attn_probs_15.transpose(1, 2);  attn_probs_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:901, code: torch.div(seq_len, window_overlap, rounding_mode="trunc"),
    div_31 = torch.div(1024, 256, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    chunked_attn_probs_6 = transpose_55.reshape(12, div_31, 256, 513);  transpose_55 = div_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_56 = value_vectors_7.transpose(1, 2);  value_vectors_7 = None
    value_3 = transpose_56.reshape(12, 1024, 64);  transpose_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    padded_value_3 = torch.nn.functional.pad(value_3, (0, 0, 256, 256), value = -1);  value_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:913, code: chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
    add_17 = chunks_count_11 + 1;  chunks_count_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    chunked_value_3 = padded_value_3.as_strided(size = (12, add_17, 768, 64), stride = (98304, 16384, 64, 1));  padded_value_3 = add_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    chunked_hidden_states_15 = torch.nn.functional.pad(chunked_attn_probs_6, (0, 257));  chunked_attn_probs_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    chunked_hidden_states_16 = chunked_hidden_states_15.view(12, 4, -1);  chunked_hidden_states_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    chunked_hidden_states_17 = chunked_hidden_states_16[(slice(None, None, None), slice(None, None, None), slice(None, -256, None))];  chunked_hidden_states_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    chunked_hidden_states_18 = chunked_hidden_states_17.view(12, 4, 256, 769);  chunked_hidden_states_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    chunked_attn_probs_7 = chunked_hidden_states_18[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, -1, None))];  chunked_hidden_states_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    context_3 = torch.functional.einsum('bcwd,bcdh->bcwh', (chunked_attn_probs_7, chunked_value_3));  chunked_attn_probs_7 = chunked_value_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    view_55 = context_3.view(1, 12, 1024, 64);  context_3 = None
    attn_output_12 = view_55.transpose(1, 2);  view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    transpose_58 = attn_output_12.transpose(0, 1);  attn_output_12 = None
    reshape_27 = transpose_58.reshape(1024, 1, 768);  transpose_58 = None
    attn_output_13 = reshape_27.contiguous();  reshape_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    transpose_59 = attn_output_13.transpose(0, 1);  attn_output_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    hidden_states_47 = self.L__self___layer_3_attention_output_dense(transpose_59);  transpose_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    hidden_states_48 = self.L__self___layer_3_attention_output_dropout(hidden_states_47);  hidden_states_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_18 = hidden_states_48 + hidden_states_41;  hidden_states_48 = hidden_states_41 = None
    attn_output_15 = self.L__self___layer_3_attention_output_LayerNorm(add_18);  add_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    hidden_states_50 = self.L__self___layer_3_intermediate_dense(attn_output_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_3 = torch._C._nn.gelu(hidden_states_50);  hidden_states_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    hidden_states_52 = self.L__self___layer_3_output_dense(intermediate_output_3);  intermediate_output_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    hidden_states_53 = self.L__self___layer_3_output_dropout(hidden_states_52);  hidden_states_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_19 = hidden_states_53 + attn_output_15;  hidden_states_53 = attn_output_15 = None
    hidden_states_55 = self.L__self___layer_3_output_LayerNorm(add_19);  add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    hidden_states_56 = hidden_states_55.transpose(0, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    query_vectors_12 = self.L__self___layer_4_attention_self_query(hidden_states_56)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    key_vectors_8 = self.L__self___layer_4_attention_self_key(hidden_states_56)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    value_vectors_8 = self.L__self___layer_4_attention_self_value(hidden_states_56);  hidden_states_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    query_vectors_12 /= 8.0;  query_vectors_13 = query_vectors_12;  query_vectors_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:568, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_56 = query_vectors_13.view(1024, 1, 12, 64);  query_vectors_13 = None
    query_vectors_14 = view_56.transpose(0, 1);  view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:569, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_57 = key_vectors_8.view(1024, 1, 12, 64);  key_vectors_8 = None
    key_vectors_9 = view_57.transpose(0, 1);  view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:830, code: chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
    div_32 = torch.div(1024, 256, rounding_mode = 'trunc')
    chunks_count_12 = div_32 - 1;  div_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:833, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_63 = query_vectors_14.transpose(1, 2)
    query_16 = transpose_63.reshape(12, 1024, 64);  transpose_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_64 = key_vectors_9.transpose(1, 2);  key_vectors_9 = None
    key_16 = transpose_64.reshape(12, 1024, 64);  transpose_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_33 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_57 = query_16.view(12, div_33, 512, 64);  query_16 = div_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    query_17 = hidden_states_57.as_strided(size = [12, 3, 512, 64], stride = [64, 196608, 768, 1]);  hidden_states_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_34 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_58 = key_16.view(12, div_34, 512, 64);  key_16 = div_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    key_17 = hidden_states_58.as_strided(size = [12, 3, 512, 64], stride = [64, 196608, 768, 1]);  hidden_states_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    diagonal_chunked_attention_scores_16 = torch.functional.einsum('bcxd,bcyd->bcxy', (query_17, key_17));  query_17 = key_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    hidden_states_padded_16 = torch.nn.functional.pad(diagonal_chunked_attention_scores_16, (0, 0, 0, 1));  diagonal_chunked_attention_scores_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    diagonal_chunked_attention_scores_17 = hidden_states_padded_16.view(12, 3, 512, 513);  hidden_states_padded_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:856, code: (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
    add_20 = chunks_count_12 + 1;  chunks_count_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    diagonal_attention_scores_16 = diagonal_chunked_attention_scores_17.new_zeros((12, add_20, 256, 513));  add_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_72 = diagonal_chunked_attention_scores_17[(slice(None, None, None), slice(None, None, None), slice(None, 256, None), slice(None, 257, None))]
    diagonal_attention_scores_16[(slice(None, None, None), slice(None, -1, None), slice(None, None, None), slice(256, None, None))] = getitem_72;  setitem_48 = diagonal_attention_scores_16;  getitem_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_73 = diagonal_chunked_attention_scores_17[(slice(None, None, None), -1, slice(256, None, None), slice(None, 257, None))]
    diagonal_attention_scores_16[(slice(None, None, None), -1, slice(None, None, None), slice(256, None, None))] = getitem_73;  setitem_49 = diagonal_attention_scores_16;  getitem_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    getitem_74 = diagonal_chunked_attention_scores_17[(slice(None, None, None), slice(None, None, None), slice(-257, -1, None), slice(257, None, None))]
    diagonal_attention_scores_16[(slice(None, None, None), slice(1, None, None), slice(None, None, None), slice(None, 256, None))] = getitem_74;  setitem_50 = diagonal_attention_scores_16;  getitem_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    getitem_75 = diagonal_chunked_attention_scores_17[(slice(None, None, None), 0, slice(None, 255, None), slice(-255, None, None))];  diagonal_chunked_attention_scores_17 = None
    diagonal_attention_scores_16[(slice(None, None, None), 0, slice(1, 256, None), slice(1, 256, None))] = getitem_75;  setitem_51 = diagonal_attention_scores_16;  getitem_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    view_61 = diagonal_attention_scores_16.view(1, 12, 1024, 513);  diagonal_attention_scores_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:879, code: ).transpose(2, 1)
    attn_scores_8 = view_61.transpose(2, 1);  view_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    new_ones_12 = attn_scores_8.new_ones(256, 257)
    tril_8 = new_ones_12.tril();  new_ones_12 = None
    beginning_mask_2d_8 = tril_8.flip(dims = [0]);  tril_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    beginning_mask_16 = beginning_mask_2d_8[(None, slice(None, None, None), None, slice(None, None, None))];  beginning_mask_2d_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    ending_mask_16 = beginning_mask_16.flip(dims = (1, 3))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    beginning_input_8 = attn_scores_8[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    beginning_mask_17 = beginning_mask_16.expand((1, 256, 12, 257));  beginning_mask_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    full_like_16 = torch.full_like(beginning_input_8, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    bool_17 = beginning_mask_17.bool();  beginning_mask_17 = None
    where_16 = full_like_16.where(bool_17, beginning_input_8);  full_like_16 = bool_17 = beginning_input_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    attn_scores_8[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))] = where_16;  setitem_52 = attn_scores_8;  where_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    ending_input_8 = attn_scores_8[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    ending_mask_17 = ending_mask_16.expand((1, 256, 12, 257));  ending_mask_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    full_like_17 = torch.full_like(ending_input_8, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    bool_18 = ending_mask_17.bool();  ending_mask_17 = None
    where_17 = full_like_17.where(bool_18, ending_input_8);  full_like_17 = bool_18 = ending_input_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    attn_scores_8[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))] = where_17;  setitem_53 = attn_scores_8;  where_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:576, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
    ne_4 = l_attention_mask_ != 0
    remove_from_windowed_attention_mask_4 = ne_4[(slice(None, None, None), slice(None, None, None), None, None)];  ne_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
    type_as_8 = remove_from_windowed_attention_mask_4.type_as(query_vectors_14);  query_vectors_14 = None
    float_mask_4 = type_as_8.masked_fill(remove_from_windowed_attention_mask_4, -3.4028234663852886e+38);  type_as_8 = remove_from_windowed_attention_mask_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:584, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
    new_ones_13 = float_mask_4.new_ones(size = (1, 1024, 1, 1))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:830, code: chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
    div_35 = torch.div(1024, 256, rounding_mode = 'trunc')
    chunks_count_13 = div_35 - 1;  div_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:833, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_66 = new_ones_13.transpose(1, 2);  new_ones_13 = None
    query_18 = transpose_66.reshape(1, 1024, 1);  transpose_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_67 = float_mask_4.transpose(1, 2);  float_mask_4 = None
    key_18 = transpose_67.reshape(1, 1024, 1);  transpose_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_36 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_59 = query_18.view(1, div_36, 512, 1);  query_18 = div_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    query_19 = hidden_states_59.as_strided(size = [1, 3, 512, 1], stride = [1024, 256, 1, 1]);  hidden_states_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_37 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_60 = key_18.view(1, div_37, 512, 1);  key_18 = div_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    key_19 = hidden_states_60.as_strided(size = [1, 3, 512, 1], stride = [1024, 256, 1, 1]);  hidden_states_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    diagonal_chunked_attention_scores_18 = torch.functional.einsum('bcxd,bcyd->bcxy', (query_19, key_19));  query_19 = key_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    hidden_states_padded_18 = torch.nn.functional.pad(diagonal_chunked_attention_scores_18, (0, 0, 0, 1));  diagonal_chunked_attention_scores_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    diagonal_chunked_attention_scores_19 = hidden_states_padded_18.view(1, 3, 512, 513);  hidden_states_padded_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:856, code: (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
    add_21 = chunks_count_13 + 1;  chunks_count_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    diagonal_attention_scores_18 = diagonal_chunked_attention_scores_19.new_zeros((1, add_21, 256, 513));  add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_80 = diagonal_chunked_attention_scores_19[(slice(None, None, None), slice(None, None, None), slice(None, 256, None), slice(None, 257, None))]
    diagonal_attention_scores_18[(slice(None, None, None), slice(None, -1, None), slice(None, None, None), slice(256, None, None))] = getitem_80;  setitem_54 = diagonal_attention_scores_18;  getitem_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_81 = diagonal_chunked_attention_scores_19[(slice(None, None, None), -1, slice(256, None, None), slice(None, 257, None))]
    diagonal_attention_scores_18[(slice(None, None, None), -1, slice(None, None, None), slice(256, None, None))] = getitem_81;  setitem_55 = diagonal_attention_scores_18;  getitem_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    getitem_82 = diagonal_chunked_attention_scores_19[(slice(None, None, None), slice(None, None, None), slice(-257, -1, None), slice(257, None, None))]
    diagonal_attention_scores_18[(slice(None, None, None), slice(1, None, None), slice(None, None, None), slice(None, 256, None))] = getitem_82;  setitem_56 = diagonal_attention_scores_18;  getitem_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    getitem_83 = diagonal_chunked_attention_scores_19[(slice(None, None, None), 0, slice(None, 255, None), slice(-255, None, None))];  diagonal_chunked_attention_scores_19 = None
    diagonal_attention_scores_18[(slice(None, None, None), 0, slice(1, 256, None), slice(1, 256, None))] = getitem_83;  setitem_57 = diagonal_attention_scores_18;  getitem_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    view_65 = diagonal_attention_scores_18.view(1, 1, 1024, 513);  diagonal_attention_scores_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:879, code: ).transpose(2, 1)
    diagonal_mask_4 = view_65.transpose(2, 1);  view_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    new_ones_14 = diagonal_mask_4.new_ones(256, 257)
    tril_9 = new_ones_14.tril();  new_ones_14 = None
    beginning_mask_2d_9 = tril_9.flip(dims = [0]);  tril_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    beginning_mask_18 = beginning_mask_2d_9[(None, slice(None, None, None), None, slice(None, None, None))];  beginning_mask_2d_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    ending_mask_18 = beginning_mask_18.flip(dims = (1, 3))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    beginning_input_9 = diagonal_mask_4[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    beginning_mask_19 = beginning_mask_18.expand((1, 256, 1, 257));  beginning_mask_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    full_like_18 = torch.full_like(beginning_input_9, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    bool_19 = beginning_mask_19.bool();  beginning_mask_19 = None
    where_18 = full_like_18.where(bool_19, beginning_input_9);  full_like_18 = bool_19 = beginning_input_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    diagonal_mask_4[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))] = where_18;  setitem_58 = diagonal_mask_4;  where_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    ending_input_9 = diagonal_mask_4[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    ending_mask_19 = ending_mask_18.expand((1, 256, 1, 257));  ending_mask_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    full_like_19 = torch.full_like(ending_input_9, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    bool_20 = ending_mask_19.bool();  ending_mask_19 = None
    where_19 = full_like_19.where(bool_20, ending_input_9);  full_like_19 = bool_20 = ending_input_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    diagonal_mask_4[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))] = where_19;  setitem_59 = diagonal_mask_4;  where_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    attn_scores_8 += diagonal_mask_4;  attn_scores_9 = attn_scores_8;  attn_scores_8 = diagonal_mask_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    attn_probs_16 = torch.nn.functional.softmax(attn_scores_9, dim = -1, dtype = torch.float32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    getitem_87 = l_is_index_masked_[(slice(None, None, None), slice(None, None, None), None, None)]
    attn_probs_17 = torch.masked_fill(attn_probs_16, getitem_87, 0.0);  attn_probs_16 = getitem_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:638, code: attn_probs = attn_probs.type_as(attn_scores)
    attn_probs_18 = attn_probs_17.type_as(attn_scores_9);  attn_probs_17 = attn_scores_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    attn_probs_19 = torch.nn.functional.dropout(attn_probs_18, p = 0.1, training = True);  attn_probs_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_66 = value_vectors_8.view(1024, 1, 12, 64);  value_vectors_8 = None
    value_vectors_9 = view_66.transpose(0, 1);  view_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:896, code: chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
    div_38 = torch.div(1024, 256, rounding_mode = 'trunc')
    chunks_count_14 = div_38 - 1;  div_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    transpose_70 = attn_probs_19.transpose(1, 2);  attn_probs_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:901, code: torch.div(seq_len, window_overlap, rounding_mode="trunc"),
    div_39 = torch.div(1024, 256, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    chunked_attn_probs_8 = transpose_70.reshape(12, div_39, 256, 513);  transpose_70 = div_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_71 = value_vectors_9.transpose(1, 2);  value_vectors_9 = None
    value_4 = transpose_71.reshape(12, 1024, 64);  transpose_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    padded_value_4 = torch.nn.functional.pad(value_4, (0, 0, 256, 256), value = -1);  value_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:913, code: chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
    add_22 = chunks_count_14 + 1;  chunks_count_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    chunked_value_4 = padded_value_4.as_strided(size = (12, add_22, 768, 64), stride = (98304, 16384, 64, 1));  padded_value_4 = add_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    chunked_hidden_states_20 = torch.nn.functional.pad(chunked_attn_probs_8, (0, 257));  chunked_attn_probs_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    chunked_hidden_states_21 = chunked_hidden_states_20.view(12, 4, -1);  chunked_hidden_states_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    chunked_hidden_states_22 = chunked_hidden_states_21[(slice(None, None, None), slice(None, None, None), slice(None, -256, None))];  chunked_hidden_states_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    chunked_hidden_states_23 = chunked_hidden_states_22.view(12, 4, 256, 769);  chunked_hidden_states_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    chunked_attn_probs_9 = chunked_hidden_states_23[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, -1, None))];  chunked_hidden_states_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    context_4 = torch.functional.einsum('bcwd,bcdh->bcwh', (chunked_attn_probs_9, chunked_value_4));  chunked_attn_probs_9 = chunked_value_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    view_69 = context_4.view(1, 12, 1024, 64);  context_4 = None
    attn_output_16 = view_69.transpose(1, 2);  view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    transpose_73 = attn_output_16.transpose(0, 1);  attn_output_16 = None
    reshape_34 = transpose_73.reshape(1024, 1, 768);  transpose_73 = None
    attn_output_17 = reshape_34.contiguous();  reshape_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    transpose_74 = attn_output_17.transpose(0, 1);  attn_output_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    hidden_states_61 = self.L__self___layer_4_attention_output_dense(transpose_74);  transpose_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    hidden_states_62 = self.L__self___layer_4_attention_output_dropout(hidden_states_61);  hidden_states_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_23 = hidden_states_62 + hidden_states_55;  hidden_states_62 = hidden_states_55 = None
    attn_output_19 = self.L__self___layer_4_attention_output_LayerNorm(add_23);  add_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    hidden_states_64 = self.L__self___layer_4_intermediate_dense(attn_output_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_4 = torch._C._nn.gelu(hidden_states_64);  hidden_states_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    hidden_states_66 = self.L__self___layer_4_output_dense(intermediate_output_4);  intermediate_output_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    hidden_states_67 = self.L__self___layer_4_output_dropout(hidden_states_66);  hidden_states_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_24 = hidden_states_67 + attn_output_19;  hidden_states_67 = attn_output_19 = None
    hidden_states_69 = self.L__self___layer_4_output_LayerNorm(add_24);  add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    hidden_states_70 = hidden_states_69.transpose(0, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    query_vectors_15 = self.L__self___layer_5_attention_self_query(hidden_states_70)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    key_vectors_10 = self.L__self___layer_5_attention_self_key(hidden_states_70)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    value_vectors_10 = self.L__self___layer_5_attention_self_value(hidden_states_70);  hidden_states_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    query_vectors_15 /= 8.0;  query_vectors_16 = query_vectors_15;  query_vectors_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:568, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_70 = query_vectors_16.view(1024, 1, 12, 64);  query_vectors_16 = None
    query_vectors_17 = view_70.transpose(0, 1);  view_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:569, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_71 = key_vectors_10.view(1024, 1, 12, 64);  key_vectors_10 = None
    key_vectors_11 = view_71.transpose(0, 1);  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:830, code: chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
    div_40 = torch.div(1024, 256, rounding_mode = 'trunc')
    chunks_count_15 = div_40 - 1;  div_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:833, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_78 = query_vectors_17.transpose(1, 2)
    query_20 = transpose_78.reshape(12, 1024, 64);  transpose_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_79 = key_vectors_11.transpose(1, 2);  key_vectors_11 = None
    key_20 = transpose_79.reshape(12, 1024, 64);  transpose_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_41 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_71 = query_20.view(12, div_41, 512, 64);  query_20 = div_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    query_21 = hidden_states_71.as_strided(size = [12, 3, 512, 64], stride = [64, 196608, 768, 1]);  hidden_states_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_42 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_72 = key_20.view(12, div_42, 512, 64);  key_20 = div_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    key_21 = hidden_states_72.as_strided(size = [12, 3, 512, 64], stride = [64, 196608, 768, 1]);  hidden_states_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    diagonal_chunked_attention_scores_20 = torch.functional.einsum('bcxd,bcyd->bcxy', (query_21, key_21));  query_21 = key_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    hidden_states_padded_20 = torch.nn.functional.pad(diagonal_chunked_attention_scores_20, (0, 0, 0, 1));  diagonal_chunked_attention_scores_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    diagonal_chunked_attention_scores_21 = hidden_states_padded_20.view(12, 3, 512, 513);  hidden_states_padded_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:856, code: (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
    add_25 = chunks_count_15 + 1;  chunks_count_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    diagonal_attention_scores_20 = diagonal_chunked_attention_scores_21.new_zeros((12, add_25, 256, 513));  add_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_90 = diagonal_chunked_attention_scores_21[(slice(None, None, None), slice(None, None, None), slice(None, 256, None), slice(None, 257, None))]
    diagonal_attention_scores_20[(slice(None, None, None), slice(None, -1, None), slice(None, None, None), slice(256, None, None))] = getitem_90;  setitem_60 = diagonal_attention_scores_20;  getitem_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_91 = diagonal_chunked_attention_scores_21[(slice(None, None, None), -1, slice(256, None, None), slice(None, 257, None))]
    diagonal_attention_scores_20[(slice(None, None, None), -1, slice(None, None, None), slice(256, None, None))] = getitem_91;  setitem_61 = diagonal_attention_scores_20;  getitem_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    getitem_92 = diagonal_chunked_attention_scores_21[(slice(None, None, None), slice(None, None, None), slice(-257, -1, None), slice(257, None, None))]
    diagonal_attention_scores_20[(slice(None, None, None), slice(1, None, None), slice(None, None, None), slice(None, 256, None))] = getitem_92;  setitem_62 = diagonal_attention_scores_20;  getitem_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    getitem_93 = diagonal_chunked_attention_scores_21[(slice(None, None, None), 0, slice(None, 255, None), slice(-255, None, None))];  diagonal_chunked_attention_scores_21 = None
    diagonal_attention_scores_20[(slice(None, None, None), 0, slice(1, 256, None), slice(1, 256, None))] = getitem_93;  setitem_63 = diagonal_attention_scores_20;  getitem_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    view_75 = diagonal_attention_scores_20.view(1, 12, 1024, 513);  diagonal_attention_scores_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:879, code: ).transpose(2, 1)
    attn_scores_10 = view_75.transpose(2, 1);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    new_ones_15 = attn_scores_10.new_ones(256, 257)
    tril_10 = new_ones_15.tril();  new_ones_15 = None
    beginning_mask_2d_10 = tril_10.flip(dims = [0]);  tril_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    beginning_mask_20 = beginning_mask_2d_10[(None, slice(None, None, None), None, slice(None, None, None))];  beginning_mask_2d_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    ending_mask_20 = beginning_mask_20.flip(dims = (1, 3))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    beginning_input_10 = attn_scores_10[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    beginning_mask_21 = beginning_mask_20.expand((1, 256, 12, 257));  beginning_mask_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    full_like_20 = torch.full_like(beginning_input_10, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    bool_21 = beginning_mask_21.bool();  beginning_mask_21 = None
    where_20 = full_like_20.where(bool_21, beginning_input_10);  full_like_20 = bool_21 = beginning_input_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    attn_scores_10[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))] = where_20;  setitem_64 = attn_scores_10;  where_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    ending_input_10 = attn_scores_10[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    ending_mask_21 = ending_mask_20.expand((1, 256, 12, 257));  ending_mask_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    full_like_21 = torch.full_like(ending_input_10, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    bool_22 = ending_mask_21.bool();  ending_mask_21 = None
    where_21 = full_like_21.where(bool_22, ending_input_10);  full_like_21 = bool_22 = ending_input_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    attn_scores_10[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))] = where_21;  setitem_65 = attn_scores_10;  where_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:576, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
    ne_5 = l_attention_mask_ != 0
    remove_from_windowed_attention_mask_5 = ne_5[(slice(None, None, None), slice(None, None, None), None, None)];  ne_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
    type_as_10 = remove_from_windowed_attention_mask_5.type_as(query_vectors_17);  query_vectors_17 = None
    float_mask_5 = type_as_10.masked_fill(remove_from_windowed_attention_mask_5, -3.4028234663852886e+38);  type_as_10 = remove_from_windowed_attention_mask_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:584, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
    new_ones_16 = float_mask_5.new_ones(size = (1, 1024, 1, 1))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:830, code: chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
    div_43 = torch.div(1024, 256, rounding_mode = 'trunc')
    chunks_count_16 = div_43 - 1;  div_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:833, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_81 = new_ones_16.transpose(1, 2);  new_ones_16 = None
    query_22 = transpose_81.reshape(1, 1024, 1);  transpose_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_82 = float_mask_5.transpose(1, 2);  float_mask_5 = None
    key_22 = transpose_82.reshape(1, 1024, 1);  transpose_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_44 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_73 = query_22.view(1, div_44, 512, 1);  query_22 = div_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    query_23 = hidden_states_73.as_strided(size = [1, 3, 512, 1], stride = [1024, 256, 1, 1]);  hidden_states_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_45 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_74 = key_22.view(1, div_45, 512, 1);  key_22 = div_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    key_23 = hidden_states_74.as_strided(size = [1, 3, 512, 1], stride = [1024, 256, 1, 1]);  hidden_states_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    diagonal_chunked_attention_scores_22 = torch.functional.einsum('bcxd,bcyd->bcxy', (query_23, key_23));  query_23 = key_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    hidden_states_padded_22 = torch.nn.functional.pad(diagonal_chunked_attention_scores_22, (0, 0, 0, 1));  diagonal_chunked_attention_scores_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    diagonal_chunked_attention_scores_23 = hidden_states_padded_22.view(1, 3, 512, 513);  hidden_states_padded_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:856, code: (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
    add_26 = chunks_count_16 + 1;  chunks_count_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    diagonal_attention_scores_22 = diagonal_chunked_attention_scores_23.new_zeros((1, add_26, 256, 513));  add_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_98 = diagonal_chunked_attention_scores_23[(slice(None, None, None), slice(None, None, None), slice(None, 256, None), slice(None, 257, None))]
    diagonal_attention_scores_22[(slice(None, None, None), slice(None, -1, None), slice(None, None, None), slice(256, None, None))] = getitem_98;  setitem_66 = diagonal_attention_scores_22;  getitem_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_99 = diagonal_chunked_attention_scores_23[(slice(None, None, None), -1, slice(256, None, None), slice(None, 257, None))]
    diagonal_attention_scores_22[(slice(None, None, None), -1, slice(None, None, None), slice(256, None, None))] = getitem_99;  setitem_67 = diagonal_attention_scores_22;  getitem_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    getitem_100 = diagonal_chunked_attention_scores_23[(slice(None, None, None), slice(None, None, None), slice(-257, -1, None), slice(257, None, None))]
    diagonal_attention_scores_22[(slice(None, None, None), slice(1, None, None), slice(None, None, None), slice(None, 256, None))] = getitem_100;  setitem_68 = diagonal_attention_scores_22;  getitem_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    getitem_101 = diagonal_chunked_attention_scores_23[(slice(None, None, None), 0, slice(None, 255, None), slice(-255, None, None))];  diagonal_chunked_attention_scores_23 = None
    diagonal_attention_scores_22[(slice(None, None, None), 0, slice(1, 256, None), slice(1, 256, None))] = getitem_101;  setitem_69 = diagonal_attention_scores_22;  getitem_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    view_79 = diagonal_attention_scores_22.view(1, 1, 1024, 513);  diagonal_attention_scores_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:879, code: ).transpose(2, 1)
    diagonal_mask_5 = view_79.transpose(2, 1);  view_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    new_ones_17 = diagonal_mask_5.new_ones(256, 257)
    tril_11 = new_ones_17.tril();  new_ones_17 = None
    beginning_mask_2d_11 = tril_11.flip(dims = [0]);  tril_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    beginning_mask_22 = beginning_mask_2d_11[(None, slice(None, None, None), None, slice(None, None, None))];  beginning_mask_2d_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    ending_mask_22 = beginning_mask_22.flip(dims = (1, 3))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    beginning_input_11 = diagonal_mask_5[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    beginning_mask_23 = beginning_mask_22.expand((1, 256, 1, 257));  beginning_mask_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    full_like_22 = torch.full_like(beginning_input_11, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    bool_23 = beginning_mask_23.bool();  beginning_mask_23 = None
    where_22 = full_like_22.where(bool_23, beginning_input_11);  full_like_22 = bool_23 = beginning_input_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    diagonal_mask_5[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))] = where_22;  setitem_70 = diagonal_mask_5;  where_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    ending_input_11 = diagonal_mask_5[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    ending_mask_23 = ending_mask_22.expand((1, 256, 1, 257));  ending_mask_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    full_like_23 = torch.full_like(ending_input_11, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    bool_24 = ending_mask_23.bool();  ending_mask_23 = None
    where_23 = full_like_23.where(bool_24, ending_input_11);  full_like_23 = bool_24 = ending_input_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    diagonal_mask_5[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))] = where_23;  setitem_71 = diagonal_mask_5;  where_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    attn_scores_10 += diagonal_mask_5;  attn_scores_11 = attn_scores_10;  attn_scores_10 = diagonal_mask_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    attn_probs_20 = torch.nn.functional.softmax(attn_scores_11, dim = -1, dtype = torch.float32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    getitem_105 = l_is_index_masked_[(slice(None, None, None), slice(None, None, None), None, None)]
    attn_probs_21 = torch.masked_fill(attn_probs_20, getitem_105, 0.0);  attn_probs_20 = getitem_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:638, code: attn_probs = attn_probs.type_as(attn_scores)
    attn_probs_22 = attn_probs_21.type_as(attn_scores_11);  attn_probs_21 = attn_scores_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    attn_probs_23 = torch.nn.functional.dropout(attn_probs_22, p = 0.1, training = True);  attn_probs_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_80 = value_vectors_10.view(1024, 1, 12, 64);  value_vectors_10 = None
    value_vectors_11 = view_80.transpose(0, 1);  view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:896, code: chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
    div_46 = torch.div(1024, 256, rounding_mode = 'trunc')
    chunks_count_17 = div_46 - 1;  div_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    transpose_85 = attn_probs_23.transpose(1, 2);  attn_probs_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:901, code: torch.div(seq_len, window_overlap, rounding_mode="trunc"),
    div_47 = torch.div(1024, 256, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    chunked_attn_probs_10 = transpose_85.reshape(12, div_47, 256, 513);  transpose_85 = div_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_86 = value_vectors_11.transpose(1, 2);  value_vectors_11 = None
    value_5 = transpose_86.reshape(12, 1024, 64);  transpose_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    padded_value_5 = torch.nn.functional.pad(value_5, (0, 0, 256, 256), value = -1);  value_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:913, code: chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
    add_27 = chunks_count_17 + 1;  chunks_count_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    chunked_value_5 = padded_value_5.as_strided(size = (12, add_27, 768, 64), stride = (98304, 16384, 64, 1));  padded_value_5 = add_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    chunked_hidden_states_25 = torch.nn.functional.pad(chunked_attn_probs_10, (0, 257));  chunked_attn_probs_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    chunked_hidden_states_26 = chunked_hidden_states_25.view(12, 4, -1);  chunked_hidden_states_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    chunked_hidden_states_27 = chunked_hidden_states_26[(slice(None, None, None), slice(None, None, None), slice(None, -256, None))];  chunked_hidden_states_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    chunked_hidden_states_28 = chunked_hidden_states_27.view(12, 4, 256, 769);  chunked_hidden_states_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    chunked_attn_probs_11 = chunked_hidden_states_28[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, -1, None))];  chunked_hidden_states_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    context_5 = torch.functional.einsum('bcwd,bcdh->bcwh', (chunked_attn_probs_11, chunked_value_5));  chunked_attn_probs_11 = chunked_value_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    view_83 = context_5.view(1, 12, 1024, 64);  context_5 = None
    attn_output_20 = view_83.transpose(1, 2);  view_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    transpose_88 = attn_output_20.transpose(0, 1);  attn_output_20 = None
    reshape_41 = transpose_88.reshape(1024, 1, 768);  transpose_88 = None
    attn_output_21 = reshape_41.contiguous();  reshape_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    transpose_89 = attn_output_21.transpose(0, 1);  attn_output_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    hidden_states_75 = self.L__self___layer_5_attention_output_dense(transpose_89);  transpose_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    hidden_states_76 = self.L__self___layer_5_attention_output_dropout(hidden_states_75);  hidden_states_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_28 = hidden_states_76 + hidden_states_69;  hidden_states_76 = hidden_states_69 = None
    attn_output_23 = self.L__self___layer_5_attention_output_LayerNorm(add_28);  add_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    hidden_states_78 = self.L__self___layer_5_intermediate_dense(attn_output_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_5 = torch._C._nn.gelu(hidden_states_78);  hidden_states_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    hidden_states_80 = self.L__self___layer_5_output_dense(intermediate_output_5);  intermediate_output_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    hidden_states_81 = self.L__self___layer_5_output_dropout(hidden_states_80);  hidden_states_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_29 = hidden_states_81 + attn_output_23;  hidden_states_81 = attn_output_23 = None
    hidden_states_83 = self.L__self___layer_5_output_LayerNorm(add_29);  add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    hidden_states_84 = hidden_states_83.transpose(0, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    query_vectors_18 = self.L__self___layer_6_attention_self_query(hidden_states_84)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    key_vectors_12 = self.L__self___layer_6_attention_self_key(hidden_states_84)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    value_vectors_12 = self.L__self___layer_6_attention_self_value(hidden_states_84);  hidden_states_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    query_vectors_18 /= 8.0;  query_vectors_19 = query_vectors_18;  query_vectors_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:568, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_84 = query_vectors_19.view(1024, 1, 12, 64);  query_vectors_19 = None
    query_vectors_20 = view_84.transpose(0, 1);  view_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:569, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_85 = key_vectors_12.view(1024, 1, 12, 64);  key_vectors_12 = None
    key_vectors_13 = view_85.transpose(0, 1);  view_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:830, code: chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
    div_48 = torch.div(1024, 256, rounding_mode = 'trunc')
    chunks_count_18 = div_48 - 1;  div_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:833, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_93 = query_vectors_20.transpose(1, 2)
    query_24 = transpose_93.reshape(12, 1024, 64);  transpose_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_94 = key_vectors_13.transpose(1, 2);  key_vectors_13 = None
    key_24 = transpose_94.reshape(12, 1024, 64);  transpose_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_49 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_85 = query_24.view(12, div_49, 512, 64);  query_24 = div_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    query_25 = hidden_states_85.as_strided(size = [12, 3, 512, 64], stride = [64, 196608, 768, 1]);  hidden_states_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_50 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_86 = key_24.view(12, div_50, 512, 64);  key_24 = div_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    key_25 = hidden_states_86.as_strided(size = [12, 3, 512, 64], stride = [64, 196608, 768, 1]);  hidden_states_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    diagonal_chunked_attention_scores_24 = torch.functional.einsum('bcxd,bcyd->bcxy', (query_25, key_25));  query_25 = key_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    hidden_states_padded_24 = torch.nn.functional.pad(diagonal_chunked_attention_scores_24, (0, 0, 0, 1));  diagonal_chunked_attention_scores_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    diagonal_chunked_attention_scores_25 = hidden_states_padded_24.view(12, 3, 512, 513);  hidden_states_padded_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:856, code: (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
    add_30 = chunks_count_18 + 1;  chunks_count_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    diagonal_attention_scores_24 = diagonal_chunked_attention_scores_25.new_zeros((12, add_30, 256, 513));  add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_108 = diagonal_chunked_attention_scores_25[(slice(None, None, None), slice(None, None, None), slice(None, 256, None), slice(None, 257, None))]
    diagonal_attention_scores_24[(slice(None, None, None), slice(None, -1, None), slice(None, None, None), slice(256, None, None))] = getitem_108;  setitem_72 = diagonal_attention_scores_24;  getitem_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_109 = diagonal_chunked_attention_scores_25[(slice(None, None, None), -1, slice(256, None, None), slice(None, 257, None))]
    diagonal_attention_scores_24[(slice(None, None, None), -1, slice(None, None, None), slice(256, None, None))] = getitem_109;  setitem_73 = diagonal_attention_scores_24;  getitem_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    getitem_110 = diagonal_chunked_attention_scores_25[(slice(None, None, None), slice(None, None, None), slice(-257, -1, None), slice(257, None, None))]
    diagonal_attention_scores_24[(slice(None, None, None), slice(1, None, None), slice(None, None, None), slice(None, 256, None))] = getitem_110;  setitem_74 = diagonal_attention_scores_24;  getitem_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    getitem_111 = diagonal_chunked_attention_scores_25[(slice(None, None, None), 0, slice(None, 255, None), slice(-255, None, None))];  diagonal_chunked_attention_scores_25 = None
    diagonal_attention_scores_24[(slice(None, None, None), 0, slice(1, 256, None), slice(1, 256, None))] = getitem_111;  setitem_75 = diagonal_attention_scores_24;  getitem_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    view_89 = diagonal_attention_scores_24.view(1, 12, 1024, 513);  diagonal_attention_scores_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:879, code: ).transpose(2, 1)
    attn_scores_12 = view_89.transpose(2, 1);  view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    new_ones_18 = attn_scores_12.new_ones(256, 257)
    tril_12 = new_ones_18.tril();  new_ones_18 = None
    beginning_mask_2d_12 = tril_12.flip(dims = [0]);  tril_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    beginning_mask_24 = beginning_mask_2d_12[(None, slice(None, None, None), None, slice(None, None, None))];  beginning_mask_2d_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    ending_mask_24 = beginning_mask_24.flip(dims = (1, 3))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    beginning_input_12 = attn_scores_12[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    beginning_mask_25 = beginning_mask_24.expand((1, 256, 12, 257));  beginning_mask_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    full_like_24 = torch.full_like(beginning_input_12, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    bool_25 = beginning_mask_25.bool();  beginning_mask_25 = None
    where_24 = full_like_24.where(bool_25, beginning_input_12);  full_like_24 = bool_25 = beginning_input_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    attn_scores_12[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))] = where_24;  setitem_76 = attn_scores_12;  where_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    ending_input_12 = attn_scores_12[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    ending_mask_25 = ending_mask_24.expand((1, 256, 12, 257));  ending_mask_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    full_like_25 = torch.full_like(ending_input_12, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    bool_26 = ending_mask_25.bool();  ending_mask_25 = None
    where_25 = full_like_25.where(bool_26, ending_input_12);  full_like_25 = bool_26 = ending_input_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    attn_scores_12[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))] = where_25;  setitem_77 = attn_scores_12;  where_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:576, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
    ne_6 = l_attention_mask_ != 0
    remove_from_windowed_attention_mask_6 = ne_6[(slice(None, None, None), slice(None, None, None), None, None)];  ne_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
    type_as_12 = remove_from_windowed_attention_mask_6.type_as(query_vectors_20);  query_vectors_20 = None
    float_mask_6 = type_as_12.masked_fill(remove_from_windowed_attention_mask_6, -3.4028234663852886e+38);  type_as_12 = remove_from_windowed_attention_mask_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:584, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
    new_ones_19 = float_mask_6.new_ones(size = (1, 1024, 1, 1))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:830, code: chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
    div_51 = torch.div(1024, 256, rounding_mode = 'trunc')
    chunks_count_19 = div_51 - 1;  div_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:833, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_96 = new_ones_19.transpose(1, 2);  new_ones_19 = None
    query_26 = transpose_96.reshape(1, 1024, 1);  transpose_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_97 = float_mask_6.transpose(1, 2);  float_mask_6 = None
    key_26 = transpose_97.reshape(1, 1024, 1);  transpose_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_52 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_87 = query_26.view(1, div_52, 512, 1);  query_26 = div_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    query_27 = hidden_states_87.as_strided(size = [1, 3, 512, 1], stride = [1024, 256, 1, 1]);  hidden_states_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_53 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_88 = key_26.view(1, div_53, 512, 1);  key_26 = div_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    key_27 = hidden_states_88.as_strided(size = [1, 3, 512, 1], stride = [1024, 256, 1, 1]);  hidden_states_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    diagonal_chunked_attention_scores_26 = torch.functional.einsum('bcxd,bcyd->bcxy', (query_27, key_27));  query_27 = key_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    hidden_states_padded_26 = torch.nn.functional.pad(diagonal_chunked_attention_scores_26, (0, 0, 0, 1));  diagonal_chunked_attention_scores_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    diagonal_chunked_attention_scores_27 = hidden_states_padded_26.view(1, 3, 512, 513);  hidden_states_padded_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:856, code: (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
    add_31 = chunks_count_19 + 1;  chunks_count_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    diagonal_attention_scores_26 = diagonal_chunked_attention_scores_27.new_zeros((1, add_31, 256, 513));  add_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_116 = diagonal_chunked_attention_scores_27[(slice(None, None, None), slice(None, None, None), slice(None, 256, None), slice(None, 257, None))]
    diagonal_attention_scores_26[(slice(None, None, None), slice(None, -1, None), slice(None, None, None), slice(256, None, None))] = getitem_116;  setitem_78 = diagonal_attention_scores_26;  getitem_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_117 = diagonal_chunked_attention_scores_27[(slice(None, None, None), -1, slice(256, None, None), slice(None, 257, None))]
    diagonal_attention_scores_26[(slice(None, None, None), -1, slice(None, None, None), slice(256, None, None))] = getitem_117;  setitem_79 = diagonal_attention_scores_26;  getitem_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    getitem_118 = diagonal_chunked_attention_scores_27[(slice(None, None, None), slice(None, None, None), slice(-257, -1, None), slice(257, None, None))]
    diagonal_attention_scores_26[(slice(None, None, None), slice(1, None, None), slice(None, None, None), slice(None, 256, None))] = getitem_118;  setitem_80 = diagonal_attention_scores_26;  getitem_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    getitem_119 = diagonal_chunked_attention_scores_27[(slice(None, None, None), 0, slice(None, 255, None), slice(-255, None, None))];  diagonal_chunked_attention_scores_27 = None
    diagonal_attention_scores_26[(slice(None, None, None), 0, slice(1, 256, None), slice(1, 256, None))] = getitem_119;  setitem_81 = diagonal_attention_scores_26;  getitem_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    view_93 = diagonal_attention_scores_26.view(1, 1, 1024, 513);  diagonal_attention_scores_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:879, code: ).transpose(2, 1)
    diagonal_mask_6 = view_93.transpose(2, 1);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    new_ones_20 = diagonal_mask_6.new_ones(256, 257)
    tril_13 = new_ones_20.tril();  new_ones_20 = None
    beginning_mask_2d_13 = tril_13.flip(dims = [0]);  tril_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    beginning_mask_26 = beginning_mask_2d_13[(None, slice(None, None, None), None, slice(None, None, None))];  beginning_mask_2d_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    ending_mask_26 = beginning_mask_26.flip(dims = (1, 3))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    beginning_input_13 = diagonal_mask_6[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    beginning_mask_27 = beginning_mask_26.expand((1, 256, 1, 257));  beginning_mask_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    full_like_26 = torch.full_like(beginning_input_13, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    bool_27 = beginning_mask_27.bool();  beginning_mask_27 = None
    where_26 = full_like_26.where(bool_27, beginning_input_13);  full_like_26 = bool_27 = beginning_input_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    diagonal_mask_6[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))] = where_26;  setitem_82 = diagonal_mask_6;  where_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    ending_input_13 = diagonal_mask_6[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    ending_mask_27 = ending_mask_26.expand((1, 256, 1, 257));  ending_mask_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    full_like_27 = torch.full_like(ending_input_13, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    bool_28 = ending_mask_27.bool();  ending_mask_27 = None
    where_27 = full_like_27.where(bool_28, ending_input_13);  full_like_27 = bool_28 = ending_input_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    diagonal_mask_6[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))] = where_27;  setitem_83 = diagonal_mask_6;  where_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    attn_scores_12 += diagonal_mask_6;  attn_scores_13 = attn_scores_12;  attn_scores_12 = diagonal_mask_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    attn_probs_24 = torch.nn.functional.softmax(attn_scores_13, dim = -1, dtype = torch.float32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    getitem_123 = l_is_index_masked_[(slice(None, None, None), slice(None, None, None), None, None)]
    attn_probs_25 = torch.masked_fill(attn_probs_24, getitem_123, 0.0);  attn_probs_24 = getitem_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:638, code: attn_probs = attn_probs.type_as(attn_scores)
    attn_probs_26 = attn_probs_25.type_as(attn_scores_13);  attn_probs_25 = attn_scores_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    attn_probs_27 = torch.nn.functional.dropout(attn_probs_26, p = 0.1, training = True);  attn_probs_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_94 = value_vectors_12.view(1024, 1, 12, 64);  value_vectors_12 = None
    value_vectors_13 = view_94.transpose(0, 1);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:896, code: chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
    div_54 = torch.div(1024, 256, rounding_mode = 'trunc')
    chunks_count_20 = div_54 - 1;  div_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    transpose_100 = attn_probs_27.transpose(1, 2);  attn_probs_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:901, code: torch.div(seq_len, window_overlap, rounding_mode="trunc"),
    div_55 = torch.div(1024, 256, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    chunked_attn_probs_12 = transpose_100.reshape(12, div_55, 256, 513);  transpose_100 = div_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_101 = value_vectors_13.transpose(1, 2);  value_vectors_13 = None
    value_6 = transpose_101.reshape(12, 1024, 64);  transpose_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    padded_value_6 = torch.nn.functional.pad(value_6, (0, 0, 256, 256), value = -1);  value_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:913, code: chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
    add_32 = chunks_count_20 + 1;  chunks_count_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    chunked_value_6 = padded_value_6.as_strided(size = (12, add_32, 768, 64), stride = (98304, 16384, 64, 1));  padded_value_6 = add_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    chunked_hidden_states_30 = torch.nn.functional.pad(chunked_attn_probs_12, (0, 257));  chunked_attn_probs_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    chunked_hidden_states_31 = chunked_hidden_states_30.view(12, 4, -1);  chunked_hidden_states_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    chunked_hidden_states_32 = chunked_hidden_states_31[(slice(None, None, None), slice(None, None, None), slice(None, -256, None))];  chunked_hidden_states_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    chunked_hidden_states_33 = chunked_hidden_states_32.view(12, 4, 256, 769);  chunked_hidden_states_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    chunked_attn_probs_13 = chunked_hidden_states_33[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, -1, None))];  chunked_hidden_states_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    context_6 = torch.functional.einsum('bcwd,bcdh->bcwh', (chunked_attn_probs_13, chunked_value_6));  chunked_attn_probs_13 = chunked_value_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    view_97 = context_6.view(1, 12, 1024, 64);  context_6 = None
    attn_output_24 = view_97.transpose(1, 2);  view_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    transpose_103 = attn_output_24.transpose(0, 1);  attn_output_24 = None
    reshape_48 = transpose_103.reshape(1024, 1, 768);  transpose_103 = None
    attn_output_25 = reshape_48.contiguous();  reshape_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    transpose_104 = attn_output_25.transpose(0, 1);  attn_output_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    hidden_states_89 = self.L__self___layer_6_attention_output_dense(transpose_104);  transpose_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    hidden_states_90 = self.L__self___layer_6_attention_output_dropout(hidden_states_89);  hidden_states_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_33 = hidden_states_90 + hidden_states_83;  hidden_states_90 = hidden_states_83 = None
    attn_output_27 = self.L__self___layer_6_attention_output_LayerNorm(add_33);  add_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    hidden_states_92 = self.L__self___layer_6_intermediate_dense(attn_output_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_6 = torch._C._nn.gelu(hidden_states_92);  hidden_states_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    hidden_states_94 = self.L__self___layer_6_output_dense(intermediate_output_6);  intermediate_output_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    hidden_states_95 = self.L__self___layer_6_output_dropout(hidden_states_94);  hidden_states_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_34 = hidden_states_95 + attn_output_27;  hidden_states_95 = attn_output_27 = None
    hidden_states_97 = self.L__self___layer_6_output_LayerNorm(add_34);  add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    hidden_states_98 = hidden_states_97.transpose(0, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    query_vectors_21 = self.L__self___layer_7_attention_self_query(hidden_states_98)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    key_vectors_14 = self.L__self___layer_7_attention_self_key(hidden_states_98)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    value_vectors_14 = self.L__self___layer_7_attention_self_value(hidden_states_98);  hidden_states_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    query_vectors_21 /= 8.0;  query_vectors_22 = query_vectors_21;  query_vectors_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:568, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_98 = query_vectors_22.view(1024, 1, 12, 64);  query_vectors_22 = None
    query_vectors_23 = view_98.transpose(0, 1);  view_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:569, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_99 = key_vectors_14.view(1024, 1, 12, 64);  key_vectors_14 = None
    key_vectors_15 = view_99.transpose(0, 1);  view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:830, code: chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
    div_56 = torch.div(1024, 256, rounding_mode = 'trunc')
    chunks_count_21 = div_56 - 1;  div_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:833, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_108 = query_vectors_23.transpose(1, 2)
    query_28 = transpose_108.reshape(12, 1024, 64);  transpose_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_109 = key_vectors_15.transpose(1, 2);  key_vectors_15 = None
    key_28 = transpose_109.reshape(12, 1024, 64);  transpose_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_57 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_99 = query_28.view(12, div_57, 512, 64);  query_28 = div_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    query_29 = hidden_states_99.as_strided(size = [12, 3, 512, 64], stride = [64, 196608, 768, 1]);  hidden_states_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_58 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_100 = key_28.view(12, div_58, 512, 64);  key_28 = div_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    key_29 = hidden_states_100.as_strided(size = [12, 3, 512, 64], stride = [64, 196608, 768, 1]);  hidden_states_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    diagonal_chunked_attention_scores_28 = torch.functional.einsum('bcxd,bcyd->bcxy', (query_29, key_29));  query_29 = key_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    hidden_states_padded_28 = torch.nn.functional.pad(diagonal_chunked_attention_scores_28, (0, 0, 0, 1));  diagonal_chunked_attention_scores_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    diagonal_chunked_attention_scores_29 = hidden_states_padded_28.view(12, 3, 512, 513);  hidden_states_padded_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:856, code: (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
    add_35 = chunks_count_21 + 1;  chunks_count_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    diagonal_attention_scores_28 = diagonal_chunked_attention_scores_29.new_zeros((12, add_35, 256, 513));  add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_126 = diagonal_chunked_attention_scores_29[(slice(None, None, None), slice(None, None, None), slice(None, 256, None), slice(None, 257, None))]
    diagonal_attention_scores_28[(slice(None, None, None), slice(None, -1, None), slice(None, None, None), slice(256, None, None))] = getitem_126;  setitem_84 = diagonal_attention_scores_28;  getitem_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_127 = diagonal_chunked_attention_scores_29[(slice(None, None, None), -1, slice(256, None, None), slice(None, 257, None))]
    diagonal_attention_scores_28[(slice(None, None, None), -1, slice(None, None, None), slice(256, None, None))] = getitem_127;  setitem_85 = diagonal_attention_scores_28;  getitem_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    getitem_128 = diagonal_chunked_attention_scores_29[(slice(None, None, None), slice(None, None, None), slice(-257, -1, None), slice(257, None, None))]
    diagonal_attention_scores_28[(slice(None, None, None), slice(1, None, None), slice(None, None, None), slice(None, 256, None))] = getitem_128;  setitem_86 = diagonal_attention_scores_28;  getitem_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    getitem_129 = diagonal_chunked_attention_scores_29[(slice(None, None, None), 0, slice(None, 255, None), slice(-255, None, None))];  diagonal_chunked_attention_scores_29 = None
    diagonal_attention_scores_28[(slice(None, None, None), 0, slice(1, 256, None), slice(1, 256, None))] = getitem_129;  setitem_87 = diagonal_attention_scores_28;  getitem_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    view_103 = diagonal_attention_scores_28.view(1, 12, 1024, 513);  diagonal_attention_scores_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:879, code: ).transpose(2, 1)
    attn_scores_14 = view_103.transpose(2, 1);  view_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    new_ones_21 = attn_scores_14.new_ones(256, 257)
    tril_14 = new_ones_21.tril();  new_ones_21 = None
    beginning_mask_2d_14 = tril_14.flip(dims = [0]);  tril_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    beginning_mask_28 = beginning_mask_2d_14[(None, slice(None, None, None), None, slice(None, None, None))];  beginning_mask_2d_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    ending_mask_28 = beginning_mask_28.flip(dims = (1, 3))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    beginning_input_14 = attn_scores_14[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    beginning_mask_29 = beginning_mask_28.expand((1, 256, 12, 257));  beginning_mask_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    full_like_28 = torch.full_like(beginning_input_14, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    bool_29 = beginning_mask_29.bool();  beginning_mask_29 = None
    where_28 = full_like_28.where(bool_29, beginning_input_14);  full_like_28 = bool_29 = beginning_input_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    attn_scores_14[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))] = where_28;  setitem_88 = attn_scores_14;  where_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    ending_input_14 = attn_scores_14[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    ending_mask_29 = ending_mask_28.expand((1, 256, 12, 257));  ending_mask_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    full_like_29 = torch.full_like(ending_input_14, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    bool_30 = ending_mask_29.bool();  ending_mask_29 = None
    where_29 = full_like_29.where(bool_30, ending_input_14);  full_like_29 = bool_30 = ending_input_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    attn_scores_14[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))] = where_29;  setitem_89 = attn_scores_14;  where_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:576, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
    ne_7 = l_attention_mask_ != 0
    remove_from_windowed_attention_mask_7 = ne_7[(slice(None, None, None), slice(None, None, None), None, None)];  ne_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
    type_as_14 = remove_from_windowed_attention_mask_7.type_as(query_vectors_23);  query_vectors_23 = None
    float_mask_7 = type_as_14.masked_fill(remove_from_windowed_attention_mask_7, -3.4028234663852886e+38);  type_as_14 = remove_from_windowed_attention_mask_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:584, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
    new_ones_22 = float_mask_7.new_ones(size = (1, 1024, 1, 1))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:830, code: chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
    div_59 = torch.div(1024, 256, rounding_mode = 'trunc')
    chunks_count_22 = div_59 - 1;  div_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:833, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_111 = new_ones_22.transpose(1, 2);  new_ones_22 = None
    query_30 = transpose_111.reshape(1, 1024, 1);  transpose_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_112 = float_mask_7.transpose(1, 2);  float_mask_7 = None
    key_30 = transpose_112.reshape(1, 1024, 1);  transpose_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_60 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_101 = query_30.view(1, div_60, 512, 1);  query_30 = div_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    query_31 = hidden_states_101.as_strided(size = [1, 3, 512, 1], stride = [1024, 256, 1, 1]);  hidden_states_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_61 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_102 = key_30.view(1, div_61, 512, 1);  key_30 = div_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    key_31 = hidden_states_102.as_strided(size = [1, 3, 512, 1], stride = [1024, 256, 1, 1]);  hidden_states_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    diagonal_chunked_attention_scores_30 = torch.functional.einsum('bcxd,bcyd->bcxy', (query_31, key_31));  query_31 = key_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    hidden_states_padded_30 = torch.nn.functional.pad(diagonal_chunked_attention_scores_30, (0, 0, 0, 1));  diagonal_chunked_attention_scores_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    diagonal_chunked_attention_scores_31 = hidden_states_padded_30.view(1, 3, 512, 513);  hidden_states_padded_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:856, code: (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
    add_36 = chunks_count_22 + 1;  chunks_count_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    diagonal_attention_scores_30 = diagonal_chunked_attention_scores_31.new_zeros((1, add_36, 256, 513));  add_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_134 = diagonal_chunked_attention_scores_31[(slice(None, None, None), slice(None, None, None), slice(None, 256, None), slice(None, 257, None))]
    diagonal_attention_scores_30[(slice(None, None, None), slice(None, -1, None), slice(None, None, None), slice(256, None, None))] = getitem_134;  setitem_90 = diagonal_attention_scores_30;  getitem_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_135 = diagonal_chunked_attention_scores_31[(slice(None, None, None), -1, slice(256, None, None), slice(None, 257, None))]
    diagonal_attention_scores_30[(slice(None, None, None), -1, slice(None, None, None), slice(256, None, None))] = getitem_135;  setitem_91 = diagonal_attention_scores_30;  getitem_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    getitem_136 = diagonal_chunked_attention_scores_31[(slice(None, None, None), slice(None, None, None), slice(-257, -1, None), slice(257, None, None))]
    diagonal_attention_scores_30[(slice(None, None, None), slice(1, None, None), slice(None, None, None), slice(None, 256, None))] = getitem_136;  setitem_92 = diagonal_attention_scores_30;  getitem_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    getitem_137 = diagonal_chunked_attention_scores_31[(slice(None, None, None), 0, slice(None, 255, None), slice(-255, None, None))];  diagonal_chunked_attention_scores_31 = None
    diagonal_attention_scores_30[(slice(None, None, None), 0, slice(1, 256, None), slice(1, 256, None))] = getitem_137;  setitem_93 = diagonal_attention_scores_30;  getitem_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    view_107 = diagonal_attention_scores_30.view(1, 1, 1024, 513);  diagonal_attention_scores_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:879, code: ).transpose(2, 1)
    diagonal_mask_7 = view_107.transpose(2, 1);  view_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    new_ones_23 = diagonal_mask_7.new_ones(256, 257)
    tril_15 = new_ones_23.tril();  new_ones_23 = None
    beginning_mask_2d_15 = tril_15.flip(dims = [0]);  tril_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    beginning_mask_30 = beginning_mask_2d_15[(None, slice(None, None, None), None, slice(None, None, None))];  beginning_mask_2d_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    ending_mask_30 = beginning_mask_30.flip(dims = (1, 3))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    beginning_input_15 = diagonal_mask_7[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    beginning_mask_31 = beginning_mask_30.expand((1, 256, 1, 257));  beginning_mask_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    full_like_30 = torch.full_like(beginning_input_15, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    bool_31 = beginning_mask_31.bool();  beginning_mask_31 = None
    where_30 = full_like_30.where(bool_31, beginning_input_15);  full_like_30 = bool_31 = beginning_input_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    diagonal_mask_7[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))] = where_30;  setitem_94 = diagonal_mask_7;  where_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    ending_input_15 = diagonal_mask_7[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    ending_mask_31 = ending_mask_30.expand((1, 256, 1, 257));  ending_mask_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    full_like_31 = torch.full_like(ending_input_15, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    bool_32 = ending_mask_31.bool();  ending_mask_31 = None
    where_31 = full_like_31.where(bool_32, ending_input_15);  full_like_31 = bool_32 = ending_input_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    diagonal_mask_7[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))] = where_31;  setitem_95 = diagonal_mask_7;  where_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    attn_scores_14 += diagonal_mask_7;  attn_scores_15 = attn_scores_14;  attn_scores_14 = diagonal_mask_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    attn_probs_28 = torch.nn.functional.softmax(attn_scores_15, dim = -1, dtype = torch.float32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    getitem_141 = l_is_index_masked_[(slice(None, None, None), slice(None, None, None), None, None)]
    attn_probs_29 = torch.masked_fill(attn_probs_28, getitem_141, 0.0);  attn_probs_28 = getitem_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:638, code: attn_probs = attn_probs.type_as(attn_scores)
    attn_probs_30 = attn_probs_29.type_as(attn_scores_15);  attn_probs_29 = attn_scores_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    attn_probs_31 = torch.nn.functional.dropout(attn_probs_30, p = 0.1, training = True);  attn_probs_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_108 = value_vectors_14.view(1024, 1, 12, 64);  value_vectors_14 = None
    value_vectors_15 = view_108.transpose(0, 1);  view_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:896, code: chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
    div_62 = torch.div(1024, 256, rounding_mode = 'trunc')
    chunks_count_23 = div_62 - 1;  div_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    transpose_115 = attn_probs_31.transpose(1, 2);  attn_probs_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:901, code: torch.div(seq_len, window_overlap, rounding_mode="trunc"),
    div_63 = torch.div(1024, 256, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    chunked_attn_probs_14 = transpose_115.reshape(12, div_63, 256, 513);  transpose_115 = div_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_116 = value_vectors_15.transpose(1, 2);  value_vectors_15 = None
    value_7 = transpose_116.reshape(12, 1024, 64);  transpose_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    padded_value_7 = torch.nn.functional.pad(value_7, (0, 0, 256, 256), value = -1);  value_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:913, code: chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
    add_37 = chunks_count_23 + 1;  chunks_count_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    chunked_value_7 = padded_value_7.as_strided(size = (12, add_37, 768, 64), stride = (98304, 16384, 64, 1));  padded_value_7 = add_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    chunked_hidden_states_35 = torch.nn.functional.pad(chunked_attn_probs_14, (0, 257));  chunked_attn_probs_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    chunked_hidden_states_36 = chunked_hidden_states_35.view(12, 4, -1);  chunked_hidden_states_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    chunked_hidden_states_37 = chunked_hidden_states_36[(slice(None, None, None), slice(None, None, None), slice(None, -256, None))];  chunked_hidden_states_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    chunked_hidden_states_38 = chunked_hidden_states_37.view(12, 4, 256, 769);  chunked_hidden_states_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    chunked_attn_probs_15 = chunked_hidden_states_38[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, -1, None))];  chunked_hidden_states_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    context_7 = torch.functional.einsum('bcwd,bcdh->bcwh', (chunked_attn_probs_15, chunked_value_7));  chunked_attn_probs_15 = chunked_value_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    view_111 = context_7.view(1, 12, 1024, 64);  context_7 = None
    attn_output_28 = view_111.transpose(1, 2);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    transpose_118 = attn_output_28.transpose(0, 1);  attn_output_28 = None
    reshape_55 = transpose_118.reshape(1024, 1, 768);  transpose_118 = None
    attn_output_29 = reshape_55.contiguous();  reshape_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    transpose_119 = attn_output_29.transpose(0, 1);  attn_output_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    hidden_states_103 = self.L__self___layer_7_attention_output_dense(transpose_119);  transpose_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    hidden_states_104 = self.L__self___layer_7_attention_output_dropout(hidden_states_103);  hidden_states_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_38 = hidden_states_104 + hidden_states_97;  hidden_states_104 = hidden_states_97 = None
    attn_output_31 = self.L__self___layer_7_attention_output_LayerNorm(add_38);  add_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    hidden_states_106 = self.L__self___layer_7_intermediate_dense(attn_output_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_7 = torch._C._nn.gelu(hidden_states_106);  hidden_states_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    hidden_states_108 = self.L__self___layer_7_output_dense(intermediate_output_7);  intermediate_output_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    hidden_states_109 = self.L__self___layer_7_output_dropout(hidden_states_108);  hidden_states_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_39 = hidden_states_109 + attn_output_31;  hidden_states_109 = attn_output_31 = None
    hidden_states_111 = self.L__self___layer_7_output_LayerNorm(add_39);  add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    hidden_states_112 = hidden_states_111.transpose(0, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    query_vectors_24 = self.L__self___layer_8_attention_self_query(hidden_states_112)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    key_vectors_16 = self.L__self___layer_8_attention_self_key(hidden_states_112)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    value_vectors_16 = self.L__self___layer_8_attention_self_value(hidden_states_112);  hidden_states_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    query_vectors_24 /= 8.0;  query_vectors_25 = query_vectors_24;  query_vectors_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:568, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_112 = query_vectors_25.view(1024, 1, 12, 64);  query_vectors_25 = None
    query_vectors_26 = view_112.transpose(0, 1);  view_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:569, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_113 = key_vectors_16.view(1024, 1, 12, 64);  key_vectors_16 = None
    key_vectors_17 = view_113.transpose(0, 1);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:830, code: chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
    div_64 = torch.div(1024, 256, rounding_mode = 'trunc')
    chunks_count_24 = div_64 - 1;  div_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:833, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_123 = query_vectors_26.transpose(1, 2)
    query_32 = transpose_123.reshape(12, 1024, 64);  transpose_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_124 = key_vectors_17.transpose(1, 2);  key_vectors_17 = None
    key_32 = transpose_124.reshape(12, 1024, 64);  transpose_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_65 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_113 = query_32.view(12, div_65, 512, 64);  query_32 = div_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    query_33 = hidden_states_113.as_strided(size = [12, 3, 512, 64], stride = [64, 196608, 768, 1]);  hidden_states_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_66 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_114 = key_32.view(12, div_66, 512, 64);  key_32 = div_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    key_33 = hidden_states_114.as_strided(size = [12, 3, 512, 64], stride = [64, 196608, 768, 1]);  hidden_states_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    diagonal_chunked_attention_scores_32 = torch.functional.einsum('bcxd,bcyd->bcxy', (query_33, key_33));  query_33 = key_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    hidden_states_padded_32 = torch.nn.functional.pad(diagonal_chunked_attention_scores_32, (0, 0, 0, 1));  diagonal_chunked_attention_scores_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    diagonal_chunked_attention_scores_33 = hidden_states_padded_32.view(12, 3, 512, 513);  hidden_states_padded_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:856, code: (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
    add_40 = chunks_count_24 + 1;  chunks_count_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    diagonal_attention_scores_32 = diagonal_chunked_attention_scores_33.new_zeros((12, add_40, 256, 513));  add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_144 = diagonal_chunked_attention_scores_33[(slice(None, None, None), slice(None, None, None), slice(None, 256, None), slice(None, 257, None))]
    diagonal_attention_scores_32[(slice(None, None, None), slice(None, -1, None), slice(None, None, None), slice(256, None, None))] = getitem_144;  setitem_96 = diagonal_attention_scores_32;  getitem_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_145 = diagonal_chunked_attention_scores_33[(slice(None, None, None), -1, slice(256, None, None), slice(None, 257, None))]
    diagonal_attention_scores_32[(slice(None, None, None), -1, slice(None, None, None), slice(256, None, None))] = getitem_145;  setitem_97 = diagonal_attention_scores_32;  getitem_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    getitem_146 = diagonal_chunked_attention_scores_33[(slice(None, None, None), slice(None, None, None), slice(-257, -1, None), slice(257, None, None))]
    diagonal_attention_scores_32[(slice(None, None, None), slice(1, None, None), slice(None, None, None), slice(None, 256, None))] = getitem_146;  setitem_98 = diagonal_attention_scores_32;  getitem_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    getitem_147 = diagonal_chunked_attention_scores_33[(slice(None, None, None), 0, slice(None, 255, None), slice(-255, None, None))];  diagonal_chunked_attention_scores_33 = None
    diagonal_attention_scores_32[(slice(None, None, None), 0, slice(1, 256, None), slice(1, 256, None))] = getitem_147;  setitem_99 = diagonal_attention_scores_32;  getitem_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    view_117 = diagonal_attention_scores_32.view(1, 12, 1024, 513);  diagonal_attention_scores_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:879, code: ).transpose(2, 1)
    attn_scores_16 = view_117.transpose(2, 1);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    new_ones_24 = attn_scores_16.new_ones(256, 257)
    tril_16 = new_ones_24.tril();  new_ones_24 = None
    beginning_mask_2d_16 = tril_16.flip(dims = [0]);  tril_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    beginning_mask_32 = beginning_mask_2d_16[(None, slice(None, None, None), None, slice(None, None, None))];  beginning_mask_2d_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    ending_mask_32 = beginning_mask_32.flip(dims = (1, 3))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    beginning_input_16 = attn_scores_16[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    beginning_mask_33 = beginning_mask_32.expand((1, 256, 12, 257));  beginning_mask_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    full_like_32 = torch.full_like(beginning_input_16, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    bool_33 = beginning_mask_33.bool();  beginning_mask_33 = None
    where_32 = full_like_32.where(bool_33, beginning_input_16);  full_like_32 = bool_33 = beginning_input_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    attn_scores_16[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))] = where_32;  setitem_100 = attn_scores_16;  where_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    ending_input_16 = attn_scores_16[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    ending_mask_33 = ending_mask_32.expand((1, 256, 12, 257));  ending_mask_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    full_like_33 = torch.full_like(ending_input_16, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    bool_34 = ending_mask_33.bool();  ending_mask_33 = None
    where_33 = full_like_33.where(bool_34, ending_input_16);  full_like_33 = bool_34 = ending_input_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    attn_scores_16[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))] = where_33;  setitem_101 = attn_scores_16;  where_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:576, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
    ne_8 = l_attention_mask_ != 0
    remove_from_windowed_attention_mask_8 = ne_8[(slice(None, None, None), slice(None, None, None), None, None)];  ne_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
    type_as_16 = remove_from_windowed_attention_mask_8.type_as(query_vectors_26);  query_vectors_26 = None
    float_mask_8 = type_as_16.masked_fill(remove_from_windowed_attention_mask_8, -3.4028234663852886e+38);  type_as_16 = remove_from_windowed_attention_mask_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:584, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
    new_ones_25 = float_mask_8.new_ones(size = (1, 1024, 1, 1))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:830, code: chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
    div_67 = torch.div(1024, 256, rounding_mode = 'trunc')
    chunks_count_25 = div_67 - 1;  div_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:833, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_126 = new_ones_25.transpose(1, 2);  new_ones_25 = None
    query_34 = transpose_126.reshape(1, 1024, 1);  transpose_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_127 = float_mask_8.transpose(1, 2);  float_mask_8 = None
    key_34 = transpose_127.reshape(1, 1024, 1);  transpose_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_68 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_115 = query_34.view(1, div_68, 512, 1);  query_34 = div_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    query_35 = hidden_states_115.as_strided(size = [1, 3, 512, 1], stride = [1024, 256, 1, 1]);  hidden_states_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_69 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_116 = key_34.view(1, div_69, 512, 1);  key_34 = div_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    key_35 = hidden_states_116.as_strided(size = [1, 3, 512, 1], stride = [1024, 256, 1, 1]);  hidden_states_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    diagonal_chunked_attention_scores_34 = torch.functional.einsum('bcxd,bcyd->bcxy', (query_35, key_35));  query_35 = key_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    hidden_states_padded_34 = torch.nn.functional.pad(diagonal_chunked_attention_scores_34, (0, 0, 0, 1));  diagonal_chunked_attention_scores_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    diagonal_chunked_attention_scores_35 = hidden_states_padded_34.view(1, 3, 512, 513);  hidden_states_padded_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:856, code: (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
    add_41 = chunks_count_25 + 1;  chunks_count_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    diagonal_attention_scores_34 = diagonal_chunked_attention_scores_35.new_zeros((1, add_41, 256, 513));  add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_152 = diagonal_chunked_attention_scores_35[(slice(None, None, None), slice(None, None, None), slice(None, 256, None), slice(None, 257, None))]
    diagonal_attention_scores_34[(slice(None, None, None), slice(None, -1, None), slice(None, None, None), slice(256, None, None))] = getitem_152;  setitem_102 = diagonal_attention_scores_34;  getitem_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_153 = diagonal_chunked_attention_scores_35[(slice(None, None, None), -1, slice(256, None, None), slice(None, 257, None))]
    diagonal_attention_scores_34[(slice(None, None, None), -1, slice(None, None, None), slice(256, None, None))] = getitem_153;  setitem_103 = diagonal_attention_scores_34;  getitem_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    getitem_154 = diagonal_chunked_attention_scores_35[(slice(None, None, None), slice(None, None, None), slice(-257, -1, None), slice(257, None, None))]
    diagonal_attention_scores_34[(slice(None, None, None), slice(1, None, None), slice(None, None, None), slice(None, 256, None))] = getitem_154;  setitem_104 = diagonal_attention_scores_34;  getitem_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    getitem_155 = diagonal_chunked_attention_scores_35[(slice(None, None, None), 0, slice(None, 255, None), slice(-255, None, None))];  diagonal_chunked_attention_scores_35 = None
    diagonal_attention_scores_34[(slice(None, None, None), 0, slice(1, 256, None), slice(1, 256, None))] = getitem_155;  setitem_105 = diagonal_attention_scores_34;  getitem_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    view_121 = diagonal_attention_scores_34.view(1, 1, 1024, 513);  diagonal_attention_scores_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:879, code: ).transpose(2, 1)
    diagonal_mask_8 = view_121.transpose(2, 1);  view_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    new_ones_26 = diagonal_mask_8.new_ones(256, 257)
    tril_17 = new_ones_26.tril();  new_ones_26 = None
    beginning_mask_2d_17 = tril_17.flip(dims = [0]);  tril_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    beginning_mask_34 = beginning_mask_2d_17[(None, slice(None, None, None), None, slice(None, None, None))];  beginning_mask_2d_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    ending_mask_34 = beginning_mask_34.flip(dims = (1, 3))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    beginning_input_17 = diagonal_mask_8[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    beginning_mask_35 = beginning_mask_34.expand((1, 256, 1, 257));  beginning_mask_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    full_like_34 = torch.full_like(beginning_input_17, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    bool_35 = beginning_mask_35.bool();  beginning_mask_35 = None
    where_34 = full_like_34.where(bool_35, beginning_input_17);  full_like_34 = bool_35 = beginning_input_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    diagonal_mask_8[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))] = where_34;  setitem_106 = diagonal_mask_8;  where_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    ending_input_17 = diagonal_mask_8[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    ending_mask_35 = ending_mask_34.expand((1, 256, 1, 257));  ending_mask_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    full_like_35 = torch.full_like(ending_input_17, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    bool_36 = ending_mask_35.bool();  ending_mask_35 = None
    where_35 = full_like_35.where(bool_36, ending_input_17);  full_like_35 = bool_36 = ending_input_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    diagonal_mask_8[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))] = where_35;  setitem_107 = diagonal_mask_8;  where_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    attn_scores_16 += diagonal_mask_8;  attn_scores_17 = attn_scores_16;  attn_scores_16 = diagonal_mask_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    attn_probs_32 = torch.nn.functional.softmax(attn_scores_17, dim = -1, dtype = torch.float32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    getitem_159 = l_is_index_masked_[(slice(None, None, None), slice(None, None, None), None, None)]
    attn_probs_33 = torch.masked_fill(attn_probs_32, getitem_159, 0.0);  attn_probs_32 = getitem_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:638, code: attn_probs = attn_probs.type_as(attn_scores)
    attn_probs_34 = attn_probs_33.type_as(attn_scores_17);  attn_probs_33 = attn_scores_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    attn_probs_35 = torch.nn.functional.dropout(attn_probs_34, p = 0.1, training = True);  attn_probs_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_122 = value_vectors_16.view(1024, 1, 12, 64);  value_vectors_16 = None
    value_vectors_17 = view_122.transpose(0, 1);  view_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:896, code: chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
    div_70 = torch.div(1024, 256, rounding_mode = 'trunc')
    chunks_count_26 = div_70 - 1;  div_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    transpose_130 = attn_probs_35.transpose(1, 2);  attn_probs_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:901, code: torch.div(seq_len, window_overlap, rounding_mode="trunc"),
    div_71 = torch.div(1024, 256, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    chunked_attn_probs_16 = transpose_130.reshape(12, div_71, 256, 513);  transpose_130 = div_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_131 = value_vectors_17.transpose(1, 2);  value_vectors_17 = None
    value_8 = transpose_131.reshape(12, 1024, 64);  transpose_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    padded_value_8 = torch.nn.functional.pad(value_8, (0, 0, 256, 256), value = -1);  value_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:913, code: chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
    add_42 = chunks_count_26 + 1;  chunks_count_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    chunked_value_8 = padded_value_8.as_strided(size = (12, add_42, 768, 64), stride = (98304, 16384, 64, 1));  padded_value_8 = add_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    chunked_hidden_states_40 = torch.nn.functional.pad(chunked_attn_probs_16, (0, 257));  chunked_attn_probs_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    chunked_hidden_states_41 = chunked_hidden_states_40.view(12, 4, -1);  chunked_hidden_states_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    chunked_hidden_states_42 = chunked_hidden_states_41[(slice(None, None, None), slice(None, None, None), slice(None, -256, None))];  chunked_hidden_states_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    chunked_hidden_states_43 = chunked_hidden_states_42.view(12, 4, 256, 769);  chunked_hidden_states_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    chunked_attn_probs_17 = chunked_hidden_states_43[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, -1, None))];  chunked_hidden_states_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    context_8 = torch.functional.einsum('bcwd,bcdh->bcwh', (chunked_attn_probs_17, chunked_value_8));  chunked_attn_probs_17 = chunked_value_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    view_125 = context_8.view(1, 12, 1024, 64);  context_8 = None
    attn_output_32 = view_125.transpose(1, 2);  view_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    transpose_133 = attn_output_32.transpose(0, 1);  attn_output_32 = None
    reshape_62 = transpose_133.reshape(1024, 1, 768);  transpose_133 = None
    attn_output_33 = reshape_62.contiguous();  reshape_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    transpose_134 = attn_output_33.transpose(0, 1);  attn_output_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    hidden_states_117 = self.L__self___layer_8_attention_output_dense(transpose_134);  transpose_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    hidden_states_118 = self.L__self___layer_8_attention_output_dropout(hidden_states_117);  hidden_states_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_43 = hidden_states_118 + hidden_states_111;  hidden_states_118 = hidden_states_111 = None
    attn_output_35 = self.L__self___layer_8_attention_output_LayerNorm(add_43);  add_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    hidden_states_120 = self.L__self___layer_8_intermediate_dense(attn_output_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_8 = torch._C._nn.gelu(hidden_states_120);  hidden_states_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    hidden_states_122 = self.L__self___layer_8_output_dense(intermediate_output_8);  intermediate_output_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    hidden_states_123 = self.L__self___layer_8_output_dropout(hidden_states_122);  hidden_states_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_44 = hidden_states_123 + attn_output_35;  hidden_states_123 = attn_output_35 = None
    hidden_states_125 = self.L__self___layer_8_output_LayerNorm(add_44);  add_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    hidden_states_126 = hidden_states_125.transpose(0, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    query_vectors_27 = self.L__self___layer_9_attention_self_query(hidden_states_126)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    key_vectors_18 = self.L__self___layer_9_attention_self_key(hidden_states_126)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    value_vectors_18 = self.L__self___layer_9_attention_self_value(hidden_states_126);  hidden_states_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    query_vectors_27 /= 8.0;  query_vectors_28 = query_vectors_27;  query_vectors_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:568, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_126 = query_vectors_28.view(1024, 1, 12, 64);  query_vectors_28 = None
    query_vectors_29 = view_126.transpose(0, 1);  view_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:569, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_127 = key_vectors_18.view(1024, 1, 12, 64);  key_vectors_18 = None
    key_vectors_19 = view_127.transpose(0, 1);  view_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:830, code: chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
    div_72 = torch.div(1024, 256, rounding_mode = 'trunc')
    chunks_count_27 = div_72 - 1;  div_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:833, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_138 = query_vectors_29.transpose(1, 2)
    query_36 = transpose_138.reshape(12, 1024, 64);  transpose_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_139 = key_vectors_19.transpose(1, 2);  key_vectors_19 = None
    key_36 = transpose_139.reshape(12, 1024, 64);  transpose_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_73 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_127 = query_36.view(12, div_73, 512, 64);  query_36 = div_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    query_37 = hidden_states_127.as_strided(size = [12, 3, 512, 64], stride = [64, 196608, 768, 1]);  hidden_states_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_74 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_128 = key_36.view(12, div_74, 512, 64);  key_36 = div_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    key_37 = hidden_states_128.as_strided(size = [12, 3, 512, 64], stride = [64, 196608, 768, 1]);  hidden_states_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    diagonal_chunked_attention_scores_36 = torch.functional.einsum('bcxd,bcyd->bcxy', (query_37, key_37));  query_37 = key_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    hidden_states_padded_36 = torch.nn.functional.pad(diagonal_chunked_attention_scores_36, (0, 0, 0, 1));  diagonal_chunked_attention_scores_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    diagonal_chunked_attention_scores_37 = hidden_states_padded_36.view(12, 3, 512, 513);  hidden_states_padded_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:856, code: (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
    add_45 = chunks_count_27 + 1;  chunks_count_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    diagonal_attention_scores_36 = diagonal_chunked_attention_scores_37.new_zeros((12, add_45, 256, 513));  add_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_162 = diagonal_chunked_attention_scores_37[(slice(None, None, None), slice(None, None, None), slice(None, 256, None), slice(None, 257, None))]
    diagonal_attention_scores_36[(slice(None, None, None), slice(None, -1, None), slice(None, None, None), slice(256, None, None))] = getitem_162;  setitem_108 = diagonal_attention_scores_36;  getitem_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_163 = diagonal_chunked_attention_scores_37[(slice(None, None, None), -1, slice(256, None, None), slice(None, 257, None))]
    diagonal_attention_scores_36[(slice(None, None, None), -1, slice(None, None, None), slice(256, None, None))] = getitem_163;  setitem_109 = diagonal_attention_scores_36;  getitem_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    getitem_164 = diagonal_chunked_attention_scores_37[(slice(None, None, None), slice(None, None, None), slice(-257, -1, None), slice(257, None, None))]
    diagonal_attention_scores_36[(slice(None, None, None), slice(1, None, None), slice(None, None, None), slice(None, 256, None))] = getitem_164;  setitem_110 = diagonal_attention_scores_36;  getitem_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    getitem_165 = diagonal_chunked_attention_scores_37[(slice(None, None, None), 0, slice(None, 255, None), slice(-255, None, None))];  diagonal_chunked_attention_scores_37 = None
    diagonal_attention_scores_36[(slice(None, None, None), 0, slice(1, 256, None), slice(1, 256, None))] = getitem_165;  setitem_111 = diagonal_attention_scores_36;  getitem_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    view_131 = diagonal_attention_scores_36.view(1, 12, 1024, 513);  diagonal_attention_scores_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:879, code: ).transpose(2, 1)
    attn_scores_18 = view_131.transpose(2, 1);  view_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    new_ones_27 = attn_scores_18.new_ones(256, 257)
    tril_18 = new_ones_27.tril();  new_ones_27 = None
    beginning_mask_2d_18 = tril_18.flip(dims = [0]);  tril_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    beginning_mask_36 = beginning_mask_2d_18[(None, slice(None, None, None), None, slice(None, None, None))];  beginning_mask_2d_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    ending_mask_36 = beginning_mask_36.flip(dims = (1, 3))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    beginning_input_18 = attn_scores_18[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    beginning_mask_37 = beginning_mask_36.expand((1, 256, 12, 257));  beginning_mask_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    full_like_36 = torch.full_like(beginning_input_18, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    bool_37 = beginning_mask_37.bool();  beginning_mask_37 = None
    where_36 = full_like_36.where(bool_37, beginning_input_18);  full_like_36 = bool_37 = beginning_input_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    attn_scores_18[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))] = where_36;  setitem_112 = attn_scores_18;  where_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    ending_input_18 = attn_scores_18[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    ending_mask_37 = ending_mask_36.expand((1, 256, 12, 257));  ending_mask_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    full_like_37 = torch.full_like(ending_input_18, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    bool_38 = ending_mask_37.bool();  ending_mask_37 = None
    where_37 = full_like_37.where(bool_38, ending_input_18);  full_like_37 = bool_38 = ending_input_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    attn_scores_18[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))] = where_37;  setitem_113 = attn_scores_18;  where_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:576, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
    ne_9 = l_attention_mask_ != 0
    remove_from_windowed_attention_mask_9 = ne_9[(slice(None, None, None), slice(None, None, None), None, None)];  ne_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
    type_as_18 = remove_from_windowed_attention_mask_9.type_as(query_vectors_29);  query_vectors_29 = None
    float_mask_9 = type_as_18.masked_fill(remove_from_windowed_attention_mask_9, -3.4028234663852886e+38);  type_as_18 = remove_from_windowed_attention_mask_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:584, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
    new_ones_28 = float_mask_9.new_ones(size = (1, 1024, 1, 1))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:830, code: chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
    div_75 = torch.div(1024, 256, rounding_mode = 'trunc')
    chunks_count_28 = div_75 - 1;  div_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:833, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_141 = new_ones_28.transpose(1, 2);  new_ones_28 = None
    query_38 = transpose_141.reshape(1, 1024, 1);  transpose_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_142 = float_mask_9.transpose(1, 2);  float_mask_9 = None
    key_38 = transpose_142.reshape(1, 1024, 1);  transpose_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_76 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_129 = query_38.view(1, div_76, 512, 1);  query_38 = div_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    query_39 = hidden_states_129.as_strided(size = [1, 3, 512, 1], stride = [1024, 256, 1, 1]);  hidden_states_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_77 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_130 = key_38.view(1, div_77, 512, 1);  key_38 = div_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    key_39 = hidden_states_130.as_strided(size = [1, 3, 512, 1], stride = [1024, 256, 1, 1]);  hidden_states_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    diagonal_chunked_attention_scores_38 = torch.functional.einsum('bcxd,bcyd->bcxy', (query_39, key_39));  query_39 = key_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    hidden_states_padded_38 = torch.nn.functional.pad(diagonal_chunked_attention_scores_38, (0, 0, 0, 1));  diagonal_chunked_attention_scores_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    diagonal_chunked_attention_scores_39 = hidden_states_padded_38.view(1, 3, 512, 513);  hidden_states_padded_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:856, code: (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
    add_46 = chunks_count_28 + 1;  chunks_count_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    diagonal_attention_scores_38 = diagonal_chunked_attention_scores_39.new_zeros((1, add_46, 256, 513));  add_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_170 = diagonal_chunked_attention_scores_39[(slice(None, None, None), slice(None, None, None), slice(None, 256, None), slice(None, 257, None))]
    diagonal_attention_scores_38[(slice(None, None, None), slice(None, -1, None), slice(None, None, None), slice(256, None, None))] = getitem_170;  setitem_114 = diagonal_attention_scores_38;  getitem_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_171 = diagonal_chunked_attention_scores_39[(slice(None, None, None), -1, slice(256, None, None), slice(None, 257, None))]
    diagonal_attention_scores_38[(slice(None, None, None), -1, slice(None, None, None), slice(256, None, None))] = getitem_171;  setitem_115 = diagonal_attention_scores_38;  getitem_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    getitem_172 = diagonal_chunked_attention_scores_39[(slice(None, None, None), slice(None, None, None), slice(-257, -1, None), slice(257, None, None))]
    diagonal_attention_scores_38[(slice(None, None, None), slice(1, None, None), slice(None, None, None), slice(None, 256, None))] = getitem_172;  setitem_116 = diagonal_attention_scores_38;  getitem_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    getitem_173 = diagonal_chunked_attention_scores_39[(slice(None, None, None), 0, slice(None, 255, None), slice(-255, None, None))];  diagonal_chunked_attention_scores_39 = None
    diagonal_attention_scores_38[(slice(None, None, None), 0, slice(1, 256, None), slice(1, 256, None))] = getitem_173;  setitem_117 = diagonal_attention_scores_38;  getitem_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    view_135 = diagonal_attention_scores_38.view(1, 1, 1024, 513);  diagonal_attention_scores_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:879, code: ).transpose(2, 1)
    diagonal_mask_9 = view_135.transpose(2, 1);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    new_ones_29 = diagonal_mask_9.new_ones(256, 257)
    tril_19 = new_ones_29.tril();  new_ones_29 = None
    beginning_mask_2d_19 = tril_19.flip(dims = [0]);  tril_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    beginning_mask_38 = beginning_mask_2d_19[(None, slice(None, None, None), None, slice(None, None, None))];  beginning_mask_2d_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    ending_mask_38 = beginning_mask_38.flip(dims = (1, 3))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    beginning_input_19 = diagonal_mask_9[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    beginning_mask_39 = beginning_mask_38.expand((1, 256, 1, 257));  beginning_mask_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    full_like_38 = torch.full_like(beginning_input_19, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    bool_39 = beginning_mask_39.bool();  beginning_mask_39 = None
    where_38 = full_like_38.where(bool_39, beginning_input_19);  full_like_38 = bool_39 = beginning_input_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    diagonal_mask_9[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))] = where_38;  setitem_118 = diagonal_mask_9;  where_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    ending_input_19 = diagonal_mask_9[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    ending_mask_39 = ending_mask_38.expand((1, 256, 1, 257));  ending_mask_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    full_like_39 = torch.full_like(ending_input_19, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    bool_40 = ending_mask_39.bool();  ending_mask_39 = None
    where_39 = full_like_39.where(bool_40, ending_input_19);  full_like_39 = bool_40 = ending_input_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    diagonal_mask_9[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))] = where_39;  setitem_119 = diagonal_mask_9;  where_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    attn_scores_18 += diagonal_mask_9;  attn_scores_19 = attn_scores_18;  attn_scores_18 = diagonal_mask_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    attn_probs_36 = torch.nn.functional.softmax(attn_scores_19, dim = -1, dtype = torch.float32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    getitem_177 = l_is_index_masked_[(slice(None, None, None), slice(None, None, None), None, None)]
    attn_probs_37 = torch.masked_fill(attn_probs_36, getitem_177, 0.0);  attn_probs_36 = getitem_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:638, code: attn_probs = attn_probs.type_as(attn_scores)
    attn_probs_38 = attn_probs_37.type_as(attn_scores_19);  attn_probs_37 = attn_scores_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    attn_probs_39 = torch.nn.functional.dropout(attn_probs_38, p = 0.1, training = True);  attn_probs_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_136 = value_vectors_18.view(1024, 1, 12, 64);  value_vectors_18 = None
    value_vectors_19 = view_136.transpose(0, 1);  view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:896, code: chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
    div_78 = torch.div(1024, 256, rounding_mode = 'trunc')
    chunks_count_29 = div_78 - 1;  div_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    transpose_145 = attn_probs_39.transpose(1, 2);  attn_probs_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:901, code: torch.div(seq_len, window_overlap, rounding_mode="trunc"),
    div_79 = torch.div(1024, 256, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    chunked_attn_probs_18 = transpose_145.reshape(12, div_79, 256, 513);  transpose_145 = div_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_146 = value_vectors_19.transpose(1, 2);  value_vectors_19 = None
    value_9 = transpose_146.reshape(12, 1024, 64);  transpose_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    padded_value_9 = torch.nn.functional.pad(value_9, (0, 0, 256, 256), value = -1);  value_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:913, code: chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
    add_47 = chunks_count_29 + 1;  chunks_count_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    chunked_value_9 = padded_value_9.as_strided(size = (12, add_47, 768, 64), stride = (98304, 16384, 64, 1));  padded_value_9 = add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    chunked_hidden_states_45 = torch.nn.functional.pad(chunked_attn_probs_18, (0, 257));  chunked_attn_probs_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    chunked_hidden_states_46 = chunked_hidden_states_45.view(12, 4, -1);  chunked_hidden_states_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    chunked_hidden_states_47 = chunked_hidden_states_46[(slice(None, None, None), slice(None, None, None), slice(None, -256, None))];  chunked_hidden_states_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    chunked_hidden_states_48 = chunked_hidden_states_47.view(12, 4, 256, 769);  chunked_hidden_states_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    chunked_attn_probs_19 = chunked_hidden_states_48[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, -1, None))];  chunked_hidden_states_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    context_9 = torch.functional.einsum('bcwd,bcdh->bcwh', (chunked_attn_probs_19, chunked_value_9));  chunked_attn_probs_19 = chunked_value_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    view_139 = context_9.view(1, 12, 1024, 64);  context_9 = None
    attn_output_36 = view_139.transpose(1, 2);  view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    transpose_148 = attn_output_36.transpose(0, 1);  attn_output_36 = None
    reshape_69 = transpose_148.reshape(1024, 1, 768);  transpose_148 = None
    attn_output_37 = reshape_69.contiguous();  reshape_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    transpose_149 = attn_output_37.transpose(0, 1);  attn_output_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    hidden_states_131 = self.L__self___layer_9_attention_output_dense(transpose_149);  transpose_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    hidden_states_132 = self.L__self___layer_9_attention_output_dropout(hidden_states_131);  hidden_states_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_48 = hidden_states_132 + hidden_states_125;  hidden_states_132 = hidden_states_125 = None
    attn_output_39 = self.L__self___layer_9_attention_output_LayerNorm(add_48);  add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    hidden_states_134 = self.L__self___layer_9_intermediate_dense(attn_output_39)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_9 = torch._C._nn.gelu(hidden_states_134);  hidden_states_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    hidden_states_136 = self.L__self___layer_9_output_dense(intermediate_output_9);  intermediate_output_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    hidden_states_137 = self.L__self___layer_9_output_dropout(hidden_states_136);  hidden_states_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_49 = hidden_states_137 + attn_output_39;  hidden_states_137 = attn_output_39 = None
    hidden_states_139 = self.L__self___layer_9_output_LayerNorm(add_49);  add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    hidden_states_140 = hidden_states_139.transpose(0, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    query_vectors_30 = self.L__self___layer_10_attention_self_query(hidden_states_140)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    key_vectors_20 = self.L__self___layer_10_attention_self_key(hidden_states_140)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    value_vectors_20 = self.L__self___layer_10_attention_self_value(hidden_states_140);  hidden_states_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    query_vectors_30 /= 8.0;  query_vectors_31 = query_vectors_30;  query_vectors_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:568, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_140 = query_vectors_31.view(1024, 1, 12, 64);  query_vectors_31 = None
    query_vectors_32 = view_140.transpose(0, 1);  view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:569, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_141 = key_vectors_20.view(1024, 1, 12, 64);  key_vectors_20 = None
    key_vectors_21 = view_141.transpose(0, 1);  view_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:830, code: chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
    div_80 = torch.div(1024, 256, rounding_mode = 'trunc')
    chunks_count_30 = div_80 - 1;  div_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:833, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_153 = query_vectors_32.transpose(1, 2)
    query_40 = transpose_153.reshape(12, 1024, 64);  transpose_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_154 = key_vectors_21.transpose(1, 2);  key_vectors_21 = None
    key_40 = transpose_154.reshape(12, 1024, 64);  transpose_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_81 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_141 = query_40.view(12, div_81, 512, 64);  query_40 = div_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    query_41 = hidden_states_141.as_strided(size = [12, 3, 512, 64], stride = [64, 196608, 768, 1]);  hidden_states_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_82 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_142 = key_40.view(12, div_82, 512, 64);  key_40 = div_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    key_41 = hidden_states_142.as_strided(size = [12, 3, 512, 64], stride = [64, 196608, 768, 1]);  hidden_states_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    diagonal_chunked_attention_scores_40 = torch.functional.einsum('bcxd,bcyd->bcxy', (query_41, key_41));  query_41 = key_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    hidden_states_padded_40 = torch.nn.functional.pad(diagonal_chunked_attention_scores_40, (0, 0, 0, 1));  diagonal_chunked_attention_scores_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    diagonal_chunked_attention_scores_41 = hidden_states_padded_40.view(12, 3, 512, 513);  hidden_states_padded_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:856, code: (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
    add_50 = chunks_count_30 + 1;  chunks_count_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    diagonal_attention_scores_40 = diagonal_chunked_attention_scores_41.new_zeros((12, add_50, 256, 513));  add_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_180 = diagonal_chunked_attention_scores_41[(slice(None, None, None), slice(None, None, None), slice(None, 256, None), slice(None, 257, None))]
    diagonal_attention_scores_40[(slice(None, None, None), slice(None, -1, None), slice(None, None, None), slice(256, None, None))] = getitem_180;  setitem_120 = diagonal_attention_scores_40;  getitem_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_181 = diagonal_chunked_attention_scores_41[(slice(None, None, None), -1, slice(256, None, None), slice(None, 257, None))]
    diagonal_attention_scores_40[(slice(None, None, None), -1, slice(None, None, None), slice(256, None, None))] = getitem_181;  setitem_121 = diagonal_attention_scores_40;  getitem_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    getitem_182 = diagonal_chunked_attention_scores_41[(slice(None, None, None), slice(None, None, None), slice(-257, -1, None), slice(257, None, None))]
    diagonal_attention_scores_40[(slice(None, None, None), slice(1, None, None), slice(None, None, None), slice(None, 256, None))] = getitem_182;  setitem_122 = diagonal_attention_scores_40;  getitem_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    getitem_183 = diagonal_chunked_attention_scores_41[(slice(None, None, None), 0, slice(None, 255, None), slice(-255, None, None))];  diagonal_chunked_attention_scores_41 = None
    diagonal_attention_scores_40[(slice(None, None, None), 0, slice(1, 256, None), slice(1, 256, None))] = getitem_183;  setitem_123 = diagonal_attention_scores_40;  getitem_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    view_145 = diagonal_attention_scores_40.view(1, 12, 1024, 513);  diagonal_attention_scores_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:879, code: ).transpose(2, 1)
    attn_scores_20 = view_145.transpose(2, 1);  view_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    new_ones_30 = attn_scores_20.new_ones(256, 257)
    tril_20 = new_ones_30.tril();  new_ones_30 = None
    beginning_mask_2d_20 = tril_20.flip(dims = [0]);  tril_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    beginning_mask_40 = beginning_mask_2d_20[(None, slice(None, None, None), None, slice(None, None, None))];  beginning_mask_2d_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    ending_mask_40 = beginning_mask_40.flip(dims = (1, 3))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    beginning_input_20 = attn_scores_20[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    beginning_mask_41 = beginning_mask_40.expand((1, 256, 12, 257));  beginning_mask_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    full_like_40 = torch.full_like(beginning_input_20, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    bool_41 = beginning_mask_41.bool();  beginning_mask_41 = None
    where_40 = full_like_40.where(bool_41, beginning_input_20);  full_like_40 = bool_41 = beginning_input_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    attn_scores_20[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))] = where_40;  setitem_124 = attn_scores_20;  where_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    ending_input_20 = attn_scores_20[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    ending_mask_41 = ending_mask_40.expand((1, 256, 12, 257));  ending_mask_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    full_like_41 = torch.full_like(ending_input_20, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    bool_42 = ending_mask_41.bool();  ending_mask_41 = None
    where_41 = full_like_41.where(bool_42, ending_input_20);  full_like_41 = bool_42 = ending_input_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    attn_scores_20[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))] = where_41;  setitem_125 = attn_scores_20;  where_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:576, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
    ne_10 = l_attention_mask_ != 0
    remove_from_windowed_attention_mask_10 = ne_10[(slice(None, None, None), slice(None, None, None), None, None)];  ne_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
    type_as_20 = remove_from_windowed_attention_mask_10.type_as(query_vectors_32);  query_vectors_32 = None
    float_mask_10 = type_as_20.masked_fill(remove_from_windowed_attention_mask_10, -3.4028234663852886e+38);  type_as_20 = remove_from_windowed_attention_mask_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:584, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
    new_ones_31 = float_mask_10.new_ones(size = (1, 1024, 1, 1))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:830, code: chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
    div_83 = torch.div(1024, 256, rounding_mode = 'trunc')
    chunks_count_31 = div_83 - 1;  div_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:833, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_156 = new_ones_31.transpose(1, 2);  new_ones_31 = None
    query_42 = transpose_156.reshape(1, 1024, 1);  transpose_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_157 = float_mask_10.transpose(1, 2);  float_mask_10 = None
    key_42 = transpose_157.reshape(1, 1024, 1);  transpose_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_84 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_143 = query_42.view(1, div_84, 512, 1);  query_42 = div_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    query_43 = hidden_states_143.as_strided(size = [1, 3, 512, 1], stride = [1024, 256, 1, 1]);  hidden_states_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_85 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_144 = key_42.view(1, div_85, 512, 1);  key_42 = div_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    key_43 = hidden_states_144.as_strided(size = [1, 3, 512, 1], stride = [1024, 256, 1, 1]);  hidden_states_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    diagonal_chunked_attention_scores_42 = torch.functional.einsum('bcxd,bcyd->bcxy', (query_43, key_43));  query_43 = key_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    hidden_states_padded_42 = torch.nn.functional.pad(diagonal_chunked_attention_scores_42, (0, 0, 0, 1));  diagonal_chunked_attention_scores_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    diagonal_chunked_attention_scores_43 = hidden_states_padded_42.view(1, 3, 512, 513);  hidden_states_padded_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:856, code: (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
    add_51 = chunks_count_31 + 1;  chunks_count_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    diagonal_attention_scores_42 = diagonal_chunked_attention_scores_43.new_zeros((1, add_51, 256, 513));  add_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_188 = diagonal_chunked_attention_scores_43[(slice(None, None, None), slice(None, None, None), slice(None, 256, None), slice(None, 257, None))]
    diagonal_attention_scores_42[(slice(None, None, None), slice(None, -1, None), slice(None, None, None), slice(256, None, None))] = getitem_188;  setitem_126 = diagonal_attention_scores_42;  getitem_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_189 = diagonal_chunked_attention_scores_43[(slice(None, None, None), -1, slice(256, None, None), slice(None, 257, None))]
    diagonal_attention_scores_42[(slice(None, None, None), -1, slice(None, None, None), slice(256, None, None))] = getitem_189;  setitem_127 = diagonal_attention_scores_42;  getitem_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    getitem_190 = diagonal_chunked_attention_scores_43[(slice(None, None, None), slice(None, None, None), slice(-257, -1, None), slice(257, None, None))]
    diagonal_attention_scores_42[(slice(None, None, None), slice(1, None, None), slice(None, None, None), slice(None, 256, None))] = getitem_190;  setitem_128 = diagonal_attention_scores_42;  getitem_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    getitem_191 = diagonal_chunked_attention_scores_43[(slice(None, None, None), 0, slice(None, 255, None), slice(-255, None, None))];  diagonal_chunked_attention_scores_43 = None
    diagonal_attention_scores_42[(slice(None, None, None), 0, slice(1, 256, None), slice(1, 256, None))] = getitem_191;  setitem_129 = diagonal_attention_scores_42;  getitem_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    view_149 = diagonal_attention_scores_42.view(1, 1, 1024, 513);  diagonal_attention_scores_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:879, code: ).transpose(2, 1)
    diagonal_mask_10 = view_149.transpose(2, 1);  view_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    new_ones_32 = diagonal_mask_10.new_ones(256, 257)
    tril_21 = new_ones_32.tril();  new_ones_32 = None
    beginning_mask_2d_21 = tril_21.flip(dims = [0]);  tril_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    beginning_mask_42 = beginning_mask_2d_21[(None, slice(None, None, None), None, slice(None, None, None))];  beginning_mask_2d_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    ending_mask_42 = beginning_mask_42.flip(dims = (1, 3))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    beginning_input_21 = diagonal_mask_10[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    beginning_mask_43 = beginning_mask_42.expand((1, 256, 1, 257));  beginning_mask_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    full_like_42 = torch.full_like(beginning_input_21, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    bool_43 = beginning_mask_43.bool();  beginning_mask_43 = None
    where_42 = full_like_42.where(bool_43, beginning_input_21);  full_like_42 = bool_43 = beginning_input_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    diagonal_mask_10[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))] = where_42;  setitem_130 = diagonal_mask_10;  where_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    ending_input_21 = diagonal_mask_10[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    ending_mask_43 = ending_mask_42.expand((1, 256, 1, 257));  ending_mask_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    full_like_43 = torch.full_like(ending_input_21, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    bool_44 = ending_mask_43.bool();  ending_mask_43 = None
    where_43 = full_like_43.where(bool_44, ending_input_21);  full_like_43 = bool_44 = ending_input_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    diagonal_mask_10[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))] = where_43;  setitem_131 = diagonal_mask_10;  where_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    attn_scores_20 += diagonal_mask_10;  attn_scores_21 = attn_scores_20;  attn_scores_20 = diagonal_mask_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    attn_probs_40 = torch.nn.functional.softmax(attn_scores_21, dim = -1, dtype = torch.float32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    getitem_195 = l_is_index_masked_[(slice(None, None, None), slice(None, None, None), None, None)]
    attn_probs_41 = torch.masked_fill(attn_probs_40, getitem_195, 0.0);  attn_probs_40 = getitem_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:638, code: attn_probs = attn_probs.type_as(attn_scores)
    attn_probs_42 = attn_probs_41.type_as(attn_scores_21);  attn_probs_41 = attn_scores_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    attn_probs_43 = torch.nn.functional.dropout(attn_probs_42, p = 0.1, training = True);  attn_probs_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_150 = value_vectors_20.view(1024, 1, 12, 64);  value_vectors_20 = None
    value_vectors_21 = view_150.transpose(0, 1);  view_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:896, code: chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
    div_86 = torch.div(1024, 256, rounding_mode = 'trunc')
    chunks_count_32 = div_86 - 1;  div_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    transpose_160 = attn_probs_43.transpose(1, 2);  attn_probs_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:901, code: torch.div(seq_len, window_overlap, rounding_mode="trunc"),
    div_87 = torch.div(1024, 256, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    chunked_attn_probs_20 = transpose_160.reshape(12, div_87, 256, 513);  transpose_160 = div_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_161 = value_vectors_21.transpose(1, 2);  value_vectors_21 = None
    value_10 = transpose_161.reshape(12, 1024, 64);  transpose_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    padded_value_10 = torch.nn.functional.pad(value_10, (0, 0, 256, 256), value = -1);  value_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:913, code: chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
    add_52 = chunks_count_32 + 1;  chunks_count_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    chunked_value_10 = padded_value_10.as_strided(size = (12, add_52, 768, 64), stride = (98304, 16384, 64, 1));  padded_value_10 = add_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    chunked_hidden_states_50 = torch.nn.functional.pad(chunked_attn_probs_20, (0, 257));  chunked_attn_probs_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    chunked_hidden_states_51 = chunked_hidden_states_50.view(12, 4, -1);  chunked_hidden_states_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    chunked_hidden_states_52 = chunked_hidden_states_51[(slice(None, None, None), slice(None, None, None), slice(None, -256, None))];  chunked_hidden_states_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    chunked_hidden_states_53 = chunked_hidden_states_52.view(12, 4, 256, 769);  chunked_hidden_states_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    chunked_attn_probs_21 = chunked_hidden_states_53[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, -1, None))];  chunked_hidden_states_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    context_10 = torch.functional.einsum('bcwd,bcdh->bcwh', (chunked_attn_probs_21, chunked_value_10));  chunked_attn_probs_21 = chunked_value_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    view_153 = context_10.view(1, 12, 1024, 64);  context_10 = None
    attn_output_40 = view_153.transpose(1, 2);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    transpose_163 = attn_output_40.transpose(0, 1);  attn_output_40 = None
    reshape_76 = transpose_163.reshape(1024, 1, 768);  transpose_163 = None
    attn_output_41 = reshape_76.contiguous();  reshape_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    transpose_164 = attn_output_41.transpose(0, 1);  attn_output_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    hidden_states_145 = self.L__self___layer_10_attention_output_dense(transpose_164);  transpose_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    hidden_states_146 = self.L__self___layer_10_attention_output_dropout(hidden_states_145);  hidden_states_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_53 = hidden_states_146 + hidden_states_139;  hidden_states_146 = hidden_states_139 = None
    attn_output_43 = self.L__self___layer_10_attention_output_LayerNorm(add_53);  add_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    hidden_states_148 = self.L__self___layer_10_intermediate_dense(attn_output_43)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_10 = torch._C._nn.gelu(hidden_states_148);  hidden_states_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    hidden_states_150 = self.L__self___layer_10_output_dense(intermediate_output_10);  intermediate_output_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    hidden_states_151 = self.L__self___layer_10_output_dropout(hidden_states_150);  hidden_states_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_54 = hidden_states_151 + attn_output_43;  hidden_states_151 = attn_output_43 = None
    hidden_states_153 = self.L__self___layer_10_output_LayerNorm(add_54);  add_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    hidden_states_154 = hidden_states_153.transpose(0, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    query_vectors_33 = self.L__self___layer_11_attention_self_query(hidden_states_154)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    key_vectors_22 = self.L__self___layer_11_attention_self_key(hidden_states_154)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    value_vectors_22 = self.L__self___layer_11_attention_self_value(hidden_states_154);  hidden_states_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    query_vectors_33 /= 8.0;  query_vectors_34 = query_vectors_33;  query_vectors_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:568, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_154 = query_vectors_34.view(1024, 1, 12, 64);  query_vectors_34 = None
    query_vectors_35 = view_154.transpose(0, 1);  view_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:569, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_155 = key_vectors_22.view(1024, 1, 12, 64);  key_vectors_22 = None
    key_vectors_23 = view_155.transpose(0, 1);  view_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:830, code: chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
    div_88 = torch.div(1024, 256, rounding_mode = 'trunc')
    chunks_count_33 = div_88 - 1;  div_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:833, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_168 = query_vectors_35.transpose(1, 2)
    query_44 = transpose_168.reshape(12, 1024, 64);  transpose_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_169 = key_vectors_23.transpose(1, 2);  key_vectors_23 = None
    key_44 = transpose_169.reshape(12, 1024, 64);  transpose_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_89 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_155 = query_44.view(12, div_89, 512, 64);  query_44 = div_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    query_45 = hidden_states_155.as_strided(size = [12, 3, 512, 64], stride = [64, 196608, 768, 1]);  hidden_states_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_90 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_156 = key_44.view(12, div_90, 512, 64);  key_44 = div_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    key_45 = hidden_states_156.as_strided(size = [12, 3, 512, 64], stride = [64, 196608, 768, 1]);  hidden_states_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    diagonal_chunked_attention_scores_44 = torch.functional.einsum('bcxd,bcyd->bcxy', (query_45, key_45));  query_45 = key_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    hidden_states_padded_44 = torch.nn.functional.pad(diagonal_chunked_attention_scores_44, (0, 0, 0, 1));  diagonal_chunked_attention_scores_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    diagonal_chunked_attention_scores_45 = hidden_states_padded_44.view(12, 3, 512, 513);  hidden_states_padded_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:856, code: (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
    add_55 = chunks_count_33 + 1;  chunks_count_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    diagonal_attention_scores_44 = diagonal_chunked_attention_scores_45.new_zeros((12, add_55, 256, 513));  add_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_198 = diagonal_chunked_attention_scores_45[(slice(None, None, None), slice(None, None, None), slice(None, 256, None), slice(None, 257, None))]
    diagonal_attention_scores_44[(slice(None, None, None), slice(None, -1, None), slice(None, None, None), slice(256, None, None))] = getitem_198;  setitem_132 = diagonal_attention_scores_44;  getitem_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_199 = diagonal_chunked_attention_scores_45[(slice(None, None, None), -1, slice(256, None, None), slice(None, 257, None))]
    diagonal_attention_scores_44[(slice(None, None, None), -1, slice(None, None, None), slice(256, None, None))] = getitem_199;  setitem_133 = diagonal_attention_scores_44;  getitem_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    getitem_200 = diagonal_chunked_attention_scores_45[(slice(None, None, None), slice(None, None, None), slice(-257, -1, None), slice(257, None, None))]
    diagonal_attention_scores_44[(slice(None, None, None), slice(1, None, None), slice(None, None, None), slice(None, 256, None))] = getitem_200;  setitem_134 = diagonal_attention_scores_44;  getitem_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    getitem_201 = diagonal_chunked_attention_scores_45[(slice(None, None, None), 0, slice(None, 255, None), slice(-255, None, None))];  diagonal_chunked_attention_scores_45 = None
    diagonal_attention_scores_44[(slice(None, None, None), 0, slice(1, 256, None), slice(1, 256, None))] = getitem_201;  setitem_135 = diagonal_attention_scores_44;  getitem_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    view_159 = diagonal_attention_scores_44.view(1, 12, 1024, 513);  diagonal_attention_scores_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:879, code: ).transpose(2, 1)
    attn_scores_22 = view_159.transpose(2, 1);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    new_ones_33 = attn_scores_22.new_ones(256, 257)
    tril_22 = new_ones_33.tril();  new_ones_33 = None
    beginning_mask_2d_22 = tril_22.flip(dims = [0]);  tril_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    beginning_mask_44 = beginning_mask_2d_22[(None, slice(None, None, None), None, slice(None, None, None))];  beginning_mask_2d_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    ending_mask_44 = beginning_mask_44.flip(dims = (1, 3))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    beginning_input_22 = attn_scores_22[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    beginning_mask_45 = beginning_mask_44.expand((1, 256, 12, 257));  beginning_mask_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    full_like_44 = torch.full_like(beginning_input_22, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    bool_45 = beginning_mask_45.bool();  beginning_mask_45 = None
    where_44 = full_like_44.where(bool_45, beginning_input_22);  full_like_44 = bool_45 = beginning_input_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    attn_scores_22[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))] = where_44;  setitem_136 = attn_scores_22;  where_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    ending_input_22 = attn_scores_22[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    ending_mask_45 = ending_mask_44.expand((1, 256, 12, 257));  ending_mask_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    full_like_45 = torch.full_like(ending_input_22, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    bool_46 = ending_mask_45.bool();  ending_mask_45 = None
    where_45 = full_like_45.where(bool_46, ending_input_22);  full_like_45 = bool_46 = ending_input_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    attn_scores_22[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))] = where_45;  setitem_137 = attn_scores_22;  where_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:576, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
    ne_11 = l_attention_mask_ != 0;  l_attention_mask_ = None
    remove_from_windowed_attention_mask_11 = ne_11[(slice(None, None, None), slice(None, None, None), None, None)];  ne_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
    type_as_22 = remove_from_windowed_attention_mask_11.type_as(query_vectors_35);  query_vectors_35 = None
    float_mask_11 = type_as_22.masked_fill(remove_from_windowed_attention_mask_11, -3.4028234663852886e+38);  type_as_22 = remove_from_windowed_attention_mask_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:584, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
    new_ones_34 = float_mask_11.new_ones(size = (1, 1024, 1, 1))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:830, code: chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
    div_91 = torch.div(1024, 256, rounding_mode = 'trunc')
    chunks_count_34 = div_91 - 1;  div_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:833, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_171 = new_ones_34.transpose(1, 2);  new_ones_34 = None
    query_46 = transpose_171.reshape(1, 1024, 1);  transpose_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_172 = float_mask_11.transpose(1, 2);  float_mask_11 = None
    key_46 = transpose_172.reshape(1, 1024, 1);  transpose_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_92 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_157 = query_46.view(1, div_92, 512, 1);  query_46 = div_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    query_47 = hidden_states_157.as_strided(size = [1, 3, 512, 1], stride = [1024, 256, 1, 1]);  hidden_states_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:768, code: torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
    div_93 = torch.div(1024, 512, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    hidden_states_158 = key_46.view(1, div_93, 512, 1);  key_46 = div_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    key_47 = hidden_states_158.as_strided(size = [1, 3, 512, 1], stride = [1024, 256, 1, 1]);  hidden_states_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    diagonal_chunked_attention_scores_46 = torch.functional.einsum('bcxd,bcyd->bcxy', (query_47, key_47));  query_47 = key_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    hidden_states_padded_46 = torch.nn.functional.pad(diagonal_chunked_attention_scores_46, (0, 0, 0, 1));  diagonal_chunked_attention_scores_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    diagonal_chunked_attention_scores_47 = hidden_states_padded_46.view(1, 3, 512, 513);  hidden_states_padded_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:856, code: (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
    add_56 = chunks_count_34 + 1;  chunks_count_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    diagonal_attention_scores_46 = diagonal_chunked_attention_scores_47.new_zeros((1, add_56, 256, 513));  add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_206 = diagonal_chunked_attention_scores_47[(slice(None, None, None), slice(None, None, None), slice(None, 256, None), slice(None, 257, None))]
    diagonal_attention_scores_46[(slice(None, None, None), slice(None, -1, None), slice(None, None, None), slice(256, None, None))] = getitem_206;  setitem_138 = diagonal_attention_scores_46;  getitem_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    getitem_207 = diagonal_chunked_attention_scores_47[(slice(None, None, None), -1, slice(256, None, None), slice(None, 257, None))]
    diagonal_attention_scores_46[(slice(None, None, None), -1, slice(None, None, None), slice(256, None, None))] = getitem_207;  setitem_139 = diagonal_attention_scores_46;  getitem_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    getitem_208 = diagonal_chunked_attention_scores_47[(slice(None, None, None), slice(None, None, None), slice(-257, -1, None), slice(257, None, None))]
    diagonal_attention_scores_46[(slice(None, None, None), slice(1, None, None), slice(None, None, None), slice(None, 256, None))] = getitem_208;  setitem_140 = diagonal_attention_scores_46;  getitem_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    getitem_209 = diagonal_chunked_attention_scores_47[(slice(None, None, None), 0, slice(None, 255, None), slice(-255, None, None))];  diagonal_chunked_attention_scores_47 = None
    diagonal_attention_scores_46[(slice(None, None, None), 0, slice(1, 256, None), slice(1, 256, None))] = getitem_209;  setitem_141 = diagonal_attention_scores_46;  getitem_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    view_163 = diagonal_attention_scores_46.view(1, 1, 1024, 513);  diagonal_attention_scores_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:879, code: ).transpose(2, 1)
    diagonal_mask_11 = view_163.transpose(2, 1);  view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    new_ones_35 = diagonal_mask_11.new_ones(256, 257)
    tril_23 = new_ones_35.tril();  new_ones_35 = None
    beginning_mask_2d_23 = tril_23.flip(dims = [0]);  tril_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    beginning_mask_46 = beginning_mask_2d_23[(None, slice(None, None, None), None, slice(None, None, None))];  beginning_mask_2d_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    ending_mask_46 = beginning_mask_46.flip(dims = (1, 3))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    beginning_input_23 = diagonal_mask_11[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    beginning_mask_47 = beginning_mask_46.expand((1, 256, 1, 257));  beginning_mask_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    full_like_46 = torch.full_like(beginning_input_23, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    bool_47 = beginning_mask_47.bool();  beginning_mask_47 = None
    where_46 = full_like_46.where(bool_47, beginning_input_23);  full_like_46 = bool_47 = beginning_input_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    diagonal_mask_11[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, 257, None))] = where_46;  setitem_142 = diagonal_mask_11;  where_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    ending_input_23 = diagonal_mask_11[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    ending_mask_47 = ending_mask_46.expand((1, 256, 1, 257));  ending_mask_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    full_like_47 = torch.full_like(ending_input_23, -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    bool_48 = ending_mask_47.bool();  ending_mask_47 = None
    where_47 = full_like_47.where(bool_48, ending_input_23);  full_like_47 = bool_48 = ending_input_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    diagonal_mask_11[(slice(None, None, None), slice(-256, None, None), slice(None, None, None), slice(-257, None, None))] = where_47;  setitem_143 = diagonal_mask_11;  where_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    attn_scores_22 += diagonal_mask_11;  attn_scores_23 = attn_scores_22;  attn_scores_22 = diagonal_mask_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    attn_probs_44 = torch.nn.functional.softmax(attn_scores_23, dim = -1, dtype = torch.float32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    getitem_213 = l_is_index_masked_[(slice(None, None, None), slice(None, None, None), None, None)];  l_is_index_masked_ = None
    attn_probs_45 = torch.masked_fill(attn_probs_44, getitem_213, 0.0);  attn_probs_44 = getitem_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:638, code: attn_probs = attn_probs.type_as(attn_scores)
    attn_probs_46 = attn_probs_45.type_as(attn_scores_23);  attn_probs_45 = attn_scores_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    attn_probs_47 = torch.nn.functional.dropout(attn_probs_46, p = 0.1, training = True);  attn_probs_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_164 = value_vectors_22.view(1024, 1, 12, 64);  value_vectors_22 = None
    value_vectors_23 = view_164.transpose(0, 1);  view_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:896, code: chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
    div_94 = torch.div(1024, 256, rounding_mode = 'trunc')
    chunks_count_35 = div_94 - 1;  div_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    transpose_175 = attn_probs_47.transpose(1, 2);  attn_probs_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:901, code: torch.div(seq_len, window_overlap, rounding_mode="trunc"),
    div_95 = torch.div(1024, 256, rounding_mode = 'trunc')
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    chunked_attn_probs_22 = transpose_175.reshape(12, div_95, 256, 513);  transpose_175 = div_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    transpose_176 = value_vectors_23.transpose(1, 2);  value_vectors_23 = None
    value_11 = transpose_176.reshape(12, 1024, 64);  transpose_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    padded_value_11 = torch.nn.functional.pad(value_11, (0, 0, 256, 256), value = -1);  value_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:913, code: chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
    add_57 = chunks_count_35 + 1;  chunks_count_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    chunked_value_11 = padded_value_11.as_strided(size = (12, add_57, 768, 64), stride = (98304, 16384, 64, 1));  padded_value_11 = add_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    chunked_hidden_states_55 = torch.nn.functional.pad(chunked_attn_probs_22, (0, 257));  chunked_attn_probs_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    chunked_hidden_states_56 = chunked_hidden_states_55.view(12, 4, -1);  chunked_hidden_states_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    chunked_hidden_states_57 = chunked_hidden_states_56[(slice(None, None, None), slice(None, None, None), slice(None, -256, None))];  chunked_hidden_states_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    chunked_hidden_states_58 = chunked_hidden_states_57.view(12, 4, 256, 769);  chunked_hidden_states_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    chunked_attn_probs_23 = chunked_hidden_states_58[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, -1, None))];  chunked_hidden_states_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    context_11 = torch.functional.einsum('bcwd,bcdh->bcwh', (chunked_attn_probs_23, chunked_value_11));  chunked_attn_probs_23 = chunked_value_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    view_167 = context_11.view(1, 12, 1024, 64);  context_11 = None
    attn_output_44 = view_167.transpose(1, 2);  view_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    transpose_178 = attn_output_44.transpose(0, 1);  attn_output_44 = None
    reshape_83 = transpose_178.reshape(1024, 1, 768);  transpose_178 = None
    attn_output_45 = reshape_83.contiguous();  reshape_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    transpose_179 = attn_output_45.transpose(0, 1);  attn_output_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    hidden_states_159 = self.L__self___layer_11_attention_output_dense(transpose_179);  transpose_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    hidden_states_160 = self.L__self___layer_11_attention_output_dropout(hidden_states_159);  hidden_states_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_58 = hidden_states_160 + hidden_states_153;  hidden_states_160 = hidden_states_153 = None
    attn_output_47 = self.L__self___layer_11_attention_output_LayerNorm(add_58);  add_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    hidden_states_162 = self.L__self___layer_11_intermediate_dense(attn_output_47)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_11 = torch._C._nn.gelu(hidden_states_162);  hidden_states_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    hidden_states_164 = self.L__self___layer_11_output_dense(intermediate_output_11);  intermediate_output_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    hidden_states_165 = self.L__self___layer_11_output_dropout(hidden_states_164);  hidden_states_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_59 = hidden_states_165 + attn_output_47;  hidden_states_165 = attn_output_47 = None
    hidden_states_167 = self.L__self___layer_11_output_LayerNorm(add_59);  add_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1348, code: hidden_states = hidden_states[:, : hidden_states.shape[1] - padding_len]
    hidden_states_168 = hidden_states_167[(slice(None, None, None), slice(None, 1024, None))];  hidden_states_167 = None
    return (hidden_states_168,)
    