from __future__ import annotations



def forward(self, L_inputs_labels_ : torch.Tensor, L_inputs_input_ids_ : torch.Tensor):
    l_inputs_labels_ = L_inputs_labels_
    input_1 = L_inputs_input_ids_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:66, code: prev_output_tokens = input_ids.clone()
    input_2 = l_inputs_labels_.clone()
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:71, code: prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)
    eq = input_2 == -100
    masked_fill_ = input_2.masked_fill_(eq, 1);  eq = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:73, code: index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    ne = input_2.ne(1)
    sum_1 = ne.sum(dim = 1);  ne = None
    sub = sum_1 - 1;  sum_1 = None
    index_of_eos = sub.unsqueeze(-1);  sub = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:74, code: decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
    gather = input_2.gather(1, index_of_eos);  index_of_eos = None
    decoder_start_tokens = gather.squeeze();  gather = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:75, code: prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
    getitem = input_2[(slice(None, None, None), slice(None, -1, None))]
    clone_1 = getitem.clone();  getitem = None
    input_2[(slice(None, None, None), slice(1, None, None))] = clone_1;  setitem = input_2;  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:76, code: prev_output_tokens[:, 0] = decoder_start_tokens
    input_2[(slice(None, None, None), 0)] = decoder_start_tokens;  setitem_1 = input_2;  decoder_start_tokens = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:787, code: input_ids = input_ids.view(-1, input_shape[-1])
    input_ids = input_1.view(-1, 1024);  input_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:794, code: inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
    l__mod___model_encoder_embed_tokens = self.L__mod___model_encoder_embed_tokens(input_ids);  input_ids = None
    inputs_embeds = l__mod___model_encoder_embed_tokens * 1.0;  l__mod___model_encoder_embed_tokens = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:131, code: past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
    l__mod___model_encoder_embed_positions_weight = self.L__mod___model_encoder_embed_positions_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:130, code: positions = torch.arange(
    arange = torch.arange(0, 1024, dtype = torch.int64, device = device(type='cpu'))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:132, code: ).expand(bsz, -1)
    positions = arange.expand(1, -1);  arange = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:134, code: return super().forward(positions + self.offset)
    add = positions + 2;  positions = None
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    embed_pos = torch.nn.functional.embedding(add, l__mod___model_encoder_embed_positions_weight, None, None, 2.0, False, False);  add = l__mod___model_encoder_embed_positions_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:798, code: hidden_states = inputs_embeds + embed_pos.to(inputs_embeds.device)
    to = embed_pos.to(device(type='cpu'));  embed_pos = None
    hidden_states = inputs_embeds + to;  inputs_embeds = to = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:799, code: hidden_states = self.layernorm_embedding(hidden_states)
    hidden_states_1 = self.L__mod___model_encoder_layernorm_embedding(hidden_states);  hidden_states = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:800, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    residual = torch.nn.functional.dropout(hidden_states_1, p = 0.1, training = False);  hidden_states_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:328, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_3 = self.L__mod___model_encoder_layers_0_self_attn_layer_norm(residual)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_encoder_layers_0_self_attn_q_proj = self.L__mod___model_encoder_layers_0_self_attn_q_proj(hidden_states_3)
    query_states = l__mod___model_encoder_layers_0_self_attn_q_proj * 0.125;  l__mod___model_encoder_layers_0_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_0_self_attn_k_proj = self.L__mod___model_encoder_layers_0_self_attn_k_proj(hidden_states_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_1 = l__mod___model_encoder_layers_0_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_0_self_attn_k_proj = None
    transpose = view_1.transpose(1, 2);  view_1 = None
    key_states = transpose.contiguous();  transpose = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_0_self_attn_v_proj = self.L__mod___model_encoder_layers_0_self_attn_v_proj(hidden_states_3);  hidden_states_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_2 = l__mod___model_encoder_layers_0_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_0_self_attn_v_proj = None
    transpose_1 = view_2.transpose(1, 2);  view_2 = None
    value_states = transpose_1.contiguous();  transpose_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_3 = query_states.view(1, 1024, 16, 64);  query_states = None
    transpose_2 = view_3.transpose(1, 2);  view_3 = None
    contiguous_2 = transpose_2.contiguous();  transpose_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_1 = contiguous_2.view(16, -1, 64);  contiguous_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    key_states_1 = key_states.reshape(16, -1, 64);  key_states = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    value_states_1 = value_states.reshape(16, -1, 64);  value_states = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_3 = key_states_1.transpose(1, 2);  key_states_1 = None
    attn_weights = torch.bmm(query_states_1, transpose_3);  query_states_1 = transpose_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_1 = torch.nn.functional.softmax(attn_weights, dim = -1);  attn_weights = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:270, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs = torch.nn.functional.dropout(attn_weights_1, p = 0.0, training = False);  attn_weights_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output = torch.bmm(attn_probs, value_states_1);  attn_probs = value_states_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_1 = attn_output.view(1, 16, 1024, 64);  attn_output = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    attn_output_2 = attn_output_1.transpose(1, 2);  attn_output_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_3 = attn_output_2.reshape(1, 1024, 1024);  attn_output_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    hidden_states_4 = self.L__mod___model_encoder_layers_0_self_attn_out_proj(attn_output_3);  attn_output_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:335, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_5 = torch.nn.functional.dropout(hidden_states_4, p = 0.1, training = False);  hidden_states_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:336, code: hidden_states = residual + hidden_states
    residual_1 = residual + hidden_states_5;  residual = hidden_states_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:339, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_7 = self.L__mod___model_encoder_layers_0_final_layer_norm(residual_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:340, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_encoder_layers_0_fc1 = self.L__mod___model_encoder_layers_0_fc1(hidden_states_7);  hidden_states_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_8 = torch._C._nn.gelu(l__mod___model_encoder_layers_0_fc1);  l__mod___model_encoder_layers_0_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:341, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_9 = torch.nn.functional.dropout(hidden_states_8, p = 0.0, training = False);  hidden_states_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:342, code: hidden_states = self.fc2(hidden_states)
    hidden_states_10 = self.L__mod___model_encoder_layers_0_fc2(hidden_states_9);  hidden_states_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:343, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_11 = torch.nn.functional.dropout(hidden_states_10, p = 0.1, training = False);  hidden_states_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:344, code: hidden_states = residual + hidden_states
    residual_2 = residual_1 + hidden_states_11;  residual_1 = hidden_states_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:328, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_14 = self.L__mod___model_encoder_layers_1_self_attn_layer_norm(residual_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_encoder_layers_1_self_attn_q_proj = self.L__mod___model_encoder_layers_1_self_attn_q_proj(hidden_states_14)
    query_states_2 = l__mod___model_encoder_layers_1_self_attn_q_proj * 0.125;  l__mod___model_encoder_layers_1_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_1_self_attn_k_proj = self.L__mod___model_encoder_layers_1_self_attn_k_proj(hidden_states_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_6 = l__mod___model_encoder_layers_1_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_1_self_attn_k_proj = None
    transpose_5 = view_6.transpose(1, 2);  view_6 = None
    key_states_2 = transpose_5.contiguous();  transpose_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_1_self_attn_v_proj = self.L__mod___model_encoder_layers_1_self_attn_v_proj(hidden_states_14);  hidden_states_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_7 = l__mod___model_encoder_layers_1_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_1_self_attn_v_proj = None
    transpose_6 = view_7.transpose(1, 2);  view_7 = None
    value_states_2 = transpose_6.contiguous();  transpose_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_8 = query_states_2.view(1, 1024, 16, 64);  query_states_2 = None
    transpose_7 = view_8.transpose(1, 2);  view_8 = None
    contiguous_5 = transpose_7.contiguous();  transpose_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_3 = contiguous_5.view(16, -1, 64);  contiguous_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    key_states_3 = key_states_2.reshape(16, -1, 64);  key_states_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    value_states_3 = value_states_2.reshape(16, -1, 64);  value_states_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_8 = key_states_3.transpose(1, 2);  key_states_3 = None
    attn_weights_2 = torch.bmm(query_states_3, transpose_8);  query_states_3 = transpose_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_3 = torch.nn.functional.softmax(attn_weights_2, dim = -1);  attn_weights_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:270, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_1 = torch.nn.functional.dropout(attn_weights_3, p = 0.0, training = False);  attn_weights_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_5 = torch.bmm(attn_probs_1, value_states_3);  attn_probs_1 = value_states_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_6 = attn_output_5.view(1, 16, 1024, 64);  attn_output_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    attn_output_7 = attn_output_6.transpose(1, 2);  attn_output_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_8 = attn_output_7.reshape(1, 1024, 1024);  attn_output_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    hidden_states_15 = self.L__mod___model_encoder_layers_1_self_attn_out_proj(attn_output_8);  attn_output_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:335, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_16 = torch.nn.functional.dropout(hidden_states_15, p = 0.1, training = False);  hidden_states_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:336, code: hidden_states = residual + hidden_states
    residual_3 = residual_2 + hidden_states_16;  residual_2 = hidden_states_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:339, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_18 = self.L__mod___model_encoder_layers_1_final_layer_norm(residual_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:340, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_encoder_layers_1_fc1 = self.L__mod___model_encoder_layers_1_fc1(hidden_states_18);  hidden_states_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_19 = torch._C._nn.gelu(l__mod___model_encoder_layers_1_fc1);  l__mod___model_encoder_layers_1_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:341, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_20 = torch.nn.functional.dropout(hidden_states_19, p = 0.0, training = False);  hidden_states_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:342, code: hidden_states = self.fc2(hidden_states)
    hidden_states_21 = self.L__mod___model_encoder_layers_1_fc2(hidden_states_20);  hidden_states_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:343, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_22 = torch.nn.functional.dropout(hidden_states_21, p = 0.1, training = False);  hidden_states_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:344, code: hidden_states = residual + hidden_states
    residual_4 = residual_3 + hidden_states_22;  residual_3 = hidden_states_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:328, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_25 = self.L__mod___model_encoder_layers_2_self_attn_layer_norm(residual_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_encoder_layers_2_self_attn_q_proj = self.L__mod___model_encoder_layers_2_self_attn_q_proj(hidden_states_25)
    query_states_4 = l__mod___model_encoder_layers_2_self_attn_q_proj * 0.125;  l__mod___model_encoder_layers_2_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_2_self_attn_k_proj = self.L__mod___model_encoder_layers_2_self_attn_k_proj(hidden_states_25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_11 = l__mod___model_encoder_layers_2_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_2_self_attn_k_proj = None
    transpose_10 = view_11.transpose(1, 2);  view_11 = None
    key_states_4 = transpose_10.contiguous();  transpose_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_2_self_attn_v_proj = self.L__mod___model_encoder_layers_2_self_attn_v_proj(hidden_states_25);  hidden_states_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_12 = l__mod___model_encoder_layers_2_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_2_self_attn_v_proj = None
    transpose_11 = view_12.transpose(1, 2);  view_12 = None
    value_states_4 = transpose_11.contiguous();  transpose_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_13 = query_states_4.view(1, 1024, 16, 64);  query_states_4 = None
    transpose_12 = view_13.transpose(1, 2);  view_13 = None
    contiguous_8 = transpose_12.contiguous();  transpose_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_5 = contiguous_8.view(16, -1, 64);  contiguous_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    key_states_5 = key_states_4.reshape(16, -1, 64);  key_states_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    value_states_5 = value_states_4.reshape(16, -1, 64);  value_states_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_13 = key_states_5.transpose(1, 2);  key_states_5 = None
    attn_weights_4 = torch.bmm(query_states_5, transpose_13);  query_states_5 = transpose_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_5 = torch.nn.functional.softmax(attn_weights_4, dim = -1);  attn_weights_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:270, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_2 = torch.nn.functional.dropout(attn_weights_5, p = 0.0, training = False);  attn_weights_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_10 = torch.bmm(attn_probs_2, value_states_5);  attn_probs_2 = value_states_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_11 = attn_output_10.view(1, 16, 1024, 64);  attn_output_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    attn_output_12 = attn_output_11.transpose(1, 2);  attn_output_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_13 = attn_output_12.reshape(1, 1024, 1024);  attn_output_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    hidden_states_26 = self.L__mod___model_encoder_layers_2_self_attn_out_proj(attn_output_13);  attn_output_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:335, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_27 = torch.nn.functional.dropout(hidden_states_26, p = 0.1, training = False);  hidden_states_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:336, code: hidden_states = residual + hidden_states
    residual_5 = residual_4 + hidden_states_27;  residual_4 = hidden_states_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:339, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_29 = self.L__mod___model_encoder_layers_2_final_layer_norm(residual_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:340, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_encoder_layers_2_fc1 = self.L__mod___model_encoder_layers_2_fc1(hidden_states_29);  hidden_states_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_30 = torch._C._nn.gelu(l__mod___model_encoder_layers_2_fc1);  l__mod___model_encoder_layers_2_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:341, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_31 = torch.nn.functional.dropout(hidden_states_30, p = 0.0, training = False);  hidden_states_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:342, code: hidden_states = self.fc2(hidden_states)
    hidden_states_32 = self.L__mod___model_encoder_layers_2_fc2(hidden_states_31);  hidden_states_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:343, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_33 = torch.nn.functional.dropout(hidden_states_32, p = 0.1, training = False);  hidden_states_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:344, code: hidden_states = residual + hidden_states
    residual_6 = residual_5 + hidden_states_33;  residual_5 = hidden_states_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:328, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_36 = self.L__mod___model_encoder_layers_3_self_attn_layer_norm(residual_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_encoder_layers_3_self_attn_q_proj = self.L__mod___model_encoder_layers_3_self_attn_q_proj(hidden_states_36)
    query_states_6 = l__mod___model_encoder_layers_3_self_attn_q_proj * 0.125;  l__mod___model_encoder_layers_3_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_3_self_attn_k_proj = self.L__mod___model_encoder_layers_3_self_attn_k_proj(hidden_states_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_16 = l__mod___model_encoder_layers_3_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_3_self_attn_k_proj = None
    transpose_15 = view_16.transpose(1, 2);  view_16 = None
    key_states_6 = transpose_15.contiguous();  transpose_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_3_self_attn_v_proj = self.L__mod___model_encoder_layers_3_self_attn_v_proj(hidden_states_36);  hidden_states_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_17 = l__mod___model_encoder_layers_3_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_3_self_attn_v_proj = None
    transpose_16 = view_17.transpose(1, 2);  view_17 = None
    value_states_6 = transpose_16.contiguous();  transpose_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_18 = query_states_6.view(1, 1024, 16, 64);  query_states_6 = None
    transpose_17 = view_18.transpose(1, 2);  view_18 = None
    contiguous_11 = transpose_17.contiguous();  transpose_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_7 = contiguous_11.view(16, -1, 64);  contiguous_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    key_states_7 = key_states_6.reshape(16, -1, 64);  key_states_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    value_states_7 = value_states_6.reshape(16, -1, 64);  value_states_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_18 = key_states_7.transpose(1, 2);  key_states_7 = None
    attn_weights_6 = torch.bmm(query_states_7, transpose_18);  query_states_7 = transpose_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_7 = torch.nn.functional.softmax(attn_weights_6, dim = -1);  attn_weights_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:270, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_3 = torch.nn.functional.dropout(attn_weights_7, p = 0.0, training = False);  attn_weights_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_15 = torch.bmm(attn_probs_3, value_states_7);  attn_probs_3 = value_states_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_16 = attn_output_15.view(1, 16, 1024, 64);  attn_output_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    attn_output_17 = attn_output_16.transpose(1, 2);  attn_output_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_18 = attn_output_17.reshape(1, 1024, 1024);  attn_output_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    hidden_states_37 = self.L__mod___model_encoder_layers_3_self_attn_out_proj(attn_output_18);  attn_output_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:335, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_38 = torch.nn.functional.dropout(hidden_states_37, p = 0.1, training = False);  hidden_states_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:336, code: hidden_states = residual + hidden_states
    residual_7 = residual_6 + hidden_states_38;  residual_6 = hidden_states_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:339, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_40 = self.L__mod___model_encoder_layers_3_final_layer_norm(residual_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:340, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_encoder_layers_3_fc1 = self.L__mod___model_encoder_layers_3_fc1(hidden_states_40);  hidden_states_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_41 = torch._C._nn.gelu(l__mod___model_encoder_layers_3_fc1);  l__mod___model_encoder_layers_3_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:341, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_42 = torch.nn.functional.dropout(hidden_states_41, p = 0.0, training = False);  hidden_states_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:342, code: hidden_states = self.fc2(hidden_states)
    hidden_states_43 = self.L__mod___model_encoder_layers_3_fc2(hidden_states_42);  hidden_states_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:343, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_44 = torch.nn.functional.dropout(hidden_states_43, p = 0.1, training = False);  hidden_states_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:344, code: hidden_states = residual + hidden_states
    residual_8 = residual_7 + hidden_states_44;  residual_7 = hidden_states_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:328, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_47 = self.L__mod___model_encoder_layers_4_self_attn_layer_norm(residual_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_encoder_layers_4_self_attn_q_proj = self.L__mod___model_encoder_layers_4_self_attn_q_proj(hidden_states_47)
    query_states_8 = l__mod___model_encoder_layers_4_self_attn_q_proj * 0.125;  l__mod___model_encoder_layers_4_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_4_self_attn_k_proj = self.L__mod___model_encoder_layers_4_self_attn_k_proj(hidden_states_47)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_21 = l__mod___model_encoder_layers_4_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_4_self_attn_k_proj = None
    transpose_20 = view_21.transpose(1, 2);  view_21 = None
    key_states_8 = transpose_20.contiguous();  transpose_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_4_self_attn_v_proj = self.L__mod___model_encoder_layers_4_self_attn_v_proj(hidden_states_47);  hidden_states_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_22 = l__mod___model_encoder_layers_4_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_4_self_attn_v_proj = None
    transpose_21 = view_22.transpose(1, 2);  view_22 = None
    value_states_8 = transpose_21.contiguous();  transpose_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_23 = query_states_8.view(1, 1024, 16, 64);  query_states_8 = None
    transpose_22 = view_23.transpose(1, 2);  view_23 = None
    contiguous_14 = transpose_22.contiguous();  transpose_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_9 = contiguous_14.view(16, -1, 64);  contiguous_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    key_states_9 = key_states_8.reshape(16, -1, 64);  key_states_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    value_states_9 = value_states_8.reshape(16, -1, 64);  value_states_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_23 = key_states_9.transpose(1, 2);  key_states_9 = None
    attn_weights_8 = torch.bmm(query_states_9, transpose_23);  query_states_9 = transpose_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_9 = torch.nn.functional.softmax(attn_weights_8, dim = -1);  attn_weights_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:270, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_4 = torch.nn.functional.dropout(attn_weights_9, p = 0.0, training = False);  attn_weights_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_20 = torch.bmm(attn_probs_4, value_states_9);  attn_probs_4 = value_states_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_21 = attn_output_20.view(1, 16, 1024, 64);  attn_output_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    attn_output_22 = attn_output_21.transpose(1, 2);  attn_output_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_23 = attn_output_22.reshape(1, 1024, 1024);  attn_output_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    hidden_states_48 = self.L__mod___model_encoder_layers_4_self_attn_out_proj(attn_output_23);  attn_output_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:335, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_49 = torch.nn.functional.dropout(hidden_states_48, p = 0.1, training = False);  hidden_states_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:336, code: hidden_states = residual + hidden_states
    residual_9 = residual_8 + hidden_states_49;  residual_8 = hidden_states_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:339, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_51 = self.L__mod___model_encoder_layers_4_final_layer_norm(residual_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:340, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_encoder_layers_4_fc1 = self.L__mod___model_encoder_layers_4_fc1(hidden_states_51);  hidden_states_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_52 = torch._C._nn.gelu(l__mod___model_encoder_layers_4_fc1);  l__mod___model_encoder_layers_4_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:341, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_53 = torch.nn.functional.dropout(hidden_states_52, p = 0.0, training = False);  hidden_states_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:342, code: hidden_states = self.fc2(hidden_states)
    hidden_states_54 = self.L__mod___model_encoder_layers_4_fc2(hidden_states_53);  hidden_states_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:343, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_55 = torch.nn.functional.dropout(hidden_states_54, p = 0.1, training = False);  hidden_states_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:344, code: hidden_states = residual + hidden_states
    residual_10 = residual_9 + hidden_states_55;  residual_9 = hidden_states_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:328, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_58 = self.L__mod___model_encoder_layers_5_self_attn_layer_norm(residual_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_encoder_layers_5_self_attn_q_proj = self.L__mod___model_encoder_layers_5_self_attn_q_proj(hidden_states_58)
    query_states_10 = l__mod___model_encoder_layers_5_self_attn_q_proj * 0.125;  l__mod___model_encoder_layers_5_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_5_self_attn_k_proj = self.L__mod___model_encoder_layers_5_self_attn_k_proj(hidden_states_58)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_26 = l__mod___model_encoder_layers_5_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_5_self_attn_k_proj = None
    transpose_25 = view_26.transpose(1, 2);  view_26 = None
    key_states_10 = transpose_25.contiguous();  transpose_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_5_self_attn_v_proj = self.L__mod___model_encoder_layers_5_self_attn_v_proj(hidden_states_58);  hidden_states_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_27 = l__mod___model_encoder_layers_5_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_5_self_attn_v_proj = None
    transpose_26 = view_27.transpose(1, 2);  view_27 = None
    value_states_10 = transpose_26.contiguous();  transpose_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_28 = query_states_10.view(1, 1024, 16, 64);  query_states_10 = None
    transpose_27 = view_28.transpose(1, 2);  view_28 = None
    contiguous_17 = transpose_27.contiguous();  transpose_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_11 = contiguous_17.view(16, -1, 64);  contiguous_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    key_states_11 = key_states_10.reshape(16, -1, 64);  key_states_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    value_states_11 = value_states_10.reshape(16, -1, 64);  value_states_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_28 = key_states_11.transpose(1, 2);  key_states_11 = None
    attn_weights_10 = torch.bmm(query_states_11, transpose_28);  query_states_11 = transpose_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_11 = torch.nn.functional.softmax(attn_weights_10, dim = -1);  attn_weights_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:270, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_5 = torch.nn.functional.dropout(attn_weights_11, p = 0.0, training = False);  attn_weights_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_25 = torch.bmm(attn_probs_5, value_states_11);  attn_probs_5 = value_states_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_26 = attn_output_25.view(1, 16, 1024, 64);  attn_output_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    attn_output_27 = attn_output_26.transpose(1, 2);  attn_output_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_28 = attn_output_27.reshape(1, 1024, 1024);  attn_output_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    hidden_states_59 = self.L__mod___model_encoder_layers_5_self_attn_out_proj(attn_output_28);  attn_output_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:335, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_60 = torch.nn.functional.dropout(hidden_states_59, p = 0.1, training = False);  hidden_states_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:336, code: hidden_states = residual + hidden_states
    residual_11 = residual_10 + hidden_states_60;  residual_10 = hidden_states_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:339, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_62 = self.L__mod___model_encoder_layers_5_final_layer_norm(residual_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:340, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_encoder_layers_5_fc1 = self.L__mod___model_encoder_layers_5_fc1(hidden_states_62);  hidden_states_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_63 = torch._C._nn.gelu(l__mod___model_encoder_layers_5_fc1);  l__mod___model_encoder_layers_5_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:341, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_64 = torch.nn.functional.dropout(hidden_states_63, p = 0.0, training = False);  hidden_states_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:342, code: hidden_states = self.fc2(hidden_states)
    hidden_states_65 = self.L__mod___model_encoder_layers_5_fc2(hidden_states_64);  hidden_states_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:343, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_66 = torch.nn.functional.dropout(hidden_states_65, p = 0.1, training = False);  hidden_states_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:344, code: hidden_states = residual + hidden_states
    residual_12 = residual_11 + hidden_states_66;  residual_11 = hidden_states_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:328, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_69 = self.L__mod___model_encoder_layers_6_self_attn_layer_norm(residual_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_encoder_layers_6_self_attn_q_proj = self.L__mod___model_encoder_layers_6_self_attn_q_proj(hidden_states_69)
    query_states_12 = l__mod___model_encoder_layers_6_self_attn_q_proj * 0.125;  l__mod___model_encoder_layers_6_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_6_self_attn_k_proj = self.L__mod___model_encoder_layers_6_self_attn_k_proj(hidden_states_69)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_31 = l__mod___model_encoder_layers_6_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_6_self_attn_k_proj = None
    transpose_30 = view_31.transpose(1, 2);  view_31 = None
    key_states_12 = transpose_30.contiguous();  transpose_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_6_self_attn_v_proj = self.L__mod___model_encoder_layers_6_self_attn_v_proj(hidden_states_69);  hidden_states_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_32 = l__mod___model_encoder_layers_6_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_6_self_attn_v_proj = None
    transpose_31 = view_32.transpose(1, 2);  view_32 = None
    value_states_12 = transpose_31.contiguous();  transpose_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_33 = query_states_12.view(1, 1024, 16, 64);  query_states_12 = None
    transpose_32 = view_33.transpose(1, 2);  view_33 = None
    contiguous_20 = transpose_32.contiguous();  transpose_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_13 = contiguous_20.view(16, -1, 64);  contiguous_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    key_states_13 = key_states_12.reshape(16, -1, 64);  key_states_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    value_states_13 = value_states_12.reshape(16, -1, 64);  value_states_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_33 = key_states_13.transpose(1, 2);  key_states_13 = None
    attn_weights_12 = torch.bmm(query_states_13, transpose_33);  query_states_13 = transpose_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_13 = torch.nn.functional.softmax(attn_weights_12, dim = -1);  attn_weights_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:270, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_6 = torch.nn.functional.dropout(attn_weights_13, p = 0.0, training = False);  attn_weights_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_30 = torch.bmm(attn_probs_6, value_states_13);  attn_probs_6 = value_states_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_31 = attn_output_30.view(1, 16, 1024, 64);  attn_output_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    attn_output_32 = attn_output_31.transpose(1, 2);  attn_output_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_33 = attn_output_32.reshape(1, 1024, 1024);  attn_output_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    hidden_states_70 = self.L__mod___model_encoder_layers_6_self_attn_out_proj(attn_output_33);  attn_output_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:335, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_71 = torch.nn.functional.dropout(hidden_states_70, p = 0.1, training = False);  hidden_states_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:336, code: hidden_states = residual + hidden_states
    residual_13 = residual_12 + hidden_states_71;  residual_12 = hidden_states_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:339, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_73 = self.L__mod___model_encoder_layers_6_final_layer_norm(residual_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:340, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_encoder_layers_6_fc1 = self.L__mod___model_encoder_layers_6_fc1(hidden_states_73);  hidden_states_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_74 = torch._C._nn.gelu(l__mod___model_encoder_layers_6_fc1);  l__mod___model_encoder_layers_6_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:341, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_75 = torch.nn.functional.dropout(hidden_states_74, p = 0.0, training = False);  hidden_states_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:342, code: hidden_states = self.fc2(hidden_states)
    hidden_states_76 = self.L__mod___model_encoder_layers_6_fc2(hidden_states_75);  hidden_states_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:343, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_77 = torch.nn.functional.dropout(hidden_states_76, p = 0.1, training = False);  hidden_states_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:344, code: hidden_states = residual + hidden_states
    residual_14 = residual_13 + hidden_states_77;  residual_13 = hidden_states_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:328, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_80 = self.L__mod___model_encoder_layers_7_self_attn_layer_norm(residual_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_encoder_layers_7_self_attn_q_proj = self.L__mod___model_encoder_layers_7_self_attn_q_proj(hidden_states_80)
    query_states_14 = l__mod___model_encoder_layers_7_self_attn_q_proj * 0.125;  l__mod___model_encoder_layers_7_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_7_self_attn_k_proj = self.L__mod___model_encoder_layers_7_self_attn_k_proj(hidden_states_80)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_36 = l__mod___model_encoder_layers_7_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_7_self_attn_k_proj = None
    transpose_35 = view_36.transpose(1, 2);  view_36 = None
    key_states_14 = transpose_35.contiguous();  transpose_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_7_self_attn_v_proj = self.L__mod___model_encoder_layers_7_self_attn_v_proj(hidden_states_80);  hidden_states_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_37 = l__mod___model_encoder_layers_7_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_7_self_attn_v_proj = None
    transpose_36 = view_37.transpose(1, 2);  view_37 = None
    value_states_14 = transpose_36.contiguous();  transpose_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_38 = query_states_14.view(1, 1024, 16, 64);  query_states_14 = None
    transpose_37 = view_38.transpose(1, 2);  view_38 = None
    contiguous_23 = transpose_37.contiguous();  transpose_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_15 = contiguous_23.view(16, -1, 64);  contiguous_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    key_states_15 = key_states_14.reshape(16, -1, 64);  key_states_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    value_states_15 = value_states_14.reshape(16, -1, 64);  value_states_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_38 = key_states_15.transpose(1, 2);  key_states_15 = None
    attn_weights_14 = torch.bmm(query_states_15, transpose_38);  query_states_15 = transpose_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_15 = torch.nn.functional.softmax(attn_weights_14, dim = -1);  attn_weights_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:270, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_7 = torch.nn.functional.dropout(attn_weights_15, p = 0.0, training = False);  attn_weights_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_35 = torch.bmm(attn_probs_7, value_states_15);  attn_probs_7 = value_states_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_36 = attn_output_35.view(1, 16, 1024, 64);  attn_output_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    attn_output_37 = attn_output_36.transpose(1, 2);  attn_output_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_38 = attn_output_37.reshape(1, 1024, 1024);  attn_output_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    hidden_states_81 = self.L__mod___model_encoder_layers_7_self_attn_out_proj(attn_output_38);  attn_output_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:335, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_82 = torch.nn.functional.dropout(hidden_states_81, p = 0.1, training = False);  hidden_states_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:336, code: hidden_states = residual + hidden_states
    residual_15 = residual_14 + hidden_states_82;  residual_14 = hidden_states_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:339, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_84 = self.L__mod___model_encoder_layers_7_final_layer_norm(residual_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:340, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_encoder_layers_7_fc1 = self.L__mod___model_encoder_layers_7_fc1(hidden_states_84);  hidden_states_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_85 = torch._C._nn.gelu(l__mod___model_encoder_layers_7_fc1);  l__mod___model_encoder_layers_7_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:341, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_86 = torch.nn.functional.dropout(hidden_states_85, p = 0.0, training = False);  hidden_states_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:342, code: hidden_states = self.fc2(hidden_states)
    hidden_states_87 = self.L__mod___model_encoder_layers_7_fc2(hidden_states_86);  hidden_states_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:343, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_88 = torch.nn.functional.dropout(hidden_states_87, p = 0.1, training = False);  hidden_states_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:344, code: hidden_states = residual + hidden_states
    residual_16 = residual_15 + hidden_states_88;  residual_15 = hidden_states_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:328, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_91 = self.L__mod___model_encoder_layers_8_self_attn_layer_norm(residual_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_encoder_layers_8_self_attn_q_proj = self.L__mod___model_encoder_layers_8_self_attn_q_proj(hidden_states_91)
    query_states_16 = l__mod___model_encoder_layers_8_self_attn_q_proj * 0.125;  l__mod___model_encoder_layers_8_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_8_self_attn_k_proj = self.L__mod___model_encoder_layers_8_self_attn_k_proj(hidden_states_91)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_41 = l__mod___model_encoder_layers_8_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_8_self_attn_k_proj = None
    transpose_40 = view_41.transpose(1, 2);  view_41 = None
    key_states_16 = transpose_40.contiguous();  transpose_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_8_self_attn_v_proj = self.L__mod___model_encoder_layers_8_self_attn_v_proj(hidden_states_91);  hidden_states_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_42 = l__mod___model_encoder_layers_8_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_8_self_attn_v_proj = None
    transpose_41 = view_42.transpose(1, 2);  view_42 = None
    value_states_16 = transpose_41.contiguous();  transpose_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_43 = query_states_16.view(1, 1024, 16, 64);  query_states_16 = None
    transpose_42 = view_43.transpose(1, 2);  view_43 = None
    contiguous_26 = transpose_42.contiguous();  transpose_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_17 = contiguous_26.view(16, -1, 64);  contiguous_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    key_states_17 = key_states_16.reshape(16, -1, 64);  key_states_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    value_states_17 = value_states_16.reshape(16, -1, 64);  value_states_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_43 = key_states_17.transpose(1, 2);  key_states_17 = None
    attn_weights_16 = torch.bmm(query_states_17, transpose_43);  query_states_17 = transpose_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_17 = torch.nn.functional.softmax(attn_weights_16, dim = -1);  attn_weights_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:270, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_8 = torch.nn.functional.dropout(attn_weights_17, p = 0.0, training = False);  attn_weights_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_40 = torch.bmm(attn_probs_8, value_states_17);  attn_probs_8 = value_states_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_41 = attn_output_40.view(1, 16, 1024, 64);  attn_output_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    attn_output_42 = attn_output_41.transpose(1, 2);  attn_output_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_43 = attn_output_42.reshape(1, 1024, 1024);  attn_output_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    hidden_states_92 = self.L__mod___model_encoder_layers_8_self_attn_out_proj(attn_output_43);  attn_output_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:335, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_93 = torch.nn.functional.dropout(hidden_states_92, p = 0.1, training = False);  hidden_states_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:336, code: hidden_states = residual + hidden_states
    residual_17 = residual_16 + hidden_states_93;  residual_16 = hidden_states_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:339, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_95 = self.L__mod___model_encoder_layers_8_final_layer_norm(residual_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:340, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_encoder_layers_8_fc1 = self.L__mod___model_encoder_layers_8_fc1(hidden_states_95);  hidden_states_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_96 = torch._C._nn.gelu(l__mod___model_encoder_layers_8_fc1);  l__mod___model_encoder_layers_8_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:341, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_97 = torch.nn.functional.dropout(hidden_states_96, p = 0.0, training = False);  hidden_states_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:342, code: hidden_states = self.fc2(hidden_states)
    hidden_states_98 = self.L__mod___model_encoder_layers_8_fc2(hidden_states_97);  hidden_states_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:343, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_99 = torch.nn.functional.dropout(hidden_states_98, p = 0.1, training = False);  hidden_states_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:344, code: hidden_states = residual + hidden_states
    residual_18 = residual_17 + hidden_states_99;  residual_17 = hidden_states_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:328, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_102 = self.L__mod___model_encoder_layers_9_self_attn_layer_norm(residual_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_encoder_layers_9_self_attn_q_proj = self.L__mod___model_encoder_layers_9_self_attn_q_proj(hidden_states_102)
    query_states_18 = l__mod___model_encoder_layers_9_self_attn_q_proj * 0.125;  l__mod___model_encoder_layers_9_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_9_self_attn_k_proj = self.L__mod___model_encoder_layers_9_self_attn_k_proj(hidden_states_102)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_46 = l__mod___model_encoder_layers_9_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_9_self_attn_k_proj = None
    transpose_45 = view_46.transpose(1, 2);  view_46 = None
    key_states_18 = transpose_45.contiguous();  transpose_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_9_self_attn_v_proj = self.L__mod___model_encoder_layers_9_self_attn_v_proj(hidden_states_102);  hidden_states_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_47 = l__mod___model_encoder_layers_9_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_9_self_attn_v_proj = None
    transpose_46 = view_47.transpose(1, 2);  view_47 = None
    value_states_18 = transpose_46.contiguous();  transpose_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_48 = query_states_18.view(1, 1024, 16, 64);  query_states_18 = None
    transpose_47 = view_48.transpose(1, 2);  view_48 = None
    contiguous_29 = transpose_47.contiguous();  transpose_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_19 = contiguous_29.view(16, -1, 64);  contiguous_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    key_states_19 = key_states_18.reshape(16, -1, 64);  key_states_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    value_states_19 = value_states_18.reshape(16, -1, 64);  value_states_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_48 = key_states_19.transpose(1, 2);  key_states_19 = None
    attn_weights_18 = torch.bmm(query_states_19, transpose_48);  query_states_19 = transpose_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_19 = torch.nn.functional.softmax(attn_weights_18, dim = -1);  attn_weights_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:270, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_9 = torch.nn.functional.dropout(attn_weights_19, p = 0.0, training = False);  attn_weights_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_45 = torch.bmm(attn_probs_9, value_states_19);  attn_probs_9 = value_states_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_46 = attn_output_45.view(1, 16, 1024, 64);  attn_output_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    attn_output_47 = attn_output_46.transpose(1, 2);  attn_output_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_48 = attn_output_47.reshape(1, 1024, 1024);  attn_output_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    hidden_states_103 = self.L__mod___model_encoder_layers_9_self_attn_out_proj(attn_output_48);  attn_output_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:335, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_104 = torch.nn.functional.dropout(hidden_states_103, p = 0.1, training = False);  hidden_states_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:336, code: hidden_states = residual + hidden_states
    residual_19 = residual_18 + hidden_states_104;  residual_18 = hidden_states_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:339, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_106 = self.L__mod___model_encoder_layers_9_final_layer_norm(residual_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:340, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_encoder_layers_9_fc1 = self.L__mod___model_encoder_layers_9_fc1(hidden_states_106);  hidden_states_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_107 = torch._C._nn.gelu(l__mod___model_encoder_layers_9_fc1);  l__mod___model_encoder_layers_9_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:341, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_108 = torch.nn.functional.dropout(hidden_states_107, p = 0.0, training = False);  hidden_states_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:342, code: hidden_states = self.fc2(hidden_states)
    hidden_states_109 = self.L__mod___model_encoder_layers_9_fc2(hidden_states_108);  hidden_states_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:343, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_110 = torch.nn.functional.dropout(hidden_states_109, p = 0.1, training = False);  hidden_states_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:344, code: hidden_states = residual + hidden_states
    residual_20 = residual_19 + hidden_states_110;  residual_19 = hidden_states_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:328, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_113 = self.L__mod___model_encoder_layers_10_self_attn_layer_norm(residual_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_encoder_layers_10_self_attn_q_proj = self.L__mod___model_encoder_layers_10_self_attn_q_proj(hidden_states_113)
    query_states_20 = l__mod___model_encoder_layers_10_self_attn_q_proj * 0.125;  l__mod___model_encoder_layers_10_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_10_self_attn_k_proj = self.L__mod___model_encoder_layers_10_self_attn_k_proj(hidden_states_113)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_51 = l__mod___model_encoder_layers_10_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_10_self_attn_k_proj = None
    transpose_50 = view_51.transpose(1, 2);  view_51 = None
    key_states_20 = transpose_50.contiguous();  transpose_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_10_self_attn_v_proj = self.L__mod___model_encoder_layers_10_self_attn_v_proj(hidden_states_113);  hidden_states_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_52 = l__mod___model_encoder_layers_10_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_10_self_attn_v_proj = None
    transpose_51 = view_52.transpose(1, 2);  view_52 = None
    value_states_20 = transpose_51.contiguous();  transpose_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_53 = query_states_20.view(1, 1024, 16, 64);  query_states_20 = None
    transpose_52 = view_53.transpose(1, 2);  view_53 = None
    contiguous_32 = transpose_52.contiguous();  transpose_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_21 = contiguous_32.view(16, -1, 64);  contiguous_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    key_states_21 = key_states_20.reshape(16, -1, 64);  key_states_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    value_states_21 = value_states_20.reshape(16, -1, 64);  value_states_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_53 = key_states_21.transpose(1, 2);  key_states_21 = None
    attn_weights_20 = torch.bmm(query_states_21, transpose_53);  query_states_21 = transpose_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_21 = torch.nn.functional.softmax(attn_weights_20, dim = -1);  attn_weights_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:270, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_10 = torch.nn.functional.dropout(attn_weights_21, p = 0.0, training = False);  attn_weights_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_50 = torch.bmm(attn_probs_10, value_states_21);  attn_probs_10 = value_states_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_51 = attn_output_50.view(1, 16, 1024, 64);  attn_output_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    attn_output_52 = attn_output_51.transpose(1, 2);  attn_output_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_53 = attn_output_52.reshape(1, 1024, 1024);  attn_output_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    hidden_states_114 = self.L__mod___model_encoder_layers_10_self_attn_out_proj(attn_output_53);  attn_output_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:335, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_115 = torch.nn.functional.dropout(hidden_states_114, p = 0.1, training = False);  hidden_states_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:336, code: hidden_states = residual + hidden_states
    residual_21 = residual_20 + hidden_states_115;  residual_20 = hidden_states_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:339, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_117 = self.L__mod___model_encoder_layers_10_final_layer_norm(residual_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:340, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_encoder_layers_10_fc1 = self.L__mod___model_encoder_layers_10_fc1(hidden_states_117);  hidden_states_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_118 = torch._C._nn.gelu(l__mod___model_encoder_layers_10_fc1);  l__mod___model_encoder_layers_10_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:341, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_119 = torch.nn.functional.dropout(hidden_states_118, p = 0.0, training = False);  hidden_states_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:342, code: hidden_states = self.fc2(hidden_states)
    hidden_states_120 = self.L__mod___model_encoder_layers_10_fc2(hidden_states_119);  hidden_states_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:343, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_121 = torch.nn.functional.dropout(hidden_states_120, p = 0.1, training = False);  hidden_states_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:344, code: hidden_states = residual + hidden_states
    residual_22 = residual_21 + hidden_states_121;  residual_21 = hidden_states_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:328, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_124 = self.L__mod___model_encoder_layers_11_self_attn_layer_norm(residual_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_encoder_layers_11_self_attn_q_proj = self.L__mod___model_encoder_layers_11_self_attn_q_proj(hidden_states_124)
    query_states_22 = l__mod___model_encoder_layers_11_self_attn_q_proj * 0.125;  l__mod___model_encoder_layers_11_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_11_self_attn_k_proj = self.L__mod___model_encoder_layers_11_self_attn_k_proj(hidden_states_124)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_56 = l__mod___model_encoder_layers_11_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_11_self_attn_k_proj = None
    transpose_55 = view_56.transpose(1, 2);  view_56 = None
    key_states_22 = transpose_55.contiguous();  transpose_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_11_self_attn_v_proj = self.L__mod___model_encoder_layers_11_self_attn_v_proj(hidden_states_124);  hidden_states_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_57 = l__mod___model_encoder_layers_11_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_11_self_attn_v_proj = None
    transpose_56 = view_57.transpose(1, 2);  view_57 = None
    value_states_22 = transpose_56.contiguous();  transpose_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_58 = query_states_22.view(1, 1024, 16, 64);  query_states_22 = None
    transpose_57 = view_58.transpose(1, 2);  view_58 = None
    contiguous_35 = transpose_57.contiguous();  transpose_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_23 = contiguous_35.view(16, -1, 64);  contiguous_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    key_states_23 = key_states_22.reshape(16, -1, 64);  key_states_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    value_states_23 = value_states_22.reshape(16, -1, 64);  value_states_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_58 = key_states_23.transpose(1, 2);  key_states_23 = None
    attn_weights_22 = torch.bmm(query_states_23, transpose_58);  query_states_23 = transpose_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_23 = torch.nn.functional.softmax(attn_weights_22, dim = -1);  attn_weights_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:270, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_11 = torch.nn.functional.dropout(attn_weights_23, p = 0.0, training = False);  attn_weights_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_55 = torch.bmm(attn_probs_11, value_states_23);  attn_probs_11 = value_states_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_56 = attn_output_55.view(1, 16, 1024, 64);  attn_output_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    attn_output_57 = attn_output_56.transpose(1, 2);  attn_output_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_58 = attn_output_57.reshape(1, 1024, 1024);  attn_output_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    hidden_states_125 = self.L__mod___model_encoder_layers_11_self_attn_out_proj(attn_output_58);  attn_output_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:335, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_126 = torch.nn.functional.dropout(hidden_states_125, p = 0.1, training = False);  hidden_states_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:336, code: hidden_states = residual + hidden_states
    residual_23 = residual_22 + hidden_states_126;  residual_22 = hidden_states_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:339, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_128 = self.L__mod___model_encoder_layers_11_final_layer_norm(residual_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:340, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_encoder_layers_11_fc1 = self.L__mod___model_encoder_layers_11_fc1(hidden_states_128);  hidden_states_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_129 = torch._C._nn.gelu(l__mod___model_encoder_layers_11_fc1);  l__mod___model_encoder_layers_11_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:341, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_130 = torch.nn.functional.dropout(hidden_states_129, p = 0.0, training = False);  hidden_states_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:342, code: hidden_states = self.fc2(hidden_states)
    hidden_states_131 = self.L__mod___model_encoder_layers_11_fc2(hidden_states_130);  hidden_states_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:343, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_132 = torch.nn.functional.dropout(hidden_states_131, p = 0.1, training = False);  hidden_states_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:344, code: hidden_states = residual + hidden_states
    hidden_states_134 = residual_23 + hidden_states_132;  residual_23 = hidden_states_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:857, code: hidden_states = self.layer_norm(hidden_states)
    hidden_states_135 = self.L__mod___model_encoder_layer_norm(hidden_states_134);  hidden_states_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:1026, code: input_ids = input_ids.view(-1, input_shape[-1])
    input_ids_1 = input_2.view(-1, 1024);  input_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:1037, code: inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
    l__mod___model_decoder_embed_tokens = self.L__mod___model_decoder_embed_tokens(input_ids_1);  input_ids_1 = None
    inputs_embeds_1 = l__mod___model_decoder_embed_tokens * 1.0;  l__mod___model_decoder_embed_tokens = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:89, code: mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask = torch.full((1024, 1024), -3.4028234663852886e+38, device = device(type='cpu'))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:90, code: mask_cond = torch.arange(mask.size(-1), device=device)
    mask_cond = torch.arange(1024, device = device(type='cpu'))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:91, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    add_26 = mask_cond + 1
    view_62 = add_26.view(1024, 1);  add_26 = None
    lt = mask_cond < view_62;  mask_cond = view_62 = None
    masked_fill__1 = mask.masked_fill_(lt, 0);  lt = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:92, code: mask = mask.to(dtype)
    mask_1 = mask.to(torch.float32);  mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:96, code: return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
    getitem_1 = mask_1[(None, None, slice(None, None, None), slice(None, None, None))];  mask_1 = None
    attention_mask = getitem_1.expand(1, 1, 1024, 1024);  getitem_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:131, code: past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
    l__mod___model_decoder_embed_positions_weight = self.L__mod___model_decoder_embed_positions_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:130, code: positions = torch.arange(
    arange_2 = torch.arange(0, 1024, dtype = torch.int64, device = device(type='cpu'))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:132, code: ).expand(bsz, -1)
    positions_1 = arange_2.expand(1, -1);  arange_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:134, code: return super().forward(positions + self.offset)
    add_27 = positions_1 + 2;  positions_1 = None
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    positions_2 = torch.nn.functional.embedding(add_27, l__mod___model_decoder_embed_positions_weight, None, None, 2.0, False, False);  add_27 = l__mod___model_decoder_embed_positions_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:1051, code: hidden_states = inputs_embeds + positions.to(inputs_embeds.device)
    to_2 = positions_2.to(device(type='cpu'));  positions_2 = None
    hidden_states_136 = inputs_embeds_1 + to_2;  inputs_embeds_1 = to_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:1052, code: hidden_states = self.layernorm_embedding(hidden_states)
    hidden_states_137 = self.L__mod___model_decoder_layernorm_embedding(hidden_states_136);  hidden_states_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:1054, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    residual_24 = torch.nn.functional.dropout(hidden_states_137, p = 0.1, training = False);  hidden_states_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:418, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_139 = self.L__mod___model_decoder_layers_0_self_attn_layer_norm(residual_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_0_self_attn_q_proj = self.L__mod___model_decoder_layers_0_self_attn_q_proj(hidden_states_139)
    query_states_24 = l__mod___model_decoder_layers_0_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_0_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_0_self_attn_k_proj = self.L__mod___model_decoder_layers_0_self_attn_k_proj(hidden_states_139)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_63 = l__mod___model_decoder_layers_0_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_0_self_attn_k_proj = None
    transpose_60 = view_63.transpose(1, 2);  view_63 = None
    key_states_24 = transpose_60.contiguous();  transpose_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_0_self_attn_v_proj = self.L__mod___model_decoder_layers_0_self_attn_v_proj(hidden_states_139);  hidden_states_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_64 = l__mod___model_decoder_layers_0_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_0_self_attn_v_proj = None
    transpose_61 = view_64.transpose(1, 2);  view_64 = None
    value_states_24 = transpose_61.contiguous();  transpose_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_65 = query_states_24.view(1, 1024, 16, 64);  query_states_24 = None
    transpose_62 = view_65.transpose(1, 2);  view_65 = None
    contiguous_38 = transpose_62.contiguous();  transpose_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_25 = contiguous_38.view(16, -1, 64);  contiguous_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    key_states_25 = key_states_24.reshape(16, -1, 64);  key_states_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    value_states_25 = value_states_24.reshape(16, -1, 64);  value_states_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_63 = key_states_25.transpose(1, 2);  key_states_25 = None
    attn_weights_24 = torch.bmm(query_states_25, transpose_63);  query_states_25 = transpose_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:246, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_67 = attn_weights_24.view(1, 16, 1024, 1024);  attn_weights_24 = None
    attn_weights_25 = view_67 + attention_mask;  view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:247, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_26 = attn_weights_25.view(16, 1024, 1024);  attn_weights_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_27 = torch.nn.functional.softmax(attn_weights_26, dim = -1);  attn_weights_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:270, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_12 = torch.nn.functional.dropout(attn_weights_27, p = 0.0, training = False);  attn_weights_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_60 = torch.bmm(attn_probs_12, value_states_25);  attn_probs_12 = value_states_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_61 = attn_output_60.view(1, 16, 1024, 64);  attn_output_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    attn_output_62 = attn_output_61.transpose(1, 2);  attn_output_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_63 = attn_output_62.reshape(1, 1024, 1024);  attn_output_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    hidden_states_140 = self.L__mod___model_decoder_layers_0_self_attn_out_proj(attn_output_63);  attn_output_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:431, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_141 = torch.nn.functional.dropout(hidden_states_140, p = 0.1, training = False);  hidden_states_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:432, code: hidden_states = residual + hidden_states
    residual_25 = residual_24 + hidden_states_141;  residual_24 = hidden_states_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:439, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_143 = self.L__mod___model_decoder_layers_0_encoder_attn_layer_norm(residual_25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_0_encoder_attn_q_proj = self.L__mod___model_decoder_layers_0_encoder_attn_q_proj(hidden_states_143);  hidden_states_143 = None
    query_states_26 = l__mod___model_decoder_layers_0_encoder_attn_q_proj * 0.125;  l__mod___model_decoder_layers_0_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:204, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_0_encoder_attn_k_proj = self.L__mod___model_decoder_layers_0_encoder_attn_k_proj(hidden_states_135)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_70 = l__mod___model_decoder_layers_0_encoder_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_0_encoder_attn_k_proj = None
    transpose_65 = view_70.transpose(1, 2);  view_70 = None
    key_states_26 = transpose_65.contiguous();  transpose_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:205, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_0_encoder_attn_v_proj = self.L__mod___model_decoder_layers_0_encoder_attn_v_proj(hidden_states_135)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_71 = l__mod___model_decoder_layers_0_encoder_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_0_encoder_attn_v_proj = None
    transpose_66 = view_71.transpose(1, 2);  view_71 = None
    value_states_26 = transpose_66.contiguous();  transpose_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_72 = query_states_26.view(1, 1024, 16, 64);  query_states_26 = None
    transpose_67 = view_72.transpose(1, 2);  view_72 = None
    contiguous_41 = transpose_67.contiguous();  transpose_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_27 = contiguous_41.view(16, -1, 64);  contiguous_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    key_states_27 = key_states_26.reshape(16, -1, 64);  key_states_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    value_states_27 = value_states_26.reshape(16, -1, 64);  value_states_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_68 = key_states_27.transpose(1, 2);  key_states_27 = None
    attn_weights_28 = torch.bmm(query_states_27, transpose_68);  query_states_27 = transpose_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_29 = torch.nn.functional.softmax(attn_weights_28, dim = -1);  attn_weights_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:270, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_13 = torch.nn.functional.dropout(attn_weights_29, p = 0.0, training = False);  attn_weights_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_65 = torch.bmm(attn_probs_13, value_states_27);  attn_probs_13 = value_states_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_66 = attn_output_65.view(1, 16, 1024, 64);  attn_output_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    attn_output_67 = attn_output_66.transpose(1, 2);  attn_output_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_68 = attn_output_67.reshape(1, 1024, 1024);  attn_output_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    hidden_states_144 = self.L__mod___model_decoder_layers_0_encoder_attn_out_proj(attn_output_68);  attn_output_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:451, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_145 = torch.nn.functional.dropout(hidden_states_144, p = 0.1, training = False);  hidden_states_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:452, code: hidden_states = residual + hidden_states
    residual_26 = residual_25 + hidden_states_145;  residual_25 = hidden_states_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:459, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_147 = self.L__mod___model_decoder_layers_0_final_layer_norm(residual_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_0_fc1 = self.L__mod___model_decoder_layers_0_fc1(hidden_states_147);  hidden_states_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_148 = torch._C._nn.gelu(l__mod___model_decoder_layers_0_fc1);  l__mod___model_decoder_layers_0_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:461, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_149 = torch.nn.functional.dropout(hidden_states_148, p = 0.0, training = False);  hidden_states_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:462, code: hidden_states = self.fc2(hidden_states)
    hidden_states_150 = self.L__mod___model_decoder_layers_0_fc2(hidden_states_149);  hidden_states_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:463, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_151 = torch.nn.functional.dropout(hidden_states_150, p = 0.1, training = False);  hidden_states_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:464, code: hidden_states = residual + hidden_states
    residual_27 = residual_26 + hidden_states_151;  residual_26 = hidden_states_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:418, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_154 = self.L__mod___model_decoder_layers_1_self_attn_layer_norm(residual_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_1_self_attn_q_proj = self.L__mod___model_decoder_layers_1_self_attn_q_proj(hidden_states_154)
    query_states_28 = l__mod___model_decoder_layers_1_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_1_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_1_self_attn_k_proj = self.L__mod___model_decoder_layers_1_self_attn_k_proj(hidden_states_154)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_75 = l__mod___model_decoder_layers_1_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_1_self_attn_k_proj = None
    transpose_70 = view_75.transpose(1, 2);  view_75 = None
    key_states_28 = transpose_70.contiguous();  transpose_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_1_self_attn_v_proj = self.L__mod___model_decoder_layers_1_self_attn_v_proj(hidden_states_154);  hidden_states_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_76 = l__mod___model_decoder_layers_1_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_1_self_attn_v_proj = None
    transpose_71 = view_76.transpose(1, 2);  view_76 = None
    value_states_28 = transpose_71.contiguous();  transpose_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_77 = query_states_28.view(1, 1024, 16, 64);  query_states_28 = None
    transpose_72 = view_77.transpose(1, 2);  view_77 = None
    contiguous_44 = transpose_72.contiguous();  transpose_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_29 = contiguous_44.view(16, -1, 64);  contiguous_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    key_states_29 = key_states_28.reshape(16, -1, 64);  key_states_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    value_states_29 = value_states_28.reshape(16, -1, 64);  value_states_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_73 = key_states_29.transpose(1, 2);  key_states_29 = None
    attn_weights_30 = torch.bmm(query_states_29, transpose_73);  query_states_29 = transpose_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:246, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_79 = attn_weights_30.view(1, 16, 1024, 1024);  attn_weights_30 = None
    attn_weights_31 = view_79 + attention_mask;  view_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:247, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_32 = attn_weights_31.view(16, 1024, 1024);  attn_weights_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_33 = torch.nn.functional.softmax(attn_weights_32, dim = -1);  attn_weights_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:270, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_14 = torch.nn.functional.dropout(attn_weights_33, p = 0.0, training = False);  attn_weights_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_70 = torch.bmm(attn_probs_14, value_states_29);  attn_probs_14 = value_states_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_71 = attn_output_70.view(1, 16, 1024, 64);  attn_output_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    attn_output_72 = attn_output_71.transpose(1, 2);  attn_output_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_73 = attn_output_72.reshape(1, 1024, 1024);  attn_output_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    hidden_states_155 = self.L__mod___model_decoder_layers_1_self_attn_out_proj(attn_output_73);  attn_output_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:431, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_156 = torch.nn.functional.dropout(hidden_states_155, p = 0.1, training = False);  hidden_states_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:432, code: hidden_states = residual + hidden_states
    residual_28 = residual_27 + hidden_states_156;  residual_27 = hidden_states_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:439, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_158 = self.L__mod___model_decoder_layers_1_encoder_attn_layer_norm(residual_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_1_encoder_attn_q_proj = self.L__mod___model_decoder_layers_1_encoder_attn_q_proj(hidden_states_158);  hidden_states_158 = None
    query_states_30 = l__mod___model_decoder_layers_1_encoder_attn_q_proj * 0.125;  l__mod___model_decoder_layers_1_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:204, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_1_encoder_attn_k_proj = self.L__mod___model_decoder_layers_1_encoder_attn_k_proj(hidden_states_135)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_82 = l__mod___model_decoder_layers_1_encoder_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_1_encoder_attn_k_proj = None
    transpose_75 = view_82.transpose(1, 2);  view_82 = None
    key_states_30 = transpose_75.contiguous();  transpose_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:205, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_1_encoder_attn_v_proj = self.L__mod___model_decoder_layers_1_encoder_attn_v_proj(hidden_states_135)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_83 = l__mod___model_decoder_layers_1_encoder_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_1_encoder_attn_v_proj = None
    transpose_76 = view_83.transpose(1, 2);  view_83 = None
    value_states_30 = transpose_76.contiguous();  transpose_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_84 = query_states_30.view(1, 1024, 16, 64);  query_states_30 = None
    transpose_77 = view_84.transpose(1, 2);  view_84 = None
    contiguous_47 = transpose_77.contiguous();  transpose_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_31 = contiguous_47.view(16, -1, 64);  contiguous_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    key_states_31 = key_states_30.reshape(16, -1, 64);  key_states_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    value_states_31 = value_states_30.reshape(16, -1, 64);  value_states_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_78 = key_states_31.transpose(1, 2);  key_states_31 = None
    attn_weights_34 = torch.bmm(query_states_31, transpose_78);  query_states_31 = transpose_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_35 = torch.nn.functional.softmax(attn_weights_34, dim = -1);  attn_weights_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:270, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_15 = torch.nn.functional.dropout(attn_weights_35, p = 0.0, training = False);  attn_weights_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_75 = torch.bmm(attn_probs_15, value_states_31);  attn_probs_15 = value_states_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_76 = attn_output_75.view(1, 16, 1024, 64);  attn_output_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    attn_output_77 = attn_output_76.transpose(1, 2);  attn_output_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_78 = attn_output_77.reshape(1, 1024, 1024);  attn_output_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    hidden_states_159 = self.L__mod___model_decoder_layers_1_encoder_attn_out_proj(attn_output_78);  attn_output_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:451, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_160 = torch.nn.functional.dropout(hidden_states_159, p = 0.1, training = False);  hidden_states_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:452, code: hidden_states = residual + hidden_states
    residual_29 = residual_28 + hidden_states_160;  residual_28 = hidden_states_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:459, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_162 = self.L__mod___model_decoder_layers_1_final_layer_norm(residual_29)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_1_fc1 = self.L__mod___model_decoder_layers_1_fc1(hidden_states_162);  hidden_states_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_163 = torch._C._nn.gelu(l__mod___model_decoder_layers_1_fc1);  l__mod___model_decoder_layers_1_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:461, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_164 = torch.nn.functional.dropout(hidden_states_163, p = 0.0, training = False);  hidden_states_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:462, code: hidden_states = self.fc2(hidden_states)
    hidden_states_165 = self.L__mod___model_decoder_layers_1_fc2(hidden_states_164);  hidden_states_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:463, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_166 = torch.nn.functional.dropout(hidden_states_165, p = 0.1, training = False);  hidden_states_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:464, code: hidden_states = residual + hidden_states
    residual_30 = residual_29 + hidden_states_166;  residual_29 = hidden_states_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:418, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_169 = self.L__mod___model_decoder_layers_2_self_attn_layer_norm(residual_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_2_self_attn_q_proj = self.L__mod___model_decoder_layers_2_self_attn_q_proj(hidden_states_169)
    query_states_32 = l__mod___model_decoder_layers_2_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_2_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_2_self_attn_k_proj = self.L__mod___model_decoder_layers_2_self_attn_k_proj(hidden_states_169)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_87 = l__mod___model_decoder_layers_2_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_2_self_attn_k_proj = None
    transpose_80 = view_87.transpose(1, 2);  view_87 = None
    key_states_32 = transpose_80.contiguous();  transpose_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_2_self_attn_v_proj = self.L__mod___model_decoder_layers_2_self_attn_v_proj(hidden_states_169);  hidden_states_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_88 = l__mod___model_decoder_layers_2_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_2_self_attn_v_proj = None
    transpose_81 = view_88.transpose(1, 2);  view_88 = None
    value_states_32 = transpose_81.contiguous();  transpose_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_89 = query_states_32.view(1, 1024, 16, 64);  query_states_32 = None
    transpose_82 = view_89.transpose(1, 2);  view_89 = None
    contiguous_50 = transpose_82.contiguous();  transpose_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_33 = contiguous_50.view(16, -1, 64);  contiguous_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    key_states_33 = key_states_32.reshape(16, -1, 64);  key_states_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    value_states_33 = value_states_32.reshape(16, -1, 64);  value_states_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_83 = key_states_33.transpose(1, 2);  key_states_33 = None
    attn_weights_36 = torch.bmm(query_states_33, transpose_83);  query_states_33 = transpose_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:246, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_91 = attn_weights_36.view(1, 16, 1024, 1024);  attn_weights_36 = None
    attn_weights_37 = view_91 + attention_mask;  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:247, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_38 = attn_weights_37.view(16, 1024, 1024);  attn_weights_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_39 = torch.nn.functional.softmax(attn_weights_38, dim = -1);  attn_weights_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:270, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_16 = torch.nn.functional.dropout(attn_weights_39, p = 0.0, training = False);  attn_weights_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_80 = torch.bmm(attn_probs_16, value_states_33);  attn_probs_16 = value_states_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_81 = attn_output_80.view(1, 16, 1024, 64);  attn_output_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    attn_output_82 = attn_output_81.transpose(1, 2);  attn_output_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_83 = attn_output_82.reshape(1, 1024, 1024);  attn_output_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    hidden_states_170 = self.L__mod___model_decoder_layers_2_self_attn_out_proj(attn_output_83);  attn_output_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:431, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_171 = torch.nn.functional.dropout(hidden_states_170, p = 0.1, training = False);  hidden_states_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:432, code: hidden_states = residual + hidden_states
    residual_31 = residual_30 + hidden_states_171;  residual_30 = hidden_states_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:439, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_173 = self.L__mod___model_decoder_layers_2_encoder_attn_layer_norm(residual_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_2_encoder_attn_q_proj = self.L__mod___model_decoder_layers_2_encoder_attn_q_proj(hidden_states_173);  hidden_states_173 = None
    query_states_34 = l__mod___model_decoder_layers_2_encoder_attn_q_proj * 0.125;  l__mod___model_decoder_layers_2_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:204, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_2_encoder_attn_k_proj = self.L__mod___model_decoder_layers_2_encoder_attn_k_proj(hidden_states_135)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_94 = l__mod___model_decoder_layers_2_encoder_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_2_encoder_attn_k_proj = None
    transpose_85 = view_94.transpose(1, 2);  view_94 = None
    key_states_34 = transpose_85.contiguous();  transpose_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:205, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_2_encoder_attn_v_proj = self.L__mod___model_decoder_layers_2_encoder_attn_v_proj(hidden_states_135)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_95 = l__mod___model_decoder_layers_2_encoder_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_2_encoder_attn_v_proj = None
    transpose_86 = view_95.transpose(1, 2);  view_95 = None
    value_states_34 = transpose_86.contiguous();  transpose_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_96 = query_states_34.view(1, 1024, 16, 64);  query_states_34 = None
    transpose_87 = view_96.transpose(1, 2);  view_96 = None
    contiguous_53 = transpose_87.contiguous();  transpose_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_35 = contiguous_53.view(16, -1, 64);  contiguous_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    key_states_35 = key_states_34.reshape(16, -1, 64);  key_states_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    value_states_35 = value_states_34.reshape(16, -1, 64);  value_states_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_88 = key_states_35.transpose(1, 2);  key_states_35 = None
    attn_weights_40 = torch.bmm(query_states_35, transpose_88);  query_states_35 = transpose_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_41 = torch.nn.functional.softmax(attn_weights_40, dim = -1);  attn_weights_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:270, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_17 = torch.nn.functional.dropout(attn_weights_41, p = 0.0, training = False);  attn_weights_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_85 = torch.bmm(attn_probs_17, value_states_35);  attn_probs_17 = value_states_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_86 = attn_output_85.view(1, 16, 1024, 64);  attn_output_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    attn_output_87 = attn_output_86.transpose(1, 2);  attn_output_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_88 = attn_output_87.reshape(1, 1024, 1024);  attn_output_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    hidden_states_174 = self.L__mod___model_decoder_layers_2_encoder_attn_out_proj(attn_output_88);  attn_output_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:451, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_175 = torch.nn.functional.dropout(hidden_states_174, p = 0.1, training = False);  hidden_states_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:452, code: hidden_states = residual + hidden_states
    residual_32 = residual_31 + hidden_states_175;  residual_31 = hidden_states_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:459, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_177 = self.L__mod___model_decoder_layers_2_final_layer_norm(residual_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_2_fc1 = self.L__mod___model_decoder_layers_2_fc1(hidden_states_177);  hidden_states_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_178 = torch._C._nn.gelu(l__mod___model_decoder_layers_2_fc1);  l__mod___model_decoder_layers_2_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:461, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_179 = torch.nn.functional.dropout(hidden_states_178, p = 0.0, training = False);  hidden_states_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:462, code: hidden_states = self.fc2(hidden_states)
    hidden_states_180 = self.L__mod___model_decoder_layers_2_fc2(hidden_states_179);  hidden_states_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:463, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_181 = torch.nn.functional.dropout(hidden_states_180, p = 0.1, training = False);  hidden_states_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:464, code: hidden_states = residual + hidden_states
    residual_33 = residual_32 + hidden_states_181;  residual_32 = hidden_states_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:418, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_184 = self.L__mod___model_decoder_layers_3_self_attn_layer_norm(residual_33)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_3_self_attn_q_proj = self.L__mod___model_decoder_layers_3_self_attn_q_proj(hidden_states_184)
    query_states_36 = l__mod___model_decoder_layers_3_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_3_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_3_self_attn_k_proj = self.L__mod___model_decoder_layers_3_self_attn_k_proj(hidden_states_184)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_99 = l__mod___model_decoder_layers_3_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_3_self_attn_k_proj = None
    transpose_90 = view_99.transpose(1, 2);  view_99 = None
    key_states_36 = transpose_90.contiguous();  transpose_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_3_self_attn_v_proj = self.L__mod___model_decoder_layers_3_self_attn_v_proj(hidden_states_184);  hidden_states_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_100 = l__mod___model_decoder_layers_3_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_3_self_attn_v_proj = None
    transpose_91 = view_100.transpose(1, 2);  view_100 = None
    value_states_36 = transpose_91.contiguous();  transpose_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_101 = query_states_36.view(1, 1024, 16, 64);  query_states_36 = None
    transpose_92 = view_101.transpose(1, 2);  view_101 = None
    contiguous_56 = transpose_92.contiguous();  transpose_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_37 = contiguous_56.view(16, -1, 64);  contiguous_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    key_states_37 = key_states_36.reshape(16, -1, 64);  key_states_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    value_states_37 = value_states_36.reshape(16, -1, 64);  value_states_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_93 = key_states_37.transpose(1, 2);  key_states_37 = None
    attn_weights_42 = torch.bmm(query_states_37, transpose_93);  query_states_37 = transpose_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:246, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_103 = attn_weights_42.view(1, 16, 1024, 1024);  attn_weights_42 = None
    attn_weights_43 = view_103 + attention_mask;  view_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:247, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_44 = attn_weights_43.view(16, 1024, 1024);  attn_weights_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_45 = torch.nn.functional.softmax(attn_weights_44, dim = -1);  attn_weights_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:270, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_18 = torch.nn.functional.dropout(attn_weights_45, p = 0.0, training = False);  attn_weights_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_90 = torch.bmm(attn_probs_18, value_states_37);  attn_probs_18 = value_states_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_91 = attn_output_90.view(1, 16, 1024, 64);  attn_output_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    attn_output_92 = attn_output_91.transpose(1, 2);  attn_output_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_93 = attn_output_92.reshape(1, 1024, 1024);  attn_output_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    hidden_states_185 = self.L__mod___model_decoder_layers_3_self_attn_out_proj(attn_output_93);  attn_output_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:431, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_186 = torch.nn.functional.dropout(hidden_states_185, p = 0.1, training = False);  hidden_states_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:432, code: hidden_states = residual + hidden_states
    residual_34 = residual_33 + hidden_states_186;  residual_33 = hidden_states_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:439, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_188 = self.L__mod___model_decoder_layers_3_encoder_attn_layer_norm(residual_34)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_3_encoder_attn_q_proj = self.L__mod___model_decoder_layers_3_encoder_attn_q_proj(hidden_states_188);  hidden_states_188 = None
    query_states_38 = l__mod___model_decoder_layers_3_encoder_attn_q_proj * 0.125;  l__mod___model_decoder_layers_3_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:204, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_3_encoder_attn_k_proj = self.L__mod___model_decoder_layers_3_encoder_attn_k_proj(hidden_states_135)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_106 = l__mod___model_decoder_layers_3_encoder_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_3_encoder_attn_k_proj = None
    transpose_95 = view_106.transpose(1, 2);  view_106 = None
    key_states_38 = transpose_95.contiguous();  transpose_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:205, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_3_encoder_attn_v_proj = self.L__mod___model_decoder_layers_3_encoder_attn_v_proj(hidden_states_135)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_107 = l__mod___model_decoder_layers_3_encoder_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_3_encoder_attn_v_proj = None
    transpose_96 = view_107.transpose(1, 2);  view_107 = None
    value_states_38 = transpose_96.contiguous();  transpose_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_108 = query_states_38.view(1, 1024, 16, 64);  query_states_38 = None
    transpose_97 = view_108.transpose(1, 2);  view_108 = None
    contiguous_59 = transpose_97.contiguous();  transpose_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_39 = contiguous_59.view(16, -1, 64);  contiguous_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    key_states_39 = key_states_38.reshape(16, -1, 64);  key_states_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    value_states_39 = value_states_38.reshape(16, -1, 64);  value_states_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_98 = key_states_39.transpose(1, 2);  key_states_39 = None
    attn_weights_46 = torch.bmm(query_states_39, transpose_98);  query_states_39 = transpose_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_47 = torch.nn.functional.softmax(attn_weights_46, dim = -1);  attn_weights_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:270, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_19 = torch.nn.functional.dropout(attn_weights_47, p = 0.0, training = False);  attn_weights_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_95 = torch.bmm(attn_probs_19, value_states_39);  attn_probs_19 = value_states_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_96 = attn_output_95.view(1, 16, 1024, 64);  attn_output_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    attn_output_97 = attn_output_96.transpose(1, 2);  attn_output_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_98 = attn_output_97.reshape(1, 1024, 1024);  attn_output_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    hidden_states_189 = self.L__mod___model_decoder_layers_3_encoder_attn_out_proj(attn_output_98);  attn_output_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:451, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_190 = torch.nn.functional.dropout(hidden_states_189, p = 0.1, training = False);  hidden_states_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:452, code: hidden_states = residual + hidden_states
    residual_35 = residual_34 + hidden_states_190;  residual_34 = hidden_states_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:459, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_192 = self.L__mod___model_decoder_layers_3_final_layer_norm(residual_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_3_fc1 = self.L__mod___model_decoder_layers_3_fc1(hidden_states_192);  hidden_states_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_193 = torch._C._nn.gelu(l__mod___model_decoder_layers_3_fc1);  l__mod___model_decoder_layers_3_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:461, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_194 = torch.nn.functional.dropout(hidden_states_193, p = 0.0, training = False);  hidden_states_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:462, code: hidden_states = self.fc2(hidden_states)
    hidden_states_195 = self.L__mod___model_decoder_layers_3_fc2(hidden_states_194);  hidden_states_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:463, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_196 = torch.nn.functional.dropout(hidden_states_195, p = 0.1, training = False);  hidden_states_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:464, code: hidden_states = residual + hidden_states
    residual_36 = residual_35 + hidden_states_196;  residual_35 = hidden_states_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:418, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_199 = self.L__mod___model_decoder_layers_4_self_attn_layer_norm(residual_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_4_self_attn_q_proj = self.L__mod___model_decoder_layers_4_self_attn_q_proj(hidden_states_199)
    query_states_40 = l__mod___model_decoder_layers_4_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_4_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_4_self_attn_k_proj = self.L__mod___model_decoder_layers_4_self_attn_k_proj(hidden_states_199)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_111 = l__mod___model_decoder_layers_4_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_4_self_attn_k_proj = None
    transpose_100 = view_111.transpose(1, 2);  view_111 = None
    key_states_40 = transpose_100.contiguous();  transpose_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_4_self_attn_v_proj = self.L__mod___model_decoder_layers_4_self_attn_v_proj(hidden_states_199);  hidden_states_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_112 = l__mod___model_decoder_layers_4_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_4_self_attn_v_proj = None
    transpose_101 = view_112.transpose(1, 2);  view_112 = None
    value_states_40 = transpose_101.contiguous();  transpose_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_113 = query_states_40.view(1, 1024, 16, 64);  query_states_40 = None
    transpose_102 = view_113.transpose(1, 2);  view_113 = None
    contiguous_62 = transpose_102.contiguous();  transpose_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_41 = contiguous_62.view(16, -1, 64);  contiguous_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    key_states_41 = key_states_40.reshape(16, -1, 64);  key_states_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    value_states_41 = value_states_40.reshape(16, -1, 64);  value_states_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_103 = key_states_41.transpose(1, 2);  key_states_41 = None
    attn_weights_48 = torch.bmm(query_states_41, transpose_103);  query_states_41 = transpose_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:246, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_115 = attn_weights_48.view(1, 16, 1024, 1024);  attn_weights_48 = None
    attn_weights_49 = view_115 + attention_mask;  view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:247, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_50 = attn_weights_49.view(16, 1024, 1024);  attn_weights_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_51 = torch.nn.functional.softmax(attn_weights_50, dim = -1);  attn_weights_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:270, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_20 = torch.nn.functional.dropout(attn_weights_51, p = 0.0, training = False);  attn_weights_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_100 = torch.bmm(attn_probs_20, value_states_41);  attn_probs_20 = value_states_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_101 = attn_output_100.view(1, 16, 1024, 64);  attn_output_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    attn_output_102 = attn_output_101.transpose(1, 2);  attn_output_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_103 = attn_output_102.reshape(1, 1024, 1024);  attn_output_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    hidden_states_200 = self.L__mod___model_decoder_layers_4_self_attn_out_proj(attn_output_103);  attn_output_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:431, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_201 = torch.nn.functional.dropout(hidden_states_200, p = 0.1, training = False);  hidden_states_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:432, code: hidden_states = residual + hidden_states
    residual_37 = residual_36 + hidden_states_201;  residual_36 = hidden_states_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:439, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_203 = self.L__mod___model_decoder_layers_4_encoder_attn_layer_norm(residual_37)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_4_encoder_attn_q_proj = self.L__mod___model_decoder_layers_4_encoder_attn_q_proj(hidden_states_203);  hidden_states_203 = None
    query_states_42 = l__mod___model_decoder_layers_4_encoder_attn_q_proj * 0.125;  l__mod___model_decoder_layers_4_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:204, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_4_encoder_attn_k_proj = self.L__mod___model_decoder_layers_4_encoder_attn_k_proj(hidden_states_135)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_118 = l__mod___model_decoder_layers_4_encoder_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_4_encoder_attn_k_proj = None
    transpose_105 = view_118.transpose(1, 2);  view_118 = None
    key_states_42 = transpose_105.contiguous();  transpose_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:205, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_4_encoder_attn_v_proj = self.L__mod___model_decoder_layers_4_encoder_attn_v_proj(hidden_states_135)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_119 = l__mod___model_decoder_layers_4_encoder_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_4_encoder_attn_v_proj = None
    transpose_106 = view_119.transpose(1, 2);  view_119 = None
    value_states_42 = transpose_106.contiguous();  transpose_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_120 = query_states_42.view(1, 1024, 16, 64);  query_states_42 = None
    transpose_107 = view_120.transpose(1, 2);  view_120 = None
    contiguous_65 = transpose_107.contiguous();  transpose_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_43 = contiguous_65.view(16, -1, 64);  contiguous_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    key_states_43 = key_states_42.reshape(16, -1, 64);  key_states_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    value_states_43 = value_states_42.reshape(16, -1, 64);  value_states_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_108 = key_states_43.transpose(1, 2);  key_states_43 = None
    attn_weights_52 = torch.bmm(query_states_43, transpose_108);  query_states_43 = transpose_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_53 = torch.nn.functional.softmax(attn_weights_52, dim = -1);  attn_weights_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:270, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_21 = torch.nn.functional.dropout(attn_weights_53, p = 0.0, training = False);  attn_weights_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_105 = torch.bmm(attn_probs_21, value_states_43);  attn_probs_21 = value_states_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_106 = attn_output_105.view(1, 16, 1024, 64);  attn_output_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    attn_output_107 = attn_output_106.transpose(1, 2);  attn_output_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_108 = attn_output_107.reshape(1, 1024, 1024);  attn_output_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    hidden_states_204 = self.L__mod___model_decoder_layers_4_encoder_attn_out_proj(attn_output_108);  attn_output_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:451, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_205 = torch.nn.functional.dropout(hidden_states_204, p = 0.1, training = False);  hidden_states_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:452, code: hidden_states = residual + hidden_states
    residual_38 = residual_37 + hidden_states_205;  residual_37 = hidden_states_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:459, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_207 = self.L__mod___model_decoder_layers_4_final_layer_norm(residual_38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_4_fc1 = self.L__mod___model_decoder_layers_4_fc1(hidden_states_207);  hidden_states_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_208 = torch._C._nn.gelu(l__mod___model_decoder_layers_4_fc1);  l__mod___model_decoder_layers_4_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:461, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_209 = torch.nn.functional.dropout(hidden_states_208, p = 0.0, training = False);  hidden_states_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:462, code: hidden_states = self.fc2(hidden_states)
    hidden_states_210 = self.L__mod___model_decoder_layers_4_fc2(hidden_states_209);  hidden_states_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:463, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_211 = torch.nn.functional.dropout(hidden_states_210, p = 0.1, training = False);  hidden_states_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:464, code: hidden_states = residual + hidden_states
    residual_39 = residual_38 + hidden_states_211;  residual_38 = hidden_states_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:418, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_214 = self.L__mod___model_decoder_layers_5_self_attn_layer_norm(residual_39)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_5_self_attn_q_proj = self.L__mod___model_decoder_layers_5_self_attn_q_proj(hidden_states_214)
    query_states_44 = l__mod___model_decoder_layers_5_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_5_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_5_self_attn_k_proj = self.L__mod___model_decoder_layers_5_self_attn_k_proj(hidden_states_214)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_123 = l__mod___model_decoder_layers_5_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_5_self_attn_k_proj = None
    transpose_110 = view_123.transpose(1, 2);  view_123 = None
    key_states_44 = transpose_110.contiguous();  transpose_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_5_self_attn_v_proj = self.L__mod___model_decoder_layers_5_self_attn_v_proj(hidden_states_214);  hidden_states_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_124 = l__mod___model_decoder_layers_5_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_5_self_attn_v_proj = None
    transpose_111 = view_124.transpose(1, 2);  view_124 = None
    value_states_44 = transpose_111.contiguous();  transpose_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_125 = query_states_44.view(1, 1024, 16, 64);  query_states_44 = None
    transpose_112 = view_125.transpose(1, 2);  view_125 = None
    contiguous_68 = transpose_112.contiguous();  transpose_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_45 = contiguous_68.view(16, -1, 64);  contiguous_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    key_states_45 = key_states_44.reshape(16, -1, 64);  key_states_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    value_states_45 = value_states_44.reshape(16, -1, 64);  value_states_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_113 = key_states_45.transpose(1, 2);  key_states_45 = None
    attn_weights_54 = torch.bmm(query_states_45, transpose_113);  query_states_45 = transpose_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:246, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_127 = attn_weights_54.view(1, 16, 1024, 1024);  attn_weights_54 = None
    attn_weights_55 = view_127 + attention_mask;  view_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:247, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_56 = attn_weights_55.view(16, 1024, 1024);  attn_weights_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_57 = torch.nn.functional.softmax(attn_weights_56, dim = -1);  attn_weights_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:270, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_22 = torch.nn.functional.dropout(attn_weights_57, p = 0.0, training = False);  attn_weights_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_110 = torch.bmm(attn_probs_22, value_states_45);  attn_probs_22 = value_states_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_111 = attn_output_110.view(1, 16, 1024, 64);  attn_output_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    attn_output_112 = attn_output_111.transpose(1, 2);  attn_output_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_113 = attn_output_112.reshape(1, 1024, 1024);  attn_output_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    hidden_states_215 = self.L__mod___model_decoder_layers_5_self_attn_out_proj(attn_output_113);  attn_output_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:431, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_216 = torch.nn.functional.dropout(hidden_states_215, p = 0.1, training = False);  hidden_states_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:432, code: hidden_states = residual + hidden_states
    residual_40 = residual_39 + hidden_states_216;  residual_39 = hidden_states_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:439, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_218 = self.L__mod___model_decoder_layers_5_encoder_attn_layer_norm(residual_40)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_5_encoder_attn_q_proj = self.L__mod___model_decoder_layers_5_encoder_attn_q_proj(hidden_states_218);  hidden_states_218 = None
    query_states_46 = l__mod___model_decoder_layers_5_encoder_attn_q_proj * 0.125;  l__mod___model_decoder_layers_5_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:204, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_5_encoder_attn_k_proj = self.L__mod___model_decoder_layers_5_encoder_attn_k_proj(hidden_states_135)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_130 = l__mod___model_decoder_layers_5_encoder_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_5_encoder_attn_k_proj = None
    transpose_115 = view_130.transpose(1, 2);  view_130 = None
    key_states_46 = transpose_115.contiguous();  transpose_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:205, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_5_encoder_attn_v_proj = self.L__mod___model_decoder_layers_5_encoder_attn_v_proj(hidden_states_135)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_131 = l__mod___model_decoder_layers_5_encoder_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_5_encoder_attn_v_proj = None
    transpose_116 = view_131.transpose(1, 2);  view_131 = None
    value_states_46 = transpose_116.contiguous();  transpose_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_132 = query_states_46.view(1, 1024, 16, 64);  query_states_46 = None
    transpose_117 = view_132.transpose(1, 2);  view_132 = None
    contiguous_71 = transpose_117.contiguous();  transpose_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_47 = contiguous_71.view(16, -1, 64);  contiguous_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    key_states_47 = key_states_46.reshape(16, -1, 64);  key_states_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    value_states_47 = value_states_46.reshape(16, -1, 64);  value_states_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_118 = key_states_47.transpose(1, 2);  key_states_47 = None
    attn_weights_58 = torch.bmm(query_states_47, transpose_118);  query_states_47 = transpose_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_59 = torch.nn.functional.softmax(attn_weights_58, dim = -1);  attn_weights_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:270, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_23 = torch.nn.functional.dropout(attn_weights_59, p = 0.0, training = False);  attn_weights_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_115 = torch.bmm(attn_probs_23, value_states_47);  attn_probs_23 = value_states_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_116 = attn_output_115.view(1, 16, 1024, 64);  attn_output_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    attn_output_117 = attn_output_116.transpose(1, 2);  attn_output_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_118 = attn_output_117.reshape(1, 1024, 1024);  attn_output_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    hidden_states_219 = self.L__mod___model_decoder_layers_5_encoder_attn_out_proj(attn_output_118);  attn_output_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:451, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_220 = torch.nn.functional.dropout(hidden_states_219, p = 0.1, training = False);  hidden_states_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:452, code: hidden_states = residual + hidden_states
    residual_41 = residual_40 + hidden_states_220;  residual_40 = hidden_states_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:459, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_222 = self.L__mod___model_decoder_layers_5_final_layer_norm(residual_41)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_5_fc1 = self.L__mod___model_decoder_layers_5_fc1(hidden_states_222);  hidden_states_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_223 = torch._C._nn.gelu(l__mod___model_decoder_layers_5_fc1);  l__mod___model_decoder_layers_5_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:461, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_224 = torch.nn.functional.dropout(hidden_states_223, p = 0.0, training = False);  hidden_states_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:462, code: hidden_states = self.fc2(hidden_states)
    hidden_states_225 = self.L__mod___model_decoder_layers_5_fc2(hidden_states_224);  hidden_states_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:463, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_226 = torch.nn.functional.dropout(hidden_states_225, p = 0.1, training = False);  hidden_states_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:464, code: hidden_states = residual + hidden_states
    residual_42 = residual_41 + hidden_states_226;  residual_41 = hidden_states_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:418, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_229 = self.L__mod___model_decoder_layers_6_self_attn_layer_norm(residual_42)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_6_self_attn_q_proj = self.L__mod___model_decoder_layers_6_self_attn_q_proj(hidden_states_229)
    query_states_48 = l__mod___model_decoder_layers_6_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_6_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_6_self_attn_k_proj = self.L__mod___model_decoder_layers_6_self_attn_k_proj(hidden_states_229)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_135 = l__mod___model_decoder_layers_6_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_6_self_attn_k_proj = None
    transpose_120 = view_135.transpose(1, 2);  view_135 = None
    key_states_48 = transpose_120.contiguous();  transpose_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_6_self_attn_v_proj = self.L__mod___model_decoder_layers_6_self_attn_v_proj(hidden_states_229);  hidden_states_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_136 = l__mod___model_decoder_layers_6_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_6_self_attn_v_proj = None
    transpose_121 = view_136.transpose(1, 2);  view_136 = None
    value_states_48 = transpose_121.contiguous();  transpose_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_137 = query_states_48.view(1, 1024, 16, 64);  query_states_48 = None
    transpose_122 = view_137.transpose(1, 2);  view_137 = None
    contiguous_74 = transpose_122.contiguous();  transpose_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_49 = contiguous_74.view(16, -1, 64);  contiguous_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    key_states_49 = key_states_48.reshape(16, -1, 64);  key_states_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    value_states_49 = value_states_48.reshape(16, -1, 64);  value_states_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_123 = key_states_49.transpose(1, 2);  key_states_49 = None
    attn_weights_60 = torch.bmm(query_states_49, transpose_123);  query_states_49 = transpose_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:246, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_139 = attn_weights_60.view(1, 16, 1024, 1024);  attn_weights_60 = None
    attn_weights_61 = view_139 + attention_mask;  view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:247, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_62 = attn_weights_61.view(16, 1024, 1024);  attn_weights_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_63 = torch.nn.functional.softmax(attn_weights_62, dim = -1);  attn_weights_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:270, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_24 = torch.nn.functional.dropout(attn_weights_63, p = 0.0, training = False);  attn_weights_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_120 = torch.bmm(attn_probs_24, value_states_49);  attn_probs_24 = value_states_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_121 = attn_output_120.view(1, 16, 1024, 64);  attn_output_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    attn_output_122 = attn_output_121.transpose(1, 2);  attn_output_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_123 = attn_output_122.reshape(1, 1024, 1024);  attn_output_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    hidden_states_230 = self.L__mod___model_decoder_layers_6_self_attn_out_proj(attn_output_123);  attn_output_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:431, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_231 = torch.nn.functional.dropout(hidden_states_230, p = 0.1, training = False);  hidden_states_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:432, code: hidden_states = residual + hidden_states
    residual_43 = residual_42 + hidden_states_231;  residual_42 = hidden_states_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:439, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_233 = self.L__mod___model_decoder_layers_6_encoder_attn_layer_norm(residual_43)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_6_encoder_attn_q_proj = self.L__mod___model_decoder_layers_6_encoder_attn_q_proj(hidden_states_233);  hidden_states_233 = None
    query_states_50 = l__mod___model_decoder_layers_6_encoder_attn_q_proj * 0.125;  l__mod___model_decoder_layers_6_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:204, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_6_encoder_attn_k_proj = self.L__mod___model_decoder_layers_6_encoder_attn_k_proj(hidden_states_135)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_142 = l__mod___model_decoder_layers_6_encoder_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_6_encoder_attn_k_proj = None
    transpose_125 = view_142.transpose(1, 2);  view_142 = None
    key_states_50 = transpose_125.contiguous();  transpose_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:205, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_6_encoder_attn_v_proj = self.L__mod___model_decoder_layers_6_encoder_attn_v_proj(hidden_states_135)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_143 = l__mod___model_decoder_layers_6_encoder_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_6_encoder_attn_v_proj = None
    transpose_126 = view_143.transpose(1, 2);  view_143 = None
    value_states_50 = transpose_126.contiguous();  transpose_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_144 = query_states_50.view(1, 1024, 16, 64);  query_states_50 = None
    transpose_127 = view_144.transpose(1, 2);  view_144 = None
    contiguous_77 = transpose_127.contiguous();  transpose_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_51 = contiguous_77.view(16, -1, 64);  contiguous_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    key_states_51 = key_states_50.reshape(16, -1, 64);  key_states_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    value_states_51 = value_states_50.reshape(16, -1, 64);  value_states_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_128 = key_states_51.transpose(1, 2);  key_states_51 = None
    attn_weights_64 = torch.bmm(query_states_51, transpose_128);  query_states_51 = transpose_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_65 = torch.nn.functional.softmax(attn_weights_64, dim = -1);  attn_weights_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:270, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_25 = torch.nn.functional.dropout(attn_weights_65, p = 0.0, training = False);  attn_weights_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_125 = torch.bmm(attn_probs_25, value_states_51);  attn_probs_25 = value_states_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_126 = attn_output_125.view(1, 16, 1024, 64);  attn_output_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    attn_output_127 = attn_output_126.transpose(1, 2);  attn_output_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_128 = attn_output_127.reshape(1, 1024, 1024);  attn_output_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    hidden_states_234 = self.L__mod___model_decoder_layers_6_encoder_attn_out_proj(attn_output_128);  attn_output_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:451, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_235 = torch.nn.functional.dropout(hidden_states_234, p = 0.1, training = False);  hidden_states_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:452, code: hidden_states = residual + hidden_states
    residual_44 = residual_43 + hidden_states_235;  residual_43 = hidden_states_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:459, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_237 = self.L__mod___model_decoder_layers_6_final_layer_norm(residual_44)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_6_fc1 = self.L__mod___model_decoder_layers_6_fc1(hidden_states_237);  hidden_states_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_238 = torch._C._nn.gelu(l__mod___model_decoder_layers_6_fc1);  l__mod___model_decoder_layers_6_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:461, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_239 = torch.nn.functional.dropout(hidden_states_238, p = 0.0, training = False);  hidden_states_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:462, code: hidden_states = self.fc2(hidden_states)
    hidden_states_240 = self.L__mod___model_decoder_layers_6_fc2(hidden_states_239);  hidden_states_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:463, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_241 = torch.nn.functional.dropout(hidden_states_240, p = 0.1, training = False);  hidden_states_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:464, code: hidden_states = residual + hidden_states
    residual_45 = residual_44 + hidden_states_241;  residual_44 = hidden_states_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:418, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_244 = self.L__mod___model_decoder_layers_7_self_attn_layer_norm(residual_45)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_7_self_attn_q_proj = self.L__mod___model_decoder_layers_7_self_attn_q_proj(hidden_states_244)
    query_states_52 = l__mod___model_decoder_layers_7_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_7_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_7_self_attn_k_proj = self.L__mod___model_decoder_layers_7_self_attn_k_proj(hidden_states_244)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_147 = l__mod___model_decoder_layers_7_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_7_self_attn_k_proj = None
    transpose_130 = view_147.transpose(1, 2);  view_147 = None
    key_states_52 = transpose_130.contiguous();  transpose_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_7_self_attn_v_proj = self.L__mod___model_decoder_layers_7_self_attn_v_proj(hidden_states_244);  hidden_states_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_148 = l__mod___model_decoder_layers_7_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_7_self_attn_v_proj = None
    transpose_131 = view_148.transpose(1, 2);  view_148 = None
    value_states_52 = transpose_131.contiguous();  transpose_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_149 = query_states_52.view(1, 1024, 16, 64);  query_states_52 = None
    transpose_132 = view_149.transpose(1, 2);  view_149 = None
    contiguous_80 = transpose_132.contiguous();  transpose_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_53 = contiguous_80.view(16, -1, 64);  contiguous_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    key_states_53 = key_states_52.reshape(16, -1, 64);  key_states_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    value_states_53 = value_states_52.reshape(16, -1, 64);  value_states_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_133 = key_states_53.transpose(1, 2);  key_states_53 = None
    attn_weights_66 = torch.bmm(query_states_53, transpose_133);  query_states_53 = transpose_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:246, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_151 = attn_weights_66.view(1, 16, 1024, 1024);  attn_weights_66 = None
    attn_weights_67 = view_151 + attention_mask;  view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:247, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_68 = attn_weights_67.view(16, 1024, 1024);  attn_weights_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_69 = torch.nn.functional.softmax(attn_weights_68, dim = -1);  attn_weights_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:270, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_26 = torch.nn.functional.dropout(attn_weights_69, p = 0.0, training = False);  attn_weights_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_130 = torch.bmm(attn_probs_26, value_states_53);  attn_probs_26 = value_states_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_131 = attn_output_130.view(1, 16, 1024, 64);  attn_output_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    attn_output_132 = attn_output_131.transpose(1, 2);  attn_output_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_133 = attn_output_132.reshape(1, 1024, 1024);  attn_output_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    hidden_states_245 = self.L__mod___model_decoder_layers_7_self_attn_out_proj(attn_output_133);  attn_output_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:431, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_246 = torch.nn.functional.dropout(hidden_states_245, p = 0.1, training = False);  hidden_states_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:432, code: hidden_states = residual + hidden_states
    residual_46 = residual_45 + hidden_states_246;  residual_45 = hidden_states_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:439, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_248 = self.L__mod___model_decoder_layers_7_encoder_attn_layer_norm(residual_46)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_7_encoder_attn_q_proj = self.L__mod___model_decoder_layers_7_encoder_attn_q_proj(hidden_states_248);  hidden_states_248 = None
    query_states_54 = l__mod___model_decoder_layers_7_encoder_attn_q_proj * 0.125;  l__mod___model_decoder_layers_7_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:204, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_7_encoder_attn_k_proj = self.L__mod___model_decoder_layers_7_encoder_attn_k_proj(hidden_states_135)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_154 = l__mod___model_decoder_layers_7_encoder_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_7_encoder_attn_k_proj = None
    transpose_135 = view_154.transpose(1, 2);  view_154 = None
    key_states_54 = transpose_135.contiguous();  transpose_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:205, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_7_encoder_attn_v_proj = self.L__mod___model_decoder_layers_7_encoder_attn_v_proj(hidden_states_135)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_155 = l__mod___model_decoder_layers_7_encoder_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_7_encoder_attn_v_proj = None
    transpose_136 = view_155.transpose(1, 2);  view_155 = None
    value_states_54 = transpose_136.contiguous();  transpose_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_156 = query_states_54.view(1, 1024, 16, 64);  query_states_54 = None
    transpose_137 = view_156.transpose(1, 2);  view_156 = None
    contiguous_83 = transpose_137.contiguous();  transpose_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_55 = contiguous_83.view(16, -1, 64);  contiguous_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    key_states_55 = key_states_54.reshape(16, -1, 64);  key_states_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    value_states_55 = value_states_54.reshape(16, -1, 64);  value_states_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_138 = key_states_55.transpose(1, 2);  key_states_55 = None
    attn_weights_70 = torch.bmm(query_states_55, transpose_138);  query_states_55 = transpose_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_71 = torch.nn.functional.softmax(attn_weights_70, dim = -1);  attn_weights_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:270, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_27 = torch.nn.functional.dropout(attn_weights_71, p = 0.0, training = False);  attn_weights_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_135 = torch.bmm(attn_probs_27, value_states_55);  attn_probs_27 = value_states_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_136 = attn_output_135.view(1, 16, 1024, 64);  attn_output_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    attn_output_137 = attn_output_136.transpose(1, 2);  attn_output_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_138 = attn_output_137.reshape(1, 1024, 1024);  attn_output_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    hidden_states_249 = self.L__mod___model_decoder_layers_7_encoder_attn_out_proj(attn_output_138);  attn_output_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:451, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_250 = torch.nn.functional.dropout(hidden_states_249, p = 0.1, training = False);  hidden_states_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:452, code: hidden_states = residual + hidden_states
    residual_47 = residual_46 + hidden_states_250;  residual_46 = hidden_states_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:459, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_252 = self.L__mod___model_decoder_layers_7_final_layer_norm(residual_47)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_7_fc1 = self.L__mod___model_decoder_layers_7_fc1(hidden_states_252);  hidden_states_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_253 = torch._C._nn.gelu(l__mod___model_decoder_layers_7_fc1);  l__mod___model_decoder_layers_7_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:461, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_254 = torch.nn.functional.dropout(hidden_states_253, p = 0.0, training = False);  hidden_states_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:462, code: hidden_states = self.fc2(hidden_states)
    hidden_states_255 = self.L__mod___model_decoder_layers_7_fc2(hidden_states_254);  hidden_states_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:463, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_256 = torch.nn.functional.dropout(hidden_states_255, p = 0.1, training = False);  hidden_states_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:464, code: hidden_states = residual + hidden_states
    residual_48 = residual_47 + hidden_states_256;  residual_47 = hidden_states_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:418, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_259 = self.L__mod___model_decoder_layers_8_self_attn_layer_norm(residual_48)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_8_self_attn_q_proj = self.L__mod___model_decoder_layers_8_self_attn_q_proj(hidden_states_259)
    query_states_56 = l__mod___model_decoder_layers_8_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_8_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_8_self_attn_k_proj = self.L__mod___model_decoder_layers_8_self_attn_k_proj(hidden_states_259)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_159 = l__mod___model_decoder_layers_8_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_8_self_attn_k_proj = None
    transpose_140 = view_159.transpose(1, 2);  view_159 = None
    key_states_56 = transpose_140.contiguous();  transpose_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_8_self_attn_v_proj = self.L__mod___model_decoder_layers_8_self_attn_v_proj(hidden_states_259);  hidden_states_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_160 = l__mod___model_decoder_layers_8_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_8_self_attn_v_proj = None
    transpose_141 = view_160.transpose(1, 2);  view_160 = None
    value_states_56 = transpose_141.contiguous();  transpose_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_161 = query_states_56.view(1, 1024, 16, 64);  query_states_56 = None
    transpose_142 = view_161.transpose(1, 2);  view_161 = None
    contiguous_86 = transpose_142.contiguous();  transpose_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_57 = contiguous_86.view(16, -1, 64);  contiguous_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    key_states_57 = key_states_56.reshape(16, -1, 64);  key_states_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    value_states_57 = value_states_56.reshape(16, -1, 64);  value_states_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_143 = key_states_57.transpose(1, 2);  key_states_57 = None
    attn_weights_72 = torch.bmm(query_states_57, transpose_143);  query_states_57 = transpose_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:246, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_163 = attn_weights_72.view(1, 16, 1024, 1024);  attn_weights_72 = None
    attn_weights_73 = view_163 + attention_mask;  view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:247, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_74 = attn_weights_73.view(16, 1024, 1024);  attn_weights_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_75 = torch.nn.functional.softmax(attn_weights_74, dim = -1);  attn_weights_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:270, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_28 = torch.nn.functional.dropout(attn_weights_75, p = 0.0, training = False);  attn_weights_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_140 = torch.bmm(attn_probs_28, value_states_57);  attn_probs_28 = value_states_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_141 = attn_output_140.view(1, 16, 1024, 64);  attn_output_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    attn_output_142 = attn_output_141.transpose(1, 2);  attn_output_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_143 = attn_output_142.reshape(1, 1024, 1024);  attn_output_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    hidden_states_260 = self.L__mod___model_decoder_layers_8_self_attn_out_proj(attn_output_143);  attn_output_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:431, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_261 = torch.nn.functional.dropout(hidden_states_260, p = 0.1, training = False);  hidden_states_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:432, code: hidden_states = residual + hidden_states
    residual_49 = residual_48 + hidden_states_261;  residual_48 = hidden_states_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:439, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_263 = self.L__mod___model_decoder_layers_8_encoder_attn_layer_norm(residual_49)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_8_encoder_attn_q_proj = self.L__mod___model_decoder_layers_8_encoder_attn_q_proj(hidden_states_263);  hidden_states_263 = None
    query_states_58 = l__mod___model_decoder_layers_8_encoder_attn_q_proj * 0.125;  l__mod___model_decoder_layers_8_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:204, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_8_encoder_attn_k_proj = self.L__mod___model_decoder_layers_8_encoder_attn_k_proj(hidden_states_135)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_166 = l__mod___model_decoder_layers_8_encoder_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_8_encoder_attn_k_proj = None
    transpose_145 = view_166.transpose(1, 2);  view_166 = None
    key_states_58 = transpose_145.contiguous();  transpose_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:205, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_8_encoder_attn_v_proj = self.L__mod___model_decoder_layers_8_encoder_attn_v_proj(hidden_states_135)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_167 = l__mod___model_decoder_layers_8_encoder_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_8_encoder_attn_v_proj = None
    transpose_146 = view_167.transpose(1, 2);  view_167 = None
    value_states_58 = transpose_146.contiguous();  transpose_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_168 = query_states_58.view(1, 1024, 16, 64);  query_states_58 = None
    transpose_147 = view_168.transpose(1, 2);  view_168 = None
    contiguous_89 = transpose_147.contiguous();  transpose_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_59 = contiguous_89.view(16, -1, 64);  contiguous_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    key_states_59 = key_states_58.reshape(16, -1, 64);  key_states_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    value_states_59 = value_states_58.reshape(16, -1, 64);  value_states_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_148 = key_states_59.transpose(1, 2);  key_states_59 = None
    attn_weights_76 = torch.bmm(query_states_59, transpose_148);  query_states_59 = transpose_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_77 = torch.nn.functional.softmax(attn_weights_76, dim = -1);  attn_weights_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:270, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_29 = torch.nn.functional.dropout(attn_weights_77, p = 0.0, training = False);  attn_weights_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_145 = torch.bmm(attn_probs_29, value_states_59);  attn_probs_29 = value_states_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_146 = attn_output_145.view(1, 16, 1024, 64);  attn_output_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    attn_output_147 = attn_output_146.transpose(1, 2);  attn_output_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_148 = attn_output_147.reshape(1, 1024, 1024);  attn_output_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    hidden_states_264 = self.L__mod___model_decoder_layers_8_encoder_attn_out_proj(attn_output_148);  attn_output_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:451, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_265 = torch.nn.functional.dropout(hidden_states_264, p = 0.1, training = False);  hidden_states_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:452, code: hidden_states = residual + hidden_states
    residual_50 = residual_49 + hidden_states_265;  residual_49 = hidden_states_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:459, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_267 = self.L__mod___model_decoder_layers_8_final_layer_norm(residual_50)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_8_fc1 = self.L__mod___model_decoder_layers_8_fc1(hidden_states_267);  hidden_states_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_268 = torch._C._nn.gelu(l__mod___model_decoder_layers_8_fc1);  l__mod___model_decoder_layers_8_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:461, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_269 = torch.nn.functional.dropout(hidden_states_268, p = 0.0, training = False);  hidden_states_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:462, code: hidden_states = self.fc2(hidden_states)
    hidden_states_270 = self.L__mod___model_decoder_layers_8_fc2(hidden_states_269);  hidden_states_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:463, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_271 = torch.nn.functional.dropout(hidden_states_270, p = 0.1, training = False);  hidden_states_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:464, code: hidden_states = residual + hidden_states
    residual_51 = residual_50 + hidden_states_271;  residual_50 = hidden_states_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:418, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_274 = self.L__mod___model_decoder_layers_9_self_attn_layer_norm(residual_51)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_9_self_attn_q_proj = self.L__mod___model_decoder_layers_9_self_attn_q_proj(hidden_states_274)
    query_states_60 = l__mod___model_decoder_layers_9_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_9_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_9_self_attn_k_proj = self.L__mod___model_decoder_layers_9_self_attn_k_proj(hidden_states_274)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_171 = l__mod___model_decoder_layers_9_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_9_self_attn_k_proj = None
    transpose_150 = view_171.transpose(1, 2);  view_171 = None
    key_states_60 = transpose_150.contiguous();  transpose_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_9_self_attn_v_proj = self.L__mod___model_decoder_layers_9_self_attn_v_proj(hidden_states_274);  hidden_states_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_172 = l__mod___model_decoder_layers_9_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_9_self_attn_v_proj = None
    transpose_151 = view_172.transpose(1, 2);  view_172 = None
    value_states_60 = transpose_151.contiguous();  transpose_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_173 = query_states_60.view(1, 1024, 16, 64);  query_states_60 = None
    transpose_152 = view_173.transpose(1, 2);  view_173 = None
    contiguous_92 = transpose_152.contiguous();  transpose_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_61 = contiguous_92.view(16, -1, 64);  contiguous_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    key_states_61 = key_states_60.reshape(16, -1, 64);  key_states_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    value_states_61 = value_states_60.reshape(16, -1, 64);  value_states_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_153 = key_states_61.transpose(1, 2);  key_states_61 = None
    attn_weights_78 = torch.bmm(query_states_61, transpose_153);  query_states_61 = transpose_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:246, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_175 = attn_weights_78.view(1, 16, 1024, 1024);  attn_weights_78 = None
    attn_weights_79 = view_175 + attention_mask;  view_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:247, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_80 = attn_weights_79.view(16, 1024, 1024);  attn_weights_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_81 = torch.nn.functional.softmax(attn_weights_80, dim = -1);  attn_weights_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:270, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_30 = torch.nn.functional.dropout(attn_weights_81, p = 0.0, training = False);  attn_weights_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_150 = torch.bmm(attn_probs_30, value_states_61);  attn_probs_30 = value_states_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_151 = attn_output_150.view(1, 16, 1024, 64);  attn_output_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    attn_output_152 = attn_output_151.transpose(1, 2);  attn_output_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_153 = attn_output_152.reshape(1, 1024, 1024);  attn_output_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    hidden_states_275 = self.L__mod___model_decoder_layers_9_self_attn_out_proj(attn_output_153);  attn_output_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:431, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_276 = torch.nn.functional.dropout(hidden_states_275, p = 0.1, training = False);  hidden_states_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:432, code: hidden_states = residual + hidden_states
    residual_52 = residual_51 + hidden_states_276;  residual_51 = hidden_states_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:439, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_278 = self.L__mod___model_decoder_layers_9_encoder_attn_layer_norm(residual_52)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_9_encoder_attn_q_proj = self.L__mod___model_decoder_layers_9_encoder_attn_q_proj(hidden_states_278);  hidden_states_278 = None
    query_states_62 = l__mod___model_decoder_layers_9_encoder_attn_q_proj * 0.125;  l__mod___model_decoder_layers_9_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:204, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_9_encoder_attn_k_proj = self.L__mod___model_decoder_layers_9_encoder_attn_k_proj(hidden_states_135)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_178 = l__mod___model_decoder_layers_9_encoder_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_9_encoder_attn_k_proj = None
    transpose_155 = view_178.transpose(1, 2);  view_178 = None
    key_states_62 = transpose_155.contiguous();  transpose_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:205, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_9_encoder_attn_v_proj = self.L__mod___model_decoder_layers_9_encoder_attn_v_proj(hidden_states_135)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_179 = l__mod___model_decoder_layers_9_encoder_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_9_encoder_attn_v_proj = None
    transpose_156 = view_179.transpose(1, 2);  view_179 = None
    value_states_62 = transpose_156.contiguous();  transpose_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_180 = query_states_62.view(1, 1024, 16, 64);  query_states_62 = None
    transpose_157 = view_180.transpose(1, 2);  view_180 = None
    contiguous_95 = transpose_157.contiguous();  transpose_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_63 = contiguous_95.view(16, -1, 64);  contiguous_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    key_states_63 = key_states_62.reshape(16, -1, 64);  key_states_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    value_states_63 = value_states_62.reshape(16, -1, 64);  value_states_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_158 = key_states_63.transpose(1, 2);  key_states_63 = None
    attn_weights_82 = torch.bmm(query_states_63, transpose_158);  query_states_63 = transpose_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_83 = torch.nn.functional.softmax(attn_weights_82, dim = -1);  attn_weights_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:270, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_31 = torch.nn.functional.dropout(attn_weights_83, p = 0.0, training = False);  attn_weights_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_155 = torch.bmm(attn_probs_31, value_states_63);  attn_probs_31 = value_states_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_156 = attn_output_155.view(1, 16, 1024, 64);  attn_output_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    attn_output_157 = attn_output_156.transpose(1, 2);  attn_output_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_158 = attn_output_157.reshape(1, 1024, 1024);  attn_output_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    hidden_states_279 = self.L__mod___model_decoder_layers_9_encoder_attn_out_proj(attn_output_158);  attn_output_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:451, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_280 = torch.nn.functional.dropout(hidden_states_279, p = 0.1, training = False);  hidden_states_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:452, code: hidden_states = residual + hidden_states
    residual_53 = residual_52 + hidden_states_280;  residual_52 = hidden_states_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:459, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_282 = self.L__mod___model_decoder_layers_9_final_layer_norm(residual_53)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_9_fc1 = self.L__mod___model_decoder_layers_9_fc1(hidden_states_282);  hidden_states_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_283 = torch._C._nn.gelu(l__mod___model_decoder_layers_9_fc1);  l__mod___model_decoder_layers_9_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:461, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_284 = torch.nn.functional.dropout(hidden_states_283, p = 0.0, training = False);  hidden_states_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:462, code: hidden_states = self.fc2(hidden_states)
    hidden_states_285 = self.L__mod___model_decoder_layers_9_fc2(hidden_states_284);  hidden_states_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:463, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_286 = torch.nn.functional.dropout(hidden_states_285, p = 0.1, training = False);  hidden_states_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:464, code: hidden_states = residual + hidden_states
    residual_54 = residual_53 + hidden_states_286;  residual_53 = hidden_states_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:418, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_289 = self.L__mod___model_decoder_layers_10_self_attn_layer_norm(residual_54)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_10_self_attn_q_proj = self.L__mod___model_decoder_layers_10_self_attn_q_proj(hidden_states_289)
    query_states_64 = l__mod___model_decoder_layers_10_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_10_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_10_self_attn_k_proj = self.L__mod___model_decoder_layers_10_self_attn_k_proj(hidden_states_289)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_183 = l__mod___model_decoder_layers_10_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_10_self_attn_k_proj = None
    transpose_160 = view_183.transpose(1, 2);  view_183 = None
    key_states_64 = transpose_160.contiguous();  transpose_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_10_self_attn_v_proj = self.L__mod___model_decoder_layers_10_self_attn_v_proj(hidden_states_289);  hidden_states_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_184 = l__mod___model_decoder_layers_10_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_10_self_attn_v_proj = None
    transpose_161 = view_184.transpose(1, 2);  view_184 = None
    value_states_64 = transpose_161.contiguous();  transpose_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_185 = query_states_64.view(1, 1024, 16, 64);  query_states_64 = None
    transpose_162 = view_185.transpose(1, 2);  view_185 = None
    contiguous_98 = transpose_162.contiguous();  transpose_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_65 = contiguous_98.view(16, -1, 64);  contiguous_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    key_states_65 = key_states_64.reshape(16, -1, 64);  key_states_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    value_states_65 = value_states_64.reshape(16, -1, 64);  value_states_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_163 = key_states_65.transpose(1, 2);  key_states_65 = None
    attn_weights_84 = torch.bmm(query_states_65, transpose_163);  query_states_65 = transpose_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:246, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_187 = attn_weights_84.view(1, 16, 1024, 1024);  attn_weights_84 = None
    attn_weights_85 = view_187 + attention_mask;  view_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:247, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_86 = attn_weights_85.view(16, 1024, 1024);  attn_weights_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_87 = torch.nn.functional.softmax(attn_weights_86, dim = -1);  attn_weights_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:270, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_32 = torch.nn.functional.dropout(attn_weights_87, p = 0.0, training = False);  attn_weights_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_160 = torch.bmm(attn_probs_32, value_states_65);  attn_probs_32 = value_states_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_161 = attn_output_160.view(1, 16, 1024, 64);  attn_output_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    attn_output_162 = attn_output_161.transpose(1, 2);  attn_output_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_163 = attn_output_162.reshape(1, 1024, 1024);  attn_output_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    hidden_states_290 = self.L__mod___model_decoder_layers_10_self_attn_out_proj(attn_output_163);  attn_output_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:431, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_291 = torch.nn.functional.dropout(hidden_states_290, p = 0.1, training = False);  hidden_states_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:432, code: hidden_states = residual + hidden_states
    residual_55 = residual_54 + hidden_states_291;  residual_54 = hidden_states_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:439, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_293 = self.L__mod___model_decoder_layers_10_encoder_attn_layer_norm(residual_55)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_10_encoder_attn_q_proj = self.L__mod___model_decoder_layers_10_encoder_attn_q_proj(hidden_states_293);  hidden_states_293 = None
    query_states_66 = l__mod___model_decoder_layers_10_encoder_attn_q_proj * 0.125;  l__mod___model_decoder_layers_10_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:204, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_10_encoder_attn_k_proj = self.L__mod___model_decoder_layers_10_encoder_attn_k_proj(hidden_states_135)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_190 = l__mod___model_decoder_layers_10_encoder_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_10_encoder_attn_k_proj = None
    transpose_165 = view_190.transpose(1, 2);  view_190 = None
    key_states_66 = transpose_165.contiguous();  transpose_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:205, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_10_encoder_attn_v_proj = self.L__mod___model_decoder_layers_10_encoder_attn_v_proj(hidden_states_135)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_191 = l__mod___model_decoder_layers_10_encoder_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_10_encoder_attn_v_proj = None
    transpose_166 = view_191.transpose(1, 2);  view_191 = None
    value_states_66 = transpose_166.contiguous();  transpose_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_192 = query_states_66.view(1, 1024, 16, 64);  query_states_66 = None
    transpose_167 = view_192.transpose(1, 2);  view_192 = None
    contiguous_101 = transpose_167.contiguous();  transpose_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_67 = contiguous_101.view(16, -1, 64);  contiguous_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    key_states_67 = key_states_66.reshape(16, -1, 64);  key_states_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    value_states_67 = value_states_66.reshape(16, -1, 64);  value_states_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_168 = key_states_67.transpose(1, 2);  key_states_67 = None
    attn_weights_88 = torch.bmm(query_states_67, transpose_168);  query_states_67 = transpose_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_89 = torch.nn.functional.softmax(attn_weights_88, dim = -1);  attn_weights_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:270, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_33 = torch.nn.functional.dropout(attn_weights_89, p = 0.0, training = False);  attn_weights_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_165 = torch.bmm(attn_probs_33, value_states_67);  attn_probs_33 = value_states_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_166 = attn_output_165.view(1, 16, 1024, 64);  attn_output_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    attn_output_167 = attn_output_166.transpose(1, 2);  attn_output_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_168 = attn_output_167.reshape(1, 1024, 1024);  attn_output_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    hidden_states_294 = self.L__mod___model_decoder_layers_10_encoder_attn_out_proj(attn_output_168);  attn_output_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:451, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_295 = torch.nn.functional.dropout(hidden_states_294, p = 0.1, training = False);  hidden_states_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:452, code: hidden_states = residual + hidden_states
    residual_56 = residual_55 + hidden_states_295;  residual_55 = hidden_states_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:459, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_297 = self.L__mod___model_decoder_layers_10_final_layer_norm(residual_56)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_10_fc1 = self.L__mod___model_decoder_layers_10_fc1(hidden_states_297);  hidden_states_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_298 = torch._C._nn.gelu(l__mod___model_decoder_layers_10_fc1);  l__mod___model_decoder_layers_10_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:461, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_299 = torch.nn.functional.dropout(hidden_states_298, p = 0.0, training = False);  hidden_states_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:462, code: hidden_states = self.fc2(hidden_states)
    hidden_states_300 = self.L__mod___model_decoder_layers_10_fc2(hidden_states_299);  hidden_states_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:463, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_301 = torch.nn.functional.dropout(hidden_states_300, p = 0.1, training = False);  hidden_states_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:464, code: hidden_states = residual + hidden_states
    residual_57 = residual_56 + hidden_states_301;  residual_56 = hidden_states_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:418, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_304 = self.L__mod___model_decoder_layers_11_self_attn_layer_norm(residual_57)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_11_self_attn_q_proj = self.L__mod___model_decoder_layers_11_self_attn_q_proj(hidden_states_304)
    query_states_68 = l__mod___model_decoder_layers_11_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_11_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_11_self_attn_k_proj = self.L__mod___model_decoder_layers_11_self_attn_k_proj(hidden_states_304)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_195 = l__mod___model_decoder_layers_11_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_11_self_attn_k_proj = None
    transpose_170 = view_195.transpose(1, 2);  view_195 = None
    key_states_68 = transpose_170.contiguous();  transpose_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_11_self_attn_v_proj = self.L__mod___model_decoder_layers_11_self_attn_v_proj(hidden_states_304);  hidden_states_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_196 = l__mod___model_decoder_layers_11_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_11_self_attn_v_proj = None
    transpose_171 = view_196.transpose(1, 2);  view_196 = None
    value_states_68 = transpose_171.contiguous();  transpose_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_197 = query_states_68.view(1, 1024, 16, 64);  query_states_68 = None
    transpose_172 = view_197.transpose(1, 2);  view_197 = None
    contiguous_104 = transpose_172.contiguous();  transpose_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_69 = contiguous_104.view(16, -1, 64);  contiguous_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    key_states_69 = key_states_68.reshape(16, -1, 64);  key_states_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    value_states_69 = value_states_68.reshape(16, -1, 64);  value_states_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_173 = key_states_69.transpose(1, 2);  key_states_69 = None
    attn_weights_90 = torch.bmm(query_states_69, transpose_173);  query_states_69 = transpose_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:246, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_199 = attn_weights_90.view(1, 16, 1024, 1024);  attn_weights_90 = None
    attn_weights_91 = view_199 + attention_mask;  view_199 = attention_mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:247, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_92 = attn_weights_91.view(16, 1024, 1024);  attn_weights_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_93 = torch.nn.functional.softmax(attn_weights_92, dim = -1);  attn_weights_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:270, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_34 = torch.nn.functional.dropout(attn_weights_93, p = 0.0, training = False);  attn_weights_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_170 = torch.bmm(attn_probs_34, value_states_69);  attn_probs_34 = value_states_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_171 = attn_output_170.view(1, 16, 1024, 64);  attn_output_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    attn_output_172 = attn_output_171.transpose(1, 2);  attn_output_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_173 = attn_output_172.reshape(1, 1024, 1024);  attn_output_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    hidden_states_305 = self.L__mod___model_decoder_layers_11_self_attn_out_proj(attn_output_173);  attn_output_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:431, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_306 = torch.nn.functional.dropout(hidden_states_305, p = 0.1, training = False);  hidden_states_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:432, code: hidden_states = residual + hidden_states
    residual_58 = residual_57 + hidden_states_306;  residual_57 = hidden_states_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:439, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_308 = self.L__mod___model_decoder_layers_11_encoder_attn_layer_norm(residual_58)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_11_encoder_attn_q_proj = self.L__mod___model_decoder_layers_11_encoder_attn_q_proj(hidden_states_308);  hidden_states_308 = None
    query_states_70 = l__mod___model_decoder_layers_11_encoder_attn_q_proj * 0.125;  l__mod___model_decoder_layers_11_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:204, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_11_encoder_attn_k_proj = self.L__mod___model_decoder_layers_11_encoder_attn_k_proj(hidden_states_135)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_202 = l__mod___model_decoder_layers_11_encoder_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_11_encoder_attn_k_proj = None
    transpose_175 = view_202.transpose(1, 2);  view_202 = None
    key_states_70 = transpose_175.contiguous();  transpose_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:205, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_11_encoder_attn_v_proj = self.L__mod___model_decoder_layers_11_encoder_attn_v_proj(hidden_states_135)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_203 = l__mod___model_decoder_layers_11_encoder_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_11_encoder_attn_v_proj = None
    transpose_176 = view_203.transpose(1, 2);  view_203 = None
    value_states_70 = transpose_176.contiguous();  transpose_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_204 = query_states_70.view(1, 1024, 16, 64);  query_states_70 = None
    transpose_177 = view_204.transpose(1, 2);  view_204 = None
    contiguous_107 = transpose_177.contiguous();  transpose_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_71 = contiguous_107.view(16, -1, 64);  contiguous_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    key_states_71 = key_states_70.reshape(16, -1, 64);  key_states_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    value_states_71 = value_states_70.reshape(16, -1, 64);  value_states_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_178 = key_states_71.transpose(1, 2);  key_states_71 = None
    attn_weights_94 = torch.bmm(query_states_71, transpose_178);  query_states_71 = transpose_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_95 = torch.nn.functional.softmax(attn_weights_94, dim = -1);  attn_weights_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:270, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_35 = torch.nn.functional.dropout(attn_weights_95, p = 0.0, training = False);  attn_weights_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_175 = torch.bmm(attn_probs_35, value_states_71);  attn_probs_35 = value_states_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_176 = attn_output_175.view(1, 16, 1024, 64);  attn_output_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    attn_output_177 = attn_output_176.transpose(1, 2);  attn_output_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_178 = attn_output_177.reshape(1, 1024, 1024);  attn_output_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    hidden_states_309 = self.L__mod___model_decoder_layers_11_encoder_attn_out_proj(attn_output_178);  attn_output_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:451, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_310 = torch.nn.functional.dropout(hidden_states_309, p = 0.1, training = False);  hidden_states_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:452, code: hidden_states = residual + hidden_states
    residual_59 = residual_58 + hidden_states_310;  residual_58 = hidden_states_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:459, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_312 = self.L__mod___model_decoder_layers_11_final_layer_norm(residual_59)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_11_fc1 = self.L__mod___model_decoder_layers_11_fc1(hidden_states_312);  hidden_states_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_313 = torch._C._nn.gelu(l__mod___model_decoder_layers_11_fc1);  l__mod___model_decoder_layers_11_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:461, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_314 = torch.nn.functional.dropout(hidden_states_313, p = 0.0, training = False);  hidden_states_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:462, code: hidden_states = self.fc2(hidden_states)
    hidden_states_315 = self.L__mod___model_decoder_layers_11_fc2(hidden_states_314);  hidden_states_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:463, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_316 = torch.nn.functional.dropout(hidden_states_315, p = 0.1, training = False);  hidden_states_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:464, code: hidden_states = residual + hidden_states
    hidden_states_318 = residual_59 + hidden_states_316;  residual_59 = hidden_states_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:1132, code: hidden_states = self.layer_norm(hidden_states)
    hidden_states_319 = self.L__mod___model_decoder_layer_norm(hidden_states_318);  hidden_states_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:1374, code: lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
    l__mod___lm_head = self.L__mod___lm_head(hidden_states_319);  hidden_states_319 = None
    l__mod___final_logits_bias = self.L__mod___final_logits_bias
    lm_logits = l__mod___lm_head + l__mod___final_logits_bias;  l__mod___lm_head = l__mod___final_logits_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:1379, code: masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
    view_207 = lm_logits.view(-1, 50265)
    view_208 = l_inputs_labels_.view(-1);  l_inputs_labels_ = None
    masked_lm_loss = torch.nn.functional.cross_entropy(view_207, view_208, None, None, -100, None, 'mean', 0.0);  view_207 = view_208 = None
    return (masked_lm_loss, lm_logits, hidden_states_135)
    