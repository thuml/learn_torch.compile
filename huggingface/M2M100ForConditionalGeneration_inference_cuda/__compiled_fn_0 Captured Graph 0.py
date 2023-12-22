from __future__ import annotations



def forward(self, L_inputs_labels_ : torch.Tensor, L_inputs_decoder_input_ids_ : torch.Tensor, L_inputs_input_ids_ : torch.Tensor):
    l_inputs_labels_ = L_inputs_labels_
    l_inputs_decoder_input_ids_ = L_inputs_decoder_input_ids_
    l_inputs_input_ids_ = L_inputs_input_ids_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:779, code: input_ids = input_ids.view(-1, input_shape[-1])
    input_ids = l_inputs_input_ids_.view(-1, 128);  l_inputs_input_ids_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:786, code: inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
    l__mod___model_encoder_embed_tokens = self.L__mod___model_encoder_embed_tokens(input_ids)
    inputs_embeds = l__mod___model_encoder_embed_tokens * 32.0;  l__mod___model_encoder_embed_tokens = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:113, code: mask = input_ids.ne(padding_idx).int()
    ne = input_ids.ne(1);  input_ids = None
    mask = ne.int();  ne = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:114, code: incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    cumsum = torch.cumsum(mask, dim = 1)
    type_as = cumsum.type_as(mask);  cumsum = None
    add = type_as + 0;  type_as = None
    incremental_indices = add * mask;  add = mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:115, code: return incremental_indices.long() + padding_idx
    long = incremental_indices.long();  incremental_indices = None
    add_1 = long + 1;  long = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:164, code: position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length).to(
    position_ids = add_1.to(device(type='cuda', index=0));  add_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:173, code: if max_pos > self.weights.size(0):
    l__mod___model_encoder_embed_positions_weights = self.L__mod___model_encoder_embed_positions_weights
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:176, code: return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()
    view_1 = position_ids.view(-1);  position_ids = None
    index_select = l__mod___model_encoder_embed_positions_weights.index_select(0, view_1);  l__mod___model_encoder_embed_positions_weights = view_1 = None
    view_2 = index_select.view(1, 128, 1024);  index_select = None
    embed_pos = view_2.detach();  view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:789, code: embed_pos = embed_pos.to(inputs_embeds.device)
    embed_pos_1 = embed_pos.to(device(type='cuda', index=0));  embed_pos = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:791, code: hidden_states = inputs_embeds + embed_pos
    hidden_states = inputs_embeds + embed_pos_1;  inputs_embeds = embed_pos_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:792, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    residual = torch.nn.functional.dropout(hidden_states, p = 0.1, training = False);  hidden_states = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:816, code: dropout_probability = torch.rand([])
    dropout_probability = torch.rand([])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_2 = self.L__mod___model_encoder_layers_0_self_attn_layer_norm(residual)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_encoder_layers_0_self_attn_q_proj = self.L__mod___model_encoder_layers_0_self_attn_q_proj(hidden_states_2)
    query_states = l__mod___model_encoder_layers_0_self_attn_q_proj * 0.125;  l__mod___model_encoder_layers_0_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_0_self_attn_k_proj = self.L__mod___model_encoder_layers_0_self_attn_k_proj(hidden_states_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_3 = l__mod___model_encoder_layers_0_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_0_self_attn_k_proj = None
    transpose = view_3.transpose(1, 2);  view_3 = None
    key_states = transpose.contiguous();  transpose = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_0_self_attn_v_proj = self.L__mod___model_encoder_layers_0_self_attn_v_proj(hidden_states_2);  hidden_states_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_4 = l__mod___model_encoder_layers_0_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_0_self_attn_v_proj = None
    transpose_1 = view_4.transpose(1, 2);  view_4 = None
    value_states = transpose_1.contiguous();  transpose_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_5 = query_states.view(1, 128, 16, 64);  query_states = None
    transpose_2 = view_5.transpose(1, 2);  view_5 = None
    contiguous_2 = transpose_2.contiguous();  transpose_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_1 = contiguous_2.view(16, -1, 64);  contiguous_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    key_states_1 = key_states.reshape(16, -1, 64);  key_states = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    value_states_1 = value_states.reshape(16, -1, 64);  value_states = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_3 = key_states_1.transpose(1, 2);  key_states_1 = None
    attn_weights = torch.bmm(query_states_1, transpose_3);  query_states_1 = transpose_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_1 = torch.nn.functional.softmax(attn_weights, dim = -1);  attn_weights = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs = torch.nn.functional.dropout(attn_weights_1, p = 0.1, training = False);  attn_weights_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output = torch.bmm(attn_probs, value_states_1);  attn_probs = value_states_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_1 = attn_output.view(1, 16, 128, 64);  attn_output = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    attn_output_2 = attn_output_1.transpose(1, 2);  attn_output_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_3 = attn_output_2.reshape(1, 128, 1024);  attn_output_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    hidden_states_3 = self.L__mod___model_encoder_layers_0_self_attn_out_proj(attn_output_3);  attn_output_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:395, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_4 = torch.nn.functional.dropout(hidden_states_3, p = 0.1, training = False);  hidden_states_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:396, code: hidden_states = residual + hidden_states
    residual_1 = residual + hidden_states_4;  residual = hidden_states_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_6 = self.L__mod___model_encoder_layers_0_final_layer_norm(residual_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_encoder_layers_0_fc1 = self.L__mod___model_encoder_layers_0_fc1(hidden_states_6);  hidden_states_6 = None
    hidden_states_7 = self.L__mod___model_encoder_layers_0_activation_fn(l__mod___model_encoder_layers_0_fc1);  l__mod___model_encoder_layers_0_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:401, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_8 = torch.nn.functional.dropout(hidden_states_7, p = 0.0, training = False);  hidden_states_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    hidden_states_9 = self.L__mod___model_encoder_layers_0_fc2(hidden_states_8);  hidden_states_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:403, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_10 = torch.nn.functional.dropout(hidden_states_9, p = 0.1, training = False);  hidden_states_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:404, code: hidden_states = residual + hidden_states
    residual_2 = residual_1 + hidden_states_10;  residual_1 = hidden_states_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:816, code: dropout_probability = torch.rand([])
    dropout_probability_1 = torch.rand([])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_13 = self.L__mod___model_encoder_layers_1_self_attn_layer_norm(residual_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_encoder_layers_1_self_attn_q_proj = self.L__mod___model_encoder_layers_1_self_attn_q_proj(hidden_states_13)
    query_states_2 = l__mod___model_encoder_layers_1_self_attn_q_proj * 0.125;  l__mod___model_encoder_layers_1_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_1_self_attn_k_proj = self.L__mod___model_encoder_layers_1_self_attn_k_proj(hidden_states_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_8 = l__mod___model_encoder_layers_1_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_1_self_attn_k_proj = None
    transpose_5 = view_8.transpose(1, 2);  view_8 = None
    key_states_2 = transpose_5.contiguous();  transpose_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_1_self_attn_v_proj = self.L__mod___model_encoder_layers_1_self_attn_v_proj(hidden_states_13);  hidden_states_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_9 = l__mod___model_encoder_layers_1_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_1_self_attn_v_proj = None
    transpose_6 = view_9.transpose(1, 2);  view_9 = None
    value_states_2 = transpose_6.contiguous();  transpose_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_10 = query_states_2.view(1, 128, 16, 64);  query_states_2 = None
    transpose_7 = view_10.transpose(1, 2);  view_10 = None
    contiguous_5 = transpose_7.contiguous();  transpose_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_3 = contiguous_5.view(16, -1, 64);  contiguous_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    key_states_3 = key_states_2.reshape(16, -1, 64);  key_states_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    value_states_3 = value_states_2.reshape(16, -1, 64);  value_states_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_8 = key_states_3.transpose(1, 2);  key_states_3 = None
    attn_weights_2 = torch.bmm(query_states_3, transpose_8);  query_states_3 = transpose_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_3 = torch.nn.functional.softmax(attn_weights_2, dim = -1);  attn_weights_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_1 = torch.nn.functional.dropout(attn_weights_3, p = 0.1, training = False);  attn_weights_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_5 = torch.bmm(attn_probs_1, value_states_3);  attn_probs_1 = value_states_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_6 = attn_output_5.view(1, 16, 128, 64);  attn_output_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    attn_output_7 = attn_output_6.transpose(1, 2);  attn_output_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_8 = attn_output_7.reshape(1, 128, 1024);  attn_output_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    hidden_states_14 = self.L__mod___model_encoder_layers_1_self_attn_out_proj(attn_output_8);  attn_output_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:395, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_15 = torch.nn.functional.dropout(hidden_states_14, p = 0.1, training = False);  hidden_states_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:396, code: hidden_states = residual + hidden_states
    residual_3 = residual_2 + hidden_states_15;  residual_2 = hidden_states_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_17 = self.L__mod___model_encoder_layers_1_final_layer_norm(residual_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_encoder_layers_1_fc1 = self.L__mod___model_encoder_layers_1_fc1(hidden_states_17);  hidden_states_17 = None
    hidden_states_18 = self.L__mod___model_encoder_layers_1_activation_fn(l__mod___model_encoder_layers_1_fc1);  l__mod___model_encoder_layers_1_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:401, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_19 = torch.nn.functional.dropout(hidden_states_18, p = 0.0, training = False);  hidden_states_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    hidden_states_20 = self.L__mod___model_encoder_layers_1_fc2(hidden_states_19);  hidden_states_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:403, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_21 = torch.nn.functional.dropout(hidden_states_20, p = 0.1, training = False);  hidden_states_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:404, code: hidden_states = residual + hidden_states
    residual_4 = residual_3 + hidden_states_21;  residual_3 = hidden_states_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:816, code: dropout_probability = torch.rand([])
    dropout_probability_2 = torch.rand([])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_24 = self.L__mod___model_encoder_layers_2_self_attn_layer_norm(residual_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_encoder_layers_2_self_attn_q_proj = self.L__mod___model_encoder_layers_2_self_attn_q_proj(hidden_states_24)
    query_states_4 = l__mod___model_encoder_layers_2_self_attn_q_proj * 0.125;  l__mod___model_encoder_layers_2_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_2_self_attn_k_proj = self.L__mod___model_encoder_layers_2_self_attn_k_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_13 = l__mod___model_encoder_layers_2_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_2_self_attn_k_proj = None
    transpose_10 = view_13.transpose(1, 2);  view_13 = None
    key_states_4 = transpose_10.contiguous();  transpose_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_2_self_attn_v_proj = self.L__mod___model_encoder_layers_2_self_attn_v_proj(hidden_states_24);  hidden_states_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_14 = l__mod___model_encoder_layers_2_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_2_self_attn_v_proj = None
    transpose_11 = view_14.transpose(1, 2);  view_14 = None
    value_states_4 = transpose_11.contiguous();  transpose_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_15 = query_states_4.view(1, 128, 16, 64);  query_states_4 = None
    transpose_12 = view_15.transpose(1, 2);  view_15 = None
    contiguous_8 = transpose_12.contiguous();  transpose_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_5 = contiguous_8.view(16, -1, 64);  contiguous_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    key_states_5 = key_states_4.reshape(16, -1, 64);  key_states_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    value_states_5 = value_states_4.reshape(16, -1, 64);  value_states_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_13 = key_states_5.transpose(1, 2);  key_states_5 = None
    attn_weights_4 = torch.bmm(query_states_5, transpose_13);  query_states_5 = transpose_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_5 = torch.nn.functional.softmax(attn_weights_4, dim = -1);  attn_weights_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_2 = torch.nn.functional.dropout(attn_weights_5, p = 0.1, training = False);  attn_weights_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_10 = torch.bmm(attn_probs_2, value_states_5);  attn_probs_2 = value_states_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_11 = attn_output_10.view(1, 16, 128, 64);  attn_output_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    attn_output_12 = attn_output_11.transpose(1, 2);  attn_output_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_13 = attn_output_12.reshape(1, 128, 1024);  attn_output_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    hidden_states_25 = self.L__mod___model_encoder_layers_2_self_attn_out_proj(attn_output_13);  attn_output_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:395, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_26 = torch.nn.functional.dropout(hidden_states_25, p = 0.1, training = False);  hidden_states_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:396, code: hidden_states = residual + hidden_states
    residual_5 = residual_4 + hidden_states_26;  residual_4 = hidden_states_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_28 = self.L__mod___model_encoder_layers_2_final_layer_norm(residual_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_encoder_layers_2_fc1 = self.L__mod___model_encoder_layers_2_fc1(hidden_states_28);  hidden_states_28 = None
    hidden_states_29 = self.L__mod___model_encoder_layers_2_activation_fn(l__mod___model_encoder_layers_2_fc1);  l__mod___model_encoder_layers_2_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:401, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_30 = torch.nn.functional.dropout(hidden_states_29, p = 0.0, training = False);  hidden_states_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    hidden_states_31 = self.L__mod___model_encoder_layers_2_fc2(hidden_states_30);  hidden_states_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:403, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_32 = torch.nn.functional.dropout(hidden_states_31, p = 0.1, training = False);  hidden_states_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:404, code: hidden_states = residual + hidden_states
    residual_6 = residual_5 + hidden_states_32;  residual_5 = hidden_states_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:816, code: dropout_probability = torch.rand([])
    dropout_probability_3 = torch.rand([])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_35 = self.L__mod___model_encoder_layers_3_self_attn_layer_norm(residual_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_encoder_layers_3_self_attn_q_proj = self.L__mod___model_encoder_layers_3_self_attn_q_proj(hidden_states_35)
    query_states_6 = l__mod___model_encoder_layers_3_self_attn_q_proj * 0.125;  l__mod___model_encoder_layers_3_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_3_self_attn_k_proj = self.L__mod___model_encoder_layers_3_self_attn_k_proj(hidden_states_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_18 = l__mod___model_encoder_layers_3_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_3_self_attn_k_proj = None
    transpose_15 = view_18.transpose(1, 2);  view_18 = None
    key_states_6 = transpose_15.contiguous();  transpose_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_3_self_attn_v_proj = self.L__mod___model_encoder_layers_3_self_attn_v_proj(hidden_states_35);  hidden_states_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_19 = l__mod___model_encoder_layers_3_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_3_self_attn_v_proj = None
    transpose_16 = view_19.transpose(1, 2);  view_19 = None
    value_states_6 = transpose_16.contiguous();  transpose_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_20 = query_states_6.view(1, 128, 16, 64);  query_states_6 = None
    transpose_17 = view_20.transpose(1, 2);  view_20 = None
    contiguous_11 = transpose_17.contiguous();  transpose_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_7 = contiguous_11.view(16, -1, 64);  contiguous_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    key_states_7 = key_states_6.reshape(16, -1, 64);  key_states_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    value_states_7 = value_states_6.reshape(16, -1, 64);  value_states_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_18 = key_states_7.transpose(1, 2);  key_states_7 = None
    attn_weights_6 = torch.bmm(query_states_7, transpose_18);  query_states_7 = transpose_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_7 = torch.nn.functional.softmax(attn_weights_6, dim = -1);  attn_weights_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_3 = torch.nn.functional.dropout(attn_weights_7, p = 0.1, training = False);  attn_weights_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_15 = torch.bmm(attn_probs_3, value_states_7);  attn_probs_3 = value_states_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_16 = attn_output_15.view(1, 16, 128, 64);  attn_output_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    attn_output_17 = attn_output_16.transpose(1, 2);  attn_output_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_18 = attn_output_17.reshape(1, 128, 1024);  attn_output_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    hidden_states_36 = self.L__mod___model_encoder_layers_3_self_attn_out_proj(attn_output_18);  attn_output_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:395, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_37 = torch.nn.functional.dropout(hidden_states_36, p = 0.1, training = False);  hidden_states_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:396, code: hidden_states = residual + hidden_states
    residual_7 = residual_6 + hidden_states_37;  residual_6 = hidden_states_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_39 = self.L__mod___model_encoder_layers_3_final_layer_norm(residual_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_encoder_layers_3_fc1 = self.L__mod___model_encoder_layers_3_fc1(hidden_states_39);  hidden_states_39 = None
    hidden_states_40 = self.L__mod___model_encoder_layers_3_activation_fn(l__mod___model_encoder_layers_3_fc1);  l__mod___model_encoder_layers_3_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:401, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_41 = torch.nn.functional.dropout(hidden_states_40, p = 0.0, training = False);  hidden_states_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    hidden_states_42 = self.L__mod___model_encoder_layers_3_fc2(hidden_states_41);  hidden_states_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:403, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_43 = torch.nn.functional.dropout(hidden_states_42, p = 0.1, training = False);  hidden_states_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:404, code: hidden_states = residual + hidden_states
    residual_8 = residual_7 + hidden_states_43;  residual_7 = hidden_states_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:816, code: dropout_probability = torch.rand([])
    dropout_probability_4 = torch.rand([])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_46 = self.L__mod___model_encoder_layers_4_self_attn_layer_norm(residual_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_encoder_layers_4_self_attn_q_proj = self.L__mod___model_encoder_layers_4_self_attn_q_proj(hidden_states_46)
    query_states_8 = l__mod___model_encoder_layers_4_self_attn_q_proj * 0.125;  l__mod___model_encoder_layers_4_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_4_self_attn_k_proj = self.L__mod___model_encoder_layers_4_self_attn_k_proj(hidden_states_46)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_23 = l__mod___model_encoder_layers_4_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_4_self_attn_k_proj = None
    transpose_20 = view_23.transpose(1, 2);  view_23 = None
    key_states_8 = transpose_20.contiguous();  transpose_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_4_self_attn_v_proj = self.L__mod___model_encoder_layers_4_self_attn_v_proj(hidden_states_46);  hidden_states_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_24 = l__mod___model_encoder_layers_4_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_4_self_attn_v_proj = None
    transpose_21 = view_24.transpose(1, 2);  view_24 = None
    value_states_8 = transpose_21.contiguous();  transpose_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_25 = query_states_8.view(1, 128, 16, 64);  query_states_8 = None
    transpose_22 = view_25.transpose(1, 2);  view_25 = None
    contiguous_14 = transpose_22.contiguous();  transpose_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_9 = contiguous_14.view(16, -1, 64);  contiguous_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    key_states_9 = key_states_8.reshape(16, -1, 64);  key_states_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    value_states_9 = value_states_8.reshape(16, -1, 64);  value_states_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_23 = key_states_9.transpose(1, 2);  key_states_9 = None
    attn_weights_8 = torch.bmm(query_states_9, transpose_23);  query_states_9 = transpose_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_9 = torch.nn.functional.softmax(attn_weights_8, dim = -1);  attn_weights_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_4 = torch.nn.functional.dropout(attn_weights_9, p = 0.1, training = False);  attn_weights_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_20 = torch.bmm(attn_probs_4, value_states_9);  attn_probs_4 = value_states_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_21 = attn_output_20.view(1, 16, 128, 64);  attn_output_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    attn_output_22 = attn_output_21.transpose(1, 2);  attn_output_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_23 = attn_output_22.reshape(1, 128, 1024);  attn_output_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    hidden_states_47 = self.L__mod___model_encoder_layers_4_self_attn_out_proj(attn_output_23);  attn_output_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:395, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_48 = torch.nn.functional.dropout(hidden_states_47, p = 0.1, training = False);  hidden_states_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:396, code: hidden_states = residual + hidden_states
    residual_9 = residual_8 + hidden_states_48;  residual_8 = hidden_states_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_50 = self.L__mod___model_encoder_layers_4_final_layer_norm(residual_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_encoder_layers_4_fc1 = self.L__mod___model_encoder_layers_4_fc1(hidden_states_50);  hidden_states_50 = None
    hidden_states_51 = self.L__mod___model_encoder_layers_4_activation_fn(l__mod___model_encoder_layers_4_fc1);  l__mod___model_encoder_layers_4_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:401, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_52 = torch.nn.functional.dropout(hidden_states_51, p = 0.0, training = False);  hidden_states_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    hidden_states_53 = self.L__mod___model_encoder_layers_4_fc2(hidden_states_52);  hidden_states_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:403, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_54 = torch.nn.functional.dropout(hidden_states_53, p = 0.1, training = False);  hidden_states_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:404, code: hidden_states = residual + hidden_states
    residual_10 = residual_9 + hidden_states_54;  residual_9 = hidden_states_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:816, code: dropout_probability = torch.rand([])
    dropout_probability_5 = torch.rand([])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_57 = self.L__mod___model_encoder_layers_5_self_attn_layer_norm(residual_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_encoder_layers_5_self_attn_q_proj = self.L__mod___model_encoder_layers_5_self_attn_q_proj(hidden_states_57)
    query_states_10 = l__mod___model_encoder_layers_5_self_attn_q_proj * 0.125;  l__mod___model_encoder_layers_5_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_5_self_attn_k_proj = self.L__mod___model_encoder_layers_5_self_attn_k_proj(hidden_states_57)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_28 = l__mod___model_encoder_layers_5_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_5_self_attn_k_proj = None
    transpose_25 = view_28.transpose(1, 2);  view_28 = None
    key_states_10 = transpose_25.contiguous();  transpose_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_5_self_attn_v_proj = self.L__mod___model_encoder_layers_5_self_attn_v_proj(hidden_states_57);  hidden_states_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_29 = l__mod___model_encoder_layers_5_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_5_self_attn_v_proj = None
    transpose_26 = view_29.transpose(1, 2);  view_29 = None
    value_states_10 = transpose_26.contiguous();  transpose_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_30 = query_states_10.view(1, 128, 16, 64);  query_states_10 = None
    transpose_27 = view_30.transpose(1, 2);  view_30 = None
    contiguous_17 = transpose_27.contiguous();  transpose_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_11 = contiguous_17.view(16, -1, 64);  contiguous_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    key_states_11 = key_states_10.reshape(16, -1, 64);  key_states_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    value_states_11 = value_states_10.reshape(16, -1, 64);  value_states_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_28 = key_states_11.transpose(1, 2);  key_states_11 = None
    attn_weights_10 = torch.bmm(query_states_11, transpose_28);  query_states_11 = transpose_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_11 = torch.nn.functional.softmax(attn_weights_10, dim = -1);  attn_weights_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_5 = torch.nn.functional.dropout(attn_weights_11, p = 0.1, training = False);  attn_weights_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_25 = torch.bmm(attn_probs_5, value_states_11);  attn_probs_5 = value_states_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_26 = attn_output_25.view(1, 16, 128, 64);  attn_output_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    attn_output_27 = attn_output_26.transpose(1, 2);  attn_output_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_28 = attn_output_27.reshape(1, 128, 1024);  attn_output_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    hidden_states_58 = self.L__mod___model_encoder_layers_5_self_attn_out_proj(attn_output_28);  attn_output_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:395, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_59 = torch.nn.functional.dropout(hidden_states_58, p = 0.1, training = False);  hidden_states_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:396, code: hidden_states = residual + hidden_states
    residual_11 = residual_10 + hidden_states_59;  residual_10 = hidden_states_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_61 = self.L__mod___model_encoder_layers_5_final_layer_norm(residual_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_encoder_layers_5_fc1 = self.L__mod___model_encoder_layers_5_fc1(hidden_states_61);  hidden_states_61 = None
    hidden_states_62 = self.L__mod___model_encoder_layers_5_activation_fn(l__mod___model_encoder_layers_5_fc1);  l__mod___model_encoder_layers_5_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:401, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_63 = torch.nn.functional.dropout(hidden_states_62, p = 0.0, training = False);  hidden_states_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    hidden_states_64 = self.L__mod___model_encoder_layers_5_fc2(hidden_states_63);  hidden_states_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:403, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_65 = torch.nn.functional.dropout(hidden_states_64, p = 0.1, training = False);  hidden_states_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:404, code: hidden_states = residual + hidden_states
    residual_12 = residual_11 + hidden_states_65;  residual_11 = hidden_states_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:816, code: dropout_probability = torch.rand([])
    dropout_probability_6 = torch.rand([])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_68 = self.L__mod___model_encoder_layers_6_self_attn_layer_norm(residual_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_encoder_layers_6_self_attn_q_proj = self.L__mod___model_encoder_layers_6_self_attn_q_proj(hidden_states_68)
    query_states_12 = l__mod___model_encoder_layers_6_self_attn_q_proj * 0.125;  l__mod___model_encoder_layers_6_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_6_self_attn_k_proj = self.L__mod___model_encoder_layers_6_self_attn_k_proj(hidden_states_68)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_33 = l__mod___model_encoder_layers_6_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_6_self_attn_k_proj = None
    transpose_30 = view_33.transpose(1, 2);  view_33 = None
    key_states_12 = transpose_30.contiguous();  transpose_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_6_self_attn_v_proj = self.L__mod___model_encoder_layers_6_self_attn_v_proj(hidden_states_68);  hidden_states_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_34 = l__mod___model_encoder_layers_6_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_6_self_attn_v_proj = None
    transpose_31 = view_34.transpose(1, 2);  view_34 = None
    value_states_12 = transpose_31.contiguous();  transpose_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_35 = query_states_12.view(1, 128, 16, 64);  query_states_12 = None
    transpose_32 = view_35.transpose(1, 2);  view_35 = None
    contiguous_20 = transpose_32.contiguous();  transpose_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_13 = contiguous_20.view(16, -1, 64);  contiguous_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    key_states_13 = key_states_12.reshape(16, -1, 64);  key_states_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    value_states_13 = value_states_12.reshape(16, -1, 64);  value_states_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_33 = key_states_13.transpose(1, 2);  key_states_13 = None
    attn_weights_12 = torch.bmm(query_states_13, transpose_33);  query_states_13 = transpose_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_13 = torch.nn.functional.softmax(attn_weights_12, dim = -1);  attn_weights_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_6 = torch.nn.functional.dropout(attn_weights_13, p = 0.1, training = False);  attn_weights_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_30 = torch.bmm(attn_probs_6, value_states_13);  attn_probs_6 = value_states_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_31 = attn_output_30.view(1, 16, 128, 64);  attn_output_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    attn_output_32 = attn_output_31.transpose(1, 2);  attn_output_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_33 = attn_output_32.reshape(1, 128, 1024);  attn_output_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    hidden_states_69 = self.L__mod___model_encoder_layers_6_self_attn_out_proj(attn_output_33);  attn_output_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:395, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_70 = torch.nn.functional.dropout(hidden_states_69, p = 0.1, training = False);  hidden_states_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:396, code: hidden_states = residual + hidden_states
    residual_13 = residual_12 + hidden_states_70;  residual_12 = hidden_states_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_72 = self.L__mod___model_encoder_layers_6_final_layer_norm(residual_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_encoder_layers_6_fc1 = self.L__mod___model_encoder_layers_6_fc1(hidden_states_72);  hidden_states_72 = None
    hidden_states_73 = self.L__mod___model_encoder_layers_6_activation_fn(l__mod___model_encoder_layers_6_fc1);  l__mod___model_encoder_layers_6_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:401, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_74 = torch.nn.functional.dropout(hidden_states_73, p = 0.0, training = False);  hidden_states_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    hidden_states_75 = self.L__mod___model_encoder_layers_6_fc2(hidden_states_74);  hidden_states_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:403, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_76 = torch.nn.functional.dropout(hidden_states_75, p = 0.1, training = False);  hidden_states_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:404, code: hidden_states = residual + hidden_states
    residual_14 = residual_13 + hidden_states_76;  residual_13 = hidden_states_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:816, code: dropout_probability = torch.rand([])
    dropout_probability_7 = torch.rand([])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_79 = self.L__mod___model_encoder_layers_7_self_attn_layer_norm(residual_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_encoder_layers_7_self_attn_q_proj = self.L__mod___model_encoder_layers_7_self_attn_q_proj(hidden_states_79)
    query_states_14 = l__mod___model_encoder_layers_7_self_attn_q_proj * 0.125;  l__mod___model_encoder_layers_7_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_7_self_attn_k_proj = self.L__mod___model_encoder_layers_7_self_attn_k_proj(hidden_states_79)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_38 = l__mod___model_encoder_layers_7_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_7_self_attn_k_proj = None
    transpose_35 = view_38.transpose(1, 2);  view_38 = None
    key_states_14 = transpose_35.contiguous();  transpose_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_7_self_attn_v_proj = self.L__mod___model_encoder_layers_7_self_attn_v_proj(hidden_states_79);  hidden_states_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_39 = l__mod___model_encoder_layers_7_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_7_self_attn_v_proj = None
    transpose_36 = view_39.transpose(1, 2);  view_39 = None
    value_states_14 = transpose_36.contiguous();  transpose_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_40 = query_states_14.view(1, 128, 16, 64);  query_states_14 = None
    transpose_37 = view_40.transpose(1, 2);  view_40 = None
    contiguous_23 = transpose_37.contiguous();  transpose_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_15 = contiguous_23.view(16, -1, 64);  contiguous_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    key_states_15 = key_states_14.reshape(16, -1, 64);  key_states_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    value_states_15 = value_states_14.reshape(16, -1, 64);  value_states_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_38 = key_states_15.transpose(1, 2);  key_states_15 = None
    attn_weights_14 = torch.bmm(query_states_15, transpose_38);  query_states_15 = transpose_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_15 = torch.nn.functional.softmax(attn_weights_14, dim = -1);  attn_weights_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_7 = torch.nn.functional.dropout(attn_weights_15, p = 0.1, training = False);  attn_weights_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_35 = torch.bmm(attn_probs_7, value_states_15);  attn_probs_7 = value_states_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_36 = attn_output_35.view(1, 16, 128, 64);  attn_output_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    attn_output_37 = attn_output_36.transpose(1, 2);  attn_output_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_38 = attn_output_37.reshape(1, 128, 1024);  attn_output_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    hidden_states_80 = self.L__mod___model_encoder_layers_7_self_attn_out_proj(attn_output_38);  attn_output_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:395, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_81 = torch.nn.functional.dropout(hidden_states_80, p = 0.1, training = False);  hidden_states_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:396, code: hidden_states = residual + hidden_states
    residual_15 = residual_14 + hidden_states_81;  residual_14 = hidden_states_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_83 = self.L__mod___model_encoder_layers_7_final_layer_norm(residual_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_encoder_layers_7_fc1 = self.L__mod___model_encoder_layers_7_fc1(hidden_states_83);  hidden_states_83 = None
    hidden_states_84 = self.L__mod___model_encoder_layers_7_activation_fn(l__mod___model_encoder_layers_7_fc1);  l__mod___model_encoder_layers_7_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:401, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_85 = torch.nn.functional.dropout(hidden_states_84, p = 0.0, training = False);  hidden_states_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    hidden_states_86 = self.L__mod___model_encoder_layers_7_fc2(hidden_states_85);  hidden_states_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:403, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_87 = torch.nn.functional.dropout(hidden_states_86, p = 0.1, training = False);  hidden_states_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:404, code: hidden_states = residual + hidden_states
    residual_16 = residual_15 + hidden_states_87;  residual_15 = hidden_states_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:816, code: dropout_probability = torch.rand([])
    dropout_probability_8 = torch.rand([])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_90 = self.L__mod___model_encoder_layers_8_self_attn_layer_norm(residual_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_encoder_layers_8_self_attn_q_proj = self.L__mod___model_encoder_layers_8_self_attn_q_proj(hidden_states_90)
    query_states_16 = l__mod___model_encoder_layers_8_self_attn_q_proj * 0.125;  l__mod___model_encoder_layers_8_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_8_self_attn_k_proj = self.L__mod___model_encoder_layers_8_self_attn_k_proj(hidden_states_90)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_43 = l__mod___model_encoder_layers_8_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_8_self_attn_k_proj = None
    transpose_40 = view_43.transpose(1, 2);  view_43 = None
    key_states_16 = transpose_40.contiguous();  transpose_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_8_self_attn_v_proj = self.L__mod___model_encoder_layers_8_self_attn_v_proj(hidden_states_90);  hidden_states_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_44 = l__mod___model_encoder_layers_8_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_8_self_attn_v_proj = None
    transpose_41 = view_44.transpose(1, 2);  view_44 = None
    value_states_16 = transpose_41.contiguous();  transpose_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_45 = query_states_16.view(1, 128, 16, 64);  query_states_16 = None
    transpose_42 = view_45.transpose(1, 2);  view_45 = None
    contiguous_26 = transpose_42.contiguous();  transpose_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_17 = contiguous_26.view(16, -1, 64);  contiguous_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    key_states_17 = key_states_16.reshape(16, -1, 64);  key_states_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    value_states_17 = value_states_16.reshape(16, -1, 64);  value_states_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_43 = key_states_17.transpose(1, 2);  key_states_17 = None
    attn_weights_16 = torch.bmm(query_states_17, transpose_43);  query_states_17 = transpose_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_17 = torch.nn.functional.softmax(attn_weights_16, dim = -1);  attn_weights_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_8 = torch.nn.functional.dropout(attn_weights_17, p = 0.1, training = False);  attn_weights_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_40 = torch.bmm(attn_probs_8, value_states_17);  attn_probs_8 = value_states_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_41 = attn_output_40.view(1, 16, 128, 64);  attn_output_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    attn_output_42 = attn_output_41.transpose(1, 2);  attn_output_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_43 = attn_output_42.reshape(1, 128, 1024);  attn_output_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    hidden_states_91 = self.L__mod___model_encoder_layers_8_self_attn_out_proj(attn_output_43);  attn_output_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:395, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_92 = torch.nn.functional.dropout(hidden_states_91, p = 0.1, training = False);  hidden_states_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:396, code: hidden_states = residual + hidden_states
    residual_17 = residual_16 + hidden_states_92;  residual_16 = hidden_states_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_94 = self.L__mod___model_encoder_layers_8_final_layer_norm(residual_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_encoder_layers_8_fc1 = self.L__mod___model_encoder_layers_8_fc1(hidden_states_94);  hidden_states_94 = None
    hidden_states_95 = self.L__mod___model_encoder_layers_8_activation_fn(l__mod___model_encoder_layers_8_fc1);  l__mod___model_encoder_layers_8_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:401, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_96 = torch.nn.functional.dropout(hidden_states_95, p = 0.0, training = False);  hidden_states_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    hidden_states_97 = self.L__mod___model_encoder_layers_8_fc2(hidden_states_96);  hidden_states_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:403, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_98 = torch.nn.functional.dropout(hidden_states_97, p = 0.1, training = False);  hidden_states_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:404, code: hidden_states = residual + hidden_states
    residual_18 = residual_17 + hidden_states_98;  residual_17 = hidden_states_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:816, code: dropout_probability = torch.rand([])
    dropout_probability_9 = torch.rand([])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_101 = self.L__mod___model_encoder_layers_9_self_attn_layer_norm(residual_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_encoder_layers_9_self_attn_q_proj = self.L__mod___model_encoder_layers_9_self_attn_q_proj(hidden_states_101)
    query_states_18 = l__mod___model_encoder_layers_9_self_attn_q_proj * 0.125;  l__mod___model_encoder_layers_9_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_9_self_attn_k_proj = self.L__mod___model_encoder_layers_9_self_attn_k_proj(hidden_states_101)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_48 = l__mod___model_encoder_layers_9_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_9_self_attn_k_proj = None
    transpose_45 = view_48.transpose(1, 2);  view_48 = None
    key_states_18 = transpose_45.contiguous();  transpose_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_9_self_attn_v_proj = self.L__mod___model_encoder_layers_9_self_attn_v_proj(hidden_states_101);  hidden_states_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_49 = l__mod___model_encoder_layers_9_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_9_self_attn_v_proj = None
    transpose_46 = view_49.transpose(1, 2);  view_49 = None
    value_states_18 = transpose_46.contiguous();  transpose_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_50 = query_states_18.view(1, 128, 16, 64);  query_states_18 = None
    transpose_47 = view_50.transpose(1, 2);  view_50 = None
    contiguous_29 = transpose_47.contiguous();  transpose_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_19 = contiguous_29.view(16, -1, 64);  contiguous_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    key_states_19 = key_states_18.reshape(16, -1, 64);  key_states_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    value_states_19 = value_states_18.reshape(16, -1, 64);  value_states_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_48 = key_states_19.transpose(1, 2);  key_states_19 = None
    attn_weights_18 = torch.bmm(query_states_19, transpose_48);  query_states_19 = transpose_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_19 = torch.nn.functional.softmax(attn_weights_18, dim = -1);  attn_weights_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_9 = torch.nn.functional.dropout(attn_weights_19, p = 0.1, training = False);  attn_weights_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_45 = torch.bmm(attn_probs_9, value_states_19);  attn_probs_9 = value_states_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_46 = attn_output_45.view(1, 16, 128, 64);  attn_output_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    attn_output_47 = attn_output_46.transpose(1, 2);  attn_output_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_48 = attn_output_47.reshape(1, 128, 1024);  attn_output_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    hidden_states_102 = self.L__mod___model_encoder_layers_9_self_attn_out_proj(attn_output_48);  attn_output_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:395, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_103 = torch.nn.functional.dropout(hidden_states_102, p = 0.1, training = False);  hidden_states_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:396, code: hidden_states = residual + hidden_states
    residual_19 = residual_18 + hidden_states_103;  residual_18 = hidden_states_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_105 = self.L__mod___model_encoder_layers_9_final_layer_norm(residual_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_encoder_layers_9_fc1 = self.L__mod___model_encoder_layers_9_fc1(hidden_states_105);  hidden_states_105 = None
    hidden_states_106 = self.L__mod___model_encoder_layers_9_activation_fn(l__mod___model_encoder_layers_9_fc1);  l__mod___model_encoder_layers_9_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:401, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_107 = torch.nn.functional.dropout(hidden_states_106, p = 0.0, training = False);  hidden_states_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    hidden_states_108 = self.L__mod___model_encoder_layers_9_fc2(hidden_states_107);  hidden_states_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:403, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_109 = torch.nn.functional.dropout(hidden_states_108, p = 0.1, training = False);  hidden_states_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:404, code: hidden_states = residual + hidden_states
    residual_20 = residual_19 + hidden_states_109;  residual_19 = hidden_states_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:816, code: dropout_probability = torch.rand([])
    dropout_probability_10 = torch.rand([])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_112 = self.L__mod___model_encoder_layers_10_self_attn_layer_norm(residual_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_encoder_layers_10_self_attn_q_proj = self.L__mod___model_encoder_layers_10_self_attn_q_proj(hidden_states_112)
    query_states_20 = l__mod___model_encoder_layers_10_self_attn_q_proj * 0.125;  l__mod___model_encoder_layers_10_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_10_self_attn_k_proj = self.L__mod___model_encoder_layers_10_self_attn_k_proj(hidden_states_112)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_53 = l__mod___model_encoder_layers_10_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_10_self_attn_k_proj = None
    transpose_50 = view_53.transpose(1, 2);  view_53 = None
    key_states_20 = transpose_50.contiguous();  transpose_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_10_self_attn_v_proj = self.L__mod___model_encoder_layers_10_self_attn_v_proj(hidden_states_112);  hidden_states_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_54 = l__mod___model_encoder_layers_10_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_10_self_attn_v_proj = None
    transpose_51 = view_54.transpose(1, 2);  view_54 = None
    value_states_20 = transpose_51.contiguous();  transpose_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_55 = query_states_20.view(1, 128, 16, 64);  query_states_20 = None
    transpose_52 = view_55.transpose(1, 2);  view_55 = None
    contiguous_32 = transpose_52.contiguous();  transpose_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_21 = contiguous_32.view(16, -1, 64);  contiguous_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    key_states_21 = key_states_20.reshape(16, -1, 64);  key_states_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    value_states_21 = value_states_20.reshape(16, -1, 64);  value_states_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_53 = key_states_21.transpose(1, 2);  key_states_21 = None
    attn_weights_20 = torch.bmm(query_states_21, transpose_53);  query_states_21 = transpose_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_21 = torch.nn.functional.softmax(attn_weights_20, dim = -1);  attn_weights_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_10 = torch.nn.functional.dropout(attn_weights_21, p = 0.1, training = False);  attn_weights_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_50 = torch.bmm(attn_probs_10, value_states_21);  attn_probs_10 = value_states_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_51 = attn_output_50.view(1, 16, 128, 64);  attn_output_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    attn_output_52 = attn_output_51.transpose(1, 2);  attn_output_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_53 = attn_output_52.reshape(1, 128, 1024);  attn_output_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    hidden_states_113 = self.L__mod___model_encoder_layers_10_self_attn_out_proj(attn_output_53);  attn_output_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:395, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_114 = torch.nn.functional.dropout(hidden_states_113, p = 0.1, training = False);  hidden_states_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:396, code: hidden_states = residual + hidden_states
    residual_21 = residual_20 + hidden_states_114;  residual_20 = hidden_states_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_116 = self.L__mod___model_encoder_layers_10_final_layer_norm(residual_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_encoder_layers_10_fc1 = self.L__mod___model_encoder_layers_10_fc1(hidden_states_116);  hidden_states_116 = None
    hidden_states_117 = self.L__mod___model_encoder_layers_10_activation_fn(l__mod___model_encoder_layers_10_fc1);  l__mod___model_encoder_layers_10_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:401, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_118 = torch.nn.functional.dropout(hidden_states_117, p = 0.0, training = False);  hidden_states_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    hidden_states_119 = self.L__mod___model_encoder_layers_10_fc2(hidden_states_118);  hidden_states_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:403, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_120 = torch.nn.functional.dropout(hidden_states_119, p = 0.1, training = False);  hidden_states_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:404, code: hidden_states = residual + hidden_states
    residual_22 = residual_21 + hidden_states_120;  residual_21 = hidden_states_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:816, code: dropout_probability = torch.rand([])
    dropout_probability_11 = torch.rand([])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_123 = self.L__mod___model_encoder_layers_11_self_attn_layer_norm(residual_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_encoder_layers_11_self_attn_q_proj = self.L__mod___model_encoder_layers_11_self_attn_q_proj(hidden_states_123)
    query_states_22 = l__mod___model_encoder_layers_11_self_attn_q_proj * 0.125;  l__mod___model_encoder_layers_11_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_11_self_attn_k_proj = self.L__mod___model_encoder_layers_11_self_attn_k_proj(hidden_states_123)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_58 = l__mod___model_encoder_layers_11_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_11_self_attn_k_proj = None
    transpose_55 = view_58.transpose(1, 2);  view_58 = None
    key_states_22 = transpose_55.contiguous();  transpose_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_11_self_attn_v_proj = self.L__mod___model_encoder_layers_11_self_attn_v_proj(hidden_states_123);  hidden_states_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_59 = l__mod___model_encoder_layers_11_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_encoder_layers_11_self_attn_v_proj = None
    transpose_56 = view_59.transpose(1, 2);  view_59 = None
    value_states_22 = transpose_56.contiguous();  transpose_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_60 = query_states_22.view(1, 128, 16, 64);  query_states_22 = None
    transpose_57 = view_60.transpose(1, 2);  view_60 = None
    contiguous_35 = transpose_57.contiguous();  transpose_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_23 = contiguous_35.view(16, -1, 64);  contiguous_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    key_states_23 = key_states_22.reshape(16, -1, 64);  key_states_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    value_states_23 = value_states_22.reshape(16, -1, 64);  value_states_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_58 = key_states_23.transpose(1, 2);  key_states_23 = None
    attn_weights_22 = torch.bmm(query_states_23, transpose_58);  query_states_23 = transpose_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_23 = torch.nn.functional.softmax(attn_weights_22, dim = -1);  attn_weights_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_11 = torch.nn.functional.dropout(attn_weights_23, p = 0.1, training = False);  attn_weights_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_55 = torch.bmm(attn_probs_11, value_states_23);  attn_probs_11 = value_states_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_56 = attn_output_55.view(1, 16, 128, 64);  attn_output_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    attn_output_57 = attn_output_56.transpose(1, 2);  attn_output_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_58 = attn_output_57.reshape(1, 128, 1024);  attn_output_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    hidden_states_124 = self.L__mod___model_encoder_layers_11_self_attn_out_proj(attn_output_58);  attn_output_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:395, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_125 = torch.nn.functional.dropout(hidden_states_124, p = 0.1, training = False);  hidden_states_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:396, code: hidden_states = residual + hidden_states
    residual_23 = residual_22 + hidden_states_125;  residual_22 = hidden_states_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_127 = self.L__mod___model_encoder_layers_11_final_layer_norm(residual_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_encoder_layers_11_fc1 = self.L__mod___model_encoder_layers_11_fc1(hidden_states_127);  hidden_states_127 = None
    hidden_states_128 = self.L__mod___model_encoder_layers_11_activation_fn(l__mod___model_encoder_layers_11_fc1);  l__mod___model_encoder_layers_11_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:401, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_129 = torch.nn.functional.dropout(hidden_states_128, p = 0.0, training = False);  hidden_states_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    hidden_states_130 = self.L__mod___model_encoder_layers_11_fc2(hidden_states_129);  hidden_states_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:403, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_131 = torch.nn.functional.dropout(hidden_states_130, p = 0.1, training = False);  hidden_states_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:404, code: hidden_states = residual + hidden_states
    hidden_states_133 = residual_23 + hidden_states_131;  residual_23 = hidden_states_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:852, code: hidden_states = self.layer_norm(hidden_states)
    hidden_states_134 = self.L__mod___model_encoder_layer_norm(hidden_states_133);  hidden_states_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:990, code: input_ids = input_ids.view(-1, input_shape[-1])
    input_ids_1 = l_inputs_decoder_input_ids_.view(-1, 128);  l_inputs_decoder_input_ids_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1000, code: inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
    l__mod___model_decoder_embed_tokens = self.L__mod___model_decoder_embed_tokens(input_ids_1)
    inputs_embeds_1 = l__mod___model_decoder_embed_tokens * 32.0;  l__mod___model_decoder_embed_tokens = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:82, code: mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_1 = torch.full((128, 128), -3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:83, code: mask_cond = torch.arange(mask.size(-1), device=device)
    mask_cond = torch.arange(128, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:84, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    add_27 = mask_cond + 1
    view_64 = add_27.view(128, 1);  add_27 = None
    lt = mask_cond < view_64;  mask_cond = view_64 = None
    masked_fill_ = mask_1.masked_fill_(lt, 0);  lt = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:85, code: mask = mask.to(dtype)
    mask_2 = mask_1.to(torch.float32);  mask_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:89, code: return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
    getitem = mask_2[(None, None, slice(None, None, None), slice(None, None, None))];  mask_2 = None
    combined_attention_mask = getitem.expand(1, 1, 128, 128);  getitem = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:113, code: mask = input_ids.ne(padding_idx).int()
    ne_1 = input_ids_1.ne(1);  input_ids_1 = None
    mask_3 = ne_1.int();  ne_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:114, code: incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    cumsum_1 = torch.cumsum(mask_3, dim = 1)
    type_as_1 = cumsum_1.type_as(mask_3);  cumsum_1 = None
    add_28 = type_as_1 + 0;  type_as_1 = None
    incremental_indices_1 = add_28 * mask_3;  add_28 = mask_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:115, code: return incremental_indices.long() + padding_idx
    long_1 = incremental_indices_1.long();  incremental_indices_1 = None
    add_29 = long_1 + 1;  long_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:164, code: position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length).to(
    position_ids_1 = add_29.to(device(type='cuda', index=0));  add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:173, code: if max_pos > self.weights.size(0):
    l__mod___model_decoder_embed_positions_weights = self.L__mod___model_decoder_embed_positions_weights
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:176, code: return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()
    view_65 = position_ids_1.view(-1);  position_ids_1 = None
    index_select_1 = l__mod___model_decoder_embed_positions_weights.index_select(0, view_65);  l__mod___model_decoder_embed_positions_weights = view_65 = None
    view_66 = index_select_1.view(1, 128, 1024);  index_select_1 = None
    positions = view_66.detach();  view_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1026, code: positions = positions.to(inputs_embeds.device)
    positions_1 = positions.to(device(type='cuda', index=0));  positions = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1028, code: hidden_states = inputs_embeds + positions
    hidden_states_135 = inputs_embeds_1 + positions_1;  inputs_embeds_1 = positions_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1030, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    residual_24 = torch.nn.functional.dropout(hidden_states_135, p = 0.1, training = False);  hidden_states_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1060, code: dropout_probability = torch.rand([])
    dropout_probability_12 = torch.rand([])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_137 = self.L__mod___model_decoder_layers_0_self_attn_layer_norm(residual_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_0_self_attn_q_proj = self.L__mod___model_decoder_layers_0_self_attn_q_proj(hidden_states_137)
    query_states_24 = l__mod___model_decoder_layers_0_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_0_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_0_self_attn_k_proj = self.L__mod___model_decoder_layers_0_self_attn_k_proj(hidden_states_137)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_67 = l__mod___model_decoder_layers_0_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_0_self_attn_k_proj = None
    transpose_60 = view_67.transpose(1, 2);  view_67 = None
    key_states_24 = transpose_60.contiguous();  transpose_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_0_self_attn_v_proj = self.L__mod___model_decoder_layers_0_self_attn_v_proj(hidden_states_137);  hidden_states_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_68 = l__mod___model_decoder_layers_0_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_0_self_attn_v_proj = None
    transpose_61 = view_68.transpose(1, 2);  view_68 = None
    value_states_24 = transpose_61.contiguous();  transpose_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_69 = query_states_24.view(1, 128, 16, 64);  query_states_24 = None
    transpose_62 = view_69.transpose(1, 2);  view_69 = None
    contiguous_38 = transpose_62.contiguous();  transpose_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_25 = contiguous_38.view(16, -1, 64);  contiguous_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    key_states_25 = key_states_24.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    value_states_25 = value_states_24.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_63 = key_states_25.transpose(1, 2);  key_states_25 = None
    attn_weights_24 = torch.bmm(query_states_25, transpose_63);  query_states_25 = transpose_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_71 = attn_weights_24.view(1, 16, 128, 128);  attn_weights_24 = None
    attn_weights_25 = view_71 + combined_attention_mask;  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_26 = attn_weights_25.view(16, 128, 128);  attn_weights_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_27 = torch.nn.functional.softmax(attn_weights_26, dim = -1);  attn_weights_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_12 = torch.nn.functional.dropout(attn_weights_27, p = 0.1, training = False);  attn_weights_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_60 = torch.bmm(attn_probs_12, value_states_25);  attn_probs_12 = value_states_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_61 = attn_output_60.view(1, 16, 128, 64);  attn_output_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    attn_output_62 = attn_output_61.transpose(1, 2);  attn_output_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_63 = attn_output_62.reshape(1, 128, 1024);  attn_output_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    hidden_states_138 = self.L__mod___model_decoder_layers_0_self_attn_out_proj(attn_output_63);  attn_output_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:492, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_139 = torch.nn.functional.dropout(hidden_states_138, p = 0.1, training = False);  hidden_states_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:493, code: hidden_states = residual + hidden_states
    residual_25 = residual_24 + hidden_states_139;  residual_24 = hidden_states_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_141 = self.L__mod___model_decoder_layers_0_encoder_attn_layer_norm(residual_25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_0_encoder_attn_q_proj = self.L__mod___model_decoder_layers_0_encoder_attn_q_proj(hidden_states_141);  hidden_states_141 = None
    query_states_26 = l__mod___model_decoder_layers_0_encoder_attn_q_proj * 0.125;  l__mod___model_decoder_layers_0_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_0_encoder_attn_k_proj = self.L__mod___model_decoder_layers_0_encoder_attn_k_proj(hidden_states_134)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_74 = l__mod___model_decoder_layers_0_encoder_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_0_encoder_attn_k_proj = None
    transpose_65 = view_74.transpose(1, 2);  view_74 = None
    key_states_26 = transpose_65.contiguous();  transpose_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_0_encoder_attn_v_proj = self.L__mod___model_decoder_layers_0_encoder_attn_v_proj(hidden_states_134)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_75 = l__mod___model_decoder_layers_0_encoder_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_0_encoder_attn_v_proj = None
    transpose_66 = view_75.transpose(1, 2);  view_75 = None
    value_states_26 = transpose_66.contiguous();  transpose_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_76 = query_states_26.view(1, 128, 16, 64);  query_states_26 = None
    transpose_67 = view_76.transpose(1, 2);  view_76 = None
    contiguous_41 = transpose_67.contiguous();  transpose_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_27 = contiguous_41.view(16, -1, 64);  contiguous_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    key_states_27 = key_states_26.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    value_states_27 = value_states_26.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_68 = key_states_27.transpose(1, 2);  key_states_27 = None
    attn_weights_28 = torch.bmm(query_states_27, transpose_68);  query_states_27 = transpose_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_29 = torch.nn.functional.softmax(attn_weights_28, dim = -1);  attn_weights_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_13 = torch.nn.functional.dropout(attn_weights_29, p = 0.1, training = False);  attn_weights_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_65 = torch.bmm(attn_probs_13, value_states_27);  attn_probs_13 = value_states_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_66 = attn_output_65.view(1, 16, 128, 64);  attn_output_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    attn_output_67 = attn_output_66.transpose(1, 2);  attn_output_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_68 = attn_output_67.reshape(1, 128, 1024);  attn_output_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    hidden_states_142 = self.L__mod___model_decoder_layers_0_encoder_attn_out_proj(attn_output_68);  attn_output_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:512, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_143 = torch.nn.functional.dropout(hidden_states_142, p = 0.1, training = False);  hidden_states_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:513, code: hidden_states = residual + hidden_states
    residual_26 = residual_25 + hidden_states_143;  residual_25 = hidden_states_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_145 = self.L__mod___model_decoder_layers_0_final_layer_norm(residual_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_0_fc1 = self.L__mod___model_decoder_layers_0_fc1(hidden_states_145);  hidden_states_145 = None
    hidden_states_146 = self.L__mod___model_decoder_layers_0_activation_fn(l__mod___model_decoder_layers_0_fc1);  l__mod___model_decoder_layers_0_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:522, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_147 = torch.nn.functional.dropout(hidden_states_146, p = 0.0, training = False);  hidden_states_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    hidden_states_148 = self.L__mod___model_decoder_layers_0_fc2(hidden_states_147);  hidden_states_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:524, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_149 = torch.nn.functional.dropout(hidden_states_148, p = 0.1, training = False);  hidden_states_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:525, code: hidden_states = residual + hidden_states
    residual_27 = residual_26 + hidden_states_149;  residual_26 = hidden_states_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1060, code: dropout_probability = torch.rand([])
    dropout_probability_13 = torch.rand([])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_152 = self.L__mod___model_decoder_layers_1_self_attn_layer_norm(residual_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_1_self_attn_q_proj = self.L__mod___model_decoder_layers_1_self_attn_q_proj(hidden_states_152)
    query_states_28 = l__mod___model_decoder_layers_1_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_1_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_1_self_attn_k_proj = self.L__mod___model_decoder_layers_1_self_attn_k_proj(hidden_states_152)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_79 = l__mod___model_decoder_layers_1_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_1_self_attn_k_proj = None
    transpose_70 = view_79.transpose(1, 2);  view_79 = None
    key_states_28 = transpose_70.contiguous();  transpose_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_1_self_attn_v_proj = self.L__mod___model_decoder_layers_1_self_attn_v_proj(hidden_states_152);  hidden_states_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_80 = l__mod___model_decoder_layers_1_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_1_self_attn_v_proj = None
    transpose_71 = view_80.transpose(1, 2);  view_80 = None
    value_states_28 = transpose_71.contiguous();  transpose_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_81 = query_states_28.view(1, 128, 16, 64);  query_states_28 = None
    transpose_72 = view_81.transpose(1, 2);  view_81 = None
    contiguous_44 = transpose_72.contiguous();  transpose_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_29 = contiguous_44.view(16, -1, 64);  contiguous_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    key_states_29 = key_states_28.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    value_states_29 = value_states_28.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_73 = key_states_29.transpose(1, 2);  key_states_29 = None
    attn_weights_30 = torch.bmm(query_states_29, transpose_73);  query_states_29 = transpose_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_83 = attn_weights_30.view(1, 16, 128, 128);  attn_weights_30 = None
    attn_weights_31 = view_83 + combined_attention_mask;  view_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_32 = attn_weights_31.view(16, 128, 128);  attn_weights_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_33 = torch.nn.functional.softmax(attn_weights_32, dim = -1);  attn_weights_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_14 = torch.nn.functional.dropout(attn_weights_33, p = 0.1, training = False);  attn_weights_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_70 = torch.bmm(attn_probs_14, value_states_29);  attn_probs_14 = value_states_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_71 = attn_output_70.view(1, 16, 128, 64);  attn_output_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    attn_output_72 = attn_output_71.transpose(1, 2);  attn_output_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_73 = attn_output_72.reshape(1, 128, 1024);  attn_output_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    hidden_states_153 = self.L__mod___model_decoder_layers_1_self_attn_out_proj(attn_output_73);  attn_output_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:492, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_154 = torch.nn.functional.dropout(hidden_states_153, p = 0.1, training = False);  hidden_states_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:493, code: hidden_states = residual + hidden_states
    residual_28 = residual_27 + hidden_states_154;  residual_27 = hidden_states_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_156 = self.L__mod___model_decoder_layers_1_encoder_attn_layer_norm(residual_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_1_encoder_attn_q_proj = self.L__mod___model_decoder_layers_1_encoder_attn_q_proj(hidden_states_156);  hidden_states_156 = None
    query_states_30 = l__mod___model_decoder_layers_1_encoder_attn_q_proj * 0.125;  l__mod___model_decoder_layers_1_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_1_encoder_attn_k_proj = self.L__mod___model_decoder_layers_1_encoder_attn_k_proj(hidden_states_134)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_86 = l__mod___model_decoder_layers_1_encoder_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_1_encoder_attn_k_proj = None
    transpose_75 = view_86.transpose(1, 2);  view_86 = None
    key_states_30 = transpose_75.contiguous();  transpose_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_1_encoder_attn_v_proj = self.L__mod___model_decoder_layers_1_encoder_attn_v_proj(hidden_states_134)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_87 = l__mod___model_decoder_layers_1_encoder_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_1_encoder_attn_v_proj = None
    transpose_76 = view_87.transpose(1, 2);  view_87 = None
    value_states_30 = transpose_76.contiguous();  transpose_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_88 = query_states_30.view(1, 128, 16, 64);  query_states_30 = None
    transpose_77 = view_88.transpose(1, 2);  view_88 = None
    contiguous_47 = transpose_77.contiguous();  transpose_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_31 = contiguous_47.view(16, -1, 64);  contiguous_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    key_states_31 = key_states_30.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    value_states_31 = value_states_30.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_78 = key_states_31.transpose(1, 2);  key_states_31 = None
    attn_weights_34 = torch.bmm(query_states_31, transpose_78);  query_states_31 = transpose_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_35 = torch.nn.functional.softmax(attn_weights_34, dim = -1);  attn_weights_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_15 = torch.nn.functional.dropout(attn_weights_35, p = 0.1, training = False);  attn_weights_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_75 = torch.bmm(attn_probs_15, value_states_31);  attn_probs_15 = value_states_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_76 = attn_output_75.view(1, 16, 128, 64);  attn_output_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    attn_output_77 = attn_output_76.transpose(1, 2);  attn_output_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_78 = attn_output_77.reshape(1, 128, 1024);  attn_output_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    hidden_states_157 = self.L__mod___model_decoder_layers_1_encoder_attn_out_proj(attn_output_78);  attn_output_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:512, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_158 = torch.nn.functional.dropout(hidden_states_157, p = 0.1, training = False);  hidden_states_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:513, code: hidden_states = residual + hidden_states
    residual_29 = residual_28 + hidden_states_158;  residual_28 = hidden_states_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_160 = self.L__mod___model_decoder_layers_1_final_layer_norm(residual_29)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_1_fc1 = self.L__mod___model_decoder_layers_1_fc1(hidden_states_160);  hidden_states_160 = None
    hidden_states_161 = self.L__mod___model_decoder_layers_1_activation_fn(l__mod___model_decoder_layers_1_fc1);  l__mod___model_decoder_layers_1_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:522, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_162 = torch.nn.functional.dropout(hidden_states_161, p = 0.0, training = False);  hidden_states_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    hidden_states_163 = self.L__mod___model_decoder_layers_1_fc2(hidden_states_162);  hidden_states_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:524, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_164 = torch.nn.functional.dropout(hidden_states_163, p = 0.1, training = False);  hidden_states_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:525, code: hidden_states = residual + hidden_states
    residual_30 = residual_29 + hidden_states_164;  residual_29 = hidden_states_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1060, code: dropout_probability = torch.rand([])
    dropout_probability_14 = torch.rand([])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_167 = self.L__mod___model_decoder_layers_2_self_attn_layer_norm(residual_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_2_self_attn_q_proj = self.L__mod___model_decoder_layers_2_self_attn_q_proj(hidden_states_167)
    query_states_32 = l__mod___model_decoder_layers_2_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_2_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_2_self_attn_k_proj = self.L__mod___model_decoder_layers_2_self_attn_k_proj(hidden_states_167)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_91 = l__mod___model_decoder_layers_2_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_2_self_attn_k_proj = None
    transpose_80 = view_91.transpose(1, 2);  view_91 = None
    key_states_32 = transpose_80.contiguous();  transpose_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_2_self_attn_v_proj = self.L__mod___model_decoder_layers_2_self_attn_v_proj(hidden_states_167);  hidden_states_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_92 = l__mod___model_decoder_layers_2_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_2_self_attn_v_proj = None
    transpose_81 = view_92.transpose(1, 2);  view_92 = None
    value_states_32 = transpose_81.contiguous();  transpose_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_93 = query_states_32.view(1, 128, 16, 64);  query_states_32 = None
    transpose_82 = view_93.transpose(1, 2);  view_93 = None
    contiguous_50 = transpose_82.contiguous();  transpose_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_33 = contiguous_50.view(16, -1, 64);  contiguous_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    key_states_33 = key_states_32.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    value_states_33 = value_states_32.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_83 = key_states_33.transpose(1, 2);  key_states_33 = None
    attn_weights_36 = torch.bmm(query_states_33, transpose_83);  query_states_33 = transpose_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_95 = attn_weights_36.view(1, 16, 128, 128);  attn_weights_36 = None
    attn_weights_37 = view_95 + combined_attention_mask;  view_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_38 = attn_weights_37.view(16, 128, 128);  attn_weights_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_39 = torch.nn.functional.softmax(attn_weights_38, dim = -1);  attn_weights_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_16 = torch.nn.functional.dropout(attn_weights_39, p = 0.1, training = False);  attn_weights_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_80 = torch.bmm(attn_probs_16, value_states_33);  attn_probs_16 = value_states_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_81 = attn_output_80.view(1, 16, 128, 64);  attn_output_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    attn_output_82 = attn_output_81.transpose(1, 2);  attn_output_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_83 = attn_output_82.reshape(1, 128, 1024);  attn_output_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    hidden_states_168 = self.L__mod___model_decoder_layers_2_self_attn_out_proj(attn_output_83);  attn_output_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:492, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_169 = torch.nn.functional.dropout(hidden_states_168, p = 0.1, training = False);  hidden_states_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:493, code: hidden_states = residual + hidden_states
    residual_31 = residual_30 + hidden_states_169;  residual_30 = hidden_states_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_171 = self.L__mod___model_decoder_layers_2_encoder_attn_layer_norm(residual_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_2_encoder_attn_q_proj = self.L__mod___model_decoder_layers_2_encoder_attn_q_proj(hidden_states_171);  hidden_states_171 = None
    query_states_34 = l__mod___model_decoder_layers_2_encoder_attn_q_proj * 0.125;  l__mod___model_decoder_layers_2_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_2_encoder_attn_k_proj = self.L__mod___model_decoder_layers_2_encoder_attn_k_proj(hidden_states_134)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_98 = l__mod___model_decoder_layers_2_encoder_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_2_encoder_attn_k_proj = None
    transpose_85 = view_98.transpose(1, 2);  view_98 = None
    key_states_34 = transpose_85.contiguous();  transpose_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_2_encoder_attn_v_proj = self.L__mod___model_decoder_layers_2_encoder_attn_v_proj(hidden_states_134)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_99 = l__mod___model_decoder_layers_2_encoder_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_2_encoder_attn_v_proj = None
    transpose_86 = view_99.transpose(1, 2);  view_99 = None
    value_states_34 = transpose_86.contiguous();  transpose_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_100 = query_states_34.view(1, 128, 16, 64);  query_states_34 = None
    transpose_87 = view_100.transpose(1, 2);  view_100 = None
    contiguous_53 = transpose_87.contiguous();  transpose_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_35 = contiguous_53.view(16, -1, 64);  contiguous_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    key_states_35 = key_states_34.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    value_states_35 = value_states_34.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_88 = key_states_35.transpose(1, 2);  key_states_35 = None
    attn_weights_40 = torch.bmm(query_states_35, transpose_88);  query_states_35 = transpose_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_41 = torch.nn.functional.softmax(attn_weights_40, dim = -1);  attn_weights_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_17 = torch.nn.functional.dropout(attn_weights_41, p = 0.1, training = False);  attn_weights_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_85 = torch.bmm(attn_probs_17, value_states_35);  attn_probs_17 = value_states_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_86 = attn_output_85.view(1, 16, 128, 64);  attn_output_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    attn_output_87 = attn_output_86.transpose(1, 2);  attn_output_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_88 = attn_output_87.reshape(1, 128, 1024);  attn_output_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    hidden_states_172 = self.L__mod___model_decoder_layers_2_encoder_attn_out_proj(attn_output_88);  attn_output_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:512, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_173 = torch.nn.functional.dropout(hidden_states_172, p = 0.1, training = False);  hidden_states_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:513, code: hidden_states = residual + hidden_states
    residual_32 = residual_31 + hidden_states_173;  residual_31 = hidden_states_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_175 = self.L__mod___model_decoder_layers_2_final_layer_norm(residual_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_2_fc1 = self.L__mod___model_decoder_layers_2_fc1(hidden_states_175);  hidden_states_175 = None
    hidden_states_176 = self.L__mod___model_decoder_layers_2_activation_fn(l__mod___model_decoder_layers_2_fc1);  l__mod___model_decoder_layers_2_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:522, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_177 = torch.nn.functional.dropout(hidden_states_176, p = 0.0, training = False);  hidden_states_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    hidden_states_178 = self.L__mod___model_decoder_layers_2_fc2(hidden_states_177);  hidden_states_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:524, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_179 = torch.nn.functional.dropout(hidden_states_178, p = 0.1, training = False);  hidden_states_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:525, code: hidden_states = residual + hidden_states
    residual_33 = residual_32 + hidden_states_179;  residual_32 = hidden_states_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1060, code: dropout_probability = torch.rand([])
    dropout_probability_15 = torch.rand([])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_182 = self.L__mod___model_decoder_layers_3_self_attn_layer_norm(residual_33)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_3_self_attn_q_proj = self.L__mod___model_decoder_layers_3_self_attn_q_proj(hidden_states_182)
    query_states_36 = l__mod___model_decoder_layers_3_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_3_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_3_self_attn_k_proj = self.L__mod___model_decoder_layers_3_self_attn_k_proj(hidden_states_182)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_103 = l__mod___model_decoder_layers_3_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_3_self_attn_k_proj = None
    transpose_90 = view_103.transpose(1, 2);  view_103 = None
    key_states_36 = transpose_90.contiguous();  transpose_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_3_self_attn_v_proj = self.L__mod___model_decoder_layers_3_self_attn_v_proj(hidden_states_182);  hidden_states_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_104 = l__mod___model_decoder_layers_3_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_3_self_attn_v_proj = None
    transpose_91 = view_104.transpose(1, 2);  view_104 = None
    value_states_36 = transpose_91.contiguous();  transpose_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_105 = query_states_36.view(1, 128, 16, 64);  query_states_36 = None
    transpose_92 = view_105.transpose(1, 2);  view_105 = None
    contiguous_56 = transpose_92.contiguous();  transpose_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_37 = contiguous_56.view(16, -1, 64);  contiguous_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    key_states_37 = key_states_36.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    value_states_37 = value_states_36.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_93 = key_states_37.transpose(1, 2);  key_states_37 = None
    attn_weights_42 = torch.bmm(query_states_37, transpose_93);  query_states_37 = transpose_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_107 = attn_weights_42.view(1, 16, 128, 128);  attn_weights_42 = None
    attn_weights_43 = view_107 + combined_attention_mask;  view_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_44 = attn_weights_43.view(16, 128, 128);  attn_weights_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_45 = torch.nn.functional.softmax(attn_weights_44, dim = -1);  attn_weights_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_18 = torch.nn.functional.dropout(attn_weights_45, p = 0.1, training = False);  attn_weights_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_90 = torch.bmm(attn_probs_18, value_states_37);  attn_probs_18 = value_states_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_91 = attn_output_90.view(1, 16, 128, 64);  attn_output_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    attn_output_92 = attn_output_91.transpose(1, 2);  attn_output_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_93 = attn_output_92.reshape(1, 128, 1024);  attn_output_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    hidden_states_183 = self.L__mod___model_decoder_layers_3_self_attn_out_proj(attn_output_93);  attn_output_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:492, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_184 = torch.nn.functional.dropout(hidden_states_183, p = 0.1, training = False);  hidden_states_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:493, code: hidden_states = residual + hidden_states
    residual_34 = residual_33 + hidden_states_184;  residual_33 = hidden_states_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_186 = self.L__mod___model_decoder_layers_3_encoder_attn_layer_norm(residual_34)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_3_encoder_attn_q_proj = self.L__mod___model_decoder_layers_3_encoder_attn_q_proj(hidden_states_186);  hidden_states_186 = None
    query_states_38 = l__mod___model_decoder_layers_3_encoder_attn_q_proj * 0.125;  l__mod___model_decoder_layers_3_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_3_encoder_attn_k_proj = self.L__mod___model_decoder_layers_3_encoder_attn_k_proj(hidden_states_134)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_110 = l__mod___model_decoder_layers_3_encoder_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_3_encoder_attn_k_proj = None
    transpose_95 = view_110.transpose(1, 2);  view_110 = None
    key_states_38 = transpose_95.contiguous();  transpose_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_3_encoder_attn_v_proj = self.L__mod___model_decoder_layers_3_encoder_attn_v_proj(hidden_states_134)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_111 = l__mod___model_decoder_layers_3_encoder_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_3_encoder_attn_v_proj = None
    transpose_96 = view_111.transpose(1, 2);  view_111 = None
    value_states_38 = transpose_96.contiguous();  transpose_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_112 = query_states_38.view(1, 128, 16, 64);  query_states_38 = None
    transpose_97 = view_112.transpose(1, 2);  view_112 = None
    contiguous_59 = transpose_97.contiguous();  transpose_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_39 = contiguous_59.view(16, -1, 64);  contiguous_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    key_states_39 = key_states_38.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    value_states_39 = value_states_38.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_98 = key_states_39.transpose(1, 2);  key_states_39 = None
    attn_weights_46 = torch.bmm(query_states_39, transpose_98);  query_states_39 = transpose_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_47 = torch.nn.functional.softmax(attn_weights_46, dim = -1);  attn_weights_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_19 = torch.nn.functional.dropout(attn_weights_47, p = 0.1, training = False);  attn_weights_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_95 = torch.bmm(attn_probs_19, value_states_39);  attn_probs_19 = value_states_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_96 = attn_output_95.view(1, 16, 128, 64);  attn_output_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    attn_output_97 = attn_output_96.transpose(1, 2);  attn_output_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_98 = attn_output_97.reshape(1, 128, 1024);  attn_output_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    hidden_states_187 = self.L__mod___model_decoder_layers_3_encoder_attn_out_proj(attn_output_98);  attn_output_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:512, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_188 = torch.nn.functional.dropout(hidden_states_187, p = 0.1, training = False);  hidden_states_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:513, code: hidden_states = residual + hidden_states
    residual_35 = residual_34 + hidden_states_188;  residual_34 = hidden_states_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_190 = self.L__mod___model_decoder_layers_3_final_layer_norm(residual_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_3_fc1 = self.L__mod___model_decoder_layers_3_fc1(hidden_states_190);  hidden_states_190 = None
    hidden_states_191 = self.L__mod___model_decoder_layers_3_activation_fn(l__mod___model_decoder_layers_3_fc1);  l__mod___model_decoder_layers_3_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:522, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_192 = torch.nn.functional.dropout(hidden_states_191, p = 0.0, training = False);  hidden_states_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    hidden_states_193 = self.L__mod___model_decoder_layers_3_fc2(hidden_states_192);  hidden_states_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:524, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_194 = torch.nn.functional.dropout(hidden_states_193, p = 0.1, training = False);  hidden_states_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:525, code: hidden_states = residual + hidden_states
    residual_36 = residual_35 + hidden_states_194;  residual_35 = hidden_states_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1060, code: dropout_probability = torch.rand([])
    dropout_probability_16 = torch.rand([])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_197 = self.L__mod___model_decoder_layers_4_self_attn_layer_norm(residual_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_4_self_attn_q_proj = self.L__mod___model_decoder_layers_4_self_attn_q_proj(hidden_states_197)
    query_states_40 = l__mod___model_decoder_layers_4_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_4_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_4_self_attn_k_proj = self.L__mod___model_decoder_layers_4_self_attn_k_proj(hidden_states_197)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_115 = l__mod___model_decoder_layers_4_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_4_self_attn_k_proj = None
    transpose_100 = view_115.transpose(1, 2);  view_115 = None
    key_states_40 = transpose_100.contiguous();  transpose_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_4_self_attn_v_proj = self.L__mod___model_decoder_layers_4_self_attn_v_proj(hidden_states_197);  hidden_states_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_116 = l__mod___model_decoder_layers_4_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_4_self_attn_v_proj = None
    transpose_101 = view_116.transpose(1, 2);  view_116 = None
    value_states_40 = transpose_101.contiguous();  transpose_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_117 = query_states_40.view(1, 128, 16, 64);  query_states_40 = None
    transpose_102 = view_117.transpose(1, 2);  view_117 = None
    contiguous_62 = transpose_102.contiguous();  transpose_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_41 = contiguous_62.view(16, -1, 64);  contiguous_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    key_states_41 = key_states_40.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    value_states_41 = value_states_40.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_103 = key_states_41.transpose(1, 2);  key_states_41 = None
    attn_weights_48 = torch.bmm(query_states_41, transpose_103);  query_states_41 = transpose_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_119 = attn_weights_48.view(1, 16, 128, 128);  attn_weights_48 = None
    attn_weights_49 = view_119 + combined_attention_mask;  view_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_50 = attn_weights_49.view(16, 128, 128);  attn_weights_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_51 = torch.nn.functional.softmax(attn_weights_50, dim = -1);  attn_weights_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_20 = torch.nn.functional.dropout(attn_weights_51, p = 0.1, training = False);  attn_weights_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_100 = torch.bmm(attn_probs_20, value_states_41);  attn_probs_20 = value_states_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_101 = attn_output_100.view(1, 16, 128, 64);  attn_output_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    attn_output_102 = attn_output_101.transpose(1, 2);  attn_output_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_103 = attn_output_102.reshape(1, 128, 1024);  attn_output_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    hidden_states_198 = self.L__mod___model_decoder_layers_4_self_attn_out_proj(attn_output_103);  attn_output_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:492, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_199 = torch.nn.functional.dropout(hidden_states_198, p = 0.1, training = False);  hidden_states_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:493, code: hidden_states = residual + hidden_states
    residual_37 = residual_36 + hidden_states_199;  residual_36 = hidden_states_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_201 = self.L__mod___model_decoder_layers_4_encoder_attn_layer_norm(residual_37)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_4_encoder_attn_q_proj = self.L__mod___model_decoder_layers_4_encoder_attn_q_proj(hidden_states_201);  hidden_states_201 = None
    query_states_42 = l__mod___model_decoder_layers_4_encoder_attn_q_proj * 0.125;  l__mod___model_decoder_layers_4_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_4_encoder_attn_k_proj = self.L__mod___model_decoder_layers_4_encoder_attn_k_proj(hidden_states_134)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_122 = l__mod___model_decoder_layers_4_encoder_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_4_encoder_attn_k_proj = None
    transpose_105 = view_122.transpose(1, 2);  view_122 = None
    key_states_42 = transpose_105.contiguous();  transpose_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_4_encoder_attn_v_proj = self.L__mod___model_decoder_layers_4_encoder_attn_v_proj(hidden_states_134)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_123 = l__mod___model_decoder_layers_4_encoder_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_4_encoder_attn_v_proj = None
    transpose_106 = view_123.transpose(1, 2);  view_123 = None
    value_states_42 = transpose_106.contiguous();  transpose_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_124 = query_states_42.view(1, 128, 16, 64);  query_states_42 = None
    transpose_107 = view_124.transpose(1, 2);  view_124 = None
    contiguous_65 = transpose_107.contiguous();  transpose_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_43 = contiguous_65.view(16, -1, 64);  contiguous_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    key_states_43 = key_states_42.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    value_states_43 = value_states_42.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_108 = key_states_43.transpose(1, 2);  key_states_43 = None
    attn_weights_52 = torch.bmm(query_states_43, transpose_108);  query_states_43 = transpose_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_53 = torch.nn.functional.softmax(attn_weights_52, dim = -1);  attn_weights_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_21 = torch.nn.functional.dropout(attn_weights_53, p = 0.1, training = False);  attn_weights_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_105 = torch.bmm(attn_probs_21, value_states_43);  attn_probs_21 = value_states_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_106 = attn_output_105.view(1, 16, 128, 64);  attn_output_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    attn_output_107 = attn_output_106.transpose(1, 2);  attn_output_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_108 = attn_output_107.reshape(1, 128, 1024);  attn_output_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    hidden_states_202 = self.L__mod___model_decoder_layers_4_encoder_attn_out_proj(attn_output_108);  attn_output_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:512, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_203 = torch.nn.functional.dropout(hidden_states_202, p = 0.1, training = False);  hidden_states_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:513, code: hidden_states = residual + hidden_states
    residual_38 = residual_37 + hidden_states_203;  residual_37 = hidden_states_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_205 = self.L__mod___model_decoder_layers_4_final_layer_norm(residual_38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_4_fc1 = self.L__mod___model_decoder_layers_4_fc1(hidden_states_205);  hidden_states_205 = None
    hidden_states_206 = self.L__mod___model_decoder_layers_4_activation_fn(l__mod___model_decoder_layers_4_fc1);  l__mod___model_decoder_layers_4_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:522, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_207 = torch.nn.functional.dropout(hidden_states_206, p = 0.0, training = False);  hidden_states_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    hidden_states_208 = self.L__mod___model_decoder_layers_4_fc2(hidden_states_207);  hidden_states_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:524, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_209 = torch.nn.functional.dropout(hidden_states_208, p = 0.1, training = False);  hidden_states_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:525, code: hidden_states = residual + hidden_states
    residual_39 = residual_38 + hidden_states_209;  residual_38 = hidden_states_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1060, code: dropout_probability = torch.rand([])
    dropout_probability_17 = torch.rand([])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_212 = self.L__mod___model_decoder_layers_5_self_attn_layer_norm(residual_39)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_5_self_attn_q_proj = self.L__mod___model_decoder_layers_5_self_attn_q_proj(hidden_states_212)
    query_states_44 = l__mod___model_decoder_layers_5_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_5_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_5_self_attn_k_proj = self.L__mod___model_decoder_layers_5_self_attn_k_proj(hidden_states_212)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_127 = l__mod___model_decoder_layers_5_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_5_self_attn_k_proj = None
    transpose_110 = view_127.transpose(1, 2);  view_127 = None
    key_states_44 = transpose_110.contiguous();  transpose_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_5_self_attn_v_proj = self.L__mod___model_decoder_layers_5_self_attn_v_proj(hidden_states_212);  hidden_states_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_128 = l__mod___model_decoder_layers_5_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_5_self_attn_v_proj = None
    transpose_111 = view_128.transpose(1, 2);  view_128 = None
    value_states_44 = transpose_111.contiguous();  transpose_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_129 = query_states_44.view(1, 128, 16, 64);  query_states_44 = None
    transpose_112 = view_129.transpose(1, 2);  view_129 = None
    contiguous_68 = transpose_112.contiguous();  transpose_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_45 = contiguous_68.view(16, -1, 64);  contiguous_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    key_states_45 = key_states_44.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    value_states_45 = value_states_44.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_113 = key_states_45.transpose(1, 2);  key_states_45 = None
    attn_weights_54 = torch.bmm(query_states_45, transpose_113);  query_states_45 = transpose_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_131 = attn_weights_54.view(1, 16, 128, 128);  attn_weights_54 = None
    attn_weights_55 = view_131 + combined_attention_mask;  view_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_56 = attn_weights_55.view(16, 128, 128);  attn_weights_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_57 = torch.nn.functional.softmax(attn_weights_56, dim = -1);  attn_weights_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_22 = torch.nn.functional.dropout(attn_weights_57, p = 0.1, training = False);  attn_weights_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_110 = torch.bmm(attn_probs_22, value_states_45);  attn_probs_22 = value_states_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_111 = attn_output_110.view(1, 16, 128, 64);  attn_output_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    attn_output_112 = attn_output_111.transpose(1, 2);  attn_output_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_113 = attn_output_112.reshape(1, 128, 1024);  attn_output_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    hidden_states_213 = self.L__mod___model_decoder_layers_5_self_attn_out_proj(attn_output_113);  attn_output_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:492, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_214 = torch.nn.functional.dropout(hidden_states_213, p = 0.1, training = False);  hidden_states_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:493, code: hidden_states = residual + hidden_states
    residual_40 = residual_39 + hidden_states_214;  residual_39 = hidden_states_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_216 = self.L__mod___model_decoder_layers_5_encoder_attn_layer_norm(residual_40)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_5_encoder_attn_q_proj = self.L__mod___model_decoder_layers_5_encoder_attn_q_proj(hidden_states_216);  hidden_states_216 = None
    query_states_46 = l__mod___model_decoder_layers_5_encoder_attn_q_proj * 0.125;  l__mod___model_decoder_layers_5_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_5_encoder_attn_k_proj = self.L__mod___model_decoder_layers_5_encoder_attn_k_proj(hidden_states_134)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_134 = l__mod___model_decoder_layers_5_encoder_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_5_encoder_attn_k_proj = None
    transpose_115 = view_134.transpose(1, 2);  view_134 = None
    key_states_46 = transpose_115.contiguous();  transpose_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_5_encoder_attn_v_proj = self.L__mod___model_decoder_layers_5_encoder_attn_v_proj(hidden_states_134)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_135 = l__mod___model_decoder_layers_5_encoder_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_5_encoder_attn_v_proj = None
    transpose_116 = view_135.transpose(1, 2);  view_135 = None
    value_states_46 = transpose_116.contiguous();  transpose_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_136 = query_states_46.view(1, 128, 16, 64);  query_states_46 = None
    transpose_117 = view_136.transpose(1, 2);  view_136 = None
    contiguous_71 = transpose_117.contiguous();  transpose_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_47 = contiguous_71.view(16, -1, 64);  contiguous_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    key_states_47 = key_states_46.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    value_states_47 = value_states_46.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_118 = key_states_47.transpose(1, 2);  key_states_47 = None
    attn_weights_58 = torch.bmm(query_states_47, transpose_118);  query_states_47 = transpose_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_59 = torch.nn.functional.softmax(attn_weights_58, dim = -1);  attn_weights_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_23 = torch.nn.functional.dropout(attn_weights_59, p = 0.1, training = False);  attn_weights_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_115 = torch.bmm(attn_probs_23, value_states_47);  attn_probs_23 = value_states_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_116 = attn_output_115.view(1, 16, 128, 64);  attn_output_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    attn_output_117 = attn_output_116.transpose(1, 2);  attn_output_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_118 = attn_output_117.reshape(1, 128, 1024);  attn_output_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    hidden_states_217 = self.L__mod___model_decoder_layers_5_encoder_attn_out_proj(attn_output_118);  attn_output_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:512, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_218 = torch.nn.functional.dropout(hidden_states_217, p = 0.1, training = False);  hidden_states_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:513, code: hidden_states = residual + hidden_states
    residual_41 = residual_40 + hidden_states_218;  residual_40 = hidden_states_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_220 = self.L__mod___model_decoder_layers_5_final_layer_norm(residual_41)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_5_fc1 = self.L__mod___model_decoder_layers_5_fc1(hidden_states_220);  hidden_states_220 = None
    hidden_states_221 = self.L__mod___model_decoder_layers_5_activation_fn(l__mod___model_decoder_layers_5_fc1);  l__mod___model_decoder_layers_5_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:522, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_222 = torch.nn.functional.dropout(hidden_states_221, p = 0.0, training = False);  hidden_states_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    hidden_states_223 = self.L__mod___model_decoder_layers_5_fc2(hidden_states_222);  hidden_states_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:524, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_224 = torch.nn.functional.dropout(hidden_states_223, p = 0.1, training = False);  hidden_states_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:525, code: hidden_states = residual + hidden_states
    residual_42 = residual_41 + hidden_states_224;  residual_41 = hidden_states_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1060, code: dropout_probability = torch.rand([])
    dropout_probability_18 = torch.rand([])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_227 = self.L__mod___model_decoder_layers_6_self_attn_layer_norm(residual_42)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_6_self_attn_q_proj = self.L__mod___model_decoder_layers_6_self_attn_q_proj(hidden_states_227)
    query_states_48 = l__mod___model_decoder_layers_6_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_6_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_6_self_attn_k_proj = self.L__mod___model_decoder_layers_6_self_attn_k_proj(hidden_states_227)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_139 = l__mod___model_decoder_layers_6_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_6_self_attn_k_proj = None
    transpose_120 = view_139.transpose(1, 2);  view_139 = None
    key_states_48 = transpose_120.contiguous();  transpose_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_6_self_attn_v_proj = self.L__mod___model_decoder_layers_6_self_attn_v_proj(hidden_states_227);  hidden_states_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_140 = l__mod___model_decoder_layers_6_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_6_self_attn_v_proj = None
    transpose_121 = view_140.transpose(1, 2);  view_140 = None
    value_states_48 = transpose_121.contiguous();  transpose_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_141 = query_states_48.view(1, 128, 16, 64);  query_states_48 = None
    transpose_122 = view_141.transpose(1, 2);  view_141 = None
    contiguous_74 = transpose_122.contiguous();  transpose_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_49 = contiguous_74.view(16, -1, 64);  contiguous_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    key_states_49 = key_states_48.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    value_states_49 = value_states_48.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_123 = key_states_49.transpose(1, 2);  key_states_49 = None
    attn_weights_60 = torch.bmm(query_states_49, transpose_123);  query_states_49 = transpose_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_143 = attn_weights_60.view(1, 16, 128, 128);  attn_weights_60 = None
    attn_weights_61 = view_143 + combined_attention_mask;  view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_62 = attn_weights_61.view(16, 128, 128);  attn_weights_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_63 = torch.nn.functional.softmax(attn_weights_62, dim = -1);  attn_weights_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_24 = torch.nn.functional.dropout(attn_weights_63, p = 0.1, training = False);  attn_weights_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_120 = torch.bmm(attn_probs_24, value_states_49);  attn_probs_24 = value_states_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_121 = attn_output_120.view(1, 16, 128, 64);  attn_output_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    attn_output_122 = attn_output_121.transpose(1, 2);  attn_output_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_123 = attn_output_122.reshape(1, 128, 1024);  attn_output_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    hidden_states_228 = self.L__mod___model_decoder_layers_6_self_attn_out_proj(attn_output_123);  attn_output_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:492, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_229 = torch.nn.functional.dropout(hidden_states_228, p = 0.1, training = False);  hidden_states_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:493, code: hidden_states = residual + hidden_states
    residual_43 = residual_42 + hidden_states_229;  residual_42 = hidden_states_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_231 = self.L__mod___model_decoder_layers_6_encoder_attn_layer_norm(residual_43)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_6_encoder_attn_q_proj = self.L__mod___model_decoder_layers_6_encoder_attn_q_proj(hidden_states_231);  hidden_states_231 = None
    query_states_50 = l__mod___model_decoder_layers_6_encoder_attn_q_proj * 0.125;  l__mod___model_decoder_layers_6_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_6_encoder_attn_k_proj = self.L__mod___model_decoder_layers_6_encoder_attn_k_proj(hidden_states_134)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_146 = l__mod___model_decoder_layers_6_encoder_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_6_encoder_attn_k_proj = None
    transpose_125 = view_146.transpose(1, 2);  view_146 = None
    key_states_50 = transpose_125.contiguous();  transpose_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_6_encoder_attn_v_proj = self.L__mod___model_decoder_layers_6_encoder_attn_v_proj(hidden_states_134)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_147 = l__mod___model_decoder_layers_6_encoder_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_6_encoder_attn_v_proj = None
    transpose_126 = view_147.transpose(1, 2);  view_147 = None
    value_states_50 = transpose_126.contiguous();  transpose_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_148 = query_states_50.view(1, 128, 16, 64);  query_states_50 = None
    transpose_127 = view_148.transpose(1, 2);  view_148 = None
    contiguous_77 = transpose_127.contiguous();  transpose_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_51 = contiguous_77.view(16, -1, 64);  contiguous_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    key_states_51 = key_states_50.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    value_states_51 = value_states_50.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_128 = key_states_51.transpose(1, 2);  key_states_51 = None
    attn_weights_64 = torch.bmm(query_states_51, transpose_128);  query_states_51 = transpose_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_65 = torch.nn.functional.softmax(attn_weights_64, dim = -1);  attn_weights_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_25 = torch.nn.functional.dropout(attn_weights_65, p = 0.1, training = False);  attn_weights_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_125 = torch.bmm(attn_probs_25, value_states_51);  attn_probs_25 = value_states_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_126 = attn_output_125.view(1, 16, 128, 64);  attn_output_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    attn_output_127 = attn_output_126.transpose(1, 2);  attn_output_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_128 = attn_output_127.reshape(1, 128, 1024);  attn_output_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    hidden_states_232 = self.L__mod___model_decoder_layers_6_encoder_attn_out_proj(attn_output_128);  attn_output_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:512, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_233 = torch.nn.functional.dropout(hidden_states_232, p = 0.1, training = False);  hidden_states_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:513, code: hidden_states = residual + hidden_states
    residual_44 = residual_43 + hidden_states_233;  residual_43 = hidden_states_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_235 = self.L__mod___model_decoder_layers_6_final_layer_norm(residual_44)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_6_fc1 = self.L__mod___model_decoder_layers_6_fc1(hidden_states_235);  hidden_states_235 = None
    hidden_states_236 = self.L__mod___model_decoder_layers_6_activation_fn(l__mod___model_decoder_layers_6_fc1);  l__mod___model_decoder_layers_6_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:522, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_237 = torch.nn.functional.dropout(hidden_states_236, p = 0.0, training = False);  hidden_states_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    hidden_states_238 = self.L__mod___model_decoder_layers_6_fc2(hidden_states_237);  hidden_states_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:524, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_239 = torch.nn.functional.dropout(hidden_states_238, p = 0.1, training = False);  hidden_states_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:525, code: hidden_states = residual + hidden_states
    residual_45 = residual_44 + hidden_states_239;  residual_44 = hidden_states_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1060, code: dropout_probability = torch.rand([])
    dropout_probability_19 = torch.rand([])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_242 = self.L__mod___model_decoder_layers_7_self_attn_layer_norm(residual_45)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_7_self_attn_q_proj = self.L__mod___model_decoder_layers_7_self_attn_q_proj(hidden_states_242)
    query_states_52 = l__mod___model_decoder_layers_7_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_7_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_7_self_attn_k_proj = self.L__mod___model_decoder_layers_7_self_attn_k_proj(hidden_states_242)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_151 = l__mod___model_decoder_layers_7_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_7_self_attn_k_proj = None
    transpose_130 = view_151.transpose(1, 2);  view_151 = None
    key_states_52 = transpose_130.contiguous();  transpose_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_7_self_attn_v_proj = self.L__mod___model_decoder_layers_7_self_attn_v_proj(hidden_states_242);  hidden_states_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_152 = l__mod___model_decoder_layers_7_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_7_self_attn_v_proj = None
    transpose_131 = view_152.transpose(1, 2);  view_152 = None
    value_states_52 = transpose_131.contiguous();  transpose_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_153 = query_states_52.view(1, 128, 16, 64);  query_states_52 = None
    transpose_132 = view_153.transpose(1, 2);  view_153 = None
    contiguous_80 = transpose_132.contiguous();  transpose_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_53 = contiguous_80.view(16, -1, 64);  contiguous_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    key_states_53 = key_states_52.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    value_states_53 = value_states_52.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_133 = key_states_53.transpose(1, 2);  key_states_53 = None
    attn_weights_66 = torch.bmm(query_states_53, transpose_133);  query_states_53 = transpose_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_155 = attn_weights_66.view(1, 16, 128, 128);  attn_weights_66 = None
    attn_weights_67 = view_155 + combined_attention_mask;  view_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_68 = attn_weights_67.view(16, 128, 128);  attn_weights_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_69 = torch.nn.functional.softmax(attn_weights_68, dim = -1);  attn_weights_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_26 = torch.nn.functional.dropout(attn_weights_69, p = 0.1, training = False);  attn_weights_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_130 = torch.bmm(attn_probs_26, value_states_53);  attn_probs_26 = value_states_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_131 = attn_output_130.view(1, 16, 128, 64);  attn_output_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    attn_output_132 = attn_output_131.transpose(1, 2);  attn_output_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_133 = attn_output_132.reshape(1, 128, 1024);  attn_output_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    hidden_states_243 = self.L__mod___model_decoder_layers_7_self_attn_out_proj(attn_output_133);  attn_output_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:492, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_244 = torch.nn.functional.dropout(hidden_states_243, p = 0.1, training = False);  hidden_states_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:493, code: hidden_states = residual + hidden_states
    residual_46 = residual_45 + hidden_states_244;  residual_45 = hidden_states_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_246 = self.L__mod___model_decoder_layers_7_encoder_attn_layer_norm(residual_46)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_7_encoder_attn_q_proj = self.L__mod___model_decoder_layers_7_encoder_attn_q_proj(hidden_states_246);  hidden_states_246 = None
    query_states_54 = l__mod___model_decoder_layers_7_encoder_attn_q_proj * 0.125;  l__mod___model_decoder_layers_7_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_7_encoder_attn_k_proj = self.L__mod___model_decoder_layers_7_encoder_attn_k_proj(hidden_states_134)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_158 = l__mod___model_decoder_layers_7_encoder_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_7_encoder_attn_k_proj = None
    transpose_135 = view_158.transpose(1, 2);  view_158 = None
    key_states_54 = transpose_135.contiguous();  transpose_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_7_encoder_attn_v_proj = self.L__mod___model_decoder_layers_7_encoder_attn_v_proj(hidden_states_134)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_159 = l__mod___model_decoder_layers_7_encoder_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_7_encoder_attn_v_proj = None
    transpose_136 = view_159.transpose(1, 2);  view_159 = None
    value_states_54 = transpose_136.contiguous();  transpose_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_160 = query_states_54.view(1, 128, 16, 64);  query_states_54 = None
    transpose_137 = view_160.transpose(1, 2);  view_160 = None
    contiguous_83 = transpose_137.contiguous();  transpose_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_55 = contiguous_83.view(16, -1, 64);  contiguous_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    key_states_55 = key_states_54.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    value_states_55 = value_states_54.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_138 = key_states_55.transpose(1, 2);  key_states_55 = None
    attn_weights_70 = torch.bmm(query_states_55, transpose_138);  query_states_55 = transpose_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_71 = torch.nn.functional.softmax(attn_weights_70, dim = -1);  attn_weights_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_27 = torch.nn.functional.dropout(attn_weights_71, p = 0.1, training = False);  attn_weights_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_135 = torch.bmm(attn_probs_27, value_states_55);  attn_probs_27 = value_states_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_136 = attn_output_135.view(1, 16, 128, 64);  attn_output_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    attn_output_137 = attn_output_136.transpose(1, 2);  attn_output_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_138 = attn_output_137.reshape(1, 128, 1024);  attn_output_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    hidden_states_247 = self.L__mod___model_decoder_layers_7_encoder_attn_out_proj(attn_output_138);  attn_output_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:512, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_248 = torch.nn.functional.dropout(hidden_states_247, p = 0.1, training = False);  hidden_states_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:513, code: hidden_states = residual + hidden_states
    residual_47 = residual_46 + hidden_states_248;  residual_46 = hidden_states_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_250 = self.L__mod___model_decoder_layers_7_final_layer_norm(residual_47)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_7_fc1 = self.L__mod___model_decoder_layers_7_fc1(hidden_states_250);  hidden_states_250 = None
    hidden_states_251 = self.L__mod___model_decoder_layers_7_activation_fn(l__mod___model_decoder_layers_7_fc1);  l__mod___model_decoder_layers_7_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:522, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_252 = torch.nn.functional.dropout(hidden_states_251, p = 0.0, training = False);  hidden_states_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    hidden_states_253 = self.L__mod___model_decoder_layers_7_fc2(hidden_states_252);  hidden_states_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:524, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_254 = torch.nn.functional.dropout(hidden_states_253, p = 0.1, training = False);  hidden_states_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:525, code: hidden_states = residual + hidden_states
    residual_48 = residual_47 + hidden_states_254;  residual_47 = hidden_states_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1060, code: dropout_probability = torch.rand([])
    dropout_probability_20 = torch.rand([])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_257 = self.L__mod___model_decoder_layers_8_self_attn_layer_norm(residual_48)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_8_self_attn_q_proj = self.L__mod___model_decoder_layers_8_self_attn_q_proj(hidden_states_257)
    query_states_56 = l__mod___model_decoder_layers_8_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_8_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_8_self_attn_k_proj = self.L__mod___model_decoder_layers_8_self_attn_k_proj(hidden_states_257)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_163 = l__mod___model_decoder_layers_8_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_8_self_attn_k_proj = None
    transpose_140 = view_163.transpose(1, 2);  view_163 = None
    key_states_56 = transpose_140.contiguous();  transpose_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_8_self_attn_v_proj = self.L__mod___model_decoder_layers_8_self_attn_v_proj(hidden_states_257);  hidden_states_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_164 = l__mod___model_decoder_layers_8_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_8_self_attn_v_proj = None
    transpose_141 = view_164.transpose(1, 2);  view_164 = None
    value_states_56 = transpose_141.contiguous();  transpose_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_165 = query_states_56.view(1, 128, 16, 64);  query_states_56 = None
    transpose_142 = view_165.transpose(1, 2);  view_165 = None
    contiguous_86 = transpose_142.contiguous();  transpose_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_57 = contiguous_86.view(16, -1, 64);  contiguous_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    key_states_57 = key_states_56.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    value_states_57 = value_states_56.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_143 = key_states_57.transpose(1, 2);  key_states_57 = None
    attn_weights_72 = torch.bmm(query_states_57, transpose_143);  query_states_57 = transpose_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_167 = attn_weights_72.view(1, 16, 128, 128);  attn_weights_72 = None
    attn_weights_73 = view_167 + combined_attention_mask;  view_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_74 = attn_weights_73.view(16, 128, 128);  attn_weights_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_75 = torch.nn.functional.softmax(attn_weights_74, dim = -1);  attn_weights_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_28 = torch.nn.functional.dropout(attn_weights_75, p = 0.1, training = False);  attn_weights_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_140 = torch.bmm(attn_probs_28, value_states_57);  attn_probs_28 = value_states_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_141 = attn_output_140.view(1, 16, 128, 64);  attn_output_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    attn_output_142 = attn_output_141.transpose(1, 2);  attn_output_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_143 = attn_output_142.reshape(1, 128, 1024);  attn_output_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    hidden_states_258 = self.L__mod___model_decoder_layers_8_self_attn_out_proj(attn_output_143);  attn_output_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:492, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_259 = torch.nn.functional.dropout(hidden_states_258, p = 0.1, training = False);  hidden_states_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:493, code: hidden_states = residual + hidden_states
    residual_49 = residual_48 + hidden_states_259;  residual_48 = hidden_states_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_261 = self.L__mod___model_decoder_layers_8_encoder_attn_layer_norm(residual_49)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_8_encoder_attn_q_proj = self.L__mod___model_decoder_layers_8_encoder_attn_q_proj(hidden_states_261);  hidden_states_261 = None
    query_states_58 = l__mod___model_decoder_layers_8_encoder_attn_q_proj * 0.125;  l__mod___model_decoder_layers_8_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_8_encoder_attn_k_proj = self.L__mod___model_decoder_layers_8_encoder_attn_k_proj(hidden_states_134)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_170 = l__mod___model_decoder_layers_8_encoder_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_8_encoder_attn_k_proj = None
    transpose_145 = view_170.transpose(1, 2);  view_170 = None
    key_states_58 = transpose_145.contiguous();  transpose_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_8_encoder_attn_v_proj = self.L__mod___model_decoder_layers_8_encoder_attn_v_proj(hidden_states_134)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_171 = l__mod___model_decoder_layers_8_encoder_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_8_encoder_attn_v_proj = None
    transpose_146 = view_171.transpose(1, 2);  view_171 = None
    value_states_58 = transpose_146.contiguous();  transpose_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_172 = query_states_58.view(1, 128, 16, 64);  query_states_58 = None
    transpose_147 = view_172.transpose(1, 2);  view_172 = None
    contiguous_89 = transpose_147.contiguous();  transpose_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_59 = contiguous_89.view(16, -1, 64);  contiguous_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    key_states_59 = key_states_58.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    value_states_59 = value_states_58.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_148 = key_states_59.transpose(1, 2);  key_states_59 = None
    attn_weights_76 = torch.bmm(query_states_59, transpose_148);  query_states_59 = transpose_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_77 = torch.nn.functional.softmax(attn_weights_76, dim = -1);  attn_weights_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_29 = torch.nn.functional.dropout(attn_weights_77, p = 0.1, training = False);  attn_weights_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_145 = torch.bmm(attn_probs_29, value_states_59);  attn_probs_29 = value_states_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_146 = attn_output_145.view(1, 16, 128, 64);  attn_output_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    attn_output_147 = attn_output_146.transpose(1, 2);  attn_output_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_148 = attn_output_147.reshape(1, 128, 1024);  attn_output_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    hidden_states_262 = self.L__mod___model_decoder_layers_8_encoder_attn_out_proj(attn_output_148);  attn_output_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:512, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_263 = torch.nn.functional.dropout(hidden_states_262, p = 0.1, training = False);  hidden_states_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:513, code: hidden_states = residual + hidden_states
    residual_50 = residual_49 + hidden_states_263;  residual_49 = hidden_states_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_265 = self.L__mod___model_decoder_layers_8_final_layer_norm(residual_50)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_8_fc1 = self.L__mod___model_decoder_layers_8_fc1(hidden_states_265);  hidden_states_265 = None
    hidden_states_266 = self.L__mod___model_decoder_layers_8_activation_fn(l__mod___model_decoder_layers_8_fc1);  l__mod___model_decoder_layers_8_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:522, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_267 = torch.nn.functional.dropout(hidden_states_266, p = 0.0, training = False);  hidden_states_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    hidden_states_268 = self.L__mod___model_decoder_layers_8_fc2(hidden_states_267);  hidden_states_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:524, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_269 = torch.nn.functional.dropout(hidden_states_268, p = 0.1, training = False);  hidden_states_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:525, code: hidden_states = residual + hidden_states
    residual_51 = residual_50 + hidden_states_269;  residual_50 = hidden_states_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1060, code: dropout_probability = torch.rand([])
    dropout_probability_21 = torch.rand([])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_272 = self.L__mod___model_decoder_layers_9_self_attn_layer_norm(residual_51)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_9_self_attn_q_proj = self.L__mod___model_decoder_layers_9_self_attn_q_proj(hidden_states_272)
    query_states_60 = l__mod___model_decoder_layers_9_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_9_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_9_self_attn_k_proj = self.L__mod___model_decoder_layers_9_self_attn_k_proj(hidden_states_272)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_175 = l__mod___model_decoder_layers_9_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_9_self_attn_k_proj = None
    transpose_150 = view_175.transpose(1, 2);  view_175 = None
    key_states_60 = transpose_150.contiguous();  transpose_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_9_self_attn_v_proj = self.L__mod___model_decoder_layers_9_self_attn_v_proj(hidden_states_272);  hidden_states_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_176 = l__mod___model_decoder_layers_9_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_9_self_attn_v_proj = None
    transpose_151 = view_176.transpose(1, 2);  view_176 = None
    value_states_60 = transpose_151.contiguous();  transpose_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_177 = query_states_60.view(1, 128, 16, 64);  query_states_60 = None
    transpose_152 = view_177.transpose(1, 2);  view_177 = None
    contiguous_92 = transpose_152.contiguous();  transpose_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_61 = contiguous_92.view(16, -1, 64);  contiguous_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    key_states_61 = key_states_60.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    value_states_61 = value_states_60.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_153 = key_states_61.transpose(1, 2);  key_states_61 = None
    attn_weights_78 = torch.bmm(query_states_61, transpose_153);  query_states_61 = transpose_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_179 = attn_weights_78.view(1, 16, 128, 128);  attn_weights_78 = None
    attn_weights_79 = view_179 + combined_attention_mask;  view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_80 = attn_weights_79.view(16, 128, 128);  attn_weights_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_81 = torch.nn.functional.softmax(attn_weights_80, dim = -1);  attn_weights_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_30 = torch.nn.functional.dropout(attn_weights_81, p = 0.1, training = False);  attn_weights_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_150 = torch.bmm(attn_probs_30, value_states_61);  attn_probs_30 = value_states_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_151 = attn_output_150.view(1, 16, 128, 64);  attn_output_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    attn_output_152 = attn_output_151.transpose(1, 2);  attn_output_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_153 = attn_output_152.reshape(1, 128, 1024);  attn_output_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    hidden_states_273 = self.L__mod___model_decoder_layers_9_self_attn_out_proj(attn_output_153);  attn_output_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:492, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_274 = torch.nn.functional.dropout(hidden_states_273, p = 0.1, training = False);  hidden_states_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:493, code: hidden_states = residual + hidden_states
    residual_52 = residual_51 + hidden_states_274;  residual_51 = hidden_states_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_276 = self.L__mod___model_decoder_layers_9_encoder_attn_layer_norm(residual_52)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_9_encoder_attn_q_proj = self.L__mod___model_decoder_layers_9_encoder_attn_q_proj(hidden_states_276);  hidden_states_276 = None
    query_states_62 = l__mod___model_decoder_layers_9_encoder_attn_q_proj * 0.125;  l__mod___model_decoder_layers_9_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_9_encoder_attn_k_proj = self.L__mod___model_decoder_layers_9_encoder_attn_k_proj(hidden_states_134)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_182 = l__mod___model_decoder_layers_9_encoder_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_9_encoder_attn_k_proj = None
    transpose_155 = view_182.transpose(1, 2);  view_182 = None
    key_states_62 = transpose_155.contiguous();  transpose_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_9_encoder_attn_v_proj = self.L__mod___model_decoder_layers_9_encoder_attn_v_proj(hidden_states_134)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_183 = l__mod___model_decoder_layers_9_encoder_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_9_encoder_attn_v_proj = None
    transpose_156 = view_183.transpose(1, 2);  view_183 = None
    value_states_62 = transpose_156.contiguous();  transpose_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_184 = query_states_62.view(1, 128, 16, 64);  query_states_62 = None
    transpose_157 = view_184.transpose(1, 2);  view_184 = None
    contiguous_95 = transpose_157.contiguous();  transpose_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_63 = contiguous_95.view(16, -1, 64);  contiguous_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    key_states_63 = key_states_62.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    value_states_63 = value_states_62.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_158 = key_states_63.transpose(1, 2);  key_states_63 = None
    attn_weights_82 = torch.bmm(query_states_63, transpose_158);  query_states_63 = transpose_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_83 = torch.nn.functional.softmax(attn_weights_82, dim = -1);  attn_weights_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_31 = torch.nn.functional.dropout(attn_weights_83, p = 0.1, training = False);  attn_weights_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_155 = torch.bmm(attn_probs_31, value_states_63);  attn_probs_31 = value_states_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_156 = attn_output_155.view(1, 16, 128, 64);  attn_output_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    attn_output_157 = attn_output_156.transpose(1, 2);  attn_output_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_158 = attn_output_157.reshape(1, 128, 1024);  attn_output_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    hidden_states_277 = self.L__mod___model_decoder_layers_9_encoder_attn_out_proj(attn_output_158);  attn_output_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:512, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_278 = torch.nn.functional.dropout(hidden_states_277, p = 0.1, training = False);  hidden_states_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:513, code: hidden_states = residual + hidden_states
    residual_53 = residual_52 + hidden_states_278;  residual_52 = hidden_states_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_280 = self.L__mod___model_decoder_layers_9_final_layer_norm(residual_53)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_9_fc1 = self.L__mod___model_decoder_layers_9_fc1(hidden_states_280);  hidden_states_280 = None
    hidden_states_281 = self.L__mod___model_decoder_layers_9_activation_fn(l__mod___model_decoder_layers_9_fc1);  l__mod___model_decoder_layers_9_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:522, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_282 = torch.nn.functional.dropout(hidden_states_281, p = 0.0, training = False);  hidden_states_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    hidden_states_283 = self.L__mod___model_decoder_layers_9_fc2(hidden_states_282);  hidden_states_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:524, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_284 = torch.nn.functional.dropout(hidden_states_283, p = 0.1, training = False);  hidden_states_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:525, code: hidden_states = residual + hidden_states
    residual_54 = residual_53 + hidden_states_284;  residual_53 = hidden_states_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1060, code: dropout_probability = torch.rand([])
    dropout_probability_22 = torch.rand([])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_287 = self.L__mod___model_decoder_layers_10_self_attn_layer_norm(residual_54)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_10_self_attn_q_proj = self.L__mod___model_decoder_layers_10_self_attn_q_proj(hidden_states_287)
    query_states_64 = l__mod___model_decoder_layers_10_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_10_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_10_self_attn_k_proj = self.L__mod___model_decoder_layers_10_self_attn_k_proj(hidden_states_287)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_187 = l__mod___model_decoder_layers_10_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_10_self_attn_k_proj = None
    transpose_160 = view_187.transpose(1, 2);  view_187 = None
    key_states_64 = transpose_160.contiguous();  transpose_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_10_self_attn_v_proj = self.L__mod___model_decoder_layers_10_self_attn_v_proj(hidden_states_287);  hidden_states_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_188 = l__mod___model_decoder_layers_10_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_10_self_attn_v_proj = None
    transpose_161 = view_188.transpose(1, 2);  view_188 = None
    value_states_64 = transpose_161.contiguous();  transpose_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_189 = query_states_64.view(1, 128, 16, 64);  query_states_64 = None
    transpose_162 = view_189.transpose(1, 2);  view_189 = None
    contiguous_98 = transpose_162.contiguous();  transpose_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_65 = contiguous_98.view(16, -1, 64);  contiguous_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    key_states_65 = key_states_64.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    value_states_65 = value_states_64.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_163 = key_states_65.transpose(1, 2);  key_states_65 = None
    attn_weights_84 = torch.bmm(query_states_65, transpose_163);  query_states_65 = transpose_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_191 = attn_weights_84.view(1, 16, 128, 128);  attn_weights_84 = None
    attn_weights_85 = view_191 + combined_attention_mask;  view_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_86 = attn_weights_85.view(16, 128, 128);  attn_weights_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_87 = torch.nn.functional.softmax(attn_weights_86, dim = -1);  attn_weights_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_32 = torch.nn.functional.dropout(attn_weights_87, p = 0.1, training = False);  attn_weights_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_160 = torch.bmm(attn_probs_32, value_states_65);  attn_probs_32 = value_states_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_161 = attn_output_160.view(1, 16, 128, 64);  attn_output_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    attn_output_162 = attn_output_161.transpose(1, 2);  attn_output_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_163 = attn_output_162.reshape(1, 128, 1024);  attn_output_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    hidden_states_288 = self.L__mod___model_decoder_layers_10_self_attn_out_proj(attn_output_163);  attn_output_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:492, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_289 = torch.nn.functional.dropout(hidden_states_288, p = 0.1, training = False);  hidden_states_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:493, code: hidden_states = residual + hidden_states
    residual_55 = residual_54 + hidden_states_289;  residual_54 = hidden_states_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_291 = self.L__mod___model_decoder_layers_10_encoder_attn_layer_norm(residual_55)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_10_encoder_attn_q_proj = self.L__mod___model_decoder_layers_10_encoder_attn_q_proj(hidden_states_291);  hidden_states_291 = None
    query_states_66 = l__mod___model_decoder_layers_10_encoder_attn_q_proj * 0.125;  l__mod___model_decoder_layers_10_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_10_encoder_attn_k_proj = self.L__mod___model_decoder_layers_10_encoder_attn_k_proj(hidden_states_134)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_194 = l__mod___model_decoder_layers_10_encoder_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_10_encoder_attn_k_proj = None
    transpose_165 = view_194.transpose(1, 2);  view_194 = None
    key_states_66 = transpose_165.contiguous();  transpose_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_10_encoder_attn_v_proj = self.L__mod___model_decoder_layers_10_encoder_attn_v_proj(hidden_states_134)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_195 = l__mod___model_decoder_layers_10_encoder_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_10_encoder_attn_v_proj = None
    transpose_166 = view_195.transpose(1, 2);  view_195 = None
    value_states_66 = transpose_166.contiguous();  transpose_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_196 = query_states_66.view(1, 128, 16, 64);  query_states_66 = None
    transpose_167 = view_196.transpose(1, 2);  view_196 = None
    contiguous_101 = transpose_167.contiguous();  transpose_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_67 = contiguous_101.view(16, -1, 64);  contiguous_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    key_states_67 = key_states_66.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    value_states_67 = value_states_66.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_168 = key_states_67.transpose(1, 2);  key_states_67 = None
    attn_weights_88 = torch.bmm(query_states_67, transpose_168);  query_states_67 = transpose_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_89 = torch.nn.functional.softmax(attn_weights_88, dim = -1);  attn_weights_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_33 = torch.nn.functional.dropout(attn_weights_89, p = 0.1, training = False);  attn_weights_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_165 = torch.bmm(attn_probs_33, value_states_67);  attn_probs_33 = value_states_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_166 = attn_output_165.view(1, 16, 128, 64);  attn_output_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    attn_output_167 = attn_output_166.transpose(1, 2);  attn_output_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_168 = attn_output_167.reshape(1, 128, 1024);  attn_output_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    hidden_states_292 = self.L__mod___model_decoder_layers_10_encoder_attn_out_proj(attn_output_168);  attn_output_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:512, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_293 = torch.nn.functional.dropout(hidden_states_292, p = 0.1, training = False);  hidden_states_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:513, code: hidden_states = residual + hidden_states
    residual_56 = residual_55 + hidden_states_293;  residual_55 = hidden_states_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_295 = self.L__mod___model_decoder_layers_10_final_layer_norm(residual_56)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_10_fc1 = self.L__mod___model_decoder_layers_10_fc1(hidden_states_295);  hidden_states_295 = None
    hidden_states_296 = self.L__mod___model_decoder_layers_10_activation_fn(l__mod___model_decoder_layers_10_fc1);  l__mod___model_decoder_layers_10_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:522, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_297 = torch.nn.functional.dropout(hidden_states_296, p = 0.0, training = False);  hidden_states_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    hidden_states_298 = self.L__mod___model_decoder_layers_10_fc2(hidden_states_297);  hidden_states_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:524, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_299 = torch.nn.functional.dropout(hidden_states_298, p = 0.1, training = False);  hidden_states_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:525, code: hidden_states = residual + hidden_states
    residual_57 = residual_56 + hidden_states_299;  residual_56 = hidden_states_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1060, code: dropout_probability = torch.rand([])
    dropout_probability_23 = torch.rand([])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_302 = self.L__mod___model_decoder_layers_11_self_attn_layer_norm(residual_57)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_11_self_attn_q_proj = self.L__mod___model_decoder_layers_11_self_attn_q_proj(hidden_states_302)
    query_states_68 = l__mod___model_decoder_layers_11_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_11_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_11_self_attn_k_proj = self.L__mod___model_decoder_layers_11_self_attn_k_proj(hidden_states_302)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_199 = l__mod___model_decoder_layers_11_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_11_self_attn_k_proj = None
    transpose_170 = view_199.transpose(1, 2);  view_199 = None
    key_states_68 = transpose_170.contiguous();  transpose_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_11_self_attn_v_proj = self.L__mod___model_decoder_layers_11_self_attn_v_proj(hidden_states_302);  hidden_states_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_200 = l__mod___model_decoder_layers_11_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_11_self_attn_v_proj = None
    transpose_171 = view_200.transpose(1, 2);  view_200 = None
    value_states_68 = transpose_171.contiguous();  transpose_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_201 = query_states_68.view(1, 128, 16, 64);  query_states_68 = None
    transpose_172 = view_201.transpose(1, 2);  view_201 = None
    contiguous_104 = transpose_172.contiguous();  transpose_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_69 = contiguous_104.view(16, -1, 64);  contiguous_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    key_states_69 = key_states_68.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    value_states_69 = value_states_68.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_173 = key_states_69.transpose(1, 2);  key_states_69 = None
    attn_weights_90 = torch.bmm(query_states_69, transpose_173);  query_states_69 = transpose_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_203 = attn_weights_90.view(1, 16, 128, 128);  attn_weights_90 = None
    attn_weights_91 = view_203 + combined_attention_mask;  view_203 = combined_attention_mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_92 = attn_weights_91.view(16, 128, 128);  attn_weights_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_93 = torch.nn.functional.softmax(attn_weights_92, dim = -1);  attn_weights_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_34 = torch.nn.functional.dropout(attn_weights_93, p = 0.1, training = False);  attn_weights_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_170 = torch.bmm(attn_probs_34, value_states_69);  attn_probs_34 = value_states_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_171 = attn_output_170.view(1, 16, 128, 64);  attn_output_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    attn_output_172 = attn_output_171.transpose(1, 2);  attn_output_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_173 = attn_output_172.reshape(1, 128, 1024);  attn_output_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    hidden_states_303 = self.L__mod___model_decoder_layers_11_self_attn_out_proj(attn_output_173);  attn_output_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:492, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_304 = torch.nn.functional.dropout(hidden_states_303, p = 0.1, training = False);  hidden_states_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:493, code: hidden_states = residual + hidden_states
    residual_58 = residual_57 + hidden_states_304;  residual_57 = hidden_states_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_306 = self.L__mod___model_decoder_layers_11_encoder_attn_layer_norm(residual_58)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_11_encoder_attn_q_proj = self.L__mod___model_decoder_layers_11_encoder_attn_q_proj(hidden_states_306);  hidden_states_306 = None
    query_states_70 = l__mod___model_decoder_layers_11_encoder_attn_q_proj * 0.125;  l__mod___model_decoder_layers_11_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_11_encoder_attn_k_proj = self.L__mod___model_decoder_layers_11_encoder_attn_k_proj(hidden_states_134)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_206 = l__mod___model_decoder_layers_11_encoder_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_11_encoder_attn_k_proj = None
    transpose_175 = view_206.transpose(1, 2);  view_206 = None
    key_states_70 = transpose_175.contiguous();  transpose_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_11_encoder_attn_v_proj = self.L__mod___model_decoder_layers_11_encoder_attn_v_proj(hidden_states_134)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_207 = l__mod___model_decoder_layers_11_encoder_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_11_encoder_attn_v_proj = None
    transpose_176 = view_207.transpose(1, 2);  view_207 = None
    value_states_70 = transpose_176.contiguous();  transpose_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_208 = query_states_70.view(1, 128, 16, 64);  query_states_70 = None
    transpose_177 = view_208.transpose(1, 2);  view_208 = None
    contiguous_107 = transpose_177.contiguous();  transpose_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_71 = contiguous_107.view(16, -1, 64);  contiguous_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    key_states_71 = key_states_70.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    value_states_71 = value_states_70.reshape(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_178 = key_states_71.transpose(1, 2);  key_states_71 = None
    attn_weights_94 = torch.bmm(query_states_71, transpose_178);  query_states_71 = transpose_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_95 = torch.nn.functional.softmax(attn_weights_94, dim = -1);  attn_weights_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_35 = torch.nn.functional.dropout(attn_weights_95, p = 0.1, training = False);  attn_weights_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_175 = torch.bmm(attn_probs_35, value_states_71);  attn_probs_35 = value_states_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_176 = attn_output_175.view(1, 16, 128, 64);  attn_output_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    attn_output_177 = attn_output_176.transpose(1, 2);  attn_output_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_178 = attn_output_177.reshape(1, 128, 1024);  attn_output_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    hidden_states_307 = self.L__mod___model_decoder_layers_11_encoder_attn_out_proj(attn_output_178);  attn_output_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:512, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_308 = torch.nn.functional.dropout(hidden_states_307, p = 0.1, training = False);  hidden_states_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:513, code: hidden_states = residual + hidden_states
    residual_59 = residual_58 + hidden_states_308;  residual_58 = hidden_states_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_310 = self.L__mod___model_decoder_layers_11_final_layer_norm(residual_59)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_11_fc1 = self.L__mod___model_decoder_layers_11_fc1(hidden_states_310);  hidden_states_310 = None
    hidden_states_311 = self.L__mod___model_decoder_layers_11_activation_fn(l__mod___model_decoder_layers_11_fc1);  l__mod___model_decoder_layers_11_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:522, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_312 = torch.nn.functional.dropout(hidden_states_311, p = 0.0, training = False);  hidden_states_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    hidden_states_313 = self.L__mod___model_decoder_layers_11_fc2(hidden_states_312);  hidden_states_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:524, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_314 = torch.nn.functional.dropout(hidden_states_313, p = 0.1, training = False);  hidden_states_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:525, code: hidden_states = residual + hidden_states
    hidden_states_316 = residual_59 + hidden_states_314;  residual_59 = hidden_states_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1114, code: hidden_states = self.layer_norm(hidden_states)
    hidden_states_317 = self.L__mod___model_decoder_layer_norm(hidden_states_316);  hidden_states_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1331, code: lm_logits = self.lm_head(outputs[0])
    lm_logits = self.L__mod___lm_head(hidden_states_317);  hidden_states_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1336, code: labels = labels.to(lm_logits.device)
    labels = l_inputs_labels_.to(device(type='cuda', index=0));  l_inputs_labels_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1338, code: masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
    view_211 = lm_logits.view(-1, 128112)
    view_212 = labels.view(-1);  labels = None
    masked_lm_loss = torch.nn.functional.cross_entropy(view_211, view_212, None, None, -100, None, 'mean', 0.0);  view_211 = view_212 = None
    return (masked_lm_loss, lm_logits, key_states_24, value_states_24, key_states_26, value_states_26, key_states_28, value_states_28, key_states_30, value_states_30, key_states_32, value_states_32, key_states_34, value_states_34, key_states_36, value_states_36, key_states_38, value_states_38, key_states_40, value_states_40, key_states_42, value_states_42, key_states_44, value_states_44, key_states_46, value_states_46, key_states_48, value_states_48, key_states_50, value_states_50, key_states_52, value_states_52, key_states_54, value_states_54, key_states_56, value_states_56, key_states_58, value_states_58, key_states_60, value_states_60, key_states_62, value_states_62, key_states_64, value_states_64, key_states_66, value_states_66, key_states_68, value_states_68, key_states_70, value_states_70, hidden_states_134)
    