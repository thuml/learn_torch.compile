from __future__ import annotations



def forward(self, L_inputs_input_ids_ : torch.Tensor, L_inputs_labels_ : torch.Tensor):
    l_inputs_input_ids_ = L_inputs_input_ids_
    l_inputs_labels_ = L_inputs_labels_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:611, code: input_ids = input_ids.view(-1, input_shape[-1])
    input_ids = l_inputs_input_ids_.view(-1, 128);  l_inputs_input_ids_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:620, code: position_ids = torch.arange(
    position_ids = torch.arange(0, 128, dtype = torch.int64, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:626, code: position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
    unsqueeze = position_ids.unsqueeze(0);  position_ids = None
    position_ids_1 = unsqueeze.view(-1, 128);  unsqueeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:631, code: inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
    l__mod___model_embed_tokens = self.L__mod___model_embed_tokens(input_ids);  input_ids = None
    inputs_embeds = l__mod___model_embed_tokens * 32.0;  l__mod___model_embed_tokens = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:138, code: mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask = torch.full((128, 128), -3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:139, code: mask_cond = torch.arange(mask.size(-1), device=device)
    mask_cond = torch.arange(128, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:140, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    add = mask_cond + 1
    view_2 = add.view(128, 1);  add = None
    lt = mask_cond < view_2;  mask_cond = view_2 = None
    masked_fill_ = mask.masked_fill_(lt, 0);  lt = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:141, code: mask = mask.to(dtype)
    mask_1 = mask.to(torch.float32);  mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:145, code: return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
    getitem = mask_1[(None, None, slice(None, None, None), slice(None, None, None))];  mask_1 = None
    attention_mask = getitem.expand(1, 1, 128, 128);  getitem = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:205, code: position_ids += self.offset
    position_ids_1 += 2;  position_ids_2 = position_ids_1;  position_ids_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:209, code: if max_pos > self.weights.size(0):
    l__mod___model_embed_positions_weights = self.L__mod___model_embed_positions_weights
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:212, code: return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()
    view_3 = position_ids_2.view(-1);  position_ids_2 = None
    index_select = l__mod___model_embed_positions_weights.index_select(0, view_3);  l__mod___model_embed_positions_weights = view_3 = None
    view_4 = index_select.view(1, 128, 1024);  index_select = None
    detach = view_4.detach();  view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:642, code: hidden_states = inputs_embeds + self.embed_positions(position_ids, past_key_values_length)
    hidden_states = inputs_embeds + detach;  inputs_embeds = detach = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:643, code: hidden_states = nn.functional.dropout(hidden_states, p=float(self.dropout), training=self.training)
    residual = torch.nn.functional.dropout(hidden_states, p = 0.1, training = False);  hidden_states = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_2 = self.L__mod___model_layers_0_self_attn_layer_norm(residual)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_layers_0_self_attn_q_proj = self.L__mod___model_layers_0_self_attn_q_proj(hidden_states_2)
    query_states = l__mod___model_layers_0_self_attn_q_proj * 0.125;  l__mod___model_layers_0_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_layers_0_self_attn_k_proj = self.L__mod___model_layers_0_self_attn_k_proj(hidden_states_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_5 = l__mod___model_layers_0_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_layers_0_self_attn_k_proj = None
    transpose = view_5.transpose(1, 2);  view_5 = None
    key_states = transpose.contiguous();  transpose = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_layers_0_self_attn_v_proj = self.L__mod___model_layers_0_self_attn_v_proj(hidden_states_2);  hidden_states_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_6 = l__mod___model_layers_0_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_layers_0_self_attn_v_proj = None
    transpose_1 = view_6.transpose(1, 2);  view_6 = None
    value_states = transpose_1.contiguous();  transpose_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_7 = query_states.view(1, 128, 16, 64);  query_states = None
    transpose_2 = view_7.transpose(1, 2);  view_7 = None
    contiguous_2 = transpose_2.contiguous();  transpose_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_1 = contiguous_2.view(16, -1, 64);  contiguous_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    key_states_1 = key_states.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    value_states_1 = value_states.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_3 = key_states_1.transpose(1, 2);  key_states_1 = None
    attn_weights = torch.bmm(query_states_1, transpose_3);  query_states_1 = transpose_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_11 = attn_weights.view(1, 16, 128, 128);  attn_weights = None
    attn_weights_1 = view_11 + attention_mask;  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    tensor = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    attn_weights_2 = torch.max(attn_weights_1, tensor);  attn_weights_1 = tensor = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_3 = attn_weights_2.view(16, 128, 128);  attn_weights_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_4 = torch.nn.functional.softmax(attn_weights_3, dim = -1);  attn_weights_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs = torch.nn.functional.dropout(attn_weights_4, p = 0.1, training = False);  attn_weights_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output = torch.bmm(attn_probs, value_states_1);  attn_probs = value_states_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_1 = attn_output.view(1, 16, 128, 64);  attn_output = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    attn_output_2 = attn_output_1.transpose(1, 2);  attn_output_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_3 = attn_output_2.reshape(1, 128, 1024);  attn_output_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    hidden_states_3 = self.L__mod___model_layers_0_self_attn_out_proj(attn_output_3);  attn_output_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_4 = torch.nn.functional.dropout(hidden_states_3, p = 0.1, training = False);  hidden_states_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    residual_1 = residual + hidden_states_4;  residual = hidden_states_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_6 = self.L__mod___model_layers_0_final_layer_norm(residual_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_layers_0_fc1 = self.L__mod___model_layers_0_fc1(hidden_states_6);  hidden_states_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_7 = torch._C._nn.gelu(l__mod___model_layers_0_fc1);  l__mod___model_layers_0_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_8 = torch.nn.functional.dropout(hidden_states_7, p = 0.0, training = False);  hidden_states_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    hidden_states_9 = self.L__mod___model_layers_0_fc2(hidden_states_8);  hidden_states_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_10 = torch.nn.functional.dropout(hidden_states_9, p = 0.1, training = False);  hidden_states_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    residual_2 = residual_1 + hidden_states_10;  residual_1 = hidden_states_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_13 = self.L__mod___model_layers_1_self_attn_layer_norm(residual_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_layers_1_self_attn_q_proj = self.L__mod___model_layers_1_self_attn_q_proj(hidden_states_13)
    query_states_2 = l__mod___model_layers_1_self_attn_q_proj * 0.125;  l__mod___model_layers_1_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_layers_1_self_attn_k_proj = self.L__mod___model_layers_1_self_attn_k_proj(hidden_states_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_14 = l__mod___model_layers_1_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_layers_1_self_attn_k_proj = None
    transpose_5 = view_14.transpose(1, 2);  view_14 = None
    key_states_2 = transpose_5.contiguous();  transpose_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_layers_1_self_attn_v_proj = self.L__mod___model_layers_1_self_attn_v_proj(hidden_states_13);  hidden_states_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_15 = l__mod___model_layers_1_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_layers_1_self_attn_v_proj = None
    transpose_6 = view_15.transpose(1, 2);  view_15 = None
    value_states_2 = transpose_6.contiguous();  transpose_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_16 = query_states_2.view(1, 128, 16, 64);  query_states_2 = None
    transpose_7 = view_16.transpose(1, 2);  view_16 = None
    contiguous_5 = transpose_7.contiguous();  transpose_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_3 = contiguous_5.view(16, -1, 64);  contiguous_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    key_states_3 = key_states_2.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    value_states_3 = value_states_2.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_8 = key_states_3.transpose(1, 2);  key_states_3 = None
    attn_weights_5 = torch.bmm(query_states_3, transpose_8);  query_states_3 = transpose_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_20 = attn_weights_5.view(1, 16, 128, 128);  attn_weights_5 = None
    attn_weights_6 = view_20 + attention_mask;  view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    tensor_1 = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    attn_weights_7 = torch.max(attn_weights_6, tensor_1);  attn_weights_6 = tensor_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_8 = attn_weights_7.view(16, 128, 128);  attn_weights_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_9 = torch.nn.functional.softmax(attn_weights_8, dim = -1);  attn_weights_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_1 = torch.nn.functional.dropout(attn_weights_9, p = 0.1, training = False);  attn_weights_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_5 = torch.bmm(attn_probs_1, value_states_3);  attn_probs_1 = value_states_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_6 = attn_output_5.view(1, 16, 128, 64);  attn_output_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    attn_output_7 = attn_output_6.transpose(1, 2);  attn_output_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_8 = attn_output_7.reshape(1, 128, 1024);  attn_output_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    hidden_states_14 = self.L__mod___model_layers_1_self_attn_out_proj(attn_output_8);  attn_output_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_15 = torch.nn.functional.dropout(hidden_states_14, p = 0.1, training = False);  hidden_states_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    residual_3 = residual_2 + hidden_states_15;  residual_2 = hidden_states_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_17 = self.L__mod___model_layers_1_final_layer_norm(residual_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_layers_1_fc1 = self.L__mod___model_layers_1_fc1(hidden_states_17);  hidden_states_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_18 = torch._C._nn.gelu(l__mod___model_layers_1_fc1);  l__mod___model_layers_1_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_19 = torch.nn.functional.dropout(hidden_states_18, p = 0.0, training = False);  hidden_states_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    hidden_states_20 = self.L__mod___model_layers_1_fc2(hidden_states_19);  hidden_states_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_21 = torch.nn.functional.dropout(hidden_states_20, p = 0.1, training = False);  hidden_states_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    residual_4 = residual_3 + hidden_states_21;  residual_3 = hidden_states_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_24 = self.L__mod___model_layers_2_self_attn_layer_norm(residual_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_layers_2_self_attn_q_proj = self.L__mod___model_layers_2_self_attn_q_proj(hidden_states_24)
    query_states_4 = l__mod___model_layers_2_self_attn_q_proj * 0.125;  l__mod___model_layers_2_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_layers_2_self_attn_k_proj = self.L__mod___model_layers_2_self_attn_k_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_23 = l__mod___model_layers_2_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_layers_2_self_attn_k_proj = None
    transpose_10 = view_23.transpose(1, 2);  view_23 = None
    key_states_4 = transpose_10.contiguous();  transpose_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_layers_2_self_attn_v_proj = self.L__mod___model_layers_2_self_attn_v_proj(hidden_states_24);  hidden_states_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_24 = l__mod___model_layers_2_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_layers_2_self_attn_v_proj = None
    transpose_11 = view_24.transpose(1, 2);  view_24 = None
    value_states_4 = transpose_11.contiguous();  transpose_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_25 = query_states_4.view(1, 128, 16, 64);  query_states_4 = None
    transpose_12 = view_25.transpose(1, 2);  view_25 = None
    contiguous_8 = transpose_12.contiguous();  transpose_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_5 = contiguous_8.view(16, -1, 64);  contiguous_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    key_states_5 = key_states_4.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    value_states_5 = value_states_4.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_13 = key_states_5.transpose(1, 2);  key_states_5 = None
    attn_weights_10 = torch.bmm(query_states_5, transpose_13);  query_states_5 = transpose_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_29 = attn_weights_10.view(1, 16, 128, 128);  attn_weights_10 = None
    attn_weights_11 = view_29 + attention_mask;  view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    tensor_2 = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    attn_weights_12 = torch.max(attn_weights_11, tensor_2);  attn_weights_11 = tensor_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_13 = attn_weights_12.view(16, 128, 128);  attn_weights_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_14 = torch.nn.functional.softmax(attn_weights_13, dim = -1);  attn_weights_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_2 = torch.nn.functional.dropout(attn_weights_14, p = 0.1, training = False);  attn_weights_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_10 = torch.bmm(attn_probs_2, value_states_5);  attn_probs_2 = value_states_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_11 = attn_output_10.view(1, 16, 128, 64);  attn_output_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    attn_output_12 = attn_output_11.transpose(1, 2);  attn_output_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_13 = attn_output_12.reshape(1, 128, 1024);  attn_output_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    hidden_states_25 = self.L__mod___model_layers_2_self_attn_out_proj(attn_output_13);  attn_output_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_26 = torch.nn.functional.dropout(hidden_states_25, p = 0.1, training = False);  hidden_states_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    residual_5 = residual_4 + hidden_states_26;  residual_4 = hidden_states_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_28 = self.L__mod___model_layers_2_final_layer_norm(residual_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_layers_2_fc1 = self.L__mod___model_layers_2_fc1(hidden_states_28);  hidden_states_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_29 = torch._C._nn.gelu(l__mod___model_layers_2_fc1);  l__mod___model_layers_2_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_30 = torch.nn.functional.dropout(hidden_states_29, p = 0.0, training = False);  hidden_states_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    hidden_states_31 = self.L__mod___model_layers_2_fc2(hidden_states_30);  hidden_states_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_32 = torch.nn.functional.dropout(hidden_states_31, p = 0.1, training = False);  hidden_states_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    residual_6 = residual_5 + hidden_states_32;  residual_5 = hidden_states_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_35 = self.L__mod___model_layers_3_self_attn_layer_norm(residual_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_layers_3_self_attn_q_proj = self.L__mod___model_layers_3_self_attn_q_proj(hidden_states_35)
    query_states_6 = l__mod___model_layers_3_self_attn_q_proj * 0.125;  l__mod___model_layers_3_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_layers_3_self_attn_k_proj = self.L__mod___model_layers_3_self_attn_k_proj(hidden_states_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_32 = l__mod___model_layers_3_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_layers_3_self_attn_k_proj = None
    transpose_15 = view_32.transpose(1, 2);  view_32 = None
    key_states_6 = transpose_15.contiguous();  transpose_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_layers_3_self_attn_v_proj = self.L__mod___model_layers_3_self_attn_v_proj(hidden_states_35);  hidden_states_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_33 = l__mod___model_layers_3_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_layers_3_self_attn_v_proj = None
    transpose_16 = view_33.transpose(1, 2);  view_33 = None
    value_states_6 = transpose_16.contiguous();  transpose_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_34 = query_states_6.view(1, 128, 16, 64);  query_states_6 = None
    transpose_17 = view_34.transpose(1, 2);  view_34 = None
    contiguous_11 = transpose_17.contiguous();  transpose_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_7 = contiguous_11.view(16, -1, 64);  contiguous_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    key_states_7 = key_states_6.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    value_states_7 = value_states_6.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_18 = key_states_7.transpose(1, 2);  key_states_7 = None
    attn_weights_15 = torch.bmm(query_states_7, transpose_18);  query_states_7 = transpose_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_38 = attn_weights_15.view(1, 16, 128, 128);  attn_weights_15 = None
    attn_weights_16 = view_38 + attention_mask;  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    tensor_3 = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    attn_weights_17 = torch.max(attn_weights_16, tensor_3);  attn_weights_16 = tensor_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_18 = attn_weights_17.view(16, 128, 128);  attn_weights_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_19 = torch.nn.functional.softmax(attn_weights_18, dim = -1);  attn_weights_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_3 = torch.nn.functional.dropout(attn_weights_19, p = 0.1, training = False);  attn_weights_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_15 = torch.bmm(attn_probs_3, value_states_7);  attn_probs_3 = value_states_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_16 = attn_output_15.view(1, 16, 128, 64);  attn_output_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    attn_output_17 = attn_output_16.transpose(1, 2);  attn_output_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_18 = attn_output_17.reshape(1, 128, 1024);  attn_output_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    hidden_states_36 = self.L__mod___model_layers_3_self_attn_out_proj(attn_output_18);  attn_output_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_37 = torch.nn.functional.dropout(hidden_states_36, p = 0.1, training = False);  hidden_states_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    residual_7 = residual_6 + hidden_states_37;  residual_6 = hidden_states_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_39 = self.L__mod___model_layers_3_final_layer_norm(residual_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_layers_3_fc1 = self.L__mod___model_layers_3_fc1(hidden_states_39);  hidden_states_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_40 = torch._C._nn.gelu(l__mod___model_layers_3_fc1);  l__mod___model_layers_3_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_41 = torch.nn.functional.dropout(hidden_states_40, p = 0.0, training = False);  hidden_states_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    hidden_states_42 = self.L__mod___model_layers_3_fc2(hidden_states_41);  hidden_states_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_43 = torch.nn.functional.dropout(hidden_states_42, p = 0.1, training = False);  hidden_states_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    residual_8 = residual_7 + hidden_states_43;  residual_7 = hidden_states_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_46 = self.L__mod___model_layers_4_self_attn_layer_norm(residual_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_layers_4_self_attn_q_proj = self.L__mod___model_layers_4_self_attn_q_proj(hidden_states_46)
    query_states_8 = l__mod___model_layers_4_self_attn_q_proj * 0.125;  l__mod___model_layers_4_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_layers_4_self_attn_k_proj = self.L__mod___model_layers_4_self_attn_k_proj(hidden_states_46)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_41 = l__mod___model_layers_4_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_layers_4_self_attn_k_proj = None
    transpose_20 = view_41.transpose(1, 2);  view_41 = None
    key_states_8 = transpose_20.contiguous();  transpose_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_layers_4_self_attn_v_proj = self.L__mod___model_layers_4_self_attn_v_proj(hidden_states_46);  hidden_states_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_42 = l__mod___model_layers_4_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_layers_4_self_attn_v_proj = None
    transpose_21 = view_42.transpose(1, 2);  view_42 = None
    value_states_8 = transpose_21.contiguous();  transpose_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_43 = query_states_8.view(1, 128, 16, 64);  query_states_8 = None
    transpose_22 = view_43.transpose(1, 2);  view_43 = None
    contiguous_14 = transpose_22.contiguous();  transpose_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_9 = contiguous_14.view(16, -1, 64);  contiguous_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    key_states_9 = key_states_8.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    value_states_9 = value_states_8.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_23 = key_states_9.transpose(1, 2);  key_states_9 = None
    attn_weights_20 = torch.bmm(query_states_9, transpose_23);  query_states_9 = transpose_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_47 = attn_weights_20.view(1, 16, 128, 128);  attn_weights_20 = None
    attn_weights_21 = view_47 + attention_mask;  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    tensor_4 = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    attn_weights_22 = torch.max(attn_weights_21, tensor_4);  attn_weights_21 = tensor_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_23 = attn_weights_22.view(16, 128, 128);  attn_weights_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_24 = torch.nn.functional.softmax(attn_weights_23, dim = -1);  attn_weights_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_4 = torch.nn.functional.dropout(attn_weights_24, p = 0.1, training = False);  attn_weights_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_20 = torch.bmm(attn_probs_4, value_states_9);  attn_probs_4 = value_states_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_21 = attn_output_20.view(1, 16, 128, 64);  attn_output_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    attn_output_22 = attn_output_21.transpose(1, 2);  attn_output_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_23 = attn_output_22.reshape(1, 128, 1024);  attn_output_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    hidden_states_47 = self.L__mod___model_layers_4_self_attn_out_proj(attn_output_23);  attn_output_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_48 = torch.nn.functional.dropout(hidden_states_47, p = 0.1, training = False);  hidden_states_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    residual_9 = residual_8 + hidden_states_48;  residual_8 = hidden_states_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_50 = self.L__mod___model_layers_4_final_layer_norm(residual_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_layers_4_fc1 = self.L__mod___model_layers_4_fc1(hidden_states_50);  hidden_states_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_51 = torch._C._nn.gelu(l__mod___model_layers_4_fc1);  l__mod___model_layers_4_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_52 = torch.nn.functional.dropout(hidden_states_51, p = 0.0, training = False);  hidden_states_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    hidden_states_53 = self.L__mod___model_layers_4_fc2(hidden_states_52);  hidden_states_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_54 = torch.nn.functional.dropout(hidden_states_53, p = 0.1, training = False);  hidden_states_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    residual_10 = residual_9 + hidden_states_54;  residual_9 = hidden_states_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_57 = self.L__mod___model_layers_5_self_attn_layer_norm(residual_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_layers_5_self_attn_q_proj = self.L__mod___model_layers_5_self_attn_q_proj(hidden_states_57)
    query_states_10 = l__mod___model_layers_5_self_attn_q_proj * 0.125;  l__mod___model_layers_5_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_layers_5_self_attn_k_proj = self.L__mod___model_layers_5_self_attn_k_proj(hidden_states_57)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_50 = l__mod___model_layers_5_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_layers_5_self_attn_k_proj = None
    transpose_25 = view_50.transpose(1, 2);  view_50 = None
    key_states_10 = transpose_25.contiguous();  transpose_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_layers_5_self_attn_v_proj = self.L__mod___model_layers_5_self_attn_v_proj(hidden_states_57);  hidden_states_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_51 = l__mod___model_layers_5_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_layers_5_self_attn_v_proj = None
    transpose_26 = view_51.transpose(1, 2);  view_51 = None
    value_states_10 = transpose_26.contiguous();  transpose_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_52 = query_states_10.view(1, 128, 16, 64);  query_states_10 = None
    transpose_27 = view_52.transpose(1, 2);  view_52 = None
    contiguous_17 = transpose_27.contiguous();  transpose_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_11 = contiguous_17.view(16, -1, 64);  contiguous_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    key_states_11 = key_states_10.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    value_states_11 = value_states_10.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_28 = key_states_11.transpose(1, 2);  key_states_11 = None
    attn_weights_25 = torch.bmm(query_states_11, transpose_28);  query_states_11 = transpose_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_56 = attn_weights_25.view(1, 16, 128, 128);  attn_weights_25 = None
    attn_weights_26 = view_56 + attention_mask;  view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    tensor_5 = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    attn_weights_27 = torch.max(attn_weights_26, tensor_5);  attn_weights_26 = tensor_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_28 = attn_weights_27.view(16, 128, 128);  attn_weights_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_29 = torch.nn.functional.softmax(attn_weights_28, dim = -1);  attn_weights_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_5 = torch.nn.functional.dropout(attn_weights_29, p = 0.1, training = False);  attn_weights_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_25 = torch.bmm(attn_probs_5, value_states_11);  attn_probs_5 = value_states_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_26 = attn_output_25.view(1, 16, 128, 64);  attn_output_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    attn_output_27 = attn_output_26.transpose(1, 2);  attn_output_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_28 = attn_output_27.reshape(1, 128, 1024);  attn_output_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    hidden_states_58 = self.L__mod___model_layers_5_self_attn_out_proj(attn_output_28);  attn_output_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_59 = torch.nn.functional.dropout(hidden_states_58, p = 0.1, training = False);  hidden_states_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    residual_11 = residual_10 + hidden_states_59;  residual_10 = hidden_states_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_61 = self.L__mod___model_layers_5_final_layer_norm(residual_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_layers_5_fc1 = self.L__mod___model_layers_5_fc1(hidden_states_61);  hidden_states_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_62 = torch._C._nn.gelu(l__mod___model_layers_5_fc1);  l__mod___model_layers_5_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_63 = torch.nn.functional.dropout(hidden_states_62, p = 0.0, training = False);  hidden_states_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    hidden_states_64 = self.L__mod___model_layers_5_fc2(hidden_states_63);  hidden_states_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_65 = torch.nn.functional.dropout(hidden_states_64, p = 0.1, training = False);  hidden_states_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    residual_12 = residual_11 + hidden_states_65;  residual_11 = hidden_states_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_68 = self.L__mod___model_layers_6_self_attn_layer_norm(residual_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_layers_6_self_attn_q_proj = self.L__mod___model_layers_6_self_attn_q_proj(hidden_states_68)
    query_states_12 = l__mod___model_layers_6_self_attn_q_proj * 0.125;  l__mod___model_layers_6_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_layers_6_self_attn_k_proj = self.L__mod___model_layers_6_self_attn_k_proj(hidden_states_68)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_59 = l__mod___model_layers_6_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_layers_6_self_attn_k_proj = None
    transpose_30 = view_59.transpose(1, 2);  view_59 = None
    key_states_12 = transpose_30.contiguous();  transpose_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_layers_6_self_attn_v_proj = self.L__mod___model_layers_6_self_attn_v_proj(hidden_states_68);  hidden_states_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_60 = l__mod___model_layers_6_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_layers_6_self_attn_v_proj = None
    transpose_31 = view_60.transpose(1, 2);  view_60 = None
    value_states_12 = transpose_31.contiguous();  transpose_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_61 = query_states_12.view(1, 128, 16, 64);  query_states_12 = None
    transpose_32 = view_61.transpose(1, 2);  view_61 = None
    contiguous_20 = transpose_32.contiguous();  transpose_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_13 = contiguous_20.view(16, -1, 64);  contiguous_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    key_states_13 = key_states_12.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    value_states_13 = value_states_12.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_33 = key_states_13.transpose(1, 2);  key_states_13 = None
    attn_weights_30 = torch.bmm(query_states_13, transpose_33);  query_states_13 = transpose_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_65 = attn_weights_30.view(1, 16, 128, 128);  attn_weights_30 = None
    attn_weights_31 = view_65 + attention_mask;  view_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    tensor_6 = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    attn_weights_32 = torch.max(attn_weights_31, tensor_6);  attn_weights_31 = tensor_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_33 = attn_weights_32.view(16, 128, 128);  attn_weights_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_34 = torch.nn.functional.softmax(attn_weights_33, dim = -1);  attn_weights_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_6 = torch.nn.functional.dropout(attn_weights_34, p = 0.1, training = False);  attn_weights_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_30 = torch.bmm(attn_probs_6, value_states_13);  attn_probs_6 = value_states_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_31 = attn_output_30.view(1, 16, 128, 64);  attn_output_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    attn_output_32 = attn_output_31.transpose(1, 2);  attn_output_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_33 = attn_output_32.reshape(1, 128, 1024);  attn_output_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    hidden_states_69 = self.L__mod___model_layers_6_self_attn_out_proj(attn_output_33);  attn_output_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_70 = torch.nn.functional.dropout(hidden_states_69, p = 0.1, training = False);  hidden_states_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    residual_13 = residual_12 + hidden_states_70;  residual_12 = hidden_states_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_72 = self.L__mod___model_layers_6_final_layer_norm(residual_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_layers_6_fc1 = self.L__mod___model_layers_6_fc1(hidden_states_72);  hidden_states_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_73 = torch._C._nn.gelu(l__mod___model_layers_6_fc1);  l__mod___model_layers_6_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_74 = torch.nn.functional.dropout(hidden_states_73, p = 0.0, training = False);  hidden_states_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    hidden_states_75 = self.L__mod___model_layers_6_fc2(hidden_states_74);  hidden_states_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_76 = torch.nn.functional.dropout(hidden_states_75, p = 0.1, training = False);  hidden_states_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    residual_14 = residual_13 + hidden_states_76;  residual_13 = hidden_states_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_79 = self.L__mod___model_layers_7_self_attn_layer_norm(residual_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_layers_7_self_attn_q_proj = self.L__mod___model_layers_7_self_attn_q_proj(hidden_states_79)
    query_states_14 = l__mod___model_layers_7_self_attn_q_proj * 0.125;  l__mod___model_layers_7_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_layers_7_self_attn_k_proj = self.L__mod___model_layers_7_self_attn_k_proj(hidden_states_79)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_68 = l__mod___model_layers_7_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_layers_7_self_attn_k_proj = None
    transpose_35 = view_68.transpose(1, 2);  view_68 = None
    key_states_14 = transpose_35.contiguous();  transpose_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_layers_7_self_attn_v_proj = self.L__mod___model_layers_7_self_attn_v_proj(hidden_states_79);  hidden_states_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_69 = l__mod___model_layers_7_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_layers_7_self_attn_v_proj = None
    transpose_36 = view_69.transpose(1, 2);  view_69 = None
    value_states_14 = transpose_36.contiguous();  transpose_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_70 = query_states_14.view(1, 128, 16, 64);  query_states_14 = None
    transpose_37 = view_70.transpose(1, 2);  view_70 = None
    contiguous_23 = transpose_37.contiguous();  transpose_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_15 = contiguous_23.view(16, -1, 64);  contiguous_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    key_states_15 = key_states_14.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    value_states_15 = value_states_14.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_38 = key_states_15.transpose(1, 2);  key_states_15 = None
    attn_weights_35 = torch.bmm(query_states_15, transpose_38);  query_states_15 = transpose_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_74 = attn_weights_35.view(1, 16, 128, 128);  attn_weights_35 = None
    attn_weights_36 = view_74 + attention_mask;  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    tensor_7 = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    attn_weights_37 = torch.max(attn_weights_36, tensor_7);  attn_weights_36 = tensor_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_38 = attn_weights_37.view(16, 128, 128);  attn_weights_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_39 = torch.nn.functional.softmax(attn_weights_38, dim = -1);  attn_weights_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_7 = torch.nn.functional.dropout(attn_weights_39, p = 0.1, training = False);  attn_weights_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_35 = torch.bmm(attn_probs_7, value_states_15);  attn_probs_7 = value_states_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_36 = attn_output_35.view(1, 16, 128, 64);  attn_output_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    attn_output_37 = attn_output_36.transpose(1, 2);  attn_output_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_38 = attn_output_37.reshape(1, 128, 1024);  attn_output_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    hidden_states_80 = self.L__mod___model_layers_7_self_attn_out_proj(attn_output_38);  attn_output_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_81 = torch.nn.functional.dropout(hidden_states_80, p = 0.1, training = False);  hidden_states_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    residual_15 = residual_14 + hidden_states_81;  residual_14 = hidden_states_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_83 = self.L__mod___model_layers_7_final_layer_norm(residual_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_layers_7_fc1 = self.L__mod___model_layers_7_fc1(hidden_states_83);  hidden_states_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_84 = torch._C._nn.gelu(l__mod___model_layers_7_fc1);  l__mod___model_layers_7_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_85 = torch.nn.functional.dropout(hidden_states_84, p = 0.0, training = False);  hidden_states_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    hidden_states_86 = self.L__mod___model_layers_7_fc2(hidden_states_85);  hidden_states_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_87 = torch.nn.functional.dropout(hidden_states_86, p = 0.1, training = False);  hidden_states_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    residual_16 = residual_15 + hidden_states_87;  residual_15 = hidden_states_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_90 = self.L__mod___model_layers_8_self_attn_layer_norm(residual_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_layers_8_self_attn_q_proj = self.L__mod___model_layers_8_self_attn_q_proj(hidden_states_90)
    query_states_16 = l__mod___model_layers_8_self_attn_q_proj * 0.125;  l__mod___model_layers_8_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_layers_8_self_attn_k_proj = self.L__mod___model_layers_8_self_attn_k_proj(hidden_states_90)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_77 = l__mod___model_layers_8_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_layers_8_self_attn_k_proj = None
    transpose_40 = view_77.transpose(1, 2);  view_77 = None
    key_states_16 = transpose_40.contiguous();  transpose_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_layers_8_self_attn_v_proj = self.L__mod___model_layers_8_self_attn_v_proj(hidden_states_90);  hidden_states_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_78 = l__mod___model_layers_8_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_layers_8_self_attn_v_proj = None
    transpose_41 = view_78.transpose(1, 2);  view_78 = None
    value_states_16 = transpose_41.contiguous();  transpose_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_79 = query_states_16.view(1, 128, 16, 64);  query_states_16 = None
    transpose_42 = view_79.transpose(1, 2);  view_79 = None
    contiguous_26 = transpose_42.contiguous();  transpose_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_17 = contiguous_26.view(16, -1, 64);  contiguous_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    key_states_17 = key_states_16.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    value_states_17 = value_states_16.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_43 = key_states_17.transpose(1, 2);  key_states_17 = None
    attn_weights_40 = torch.bmm(query_states_17, transpose_43);  query_states_17 = transpose_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_83 = attn_weights_40.view(1, 16, 128, 128);  attn_weights_40 = None
    attn_weights_41 = view_83 + attention_mask;  view_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    tensor_8 = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    attn_weights_42 = torch.max(attn_weights_41, tensor_8);  attn_weights_41 = tensor_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_43 = attn_weights_42.view(16, 128, 128);  attn_weights_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_44 = torch.nn.functional.softmax(attn_weights_43, dim = -1);  attn_weights_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_8 = torch.nn.functional.dropout(attn_weights_44, p = 0.1, training = False);  attn_weights_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_40 = torch.bmm(attn_probs_8, value_states_17);  attn_probs_8 = value_states_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_41 = attn_output_40.view(1, 16, 128, 64);  attn_output_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    attn_output_42 = attn_output_41.transpose(1, 2);  attn_output_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_43 = attn_output_42.reshape(1, 128, 1024);  attn_output_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    hidden_states_91 = self.L__mod___model_layers_8_self_attn_out_proj(attn_output_43);  attn_output_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_92 = torch.nn.functional.dropout(hidden_states_91, p = 0.1, training = False);  hidden_states_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    residual_17 = residual_16 + hidden_states_92;  residual_16 = hidden_states_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_94 = self.L__mod___model_layers_8_final_layer_norm(residual_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_layers_8_fc1 = self.L__mod___model_layers_8_fc1(hidden_states_94);  hidden_states_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_95 = torch._C._nn.gelu(l__mod___model_layers_8_fc1);  l__mod___model_layers_8_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_96 = torch.nn.functional.dropout(hidden_states_95, p = 0.0, training = False);  hidden_states_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    hidden_states_97 = self.L__mod___model_layers_8_fc2(hidden_states_96);  hidden_states_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_98 = torch.nn.functional.dropout(hidden_states_97, p = 0.1, training = False);  hidden_states_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    residual_18 = residual_17 + hidden_states_98;  residual_17 = hidden_states_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_101 = self.L__mod___model_layers_9_self_attn_layer_norm(residual_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_layers_9_self_attn_q_proj = self.L__mod___model_layers_9_self_attn_q_proj(hidden_states_101)
    query_states_18 = l__mod___model_layers_9_self_attn_q_proj * 0.125;  l__mod___model_layers_9_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_layers_9_self_attn_k_proj = self.L__mod___model_layers_9_self_attn_k_proj(hidden_states_101)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_86 = l__mod___model_layers_9_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_layers_9_self_attn_k_proj = None
    transpose_45 = view_86.transpose(1, 2);  view_86 = None
    key_states_18 = transpose_45.contiguous();  transpose_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_layers_9_self_attn_v_proj = self.L__mod___model_layers_9_self_attn_v_proj(hidden_states_101);  hidden_states_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_87 = l__mod___model_layers_9_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_layers_9_self_attn_v_proj = None
    transpose_46 = view_87.transpose(1, 2);  view_87 = None
    value_states_18 = transpose_46.contiguous();  transpose_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_88 = query_states_18.view(1, 128, 16, 64);  query_states_18 = None
    transpose_47 = view_88.transpose(1, 2);  view_88 = None
    contiguous_29 = transpose_47.contiguous();  transpose_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_19 = contiguous_29.view(16, -1, 64);  contiguous_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    key_states_19 = key_states_18.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    value_states_19 = value_states_18.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_48 = key_states_19.transpose(1, 2);  key_states_19 = None
    attn_weights_45 = torch.bmm(query_states_19, transpose_48);  query_states_19 = transpose_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_92 = attn_weights_45.view(1, 16, 128, 128);  attn_weights_45 = None
    attn_weights_46 = view_92 + attention_mask;  view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    tensor_9 = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    attn_weights_47 = torch.max(attn_weights_46, tensor_9);  attn_weights_46 = tensor_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_48 = attn_weights_47.view(16, 128, 128);  attn_weights_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_49 = torch.nn.functional.softmax(attn_weights_48, dim = -1);  attn_weights_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_9 = torch.nn.functional.dropout(attn_weights_49, p = 0.1, training = False);  attn_weights_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_45 = torch.bmm(attn_probs_9, value_states_19);  attn_probs_9 = value_states_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_46 = attn_output_45.view(1, 16, 128, 64);  attn_output_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    attn_output_47 = attn_output_46.transpose(1, 2);  attn_output_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_48 = attn_output_47.reshape(1, 128, 1024);  attn_output_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    hidden_states_102 = self.L__mod___model_layers_9_self_attn_out_proj(attn_output_48);  attn_output_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_103 = torch.nn.functional.dropout(hidden_states_102, p = 0.1, training = False);  hidden_states_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    residual_19 = residual_18 + hidden_states_103;  residual_18 = hidden_states_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_105 = self.L__mod___model_layers_9_final_layer_norm(residual_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_layers_9_fc1 = self.L__mod___model_layers_9_fc1(hidden_states_105);  hidden_states_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_106 = torch._C._nn.gelu(l__mod___model_layers_9_fc1);  l__mod___model_layers_9_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_107 = torch.nn.functional.dropout(hidden_states_106, p = 0.0, training = False);  hidden_states_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    hidden_states_108 = self.L__mod___model_layers_9_fc2(hidden_states_107);  hidden_states_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_109 = torch.nn.functional.dropout(hidden_states_108, p = 0.1, training = False);  hidden_states_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    residual_20 = residual_19 + hidden_states_109;  residual_19 = hidden_states_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_112 = self.L__mod___model_layers_10_self_attn_layer_norm(residual_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_layers_10_self_attn_q_proj = self.L__mod___model_layers_10_self_attn_q_proj(hidden_states_112)
    query_states_20 = l__mod___model_layers_10_self_attn_q_proj * 0.125;  l__mod___model_layers_10_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_layers_10_self_attn_k_proj = self.L__mod___model_layers_10_self_attn_k_proj(hidden_states_112)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_95 = l__mod___model_layers_10_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_layers_10_self_attn_k_proj = None
    transpose_50 = view_95.transpose(1, 2);  view_95 = None
    key_states_20 = transpose_50.contiguous();  transpose_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_layers_10_self_attn_v_proj = self.L__mod___model_layers_10_self_attn_v_proj(hidden_states_112);  hidden_states_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_96 = l__mod___model_layers_10_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_layers_10_self_attn_v_proj = None
    transpose_51 = view_96.transpose(1, 2);  view_96 = None
    value_states_20 = transpose_51.contiguous();  transpose_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_97 = query_states_20.view(1, 128, 16, 64);  query_states_20 = None
    transpose_52 = view_97.transpose(1, 2);  view_97 = None
    contiguous_32 = transpose_52.contiguous();  transpose_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_21 = contiguous_32.view(16, -1, 64);  contiguous_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    key_states_21 = key_states_20.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    value_states_21 = value_states_20.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_53 = key_states_21.transpose(1, 2);  key_states_21 = None
    attn_weights_50 = torch.bmm(query_states_21, transpose_53);  query_states_21 = transpose_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_101 = attn_weights_50.view(1, 16, 128, 128);  attn_weights_50 = None
    attn_weights_51 = view_101 + attention_mask;  view_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    tensor_10 = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    attn_weights_52 = torch.max(attn_weights_51, tensor_10);  attn_weights_51 = tensor_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_53 = attn_weights_52.view(16, 128, 128);  attn_weights_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_54 = torch.nn.functional.softmax(attn_weights_53, dim = -1);  attn_weights_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_10 = torch.nn.functional.dropout(attn_weights_54, p = 0.1, training = False);  attn_weights_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_50 = torch.bmm(attn_probs_10, value_states_21);  attn_probs_10 = value_states_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_51 = attn_output_50.view(1, 16, 128, 64);  attn_output_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    attn_output_52 = attn_output_51.transpose(1, 2);  attn_output_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_53 = attn_output_52.reshape(1, 128, 1024);  attn_output_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    hidden_states_113 = self.L__mod___model_layers_10_self_attn_out_proj(attn_output_53);  attn_output_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_114 = torch.nn.functional.dropout(hidden_states_113, p = 0.1, training = False);  hidden_states_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    residual_21 = residual_20 + hidden_states_114;  residual_20 = hidden_states_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_116 = self.L__mod___model_layers_10_final_layer_norm(residual_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_layers_10_fc1 = self.L__mod___model_layers_10_fc1(hidden_states_116);  hidden_states_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_117 = torch._C._nn.gelu(l__mod___model_layers_10_fc1);  l__mod___model_layers_10_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_118 = torch.nn.functional.dropout(hidden_states_117, p = 0.0, training = False);  hidden_states_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    hidden_states_119 = self.L__mod___model_layers_10_fc2(hidden_states_118);  hidden_states_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_120 = torch.nn.functional.dropout(hidden_states_119, p = 0.1, training = False);  hidden_states_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    residual_22 = residual_21 + hidden_states_120;  residual_21 = hidden_states_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_123 = self.L__mod___model_layers_11_self_attn_layer_norm(residual_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_layers_11_self_attn_q_proj = self.L__mod___model_layers_11_self_attn_q_proj(hidden_states_123)
    query_states_22 = l__mod___model_layers_11_self_attn_q_proj * 0.125;  l__mod___model_layers_11_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_layers_11_self_attn_k_proj = self.L__mod___model_layers_11_self_attn_k_proj(hidden_states_123)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_104 = l__mod___model_layers_11_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_layers_11_self_attn_k_proj = None
    transpose_55 = view_104.transpose(1, 2);  view_104 = None
    key_states_22 = transpose_55.contiguous();  transpose_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_layers_11_self_attn_v_proj = self.L__mod___model_layers_11_self_attn_v_proj(hidden_states_123);  hidden_states_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_105 = l__mod___model_layers_11_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_layers_11_self_attn_v_proj = None
    transpose_56 = view_105.transpose(1, 2);  view_105 = None
    value_states_22 = transpose_56.contiguous();  transpose_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_106 = query_states_22.view(1, 128, 16, 64);  query_states_22 = None
    transpose_57 = view_106.transpose(1, 2);  view_106 = None
    contiguous_35 = transpose_57.contiguous();  transpose_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_23 = contiguous_35.view(16, -1, 64);  contiguous_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    key_states_23 = key_states_22.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    value_states_23 = value_states_22.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_58 = key_states_23.transpose(1, 2);  key_states_23 = None
    attn_weights_55 = torch.bmm(query_states_23, transpose_58);  query_states_23 = transpose_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_110 = attn_weights_55.view(1, 16, 128, 128);  attn_weights_55 = None
    attn_weights_56 = view_110 + attention_mask;  view_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    tensor_11 = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    attn_weights_57 = torch.max(attn_weights_56, tensor_11);  attn_weights_56 = tensor_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_58 = attn_weights_57.view(16, 128, 128);  attn_weights_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_59 = torch.nn.functional.softmax(attn_weights_58, dim = -1);  attn_weights_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_11 = torch.nn.functional.dropout(attn_weights_59, p = 0.1, training = False);  attn_weights_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_55 = torch.bmm(attn_probs_11, value_states_23);  attn_probs_11 = value_states_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_56 = attn_output_55.view(1, 16, 128, 64);  attn_output_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    attn_output_57 = attn_output_56.transpose(1, 2);  attn_output_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_58 = attn_output_57.reshape(1, 128, 1024);  attn_output_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    hidden_states_124 = self.L__mod___model_layers_11_self_attn_out_proj(attn_output_58);  attn_output_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_125 = torch.nn.functional.dropout(hidden_states_124, p = 0.1, training = False);  hidden_states_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    residual_23 = residual_22 + hidden_states_125;  residual_22 = hidden_states_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_127 = self.L__mod___model_layers_11_final_layer_norm(residual_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_layers_11_fc1 = self.L__mod___model_layers_11_fc1(hidden_states_127);  hidden_states_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_128 = torch._C._nn.gelu(l__mod___model_layers_11_fc1);  l__mod___model_layers_11_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_129 = torch.nn.functional.dropout(hidden_states_128, p = 0.0, training = False);  hidden_states_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    hidden_states_130 = self.L__mod___model_layers_11_fc2(hidden_states_129);  hidden_states_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_131 = torch.nn.functional.dropout(hidden_states_130, p = 0.1, training = False);  hidden_states_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    residual_24 = residual_23 + hidden_states_131;  residual_23 = hidden_states_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_134 = self.L__mod___model_layers_12_self_attn_layer_norm(residual_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_layers_12_self_attn_q_proj = self.L__mod___model_layers_12_self_attn_q_proj(hidden_states_134)
    query_states_24 = l__mod___model_layers_12_self_attn_q_proj * 0.125;  l__mod___model_layers_12_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_layers_12_self_attn_k_proj = self.L__mod___model_layers_12_self_attn_k_proj(hidden_states_134)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_113 = l__mod___model_layers_12_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_layers_12_self_attn_k_proj = None
    transpose_60 = view_113.transpose(1, 2);  view_113 = None
    key_states_24 = transpose_60.contiguous();  transpose_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_layers_12_self_attn_v_proj = self.L__mod___model_layers_12_self_attn_v_proj(hidden_states_134);  hidden_states_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_114 = l__mod___model_layers_12_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_layers_12_self_attn_v_proj = None
    transpose_61 = view_114.transpose(1, 2);  view_114 = None
    value_states_24 = transpose_61.contiguous();  transpose_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_115 = query_states_24.view(1, 128, 16, 64);  query_states_24 = None
    transpose_62 = view_115.transpose(1, 2);  view_115 = None
    contiguous_38 = transpose_62.contiguous();  transpose_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_25 = contiguous_38.view(16, -1, 64);  contiguous_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    key_states_25 = key_states_24.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    value_states_25 = value_states_24.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_63 = key_states_25.transpose(1, 2);  key_states_25 = None
    attn_weights_60 = torch.bmm(query_states_25, transpose_63);  query_states_25 = transpose_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_119 = attn_weights_60.view(1, 16, 128, 128);  attn_weights_60 = None
    attn_weights_61 = view_119 + attention_mask;  view_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    tensor_12 = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    attn_weights_62 = torch.max(attn_weights_61, tensor_12);  attn_weights_61 = tensor_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_63 = attn_weights_62.view(16, 128, 128);  attn_weights_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_64 = torch.nn.functional.softmax(attn_weights_63, dim = -1);  attn_weights_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_12 = torch.nn.functional.dropout(attn_weights_64, p = 0.1, training = False);  attn_weights_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_60 = torch.bmm(attn_probs_12, value_states_25);  attn_probs_12 = value_states_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_61 = attn_output_60.view(1, 16, 128, 64);  attn_output_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    attn_output_62 = attn_output_61.transpose(1, 2);  attn_output_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_63 = attn_output_62.reshape(1, 128, 1024);  attn_output_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    hidden_states_135 = self.L__mod___model_layers_12_self_attn_out_proj(attn_output_63);  attn_output_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_136 = torch.nn.functional.dropout(hidden_states_135, p = 0.1, training = False);  hidden_states_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    residual_25 = residual_24 + hidden_states_136;  residual_24 = hidden_states_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_138 = self.L__mod___model_layers_12_final_layer_norm(residual_25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_layers_12_fc1 = self.L__mod___model_layers_12_fc1(hidden_states_138);  hidden_states_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_139 = torch._C._nn.gelu(l__mod___model_layers_12_fc1);  l__mod___model_layers_12_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_140 = torch.nn.functional.dropout(hidden_states_139, p = 0.0, training = False);  hidden_states_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    hidden_states_141 = self.L__mod___model_layers_12_fc2(hidden_states_140);  hidden_states_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_142 = torch.nn.functional.dropout(hidden_states_141, p = 0.1, training = False);  hidden_states_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    residual_26 = residual_25 + hidden_states_142;  residual_25 = hidden_states_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_145 = self.L__mod___model_layers_13_self_attn_layer_norm(residual_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_layers_13_self_attn_q_proj = self.L__mod___model_layers_13_self_attn_q_proj(hidden_states_145)
    query_states_26 = l__mod___model_layers_13_self_attn_q_proj * 0.125;  l__mod___model_layers_13_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_layers_13_self_attn_k_proj = self.L__mod___model_layers_13_self_attn_k_proj(hidden_states_145)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_122 = l__mod___model_layers_13_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_layers_13_self_attn_k_proj = None
    transpose_65 = view_122.transpose(1, 2);  view_122 = None
    key_states_26 = transpose_65.contiguous();  transpose_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_layers_13_self_attn_v_proj = self.L__mod___model_layers_13_self_attn_v_proj(hidden_states_145);  hidden_states_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_123 = l__mod___model_layers_13_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_layers_13_self_attn_v_proj = None
    transpose_66 = view_123.transpose(1, 2);  view_123 = None
    value_states_26 = transpose_66.contiguous();  transpose_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_124 = query_states_26.view(1, 128, 16, 64);  query_states_26 = None
    transpose_67 = view_124.transpose(1, 2);  view_124 = None
    contiguous_41 = transpose_67.contiguous();  transpose_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_27 = contiguous_41.view(16, -1, 64);  contiguous_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    key_states_27 = key_states_26.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    value_states_27 = value_states_26.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_68 = key_states_27.transpose(1, 2);  key_states_27 = None
    attn_weights_65 = torch.bmm(query_states_27, transpose_68);  query_states_27 = transpose_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_128 = attn_weights_65.view(1, 16, 128, 128);  attn_weights_65 = None
    attn_weights_66 = view_128 + attention_mask;  view_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    tensor_13 = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    attn_weights_67 = torch.max(attn_weights_66, tensor_13);  attn_weights_66 = tensor_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_68 = attn_weights_67.view(16, 128, 128);  attn_weights_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_69 = torch.nn.functional.softmax(attn_weights_68, dim = -1);  attn_weights_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_13 = torch.nn.functional.dropout(attn_weights_69, p = 0.1, training = False);  attn_weights_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_65 = torch.bmm(attn_probs_13, value_states_27);  attn_probs_13 = value_states_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_66 = attn_output_65.view(1, 16, 128, 64);  attn_output_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    attn_output_67 = attn_output_66.transpose(1, 2);  attn_output_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_68 = attn_output_67.reshape(1, 128, 1024);  attn_output_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    hidden_states_146 = self.L__mod___model_layers_13_self_attn_out_proj(attn_output_68);  attn_output_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_147 = torch.nn.functional.dropout(hidden_states_146, p = 0.1, training = False);  hidden_states_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    residual_27 = residual_26 + hidden_states_147;  residual_26 = hidden_states_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_149 = self.L__mod___model_layers_13_final_layer_norm(residual_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_layers_13_fc1 = self.L__mod___model_layers_13_fc1(hidden_states_149);  hidden_states_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_150 = torch._C._nn.gelu(l__mod___model_layers_13_fc1);  l__mod___model_layers_13_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_151 = torch.nn.functional.dropout(hidden_states_150, p = 0.0, training = False);  hidden_states_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    hidden_states_152 = self.L__mod___model_layers_13_fc2(hidden_states_151);  hidden_states_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_153 = torch.nn.functional.dropout(hidden_states_152, p = 0.1, training = False);  hidden_states_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    residual_28 = residual_27 + hidden_states_153;  residual_27 = hidden_states_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_156 = self.L__mod___model_layers_14_self_attn_layer_norm(residual_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_layers_14_self_attn_q_proj = self.L__mod___model_layers_14_self_attn_q_proj(hidden_states_156)
    query_states_28 = l__mod___model_layers_14_self_attn_q_proj * 0.125;  l__mod___model_layers_14_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_layers_14_self_attn_k_proj = self.L__mod___model_layers_14_self_attn_k_proj(hidden_states_156)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_131 = l__mod___model_layers_14_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_layers_14_self_attn_k_proj = None
    transpose_70 = view_131.transpose(1, 2);  view_131 = None
    key_states_28 = transpose_70.contiguous();  transpose_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_layers_14_self_attn_v_proj = self.L__mod___model_layers_14_self_attn_v_proj(hidden_states_156);  hidden_states_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_132 = l__mod___model_layers_14_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_layers_14_self_attn_v_proj = None
    transpose_71 = view_132.transpose(1, 2);  view_132 = None
    value_states_28 = transpose_71.contiguous();  transpose_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_133 = query_states_28.view(1, 128, 16, 64);  query_states_28 = None
    transpose_72 = view_133.transpose(1, 2);  view_133 = None
    contiguous_44 = transpose_72.contiguous();  transpose_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_29 = contiguous_44.view(16, -1, 64);  contiguous_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    key_states_29 = key_states_28.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    value_states_29 = value_states_28.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_73 = key_states_29.transpose(1, 2);  key_states_29 = None
    attn_weights_70 = torch.bmm(query_states_29, transpose_73);  query_states_29 = transpose_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_137 = attn_weights_70.view(1, 16, 128, 128);  attn_weights_70 = None
    attn_weights_71 = view_137 + attention_mask;  view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    tensor_14 = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    attn_weights_72 = torch.max(attn_weights_71, tensor_14);  attn_weights_71 = tensor_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_73 = attn_weights_72.view(16, 128, 128);  attn_weights_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_74 = torch.nn.functional.softmax(attn_weights_73, dim = -1);  attn_weights_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_14 = torch.nn.functional.dropout(attn_weights_74, p = 0.1, training = False);  attn_weights_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_70 = torch.bmm(attn_probs_14, value_states_29);  attn_probs_14 = value_states_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_71 = attn_output_70.view(1, 16, 128, 64);  attn_output_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    attn_output_72 = attn_output_71.transpose(1, 2);  attn_output_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_73 = attn_output_72.reshape(1, 128, 1024);  attn_output_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    hidden_states_157 = self.L__mod___model_layers_14_self_attn_out_proj(attn_output_73);  attn_output_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_158 = torch.nn.functional.dropout(hidden_states_157, p = 0.1, training = False);  hidden_states_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    residual_29 = residual_28 + hidden_states_158;  residual_28 = hidden_states_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_160 = self.L__mod___model_layers_14_final_layer_norm(residual_29)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_layers_14_fc1 = self.L__mod___model_layers_14_fc1(hidden_states_160);  hidden_states_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_161 = torch._C._nn.gelu(l__mod___model_layers_14_fc1);  l__mod___model_layers_14_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_162 = torch.nn.functional.dropout(hidden_states_161, p = 0.0, training = False);  hidden_states_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    hidden_states_163 = self.L__mod___model_layers_14_fc2(hidden_states_162);  hidden_states_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_164 = torch.nn.functional.dropout(hidden_states_163, p = 0.1, training = False);  hidden_states_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    residual_30 = residual_29 + hidden_states_164;  residual_29 = hidden_states_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_167 = self.L__mod___model_layers_15_self_attn_layer_norm(residual_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_layers_15_self_attn_q_proj = self.L__mod___model_layers_15_self_attn_q_proj(hidden_states_167)
    query_states_30 = l__mod___model_layers_15_self_attn_q_proj * 0.125;  l__mod___model_layers_15_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_layers_15_self_attn_k_proj = self.L__mod___model_layers_15_self_attn_k_proj(hidden_states_167)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_140 = l__mod___model_layers_15_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_layers_15_self_attn_k_proj = None
    transpose_75 = view_140.transpose(1, 2);  view_140 = None
    key_states_30 = transpose_75.contiguous();  transpose_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_layers_15_self_attn_v_proj = self.L__mod___model_layers_15_self_attn_v_proj(hidden_states_167);  hidden_states_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_141 = l__mod___model_layers_15_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_layers_15_self_attn_v_proj = None
    transpose_76 = view_141.transpose(1, 2);  view_141 = None
    value_states_30 = transpose_76.contiguous();  transpose_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_142 = query_states_30.view(1, 128, 16, 64);  query_states_30 = None
    transpose_77 = view_142.transpose(1, 2);  view_142 = None
    contiguous_47 = transpose_77.contiguous();  transpose_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_31 = contiguous_47.view(16, -1, 64);  contiguous_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    key_states_31 = key_states_30.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    value_states_31 = value_states_30.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_78 = key_states_31.transpose(1, 2);  key_states_31 = None
    attn_weights_75 = torch.bmm(query_states_31, transpose_78);  query_states_31 = transpose_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_146 = attn_weights_75.view(1, 16, 128, 128);  attn_weights_75 = None
    attn_weights_76 = view_146 + attention_mask;  view_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    tensor_15 = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    attn_weights_77 = torch.max(attn_weights_76, tensor_15);  attn_weights_76 = tensor_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_78 = attn_weights_77.view(16, 128, 128);  attn_weights_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_79 = torch.nn.functional.softmax(attn_weights_78, dim = -1);  attn_weights_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_15 = torch.nn.functional.dropout(attn_weights_79, p = 0.1, training = False);  attn_weights_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_75 = torch.bmm(attn_probs_15, value_states_31);  attn_probs_15 = value_states_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_76 = attn_output_75.view(1, 16, 128, 64);  attn_output_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    attn_output_77 = attn_output_76.transpose(1, 2);  attn_output_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_78 = attn_output_77.reshape(1, 128, 1024);  attn_output_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    hidden_states_168 = self.L__mod___model_layers_15_self_attn_out_proj(attn_output_78);  attn_output_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_169 = torch.nn.functional.dropout(hidden_states_168, p = 0.1, training = False);  hidden_states_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    residual_31 = residual_30 + hidden_states_169;  residual_30 = hidden_states_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_171 = self.L__mod___model_layers_15_final_layer_norm(residual_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_layers_15_fc1 = self.L__mod___model_layers_15_fc1(hidden_states_171);  hidden_states_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_172 = torch._C._nn.gelu(l__mod___model_layers_15_fc1);  l__mod___model_layers_15_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_173 = torch.nn.functional.dropout(hidden_states_172, p = 0.0, training = False);  hidden_states_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    hidden_states_174 = self.L__mod___model_layers_15_fc2(hidden_states_173);  hidden_states_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_175 = torch.nn.functional.dropout(hidden_states_174, p = 0.1, training = False);  hidden_states_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    residual_32 = residual_31 + hidden_states_175;  residual_31 = hidden_states_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_178 = self.L__mod___model_layers_16_self_attn_layer_norm(residual_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_layers_16_self_attn_q_proj = self.L__mod___model_layers_16_self_attn_q_proj(hidden_states_178)
    query_states_32 = l__mod___model_layers_16_self_attn_q_proj * 0.125;  l__mod___model_layers_16_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_layers_16_self_attn_k_proj = self.L__mod___model_layers_16_self_attn_k_proj(hidden_states_178)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_149 = l__mod___model_layers_16_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_layers_16_self_attn_k_proj = None
    transpose_80 = view_149.transpose(1, 2);  view_149 = None
    key_states_32 = transpose_80.contiguous();  transpose_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_layers_16_self_attn_v_proj = self.L__mod___model_layers_16_self_attn_v_proj(hidden_states_178);  hidden_states_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_150 = l__mod___model_layers_16_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_layers_16_self_attn_v_proj = None
    transpose_81 = view_150.transpose(1, 2);  view_150 = None
    value_states_32 = transpose_81.contiguous();  transpose_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_151 = query_states_32.view(1, 128, 16, 64);  query_states_32 = None
    transpose_82 = view_151.transpose(1, 2);  view_151 = None
    contiguous_50 = transpose_82.contiguous();  transpose_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_33 = contiguous_50.view(16, -1, 64);  contiguous_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    key_states_33 = key_states_32.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    value_states_33 = value_states_32.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_83 = key_states_33.transpose(1, 2);  key_states_33 = None
    attn_weights_80 = torch.bmm(query_states_33, transpose_83);  query_states_33 = transpose_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_155 = attn_weights_80.view(1, 16, 128, 128);  attn_weights_80 = None
    attn_weights_81 = view_155 + attention_mask;  view_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    tensor_16 = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    attn_weights_82 = torch.max(attn_weights_81, tensor_16);  attn_weights_81 = tensor_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_83 = attn_weights_82.view(16, 128, 128);  attn_weights_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_84 = torch.nn.functional.softmax(attn_weights_83, dim = -1);  attn_weights_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_16 = torch.nn.functional.dropout(attn_weights_84, p = 0.1, training = False);  attn_weights_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_80 = torch.bmm(attn_probs_16, value_states_33);  attn_probs_16 = value_states_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_81 = attn_output_80.view(1, 16, 128, 64);  attn_output_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    attn_output_82 = attn_output_81.transpose(1, 2);  attn_output_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_83 = attn_output_82.reshape(1, 128, 1024);  attn_output_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    hidden_states_179 = self.L__mod___model_layers_16_self_attn_out_proj(attn_output_83);  attn_output_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_180 = torch.nn.functional.dropout(hidden_states_179, p = 0.1, training = False);  hidden_states_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    residual_33 = residual_32 + hidden_states_180;  residual_32 = hidden_states_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_182 = self.L__mod___model_layers_16_final_layer_norm(residual_33)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_layers_16_fc1 = self.L__mod___model_layers_16_fc1(hidden_states_182);  hidden_states_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_183 = torch._C._nn.gelu(l__mod___model_layers_16_fc1);  l__mod___model_layers_16_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_184 = torch.nn.functional.dropout(hidden_states_183, p = 0.0, training = False);  hidden_states_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    hidden_states_185 = self.L__mod___model_layers_16_fc2(hidden_states_184);  hidden_states_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_186 = torch.nn.functional.dropout(hidden_states_185, p = 0.1, training = False);  hidden_states_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    residual_34 = residual_33 + hidden_states_186;  residual_33 = hidden_states_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_189 = self.L__mod___model_layers_17_self_attn_layer_norm(residual_34)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_layers_17_self_attn_q_proj = self.L__mod___model_layers_17_self_attn_q_proj(hidden_states_189)
    query_states_34 = l__mod___model_layers_17_self_attn_q_proj * 0.125;  l__mod___model_layers_17_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_layers_17_self_attn_k_proj = self.L__mod___model_layers_17_self_attn_k_proj(hidden_states_189)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_158 = l__mod___model_layers_17_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_layers_17_self_attn_k_proj = None
    transpose_85 = view_158.transpose(1, 2);  view_158 = None
    key_states_34 = transpose_85.contiguous();  transpose_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_layers_17_self_attn_v_proj = self.L__mod___model_layers_17_self_attn_v_proj(hidden_states_189);  hidden_states_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_159 = l__mod___model_layers_17_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_layers_17_self_attn_v_proj = None
    transpose_86 = view_159.transpose(1, 2);  view_159 = None
    value_states_34 = transpose_86.contiguous();  transpose_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_160 = query_states_34.view(1, 128, 16, 64);  query_states_34 = None
    transpose_87 = view_160.transpose(1, 2);  view_160 = None
    contiguous_53 = transpose_87.contiguous();  transpose_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_35 = contiguous_53.view(16, -1, 64);  contiguous_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    key_states_35 = key_states_34.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    value_states_35 = value_states_34.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_88 = key_states_35.transpose(1, 2);  key_states_35 = None
    attn_weights_85 = torch.bmm(query_states_35, transpose_88);  query_states_35 = transpose_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_164 = attn_weights_85.view(1, 16, 128, 128);  attn_weights_85 = None
    attn_weights_86 = view_164 + attention_mask;  view_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    tensor_17 = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    attn_weights_87 = torch.max(attn_weights_86, tensor_17);  attn_weights_86 = tensor_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_88 = attn_weights_87.view(16, 128, 128);  attn_weights_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_89 = torch.nn.functional.softmax(attn_weights_88, dim = -1);  attn_weights_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_17 = torch.nn.functional.dropout(attn_weights_89, p = 0.1, training = False);  attn_weights_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_85 = torch.bmm(attn_probs_17, value_states_35);  attn_probs_17 = value_states_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_86 = attn_output_85.view(1, 16, 128, 64);  attn_output_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    attn_output_87 = attn_output_86.transpose(1, 2);  attn_output_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_88 = attn_output_87.reshape(1, 128, 1024);  attn_output_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    hidden_states_190 = self.L__mod___model_layers_17_self_attn_out_proj(attn_output_88);  attn_output_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_191 = torch.nn.functional.dropout(hidden_states_190, p = 0.1, training = False);  hidden_states_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    residual_35 = residual_34 + hidden_states_191;  residual_34 = hidden_states_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_193 = self.L__mod___model_layers_17_final_layer_norm(residual_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_layers_17_fc1 = self.L__mod___model_layers_17_fc1(hidden_states_193);  hidden_states_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_194 = torch._C._nn.gelu(l__mod___model_layers_17_fc1);  l__mod___model_layers_17_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_195 = torch.nn.functional.dropout(hidden_states_194, p = 0.0, training = False);  hidden_states_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    hidden_states_196 = self.L__mod___model_layers_17_fc2(hidden_states_195);  hidden_states_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_197 = torch.nn.functional.dropout(hidden_states_196, p = 0.1, training = False);  hidden_states_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    residual_36 = residual_35 + hidden_states_197;  residual_35 = hidden_states_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_200 = self.L__mod___model_layers_18_self_attn_layer_norm(residual_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_layers_18_self_attn_q_proj = self.L__mod___model_layers_18_self_attn_q_proj(hidden_states_200)
    query_states_36 = l__mod___model_layers_18_self_attn_q_proj * 0.125;  l__mod___model_layers_18_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_layers_18_self_attn_k_proj = self.L__mod___model_layers_18_self_attn_k_proj(hidden_states_200)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_167 = l__mod___model_layers_18_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_layers_18_self_attn_k_proj = None
    transpose_90 = view_167.transpose(1, 2);  view_167 = None
    key_states_36 = transpose_90.contiguous();  transpose_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_layers_18_self_attn_v_proj = self.L__mod___model_layers_18_self_attn_v_proj(hidden_states_200);  hidden_states_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_168 = l__mod___model_layers_18_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_layers_18_self_attn_v_proj = None
    transpose_91 = view_168.transpose(1, 2);  view_168 = None
    value_states_36 = transpose_91.contiguous();  transpose_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_169 = query_states_36.view(1, 128, 16, 64);  query_states_36 = None
    transpose_92 = view_169.transpose(1, 2);  view_169 = None
    contiguous_56 = transpose_92.contiguous();  transpose_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_37 = contiguous_56.view(16, -1, 64);  contiguous_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    key_states_37 = key_states_36.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    value_states_37 = value_states_36.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_93 = key_states_37.transpose(1, 2);  key_states_37 = None
    attn_weights_90 = torch.bmm(query_states_37, transpose_93);  query_states_37 = transpose_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_173 = attn_weights_90.view(1, 16, 128, 128);  attn_weights_90 = None
    attn_weights_91 = view_173 + attention_mask;  view_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    tensor_18 = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    attn_weights_92 = torch.max(attn_weights_91, tensor_18);  attn_weights_91 = tensor_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_93 = attn_weights_92.view(16, 128, 128);  attn_weights_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_94 = torch.nn.functional.softmax(attn_weights_93, dim = -1);  attn_weights_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_18 = torch.nn.functional.dropout(attn_weights_94, p = 0.1, training = False);  attn_weights_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_90 = torch.bmm(attn_probs_18, value_states_37);  attn_probs_18 = value_states_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_91 = attn_output_90.view(1, 16, 128, 64);  attn_output_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    attn_output_92 = attn_output_91.transpose(1, 2);  attn_output_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_93 = attn_output_92.reshape(1, 128, 1024);  attn_output_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    hidden_states_201 = self.L__mod___model_layers_18_self_attn_out_proj(attn_output_93);  attn_output_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_202 = torch.nn.functional.dropout(hidden_states_201, p = 0.1, training = False);  hidden_states_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    residual_37 = residual_36 + hidden_states_202;  residual_36 = hidden_states_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_204 = self.L__mod___model_layers_18_final_layer_norm(residual_37)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_layers_18_fc1 = self.L__mod___model_layers_18_fc1(hidden_states_204);  hidden_states_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_205 = torch._C._nn.gelu(l__mod___model_layers_18_fc1);  l__mod___model_layers_18_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_206 = torch.nn.functional.dropout(hidden_states_205, p = 0.0, training = False);  hidden_states_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    hidden_states_207 = self.L__mod___model_layers_18_fc2(hidden_states_206);  hidden_states_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_208 = torch.nn.functional.dropout(hidden_states_207, p = 0.1, training = False);  hidden_states_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    residual_38 = residual_37 + hidden_states_208;  residual_37 = hidden_states_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_211 = self.L__mod___model_layers_19_self_attn_layer_norm(residual_38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_layers_19_self_attn_q_proj = self.L__mod___model_layers_19_self_attn_q_proj(hidden_states_211)
    query_states_38 = l__mod___model_layers_19_self_attn_q_proj * 0.125;  l__mod___model_layers_19_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_layers_19_self_attn_k_proj = self.L__mod___model_layers_19_self_attn_k_proj(hidden_states_211)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_176 = l__mod___model_layers_19_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_layers_19_self_attn_k_proj = None
    transpose_95 = view_176.transpose(1, 2);  view_176 = None
    key_states_38 = transpose_95.contiguous();  transpose_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_layers_19_self_attn_v_proj = self.L__mod___model_layers_19_self_attn_v_proj(hidden_states_211);  hidden_states_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_177 = l__mod___model_layers_19_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_layers_19_self_attn_v_proj = None
    transpose_96 = view_177.transpose(1, 2);  view_177 = None
    value_states_38 = transpose_96.contiguous();  transpose_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_178 = query_states_38.view(1, 128, 16, 64);  query_states_38 = None
    transpose_97 = view_178.transpose(1, 2);  view_178 = None
    contiguous_59 = transpose_97.contiguous();  transpose_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_39 = contiguous_59.view(16, -1, 64);  contiguous_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    key_states_39 = key_states_38.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    value_states_39 = value_states_38.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_98 = key_states_39.transpose(1, 2);  key_states_39 = None
    attn_weights_95 = torch.bmm(query_states_39, transpose_98);  query_states_39 = transpose_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_182 = attn_weights_95.view(1, 16, 128, 128);  attn_weights_95 = None
    attn_weights_96 = view_182 + attention_mask;  view_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    tensor_19 = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    attn_weights_97 = torch.max(attn_weights_96, tensor_19);  attn_weights_96 = tensor_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_98 = attn_weights_97.view(16, 128, 128);  attn_weights_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_99 = torch.nn.functional.softmax(attn_weights_98, dim = -1);  attn_weights_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_19 = torch.nn.functional.dropout(attn_weights_99, p = 0.1, training = False);  attn_weights_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_95 = torch.bmm(attn_probs_19, value_states_39);  attn_probs_19 = value_states_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_96 = attn_output_95.view(1, 16, 128, 64);  attn_output_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    attn_output_97 = attn_output_96.transpose(1, 2);  attn_output_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_98 = attn_output_97.reshape(1, 128, 1024);  attn_output_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    hidden_states_212 = self.L__mod___model_layers_19_self_attn_out_proj(attn_output_98);  attn_output_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_213 = torch.nn.functional.dropout(hidden_states_212, p = 0.1, training = False);  hidden_states_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    residual_39 = residual_38 + hidden_states_213;  residual_38 = hidden_states_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_215 = self.L__mod___model_layers_19_final_layer_norm(residual_39)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_layers_19_fc1 = self.L__mod___model_layers_19_fc1(hidden_states_215);  hidden_states_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_216 = torch._C._nn.gelu(l__mod___model_layers_19_fc1);  l__mod___model_layers_19_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_217 = torch.nn.functional.dropout(hidden_states_216, p = 0.0, training = False);  hidden_states_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    hidden_states_218 = self.L__mod___model_layers_19_fc2(hidden_states_217);  hidden_states_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_219 = torch.nn.functional.dropout(hidden_states_218, p = 0.1, training = False);  hidden_states_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    residual_40 = residual_39 + hidden_states_219;  residual_39 = hidden_states_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_222 = self.L__mod___model_layers_20_self_attn_layer_norm(residual_40)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_layers_20_self_attn_q_proj = self.L__mod___model_layers_20_self_attn_q_proj(hidden_states_222)
    query_states_40 = l__mod___model_layers_20_self_attn_q_proj * 0.125;  l__mod___model_layers_20_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_layers_20_self_attn_k_proj = self.L__mod___model_layers_20_self_attn_k_proj(hidden_states_222)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_185 = l__mod___model_layers_20_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_layers_20_self_attn_k_proj = None
    transpose_100 = view_185.transpose(1, 2);  view_185 = None
    key_states_40 = transpose_100.contiguous();  transpose_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_layers_20_self_attn_v_proj = self.L__mod___model_layers_20_self_attn_v_proj(hidden_states_222);  hidden_states_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_186 = l__mod___model_layers_20_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_layers_20_self_attn_v_proj = None
    transpose_101 = view_186.transpose(1, 2);  view_186 = None
    value_states_40 = transpose_101.contiguous();  transpose_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_187 = query_states_40.view(1, 128, 16, 64);  query_states_40 = None
    transpose_102 = view_187.transpose(1, 2);  view_187 = None
    contiguous_62 = transpose_102.contiguous();  transpose_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_41 = contiguous_62.view(16, -1, 64);  contiguous_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    key_states_41 = key_states_40.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    value_states_41 = value_states_40.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_103 = key_states_41.transpose(1, 2);  key_states_41 = None
    attn_weights_100 = torch.bmm(query_states_41, transpose_103);  query_states_41 = transpose_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_191 = attn_weights_100.view(1, 16, 128, 128);  attn_weights_100 = None
    attn_weights_101 = view_191 + attention_mask;  view_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    tensor_20 = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    attn_weights_102 = torch.max(attn_weights_101, tensor_20);  attn_weights_101 = tensor_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_103 = attn_weights_102.view(16, 128, 128);  attn_weights_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_104 = torch.nn.functional.softmax(attn_weights_103, dim = -1);  attn_weights_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_20 = torch.nn.functional.dropout(attn_weights_104, p = 0.1, training = False);  attn_weights_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_100 = torch.bmm(attn_probs_20, value_states_41);  attn_probs_20 = value_states_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_101 = attn_output_100.view(1, 16, 128, 64);  attn_output_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    attn_output_102 = attn_output_101.transpose(1, 2);  attn_output_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_103 = attn_output_102.reshape(1, 128, 1024);  attn_output_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    hidden_states_223 = self.L__mod___model_layers_20_self_attn_out_proj(attn_output_103);  attn_output_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_224 = torch.nn.functional.dropout(hidden_states_223, p = 0.1, training = False);  hidden_states_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    residual_41 = residual_40 + hidden_states_224;  residual_40 = hidden_states_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_226 = self.L__mod___model_layers_20_final_layer_norm(residual_41)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_layers_20_fc1 = self.L__mod___model_layers_20_fc1(hidden_states_226);  hidden_states_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_227 = torch._C._nn.gelu(l__mod___model_layers_20_fc1);  l__mod___model_layers_20_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_228 = torch.nn.functional.dropout(hidden_states_227, p = 0.0, training = False);  hidden_states_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    hidden_states_229 = self.L__mod___model_layers_20_fc2(hidden_states_228);  hidden_states_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_230 = torch.nn.functional.dropout(hidden_states_229, p = 0.1, training = False);  hidden_states_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    residual_42 = residual_41 + hidden_states_230;  residual_41 = hidden_states_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_233 = self.L__mod___model_layers_21_self_attn_layer_norm(residual_42)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_layers_21_self_attn_q_proj = self.L__mod___model_layers_21_self_attn_q_proj(hidden_states_233)
    query_states_42 = l__mod___model_layers_21_self_attn_q_proj * 0.125;  l__mod___model_layers_21_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_layers_21_self_attn_k_proj = self.L__mod___model_layers_21_self_attn_k_proj(hidden_states_233)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_194 = l__mod___model_layers_21_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_layers_21_self_attn_k_proj = None
    transpose_105 = view_194.transpose(1, 2);  view_194 = None
    key_states_42 = transpose_105.contiguous();  transpose_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_layers_21_self_attn_v_proj = self.L__mod___model_layers_21_self_attn_v_proj(hidden_states_233);  hidden_states_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_195 = l__mod___model_layers_21_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_layers_21_self_attn_v_proj = None
    transpose_106 = view_195.transpose(1, 2);  view_195 = None
    value_states_42 = transpose_106.contiguous();  transpose_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_196 = query_states_42.view(1, 128, 16, 64);  query_states_42 = None
    transpose_107 = view_196.transpose(1, 2);  view_196 = None
    contiguous_65 = transpose_107.contiguous();  transpose_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_43 = contiguous_65.view(16, -1, 64);  contiguous_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    key_states_43 = key_states_42.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    value_states_43 = value_states_42.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_108 = key_states_43.transpose(1, 2);  key_states_43 = None
    attn_weights_105 = torch.bmm(query_states_43, transpose_108);  query_states_43 = transpose_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_200 = attn_weights_105.view(1, 16, 128, 128);  attn_weights_105 = None
    attn_weights_106 = view_200 + attention_mask;  view_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    tensor_21 = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    attn_weights_107 = torch.max(attn_weights_106, tensor_21);  attn_weights_106 = tensor_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_108 = attn_weights_107.view(16, 128, 128);  attn_weights_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_109 = torch.nn.functional.softmax(attn_weights_108, dim = -1);  attn_weights_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_21 = torch.nn.functional.dropout(attn_weights_109, p = 0.1, training = False);  attn_weights_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_105 = torch.bmm(attn_probs_21, value_states_43);  attn_probs_21 = value_states_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_106 = attn_output_105.view(1, 16, 128, 64);  attn_output_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    attn_output_107 = attn_output_106.transpose(1, 2);  attn_output_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_108 = attn_output_107.reshape(1, 128, 1024);  attn_output_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    hidden_states_234 = self.L__mod___model_layers_21_self_attn_out_proj(attn_output_108);  attn_output_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_235 = torch.nn.functional.dropout(hidden_states_234, p = 0.1, training = False);  hidden_states_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    residual_43 = residual_42 + hidden_states_235;  residual_42 = hidden_states_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_237 = self.L__mod___model_layers_21_final_layer_norm(residual_43)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_layers_21_fc1 = self.L__mod___model_layers_21_fc1(hidden_states_237);  hidden_states_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_238 = torch._C._nn.gelu(l__mod___model_layers_21_fc1);  l__mod___model_layers_21_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_239 = torch.nn.functional.dropout(hidden_states_238, p = 0.0, training = False);  hidden_states_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    hidden_states_240 = self.L__mod___model_layers_21_fc2(hidden_states_239);  hidden_states_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_241 = torch.nn.functional.dropout(hidden_states_240, p = 0.1, training = False);  hidden_states_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    residual_44 = residual_43 + hidden_states_241;  residual_43 = hidden_states_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_244 = self.L__mod___model_layers_22_self_attn_layer_norm(residual_44)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_layers_22_self_attn_q_proj = self.L__mod___model_layers_22_self_attn_q_proj(hidden_states_244)
    query_states_44 = l__mod___model_layers_22_self_attn_q_proj * 0.125;  l__mod___model_layers_22_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_layers_22_self_attn_k_proj = self.L__mod___model_layers_22_self_attn_k_proj(hidden_states_244)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_203 = l__mod___model_layers_22_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_layers_22_self_attn_k_proj = None
    transpose_110 = view_203.transpose(1, 2);  view_203 = None
    key_states_44 = transpose_110.contiguous();  transpose_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_layers_22_self_attn_v_proj = self.L__mod___model_layers_22_self_attn_v_proj(hidden_states_244);  hidden_states_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_204 = l__mod___model_layers_22_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_layers_22_self_attn_v_proj = None
    transpose_111 = view_204.transpose(1, 2);  view_204 = None
    value_states_44 = transpose_111.contiguous();  transpose_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_205 = query_states_44.view(1, 128, 16, 64);  query_states_44 = None
    transpose_112 = view_205.transpose(1, 2);  view_205 = None
    contiguous_68 = transpose_112.contiguous();  transpose_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_45 = contiguous_68.view(16, -1, 64);  contiguous_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    key_states_45 = key_states_44.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    value_states_45 = value_states_44.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_113 = key_states_45.transpose(1, 2);  key_states_45 = None
    attn_weights_110 = torch.bmm(query_states_45, transpose_113);  query_states_45 = transpose_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_209 = attn_weights_110.view(1, 16, 128, 128);  attn_weights_110 = None
    attn_weights_111 = view_209 + attention_mask;  view_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    tensor_22 = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    attn_weights_112 = torch.max(attn_weights_111, tensor_22);  attn_weights_111 = tensor_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_113 = attn_weights_112.view(16, 128, 128);  attn_weights_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_114 = torch.nn.functional.softmax(attn_weights_113, dim = -1);  attn_weights_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_22 = torch.nn.functional.dropout(attn_weights_114, p = 0.1, training = False);  attn_weights_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_110 = torch.bmm(attn_probs_22, value_states_45);  attn_probs_22 = value_states_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_111 = attn_output_110.view(1, 16, 128, 64);  attn_output_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    attn_output_112 = attn_output_111.transpose(1, 2);  attn_output_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_113 = attn_output_112.reshape(1, 128, 1024);  attn_output_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    hidden_states_245 = self.L__mod___model_layers_22_self_attn_out_proj(attn_output_113);  attn_output_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_246 = torch.nn.functional.dropout(hidden_states_245, p = 0.1, training = False);  hidden_states_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    residual_45 = residual_44 + hidden_states_246;  residual_44 = hidden_states_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_248 = self.L__mod___model_layers_22_final_layer_norm(residual_45)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_layers_22_fc1 = self.L__mod___model_layers_22_fc1(hidden_states_248);  hidden_states_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_249 = torch._C._nn.gelu(l__mod___model_layers_22_fc1);  l__mod___model_layers_22_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_250 = torch.nn.functional.dropout(hidden_states_249, p = 0.0, training = False);  hidden_states_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    hidden_states_251 = self.L__mod___model_layers_22_fc2(hidden_states_250);  hidden_states_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_252 = torch.nn.functional.dropout(hidden_states_251, p = 0.1, training = False);  hidden_states_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    residual_46 = residual_45 + hidden_states_252;  residual_45 = hidden_states_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_255 = self.L__mod___model_layers_23_self_attn_layer_norm(residual_46)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_layers_23_self_attn_q_proj = self.L__mod___model_layers_23_self_attn_q_proj(hidden_states_255)
    query_states_46 = l__mod___model_layers_23_self_attn_q_proj * 0.125;  l__mod___model_layers_23_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_layers_23_self_attn_k_proj = self.L__mod___model_layers_23_self_attn_k_proj(hidden_states_255)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_212 = l__mod___model_layers_23_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_layers_23_self_attn_k_proj = None
    transpose_115 = view_212.transpose(1, 2);  view_212 = None
    key_states_46 = transpose_115.contiguous();  transpose_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_layers_23_self_attn_v_proj = self.L__mod___model_layers_23_self_attn_v_proj(hidden_states_255);  hidden_states_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_213 = l__mod___model_layers_23_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_layers_23_self_attn_v_proj = None
    transpose_116 = view_213.transpose(1, 2);  view_213 = None
    value_states_46 = transpose_116.contiguous();  transpose_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_214 = query_states_46.view(1, 128, 16, 64);  query_states_46 = None
    transpose_117 = view_214.transpose(1, 2);  view_214 = None
    contiguous_71 = transpose_117.contiguous();  transpose_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_47 = contiguous_71.view(16, -1, 64);  contiguous_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    key_states_47 = key_states_46.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    value_states_47 = value_states_46.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_118 = key_states_47.transpose(1, 2);  key_states_47 = None
    attn_weights_115 = torch.bmm(query_states_47, transpose_118);  query_states_47 = transpose_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_218 = attn_weights_115.view(1, 16, 128, 128);  attn_weights_115 = None
    attn_weights_116 = view_218 + attention_mask;  view_218 = attention_mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    tensor_23 = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    attn_weights_117 = torch.max(attn_weights_116, tensor_23);  attn_weights_116 = tensor_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_118 = attn_weights_117.view(16, 128, 128);  attn_weights_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_119 = torch.nn.functional.softmax(attn_weights_118, dim = -1);  attn_weights_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_23 = torch.nn.functional.dropout(attn_weights_119, p = 0.1, training = False);  attn_weights_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_115 = torch.bmm(attn_probs_23, value_states_47);  attn_probs_23 = value_states_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_116 = attn_output_115.view(1, 16, 128, 64);  attn_output_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    attn_output_117 = attn_output_116.transpose(1, 2);  attn_output_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_118 = attn_output_117.reshape(1, 128, 1024);  attn_output_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    hidden_states_256 = self.L__mod___model_layers_23_self_attn_out_proj(attn_output_118);  attn_output_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_257 = torch.nn.functional.dropout(hidden_states_256, p = 0.1, training = False);  hidden_states_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    residual_47 = residual_46 + hidden_states_257;  residual_46 = hidden_states_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_259 = self.L__mod___model_layers_23_final_layer_norm(residual_47)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_layers_23_fc1 = self.L__mod___model_layers_23_fc1(hidden_states_259);  hidden_states_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_260 = torch._C._nn.gelu(l__mod___model_layers_23_fc1);  l__mod___model_layers_23_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_261 = torch.nn.functional.dropout(hidden_states_260, p = 0.0, training = False);  hidden_states_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    hidden_states_262 = self.L__mod___model_layers_23_fc2(hidden_states_261);  hidden_states_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_263 = torch.nn.functional.dropout(hidden_states_262, p = 0.1, training = False);  hidden_states_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    hidden_states_265 = residual_47 + hidden_states_263;  residual_47 = hidden_states_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:722, code: hidden_states = self.layer_norm(hidden_states)
    hidden_states_266 = self.L__mod___model_layer_norm(hidden_states_265);  hidden_states_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:828, code: logits = self.lm_head(outputs[0])
    logits = self.L__mod___lm_head(hidden_states_266);  hidden_states_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:833, code: shift_labels = labels.new_zeros(labels.shape)
    shift_labels = l_inputs_labels_.new_zeros((1, 128))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:834, code: shift_labels[:, :-1] = labels[:, 1:].clone()
    getitem_1 = l_inputs_labels_[(slice(None, None, None), slice(1, None, None))];  l_inputs_labels_ = None
    clone = getitem_1.clone();  getitem_1 = None
    shift_labels[(slice(None, None, None), slice(None, -1, None))] = clone;  setitem = shift_labels;  clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:835, code: shift_labels[:, -1] = self.config.pad_token_id
    shift_labels[(slice(None, None, None), -1)] = 1;  setitem_1 = shift_labels
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:838, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
    view_221 = logits.view(-1, 256008)
    view_222 = shift_labels.view(-1);  shift_labels = None
    loss = torch.nn.functional.cross_entropy(view_221, view_222, None, None, -100, None, 'mean', 0.0);  view_221 = view_222 = None
    return (loss, logits, key_states, value_states, key_states_2, value_states_2, key_states_4, value_states_4, key_states_6, value_states_6, key_states_8, value_states_8, key_states_10, value_states_10, key_states_12, value_states_12, key_states_14, value_states_14, key_states_16, value_states_16, key_states_18, value_states_18, key_states_20, value_states_20, key_states_22, value_states_22, key_states_24, value_states_24, key_states_26, value_states_26, key_states_28, value_states_28, key_states_30, value_states_30, key_states_32, value_states_32, key_states_34, value_states_34, key_states_36, value_states_36, key_states_38, value_states_38, key_states_40, value_states_40, key_states_42, value_states_42, key_states_44, value_states_44, key_states_46, value_states_46)
    