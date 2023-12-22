from __future__ import annotations



def forward(self, L_inputs_input_ids_ : torch.Tensor, L_inputs_labels_ : torch.Tensor):
    l_inputs_input_ids_ = L_inputs_input_ids_
    l_inputs_labels_ = L_inputs_labels_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:628, code: input_ids = input_ids.view(-1, input_shape[-1])
    input_ids = l_inputs_input_ids_.view(-1, 2048);  l_inputs_input_ids_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:635, code: inputs_embeds = self.embed_tokens(input_ids)
    inputs_embeds = self.L__mod___model_decoder_embed_tokens(input_ids);  input_ids = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:644, code: attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
    attention_mask = torch.ones(1, 2048, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:74, code: mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask = torch.full((2048, 2048), -3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:75, code: mask_cond = torch.arange(mask.size(-1), device=device)
    mask_cond = torch.arange(2048, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:76, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    add = mask_cond + 1
    view_1 = add.view(2048, 1);  add = None
    lt = mask_cond < view_1;  mask_cond = view_1 = None
    masked_fill_ = mask.masked_fill_(lt, 0);  lt = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:77, code: mask = mask.to(dtype)
    mask_1 = mask.to(torch.float32);  mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:81, code: return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
    getitem = mask_1[(None, None, slice(None, None, None), slice(None, None, None))];  mask_1 = None
    combined_attention_mask = getitem.expand(1, 1, 2048, 2048);  getitem = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:91, code: expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    getitem_1 = attention_mask[(slice(None, None, None), None, None, slice(None, None, None))]
    expand_1 = getitem_1.expand(1, 1, 2048, 2048);  getitem_1 = None
    expanded_mask = expand_1.to(torch.float32);  expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:93, code: inverted_mask = 1.0 - expanded_mask
    inverted_mask = 1.0 - expanded_mask;  expanded_mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:95, code: return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
    to_2 = inverted_mask.to(torch.bool)
    masked_fill = inverted_mask.masked_fill(to_2, -3.4028234663852886e+38);  inverted_mask = to_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:547, code: expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
    expanded_attn_mask = masked_fill.to(device(type='cuda', index=0));  masked_fill = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:551, code: expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
    causal_attention_mask = expanded_attn_mask + combined_attention_mask;  expanded_attn_mask = combined_attention_mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:111, code: attention_mask = attention_mask.long()
    attention_mask_1 = attention_mask.long();  attention_mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:114, code: positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1
    cumsum = torch.cumsum(attention_mask_1, dim = 1)
    type_as = cumsum.type_as(attention_mask_1);  cumsum = None
    mul = type_as * attention_mask_1;  type_as = attention_mask_1 = None
    long_1 = mul.long();  mul = None
    positions = long_1 - 1;  long_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:117, code: positions = positions[:, past_key_values_length:]
    positions_1 = positions[(slice(None, None, None), slice(0, None, None))];  positions = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:119, code: return super().forward(positions + self.offset)
    add_2 = positions_1 + 2;  positions_1 = None
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:164, code: input, self.weight, self.padding_idx, self.max_norm,
    l__mod___model_decoder_embed_positions_weight = self.L__mod___model_decoder_embed_positions_weight
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    pos_embeds = torch.nn.functional.embedding(add_2, l__mod___model_decoder_embed_positions_weight, None, None, 2.0, False, False);  add_2 = l__mod___model_decoder_embed_positions_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:658, code: hidden_states = inputs_embeds + pos_embeds
    residual = inputs_embeds + pos_embeds;  inputs_embeds = pos_embeds = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:327, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_1 = self.L__mod___model_decoder_layers_0_self_attn_layer_norm(residual)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:173, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_0_self_attn_q_proj = self.L__mod___model_decoder_layers_0_self_attn_q_proj(hidden_states_1)
    query_states = l__mod___model_decoder_layers_0_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_0_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:191, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_0_self_attn_k_proj = self.L__mod___model_decoder_layers_0_self_attn_k_proj(hidden_states_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_2 = l__mod___model_decoder_layers_0_self_attn_k_proj.view(1, -1, 12, 64);  l__mod___model_decoder_layers_0_self_attn_k_proj = None
    transpose = view_2.transpose(1, 2);  view_2 = None
    key_states = transpose.contiguous();  transpose = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:192, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_0_self_attn_v_proj = self.L__mod___model_decoder_layers_0_self_attn_v_proj(hidden_states_1);  hidden_states_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_3 = l__mod___model_decoder_layers_0_self_attn_v_proj.view(1, -1, 12, 64);  l__mod___model_decoder_layers_0_self_attn_v_proj = None
    transpose_1 = view_3.transpose(1, 2);  view_3 = None
    value_states = transpose_1.contiguous();  transpose_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_4 = query_states.view(1, 2048, 12, 64);  query_states = None
    transpose_2 = view_4.transpose(1, 2);  view_4 = None
    contiguous_2 = transpose_2.contiguous();  transpose_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:205, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_1 = contiguous_2.view(12, -1, 64);  contiguous_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:206, code: key_states = key_states.view(*proj_shape)
    key_states_1 = key_states.view(12, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:207, code: value_states = value_states.view(*proj_shape)
    value_states_1 = value_states.view(12, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:210, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_3 = key_states_1.transpose(1, 2);  key_states_1 = None
    attn_weights = torch.bmm(query_states_1, transpose_3);  query_states_1 = transpose_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:223, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_8 = attn_weights.view(1, 12, 2048, 2048);  attn_weights = None
    attn_weights_1 = view_8 + causal_attention_mask;  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:225, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    tensor = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:224, code: attn_weights = torch.max(
    attn_weights_2 = torch.max(attn_weights_1, tensor);  attn_weights_1 = tensor = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:227, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_3 = attn_weights_2.view(12, 2048, 2048);  attn_weights_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_4 = torch.nn.functional.softmax(attn_weights_3, dim = -1);  attn_weights_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:254, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs = torch.nn.functional.dropout(attn_weights_4, p = 0.0, training = False);  attn_weights_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:256, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output = torch.bmm(attn_probs, value_states_1);  attn_probs = value_states_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:264, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_1 = attn_output.view(1, 12, 2048, 64);  attn_output = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = attn_output.transpose(1, 2)
    attn_output_2 = attn_output_1.transpose(1, 2);  attn_output_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:269, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_3 = attn_output_2.reshape(1, 2048, 768);  attn_output_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:271, code: attn_output = self.out_proj(attn_output)
    hidden_states_2 = self.L__mod___model_decoder_layers_0_self_attn_out_proj(attn_output_3);  attn_output_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:337, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_3 = torch.nn.functional.dropout(hidden_states_2, p = 0.1, training = False);  hidden_states_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:338, code: hidden_states = residual + hidden_states
    hidden_states_4 = residual + hidden_states_3;  residual = hidden_states_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:346, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
    residual_1 = hidden_states_4.reshape(-1, 768);  hidden_states_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:351, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_6 = self.L__mod___model_decoder_layers_0_final_layer_norm(residual_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:353, code: hidden_states = self.fc1(hidden_states)
    hidden_states_7 = self.L__mod___model_decoder_layers_0_fc1(hidden_states_6);  hidden_states_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:354, code: hidden_states = self.activation_fn(hidden_states)
    hidden_states_8 = self.L__mod___model_decoder_layers_0_activation_fn(hidden_states_7);  hidden_states_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:356, code: hidden_states = self.fc2(hidden_states)
    hidden_states_9 = self.L__mod___model_decoder_layers_0_fc2(hidden_states_8);  hidden_states_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:357, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_10 = torch.nn.functional.dropout(hidden_states_9, p = 0.1, training = False);  hidden_states_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:359, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
    add_6 = residual_1 + hidden_states_10;  residual_1 = hidden_states_10 = None
    residual_2 = add_6.view((1, 2048, 768));  add_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:327, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_13 = self.L__mod___model_decoder_layers_1_self_attn_layer_norm(residual_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:173, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_1_self_attn_q_proj = self.L__mod___model_decoder_layers_1_self_attn_q_proj(hidden_states_13)
    query_states_2 = l__mod___model_decoder_layers_1_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_1_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:191, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_1_self_attn_k_proj = self.L__mod___model_decoder_layers_1_self_attn_k_proj(hidden_states_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_12 = l__mod___model_decoder_layers_1_self_attn_k_proj.view(1, -1, 12, 64);  l__mod___model_decoder_layers_1_self_attn_k_proj = None
    transpose_5 = view_12.transpose(1, 2);  view_12 = None
    key_states_2 = transpose_5.contiguous();  transpose_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:192, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_1_self_attn_v_proj = self.L__mod___model_decoder_layers_1_self_attn_v_proj(hidden_states_13);  hidden_states_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_13 = l__mod___model_decoder_layers_1_self_attn_v_proj.view(1, -1, 12, 64);  l__mod___model_decoder_layers_1_self_attn_v_proj = None
    transpose_6 = view_13.transpose(1, 2);  view_13 = None
    value_states_2 = transpose_6.contiguous();  transpose_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_14 = query_states_2.view(1, 2048, 12, 64);  query_states_2 = None
    transpose_7 = view_14.transpose(1, 2);  view_14 = None
    contiguous_5 = transpose_7.contiguous();  transpose_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:205, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_3 = contiguous_5.view(12, -1, 64);  contiguous_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:206, code: key_states = key_states.view(*proj_shape)
    key_states_3 = key_states_2.view(12, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:207, code: value_states = value_states.view(*proj_shape)
    value_states_3 = value_states_2.view(12, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:210, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_8 = key_states_3.transpose(1, 2);  key_states_3 = None
    attn_weights_5 = torch.bmm(query_states_3, transpose_8);  query_states_3 = transpose_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:223, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_18 = attn_weights_5.view(1, 12, 2048, 2048);  attn_weights_5 = None
    attn_weights_6 = view_18 + causal_attention_mask;  view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:225, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    tensor_1 = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:224, code: attn_weights = torch.max(
    attn_weights_7 = torch.max(attn_weights_6, tensor_1);  attn_weights_6 = tensor_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:227, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_8 = attn_weights_7.view(12, 2048, 2048);  attn_weights_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_9 = torch.nn.functional.softmax(attn_weights_8, dim = -1);  attn_weights_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:254, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_1 = torch.nn.functional.dropout(attn_weights_9, p = 0.0, training = False);  attn_weights_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:256, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_5 = torch.bmm(attn_probs_1, value_states_3);  attn_probs_1 = value_states_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:264, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_6 = attn_output_5.view(1, 12, 2048, 64);  attn_output_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = attn_output.transpose(1, 2)
    attn_output_7 = attn_output_6.transpose(1, 2);  attn_output_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:269, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_8 = attn_output_7.reshape(1, 2048, 768);  attn_output_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:271, code: attn_output = self.out_proj(attn_output)
    hidden_states_14 = self.L__mod___model_decoder_layers_1_self_attn_out_proj(attn_output_8);  attn_output_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:337, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_15 = torch.nn.functional.dropout(hidden_states_14, p = 0.1, training = False);  hidden_states_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:338, code: hidden_states = residual + hidden_states
    hidden_states_16 = residual_2 + hidden_states_15;  residual_2 = hidden_states_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:346, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
    residual_3 = hidden_states_16.reshape(-1, 768);  hidden_states_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:351, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_18 = self.L__mod___model_decoder_layers_1_final_layer_norm(residual_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:353, code: hidden_states = self.fc1(hidden_states)
    hidden_states_19 = self.L__mod___model_decoder_layers_1_fc1(hidden_states_18);  hidden_states_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:354, code: hidden_states = self.activation_fn(hidden_states)
    hidden_states_20 = self.L__mod___model_decoder_layers_1_activation_fn(hidden_states_19);  hidden_states_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:356, code: hidden_states = self.fc2(hidden_states)
    hidden_states_21 = self.L__mod___model_decoder_layers_1_fc2(hidden_states_20);  hidden_states_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:357, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_22 = torch.nn.functional.dropout(hidden_states_21, p = 0.1, training = False);  hidden_states_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:359, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
    add_9 = residual_3 + hidden_states_22;  residual_3 = hidden_states_22 = None
    residual_4 = add_9.view((1, 2048, 768));  add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:327, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_25 = self.L__mod___model_decoder_layers_2_self_attn_layer_norm(residual_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:173, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_2_self_attn_q_proj = self.L__mod___model_decoder_layers_2_self_attn_q_proj(hidden_states_25)
    query_states_4 = l__mod___model_decoder_layers_2_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_2_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:191, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_2_self_attn_k_proj = self.L__mod___model_decoder_layers_2_self_attn_k_proj(hidden_states_25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_22 = l__mod___model_decoder_layers_2_self_attn_k_proj.view(1, -1, 12, 64);  l__mod___model_decoder_layers_2_self_attn_k_proj = None
    transpose_10 = view_22.transpose(1, 2);  view_22 = None
    key_states_4 = transpose_10.contiguous();  transpose_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:192, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_2_self_attn_v_proj = self.L__mod___model_decoder_layers_2_self_attn_v_proj(hidden_states_25);  hidden_states_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_23 = l__mod___model_decoder_layers_2_self_attn_v_proj.view(1, -1, 12, 64);  l__mod___model_decoder_layers_2_self_attn_v_proj = None
    transpose_11 = view_23.transpose(1, 2);  view_23 = None
    value_states_4 = transpose_11.contiguous();  transpose_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_24 = query_states_4.view(1, 2048, 12, 64);  query_states_4 = None
    transpose_12 = view_24.transpose(1, 2);  view_24 = None
    contiguous_8 = transpose_12.contiguous();  transpose_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:205, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_5 = contiguous_8.view(12, -1, 64);  contiguous_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:206, code: key_states = key_states.view(*proj_shape)
    key_states_5 = key_states_4.view(12, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:207, code: value_states = value_states.view(*proj_shape)
    value_states_5 = value_states_4.view(12, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:210, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_13 = key_states_5.transpose(1, 2);  key_states_5 = None
    attn_weights_10 = torch.bmm(query_states_5, transpose_13);  query_states_5 = transpose_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:223, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_28 = attn_weights_10.view(1, 12, 2048, 2048);  attn_weights_10 = None
    attn_weights_11 = view_28 + causal_attention_mask;  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:225, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    tensor_2 = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:224, code: attn_weights = torch.max(
    attn_weights_12 = torch.max(attn_weights_11, tensor_2);  attn_weights_11 = tensor_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:227, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_13 = attn_weights_12.view(12, 2048, 2048);  attn_weights_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_14 = torch.nn.functional.softmax(attn_weights_13, dim = -1);  attn_weights_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:254, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_2 = torch.nn.functional.dropout(attn_weights_14, p = 0.0, training = False);  attn_weights_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:256, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_10 = torch.bmm(attn_probs_2, value_states_5);  attn_probs_2 = value_states_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:264, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_11 = attn_output_10.view(1, 12, 2048, 64);  attn_output_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = attn_output.transpose(1, 2)
    attn_output_12 = attn_output_11.transpose(1, 2);  attn_output_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:269, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_13 = attn_output_12.reshape(1, 2048, 768);  attn_output_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:271, code: attn_output = self.out_proj(attn_output)
    hidden_states_26 = self.L__mod___model_decoder_layers_2_self_attn_out_proj(attn_output_13);  attn_output_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:337, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_27 = torch.nn.functional.dropout(hidden_states_26, p = 0.1, training = False);  hidden_states_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:338, code: hidden_states = residual + hidden_states
    hidden_states_28 = residual_4 + hidden_states_27;  residual_4 = hidden_states_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:346, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
    residual_5 = hidden_states_28.reshape(-1, 768);  hidden_states_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:351, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_30 = self.L__mod___model_decoder_layers_2_final_layer_norm(residual_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:353, code: hidden_states = self.fc1(hidden_states)
    hidden_states_31 = self.L__mod___model_decoder_layers_2_fc1(hidden_states_30);  hidden_states_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:354, code: hidden_states = self.activation_fn(hidden_states)
    hidden_states_32 = self.L__mod___model_decoder_layers_2_activation_fn(hidden_states_31);  hidden_states_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:356, code: hidden_states = self.fc2(hidden_states)
    hidden_states_33 = self.L__mod___model_decoder_layers_2_fc2(hidden_states_32);  hidden_states_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:357, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_34 = torch.nn.functional.dropout(hidden_states_33, p = 0.1, training = False);  hidden_states_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:359, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
    add_12 = residual_5 + hidden_states_34;  residual_5 = hidden_states_34 = None
    residual_6 = add_12.view((1, 2048, 768));  add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:327, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_37 = self.L__mod___model_decoder_layers_3_self_attn_layer_norm(residual_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:173, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_3_self_attn_q_proj = self.L__mod___model_decoder_layers_3_self_attn_q_proj(hidden_states_37)
    query_states_6 = l__mod___model_decoder_layers_3_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_3_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:191, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_3_self_attn_k_proj = self.L__mod___model_decoder_layers_3_self_attn_k_proj(hidden_states_37)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_32 = l__mod___model_decoder_layers_3_self_attn_k_proj.view(1, -1, 12, 64);  l__mod___model_decoder_layers_3_self_attn_k_proj = None
    transpose_15 = view_32.transpose(1, 2);  view_32 = None
    key_states_6 = transpose_15.contiguous();  transpose_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:192, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_3_self_attn_v_proj = self.L__mod___model_decoder_layers_3_self_attn_v_proj(hidden_states_37);  hidden_states_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_33 = l__mod___model_decoder_layers_3_self_attn_v_proj.view(1, -1, 12, 64);  l__mod___model_decoder_layers_3_self_attn_v_proj = None
    transpose_16 = view_33.transpose(1, 2);  view_33 = None
    value_states_6 = transpose_16.contiguous();  transpose_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_34 = query_states_6.view(1, 2048, 12, 64);  query_states_6 = None
    transpose_17 = view_34.transpose(1, 2);  view_34 = None
    contiguous_11 = transpose_17.contiguous();  transpose_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:205, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_7 = contiguous_11.view(12, -1, 64);  contiguous_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:206, code: key_states = key_states.view(*proj_shape)
    key_states_7 = key_states_6.view(12, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:207, code: value_states = value_states.view(*proj_shape)
    value_states_7 = value_states_6.view(12, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:210, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_18 = key_states_7.transpose(1, 2);  key_states_7 = None
    attn_weights_15 = torch.bmm(query_states_7, transpose_18);  query_states_7 = transpose_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:223, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_38 = attn_weights_15.view(1, 12, 2048, 2048);  attn_weights_15 = None
    attn_weights_16 = view_38 + causal_attention_mask;  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:225, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    tensor_3 = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:224, code: attn_weights = torch.max(
    attn_weights_17 = torch.max(attn_weights_16, tensor_3);  attn_weights_16 = tensor_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:227, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_18 = attn_weights_17.view(12, 2048, 2048);  attn_weights_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_19 = torch.nn.functional.softmax(attn_weights_18, dim = -1);  attn_weights_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:254, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_3 = torch.nn.functional.dropout(attn_weights_19, p = 0.0, training = False);  attn_weights_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:256, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_15 = torch.bmm(attn_probs_3, value_states_7);  attn_probs_3 = value_states_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:264, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_16 = attn_output_15.view(1, 12, 2048, 64);  attn_output_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = attn_output.transpose(1, 2)
    attn_output_17 = attn_output_16.transpose(1, 2);  attn_output_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:269, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_18 = attn_output_17.reshape(1, 2048, 768);  attn_output_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:271, code: attn_output = self.out_proj(attn_output)
    hidden_states_38 = self.L__mod___model_decoder_layers_3_self_attn_out_proj(attn_output_18);  attn_output_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:337, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_39 = torch.nn.functional.dropout(hidden_states_38, p = 0.1, training = False);  hidden_states_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:338, code: hidden_states = residual + hidden_states
    hidden_states_40 = residual_6 + hidden_states_39;  residual_6 = hidden_states_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:346, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
    residual_7 = hidden_states_40.reshape(-1, 768);  hidden_states_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:351, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_42 = self.L__mod___model_decoder_layers_3_final_layer_norm(residual_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:353, code: hidden_states = self.fc1(hidden_states)
    hidden_states_43 = self.L__mod___model_decoder_layers_3_fc1(hidden_states_42);  hidden_states_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:354, code: hidden_states = self.activation_fn(hidden_states)
    hidden_states_44 = self.L__mod___model_decoder_layers_3_activation_fn(hidden_states_43);  hidden_states_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:356, code: hidden_states = self.fc2(hidden_states)
    hidden_states_45 = self.L__mod___model_decoder_layers_3_fc2(hidden_states_44);  hidden_states_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:357, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_46 = torch.nn.functional.dropout(hidden_states_45, p = 0.1, training = False);  hidden_states_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:359, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
    add_15 = residual_7 + hidden_states_46;  residual_7 = hidden_states_46 = None
    residual_8 = add_15.view((1, 2048, 768));  add_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:327, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_49 = self.L__mod___model_decoder_layers_4_self_attn_layer_norm(residual_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:173, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_4_self_attn_q_proj = self.L__mod___model_decoder_layers_4_self_attn_q_proj(hidden_states_49)
    query_states_8 = l__mod___model_decoder_layers_4_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_4_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:191, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_4_self_attn_k_proj = self.L__mod___model_decoder_layers_4_self_attn_k_proj(hidden_states_49)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_42 = l__mod___model_decoder_layers_4_self_attn_k_proj.view(1, -1, 12, 64);  l__mod___model_decoder_layers_4_self_attn_k_proj = None
    transpose_20 = view_42.transpose(1, 2);  view_42 = None
    key_states_8 = transpose_20.contiguous();  transpose_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:192, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_4_self_attn_v_proj = self.L__mod___model_decoder_layers_4_self_attn_v_proj(hidden_states_49);  hidden_states_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_43 = l__mod___model_decoder_layers_4_self_attn_v_proj.view(1, -1, 12, 64);  l__mod___model_decoder_layers_4_self_attn_v_proj = None
    transpose_21 = view_43.transpose(1, 2);  view_43 = None
    value_states_8 = transpose_21.contiguous();  transpose_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_44 = query_states_8.view(1, 2048, 12, 64);  query_states_8 = None
    transpose_22 = view_44.transpose(1, 2);  view_44 = None
    contiguous_14 = transpose_22.contiguous();  transpose_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:205, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_9 = contiguous_14.view(12, -1, 64);  contiguous_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:206, code: key_states = key_states.view(*proj_shape)
    key_states_9 = key_states_8.view(12, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:207, code: value_states = value_states.view(*proj_shape)
    value_states_9 = value_states_8.view(12, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:210, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_23 = key_states_9.transpose(1, 2);  key_states_9 = None
    attn_weights_20 = torch.bmm(query_states_9, transpose_23);  query_states_9 = transpose_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:223, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_48 = attn_weights_20.view(1, 12, 2048, 2048);  attn_weights_20 = None
    attn_weights_21 = view_48 + causal_attention_mask;  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:225, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    tensor_4 = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:224, code: attn_weights = torch.max(
    attn_weights_22 = torch.max(attn_weights_21, tensor_4);  attn_weights_21 = tensor_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:227, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_23 = attn_weights_22.view(12, 2048, 2048);  attn_weights_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_24 = torch.nn.functional.softmax(attn_weights_23, dim = -1);  attn_weights_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:254, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_4 = torch.nn.functional.dropout(attn_weights_24, p = 0.0, training = False);  attn_weights_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:256, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_20 = torch.bmm(attn_probs_4, value_states_9);  attn_probs_4 = value_states_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:264, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_21 = attn_output_20.view(1, 12, 2048, 64);  attn_output_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = attn_output.transpose(1, 2)
    attn_output_22 = attn_output_21.transpose(1, 2);  attn_output_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:269, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_23 = attn_output_22.reshape(1, 2048, 768);  attn_output_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:271, code: attn_output = self.out_proj(attn_output)
    hidden_states_50 = self.L__mod___model_decoder_layers_4_self_attn_out_proj(attn_output_23);  attn_output_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:337, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_51 = torch.nn.functional.dropout(hidden_states_50, p = 0.1, training = False);  hidden_states_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:338, code: hidden_states = residual + hidden_states
    hidden_states_52 = residual_8 + hidden_states_51;  residual_8 = hidden_states_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:346, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
    residual_9 = hidden_states_52.reshape(-1, 768);  hidden_states_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:351, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_54 = self.L__mod___model_decoder_layers_4_final_layer_norm(residual_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:353, code: hidden_states = self.fc1(hidden_states)
    hidden_states_55 = self.L__mod___model_decoder_layers_4_fc1(hidden_states_54);  hidden_states_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:354, code: hidden_states = self.activation_fn(hidden_states)
    hidden_states_56 = self.L__mod___model_decoder_layers_4_activation_fn(hidden_states_55);  hidden_states_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:356, code: hidden_states = self.fc2(hidden_states)
    hidden_states_57 = self.L__mod___model_decoder_layers_4_fc2(hidden_states_56);  hidden_states_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:357, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_58 = torch.nn.functional.dropout(hidden_states_57, p = 0.1, training = False);  hidden_states_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:359, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
    add_18 = residual_9 + hidden_states_58;  residual_9 = hidden_states_58 = None
    residual_10 = add_18.view((1, 2048, 768));  add_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:327, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_61 = self.L__mod___model_decoder_layers_5_self_attn_layer_norm(residual_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:173, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_5_self_attn_q_proj = self.L__mod___model_decoder_layers_5_self_attn_q_proj(hidden_states_61)
    query_states_10 = l__mod___model_decoder_layers_5_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_5_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:191, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_5_self_attn_k_proj = self.L__mod___model_decoder_layers_5_self_attn_k_proj(hidden_states_61)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_52 = l__mod___model_decoder_layers_5_self_attn_k_proj.view(1, -1, 12, 64);  l__mod___model_decoder_layers_5_self_attn_k_proj = None
    transpose_25 = view_52.transpose(1, 2);  view_52 = None
    key_states_10 = transpose_25.contiguous();  transpose_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:192, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_5_self_attn_v_proj = self.L__mod___model_decoder_layers_5_self_attn_v_proj(hidden_states_61);  hidden_states_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_53 = l__mod___model_decoder_layers_5_self_attn_v_proj.view(1, -1, 12, 64);  l__mod___model_decoder_layers_5_self_attn_v_proj = None
    transpose_26 = view_53.transpose(1, 2);  view_53 = None
    value_states_10 = transpose_26.contiguous();  transpose_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_54 = query_states_10.view(1, 2048, 12, 64);  query_states_10 = None
    transpose_27 = view_54.transpose(1, 2);  view_54 = None
    contiguous_17 = transpose_27.contiguous();  transpose_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:205, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_11 = contiguous_17.view(12, -1, 64);  contiguous_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:206, code: key_states = key_states.view(*proj_shape)
    key_states_11 = key_states_10.view(12, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:207, code: value_states = value_states.view(*proj_shape)
    value_states_11 = value_states_10.view(12, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:210, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_28 = key_states_11.transpose(1, 2);  key_states_11 = None
    attn_weights_25 = torch.bmm(query_states_11, transpose_28);  query_states_11 = transpose_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:223, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_58 = attn_weights_25.view(1, 12, 2048, 2048);  attn_weights_25 = None
    attn_weights_26 = view_58 + causal_attention_mask;  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:225, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    tensor_5 = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:224, code: attn_weights = torch.max(
    attn_weights_27 = torch.max(attn_weights_26, tensor_5);  attn_weights_26 = tensor_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:227, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_28 = attn_weights_27.view(12, 2048, 2048);  attn_weights_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_29 = torch.nn.functional.softmax(attn_weights_28, dim = -1);  attn_weights_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:254, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_5 = torch.nn.functional.dropout(attn_weights_29, p = 0.0, training = False);  attn_weights_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:256, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_25 = torch.bmm(attn_probs_5, value_states_11);  attn_probs_5 = value_states_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:264, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_26 = attn_output_25.view(1, 12, 2048, 64);  attn_output_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = attn_output.transpose(1, 2)
    attn_output_27 = attn_output_26.transpose(1, 2);  attn_output_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:269, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_28 = attn_output_27.reshape(1, 2048, 768);  attn_output_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:271, code: attn_output = self.out_proj(attn_output)
    hidden_states_62 = self.L__mod___model_decoder_layers_5_self_attn_out_proj(attn_output_28);  attn_output_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:337, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_63 = torch.nn.functional.dropout(hidden_states_62, p = 0.1, training = False);  hidden_states_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:338, code: hidden_states = residual + hidden_states
    hidden_states_64 = residual_10 + hidden_states_63;  residual_10 = hidden_states_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:346, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
    residual_11 = hidden_states_64.reshape(-1, 768);  hidden_states_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:351, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_66 = self.L__mod___model_decoder_layers_5_final_layer_norm(residual_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:353, code: hidden_states = self.fc1(hidden_states)
    hidden_states_67 = self.L__mod___model_decoder_layers_5_fc1(hidden_states_66);  hidden_states_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:354, code: hidden_states = self.activation_fn(hidden_states)
    hidden_states_68 = self.L__mod___model_decoder_layers_5_activation_fn(hidden_states_67);  hidden_states_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:356, code: hidden_states = self.fc2(hidden_states)
    hidden_states_69 = self.L__mod___model_decoder_layers_5_fc2(hidden_states_68);  hidden_states_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:357, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_70 = torch.nn.functional.dropout(hidden_states_69, p = 0.1, training = False);  hidden_states_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:359, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
    add_21 = residual_11 + hidden_states_70;  residual_11 = hidden_states_70 = None
    residual_12 = add_21.view((1, 2048, 768));  add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:327, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_73 = self.L__mod___model_decoder_layers_6_self_attn_layer_norm(residual_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:173, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_6_self_attn_q_proj = self.L__mod___model_decoder_layers_6_self_attn_q_proj(hidden_states_73)
    query_states_12 = l__mod___model_decoder_layers_6_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_6_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:191, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_6_self_attn_k_proj = self.L__mod___model_decoder_layers_6_self_attn_k_proj(hidden_states_73)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_62 = l__mod___model_decoder_layers_6_self_attn_k_proj.view(1, -1, 12, 64);  l__mod___model_decoder_layers_6_self_attn_k_proj = None
    transpose_30 = view_62.transpose(1, 2);  view_62 = None
    key_states_12 = transpose_30.contiguous();  transpose_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:192, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_6_self_attn_v_proj = self.L__mod___model_decoder_layers_6_self_attn_v_proj(hidden_states_73);  hidden_states_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_63 = l__mod___model_decoder_layers_6_self_attn_v_proj.view(1, -1, 12, 64);  l__mod___model_decoder_layers_6_self_attn_v_proj = None
    transpose_31 = view_63.transpose(1, 2);  view_63 = None
    value_states_12 = transpose_31.contiguous();  transpose_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_64 = query_states_12.view(1, 2048, 12, 64);  query_states_12 = None
    transpose_32 = view_64.transpose(1, 2);  view_64 = None
    contiguous_20 = transpose_32.contiguous();  transpose_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:205, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_13 = contiguous_20.view(12, -1, 64);  contiguous_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:206, code: key_states = key_states.view(*proj_shape)
    key_states_13 = key_states_12.view(12, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:207, code: value_states = value_states.view(*proj_shape)
    value_states_13 = value_states_12.view(12, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:210, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_33 = key_states_13.transpose(1, 2);  key_states_13 = None
    attn_weights_30 = torch.bmm(query_states_13, transpose_33);  query_states_13 = transpose_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:223, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_68 = attn_weights_30.view(1, 12, 2048, 2048);  attn_weights_30 = None
    attn_weights_31 = view_68 + causal_attention_mask;  view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:225, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    tensor_6 = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:224, code: attn_weights = torch.max(
    attn_weights_32 = torch.max(attn_weights_31, tensor_6);  attn_weights_31 = tensor_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:227, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_33 = attn_weights_32.view(12, 2048, 2048);  attn_weights_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_34 = torch.nn.functional.softmax(attn_weights_33, dim = -1);  attn_weights_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:254, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_6 = torch.nn.functional.dropout(attn_weights_34, p = 0.0, training = False);  attn_weights_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:256, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_30 = torch.bmm(attn_probs_6, value_states_13);  attn_probs_6 = value_states_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:264, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_31 = attn_output_30.view(1, 12, 2048, 64);  attn_output_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = attn_output.transpose(1, 2)
    attn_output_32 = attn_output_31.transpose(1, 2);  attn_output_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:269, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_33 = attn_output_32.reshape(1, 2048, 768);  attn_output_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:271, code: attn_output = self.out_proj(attn_output)
    hidden_states_74 = self.L__mod___model_decoder_layers_6_self_attn_out_proj(attn_output_33);  attn_output_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:337, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_75 = torch.nn.functional.dropout(hidden_states_74, p = 0.1, training = False);  hidden_states_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:338, code: hidden_states = residual + hidden_states
    hidden_states_76 = residual_12 + hidden_states_75;  residual_12 = hidden_states_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:346, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
    residual_13 = hidden_states_76.reshape(-1, 768);  hidden_states_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:351, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_78 = self.L__mod___model_decoder_layers_6_final_layer_norm(residual_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:353, code: hidden_states = self.fc1(hidden_states)
    hidden_states_79 = self.L__mod___model_decoder_layers_6_fc1(hidden_states_78);  hidden_states_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:354, code: hidden_states = self.activation_fn(hidden_states)
    hidden_states_80 = self.L__mod___model_decoder_layers_6_activation_fn(hidden_states_79);  hidden_states_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:356, code: hidden_states = self.fc2(hidden_states)
    hidden_states_81 = self.L__mod___model_decoder_layers_6_fc2(hidden_states_80);  hidden_states_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:357, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_82 = torch.nn.functional.dropout(hidden_states_81, p = 0.1, training = False);  hidden_states_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:359, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
    add_24 = residual_13 + hidden_states_82;  residual_13 = hidden_states_82 = None
    residual_14 = add_24.view((1, 2048, 768));  add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:327, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_85 = self.L__mod___model_decoder_layers_7_self_attn_layer_norm(residual_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:173, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_7_self_attn_q_proj = self.L__mod___model_decoder_layers_7_self_attn_q_proj(hidden_states_85)
    query_states_14 = l__mod___model_decoder_layers_7_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_7_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:191, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_7_self_attn_k_proj = self.L__mod___model_decoder_layers_7_self_attn_k_proj(hidden_states_85)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_72 = l__mod___model_decoder_layers_7_self_attn_k_proj.view(1, -1, 12, 64);  l__mod___model_decoder_layers_7_self_attn_k_proj = None
    transpose_35 = view_72.transpose(1, 2);  view_72 = None
    key_states_14 = transpose_35.contiguous();  transpose_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:192, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_7_self_attn_v_proj = self.L__mod___model_decoder_layers_7_self_attn_v_proj(hidden_states_85);  hidden_states_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_73 = l__mod___model_decoder_layers_7_self_attn_v_proj.view(1, -1, 12, 64);  l__mod___model_decoder_layers_7_self_attn_v_proj = None
    transpose_36 = view_73.transpose(1, 2);  view_73 = None
    value_states_14 = transpose_36.contiguous();  transpose_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_74 = query_states_14.view(1, 2048, 12, 64);  query_states_14 = None
    transpose_37 = view_74.transpose(1, 2);  view_74 = None
    contiguous_23 = transpose_37.contiguous();  transpose_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:205, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_15 = contiguous_23.view(12, -1, 64);  contiguous_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:206, code: key_states = key_states.view(*proj_shape)
    key_states_15 = key_states_14.view(12, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:207, code: value_states = value_states.view(*proj_shape)
    value_states_15 = value_states_14.view(12, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:210, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_38 = key_states_15.transpose(1, 2);  key_states_15 = None
    attn_weights_35 = torch.bmm(query_states_15, transpose_38);  query_states_15 = transpose_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:223, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_78 = attn_weights_35.view(1, 12, 2048, 2048);  attn_weights_35 = None
    attn_weights_36 = view_78 + causal_attention_mask;  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:225, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    tensor_7 = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:224, code: attn_weights = torch.max(
    attn_weights_37 = torch.max(attn_weights_36, tensor_7);  attn_weights_36 = tensor_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:227, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_38 = attn_weights_37.view(12, 2048, 2048);  attn_weights_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_39 = torch.nn.functional.softmax(attn_weights_38, dim = -1);  attn_weights_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:254, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_7 = torch.nn.functional.dropout(attn_weights_39, p = 0.0, training = False);  attn_weights_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:256, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_35 = torch.bmm(attn_probs_7, value_states_15);  attn_probs_7 = value_states_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:264, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_36 = attn_output_35.view(1, 12, 2048, 64);  attn_output_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = attn_output.transpose(1, 2)
    attn_output_37 = attn_output_36.transpose(1, 2);  attn_output_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:269, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_38 = attn_output_37.reshape(1, 2048, 768);  attn_output_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:271, code: attn_output = self.out_proj(attn_output)
    hidden_states_86 = self.L__mod___model_decoder_layers_7_self_attn_out_proj(attn_output_38);  attn_output_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:337, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_87 = torch.nn.functional.dropout(hidden_states_86, p = 0.1, training = False);  hidden_states_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:338, code: hidden_states = residual + hidden_states
    hidden_states_88 = residual_14 + hidden_states_87;  residual_14 = hidden_states_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:346, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
    residual_15 = hidden_states_88.reshape(-1, 768);  hidden_states_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:351, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_90 = self.L__mod___model_decoder_layers_7_final_layer_norm(residual_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:353, code: hidden_states = self.fc1(hidden_states)
    hidden_states_91 = self.L__mod___model_decoder_layers_7_fc1(hidden_states_90);  hidden_states_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:354, code: hidden_states = self.activation_fn(hidden_states)
    hidden_states_92 = self.L__mod___model_decoder_layers_7_activation_fn(hidden_states_91);  hidden_states_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:356, code: hidden_states = self.fc2(hidden_states)
    hidden_states_93 = self.L__mod___model_decoder_layers_7_fc2(hidden_states_92);  hidden_states_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:357, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_94 = torch.nn.functional.dropout(hidden_states_93, p = 0.1, training = False);  hidden_states_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:359, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
    add_27 = residual_15 + hidden_states_94;  residual_15 = hidden_states_94 = None
    residual_16 = add_27.view((1, 2048, 768));  add_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:327, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_97 = self.L__mod___model_decoder_layers_8_self_attn_layer_norm(residual_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:173, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_8_self_attn_q_proj = self.L__mod___model_decoder_layers_8_self_attn_q_proj(hidden_states_97)
    query_states_16 = l__mod___model_decoder_layers_8_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_8_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:191, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_8_self_attn_k_proj = self.L__mod___model_decoder_layers_8_self_attn_k_proj(hidden_states_97)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_82 = l__mod___model_decoder_layers_8_self_attn_k_proj.view(1, -1, 12, 64);  l__mod___model_decoder_layers_8_self_attn_k_proj = None
    transpose_40 = view_82.transpose(1, 2);  view_82 = None
    key_states_16 = transpose_40.contiguous();  transpose_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:192, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_8_self_attn_v_proj = self.L__mod___model_decoder_layers_8_self_attn_v_proj(hidden_states_97);  hidden_states_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_83 = l__mod___model_decoder_layers_8_self_attn_v_proj.view(1, -1, 12, 64);  l__mod___model_decoder_layers_8_self_attn_v_proj = None
    transpose_41 = view_83.transpose(1, 2);  view_83 = None
    value_states_16 = transpose_41.contiguous();  transpose_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_84 = query_states_16.view(1, 2048, 12, 64);  query_states_16 = None
    transpose_42 = view_84.transpose(1, 2);  view_84 = None
    contiguous_26 = transpose_42.contiguous();  transpose_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:205, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_17 = contiguous_26.view(12, -1, 64);  contiguous_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:206, code: key_states = key_states.view(*proj_shape)
    key_states_17 = key_states_16.view(12, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:207, code: value_states = value_states.view(*proj_shape)
    value_states_17 = value_states_16.view(12, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:210, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_43 = key_states_17.transpose(1, 2);  key_states_17 = None
    attn_weights_40 = torch.bmm(query_states_17, transpose_43);  query_states_17 = transpose_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:223, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_88 = attn_weights_40.view(1, 12, 2048, 2048);  attn_weights_40 = None
    attn_weights_41 = view_88 + causal_attention_mask;  view_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:225, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    tensor_8 = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:224, code: attn_weights = torch.max(
    attn_weights_42 = torch.max(attn_weights_41, tensor_8);  attn_weights_41 = tensor_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:227, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_43 = attn_weights_42.view(12, 2048, 2048);  attn_weights_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_44 = torch.nn.functional.softmax(attn_weights_43, dim = -1);  attn_weights_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:254, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_8 = torch.nn.functional.dropout(attn_weights_44, p = 0.0, training = False);  attn_weights_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:256, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_40 = torch.bmm(attn_probs_8, value_states_17);  attn_probs_8 = value_states_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:264, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_41 = attn_output_40.view(1, 12, 2048, 64);  attn_output_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = attn_output.transpose(1, 2)
    attn_output_42 = attn_output_41.transpose(1, 2);  attn_output_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:269, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_43 = attn_output_42.reshape(1, 2048, 768);  attn_output_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:271, code: attn_output = self.out_proj(attn_output)
    hidden_states_98 = self.L__mod___model_decoder_layers_8_self_attn_out_proj(attn_output_43);  attn_output_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:337, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_99 = torch.nn.functional.dropout(hidden_states_98, p = 0.1, training = False);  hidden_states_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:338, code: hidden_states = residual + hidden_states
    hidden_states_100 = residual_16 + hidden_states_99;  residual_16 = hidden_states_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:346, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
    residual_17 = hidden_states_100.reshape(-1, 768);  hidden_states_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:351, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_102 = self.L__mod___model_decoder_layers_8_final_layer_norm(residual_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:353, code: hidden_states = self.fc1(hidden_states)
    hidden_states_103 = self.L__mod___model_decoder_layers_8_fc1(hidden_states_102);  hidden_states_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:354, code: hidden_states = self.activation_fn(hidden_states)
    hidden_states_104 = self.L__mod___model_decoder_layers_8_activation_fn(hidden_states_103);  hidden_states_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:356, code: hidden_states = self.fc2(hidden_states)
    hidden_states_105 = self.L__mod___model_decoder_layers_8_fc2(hidden_states_104);  hidden_states_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:357, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_106 = torch.nn.functional.dropout(hidden_states_105, p = 0.1, training = False);  hidden_states_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:359, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
    add_30 = residual_17 + hidden_states_106;  residual_17 = hidden_states_106 = None
    residual_18 = add_30.view((1, 2048, 768));  add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:327, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_109 = self.L__mod___model_decoder_layers_9_self_attn_layer_norm(residual_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:173, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_9_self_attn_q_proj = self.L__mod___model_decoder_layers_9_self_attn_q_proj(hidden_states_109)
    query_states_18 = l__mod___model_decoder_layers_9_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_9_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:191, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_9_self_attn_k_proj = self.L__mod___model_decoder_layers_9_self_attn_k_proj(hidden_states_109)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_92 = l__mod___model_decoder_layers_9_self_attn_k_proj.view(1, -1, 12, 64);  l__mod___model_decoder_layers_9_self_attn_k_proj = None
    transpose_45 = view_92.transpose(1, 2);  view_92 = None
    key_states_18 = transpose_45.contiguous();  transpose_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:192, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_9_self_attn_v_proj = self.L__mod___model_decoder_layers_9_self_attn_v_proj(hidden_states_109);  hidden_states_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_93 = l__mod___model_decoder_layers_9_self_attn_v_proj.view(1, -1, 12, 64);  l__mod___model_decoder_layers_9_self_attn_v_proj = None
    transpose_46 = view_93.transpose(1, 2);  view_93 = None
    value_states_18 = transpose_46.contiguous();  transpose_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_94 = query_states_18.view(1, 2048, 12, 64);  query_states_18 = None
    transpose_47 = view_94.transpose(1, 2);  view_94 = None
    contiguous_29 = transpose_47.contiguous();  transpose_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:205, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_19 = contiguous_29.view(12, -1, 64);  contiguous_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:206, code: key_states = key_states.view(*proj_shape)
    key_states_19 = key_states_18.view(12, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:207, code: value_states = value_states.view(*proj_shape)
    value_states_19 = value_states_18.view(12, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:210, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_48 = key_states_19.transpose(1, 2);  key_states_19 = None
    attn_weights_45 = torch.bmm(query_states_19, transpose_48);  query_states_19 = transpose_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:223, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_98 = attn_weights_45.view(1, 12, 2048, 2048);  attn_weights_45 = None
    attn_weights_46 = view_98 + causal_attention_mask;  view_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:225, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    tensor_9 = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:224, code: attn_weights = torch.max(
    attn_weights_47 = torch.max(attn_weights_46, tensor_9);  attn_weights_46 = tensor_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:227, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_48 = attn_weights_47.view(12, 2048, 2048);  attn_weights_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_49 = torch.nn.functional.softmax(attn_weights_48, dim = -1);  attn_weights_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:254, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_9 = torch.nn.functional.dropout(attn_weights_49, p = 0.0, training = False);  attn_weights_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:256, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_45 = torch.bmm(attn_probs_9, value_states_19);  attn_probs_9 = value_states_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:264, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_46 = attn_output_45.view(1, 12, 2048, 64);  attn_output_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = attn_output.transpose(1, 2)
    attn_output_47 = attn_output_46.transpose(1, 2);  attn_output_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:269, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_48 = attn_output_47.reshape(1, 2048, 768);  attn_output_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:271, code: attn_output = self.out_proj(attn_output)
    hidden_states_110 = self.L__mod___model_decoder_layers_9_self_attn_out_proj(attn_output_48);  attn_output_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:337, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_111 = torch.nn.functional.dropout(hidden_states_110, p = 0.1, training = False);  hidden_states_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:338, code: hidden_states = residual + hidden_states
    hidden_states_112 = residual_18 + hidden_states_111;  residual_18 = hidden_states_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:346, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
    residual_19 = hidden_states_112.reshape(-1, 768);  hidden_states_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:351, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_114 = self.L__mod___model_decoder_layers_9_final_layer_norm(residual_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:353, code: hidden_states = self.fc1(hidden_states)
    hidden_states_115 = self.L__mod___model_decoder_layers_9_fc1(hidden_states_114);  hidden_states_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:354, code: hidden_states = self.activation_fn(hidden_states)
    hidden_states_116 = self.L__mod___model_decoder_layers_9_activation_fn(hidden_states_115);  hidden_states_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:356, code: hidden_states = self.fc2(hidden_states)
    hidden_states_117 = self.L__mod___model_decoder_layers_9_fc2(hidden_states_116);  hidden_states_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:357, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_118 = torch.nn.functional.dropout(hidden_states_117, p = 0.1, training = False);  hidden_states_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:359, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
    add_33 = residual_19 + hidden_states_118;  residual_19 = hidden_states_118 = None
    residual_20 = add_33.view((1, 2048, 768));  add_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:327, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_121 = self.L__mod___model_decoder_layers_10_self_attn_layer_norm(residual_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:173, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_10_self_attn_q_proj = self.L__mod___model_decoder_layers_10_self_attn_q_proj(hidden_states_121)
    query_states_20 = l__mod___model_decoder_layers_10_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_10_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:191, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_10_self_attn_k_proj = self.L__mod___model_decoder_layers_10_self_attn_k_proj(hidden_states_121)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_102 = l__mod___model_decoder_layers_10_self_attn_k_proj.view(1, -1, 12, 64);  l__mod___model_decoder_layers_10_self_attn_k_proj = None
    transpose_50 = view_102.transpose(1, 2);  view_102 = None
    key_states_20 = transpose_50.contiguous();  transpose_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:192, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_10_self_attn_v_proj = self.L__mod___model_decoder_layers_10_self_attn_v_proj(hidden_states_121);  hidden_states_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_103 = l__mod___model_decoder_layers_10_self_attn_v_proj.view(1, -1, 12, 64);  l__mod___model_decoder_layers_10_self_attn_v_proj = None
    transpose_51 = view_103.transpose(1, 2);  view_103 = None
    value_states_20 = transpose_51.contiguous();  transpose_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_104 = query_states_20.view(1, 2048, 12, 64);  query_states_20 = None
    transpose_52 = view_104.transpose(1, 2);  view_104 = None
    contiguous_32 = transpose_52.contiguous();  transpose_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:205, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_21 = contiguous_32.view(12, -1, 64);  contiguous_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:206, code: key_states = key_states.view(*proj_shape)
    key_states_21 = key_states_20.view(12, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:207, code: value_states = value_states.view(*proj_shape)
    value_states_21 = value_states_20.view(12, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:210, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_53 = key_states_21.transpose(1, 2);  key_states_21 = None
    attn_weights_50 = torch.bmm(query_states_21, transpose_53);  query_states_21 = transpose_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:223, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_108 = attn_weights_50.view(1, 12, 2048, 2048);  attn_weights_50 = None
    attn_weights_51 = view_108 + causal_attention_mask;  view_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:225, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    tensor_10 = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:224, code: attn_weights = torch.max(
    attn_weights_52 = torch.max(attn_weights_51, tensor_10);  attn_weights_51 = tensor_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:227, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_53 = attn_weights_52.view(12, 2048, 2048);  attn_weights_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_54 = torch.nn.functional.softmax(attn_weights_53, dim = -1);  attn_weights_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:254, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_10 = torch.nn.functional.dropout(attn_weights_54, p = 0.0, training = False);  attn_weights_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:256, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_50 = torch.bmm(attn_probs_10, value_states_21);  attn_probs_10 = value_states_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:264, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_51 = attn_output_50.view(1, 12, 2048, 64);  attn_output_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = attn_output.transpose(1, 2)
    attn_output_52 = attn_output_51.transpose(1, 2);  attn_output_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:269, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_53 = attn_output_52.reshape(1, 2048, 768);  attn_output_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:271, code: attn_output = self.out_proj(attn_output)
    hidden_states_122 = self.L__mod___model_decoder_layers_10_self_attn_out_proj(attn_output_53);  attn_output_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:337, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_123 = torch.nn.functional.dropout(hidden_states_122, p = 0.1, training = False);  hidden_states_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:338, code: hidden_states = residual + hidden_states
    hidden_states_124 = residual_20 + hidden_states_123;  residual_20 = hidden_states_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:346, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
    residual_21 = hidden_states_124.reshape(-1, 768);  hidden_states_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:351, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_126 = self.L__mod___model_decoder_layers_10_final_layer_norm(residual_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:353, code: hidden_states = self.fc1(hidden_states)
    hidden_states_127 = self.L__mod___model_decoder_layers_10_fc1(hidden_states_126);  hidden_states_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:354, code: hidden_states = self.activation_fn(hidden_states)
    hidden_states_128 = self.L__mod___model_decoder_layers_10_activation_fn(hidden_states_127);  hidden_states_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:356, code: hidden_states = self.fc2(hidden_states)
    hidden_states_129 = self.L__mod___model_decoder_layers_10_fc2(hidden_states_128);  hidden_states_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:357, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_130 = torch.nn.functional.dropout(hidden_states_129, p = 0.1, training = False);  hidden_states_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:359, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
    add_36 = residual_21 + hidden_states_130;  residual_21 = hidden_states_130 = None
    residual_22 = add_36.view((1, 2048, 768));  add_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:327, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_133 = self.L__mod___model_decoder_layers_11_self_attn_layer_norm(residual_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:173, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_11_self_attn_q_proj = self.L__mod___model_decoder_layers_11_self_attn_q_proj(hidden_states_133)
    query_states_22 = l__mod___model_decoder_layers_11_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_11_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:191, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_11_self_attn_k_proj = self.L__mod___model_decoder_layers_11_self_attn_k_proj(hidden_states_133)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_112 = l__mod___model_decoder_layers_11_self_attn_k_proj.view(1, -1, 12, 64);  l__mod___model_decoder_layers_11_self_attn_k_proj = None
    transpose_55 = view_112.transpose(1, 2);  view_112 = None
    key_states_22 = transpose_55.contiguous();  transpose_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:192, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_11_self_attn_v_proj = self.L__mod___model_decoder_layers_11_self_attn_v_proj(hidden_states_133);  hidden_states_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_113 = l__mod___model_decoder_layers_11_self_attn_v_proj.view(1, -1, 12, 64);  l__mod___model_decoder_layers_11_self_attn_v_proj = None
    transpose_56 = view_113.transpose(1, 2);  view_113 = None
    value_states_22 = transpose_56.contiguous();  transpose_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_114 = query_states_22.view(1, 2048, 12, 64);  query_states_22 = None
    transpose_57 = view_114.transpose(1, 2);  view_114 = None
    contiguous_35 = transpose_57.contiguous();  transpose_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:205, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_23 = contiguous_35.view(12, -1, 64);  contiguous_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:206, code: key_states = key_states.view(*proj_shape)
    key_states_23 = key_states_22.view(12, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:207, code: value_states = value_states.view(*proj_shape)
    value_states_23 = value_states_22.view(12, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:210, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_58 = key_states_23.transpose(1, 2);  key_states_23 = None
    attn_weights_55 = torch.bmm(query_states_23, transpose_58);  query_states_23 = transpose_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:223, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_118 = attn_weights_55.view(1, 12, 2048, 2048);  attn_weights_55 = None
    attn_weights_56 = view_118 + causal_attention_mask;  view_118 = causal_attention_mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:225, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    tensor_11 = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:224, code: attn_weights = torch.max(
    attn_weights_57 = torch.max(attn_weights_56, tensor_11);  attn_weights_56 = tensor_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:227, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_58 = attn_weights_57.view(12, 2048, 2048);  attn_weights_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_59 = torch.nn.functional.softmax(attn_weights_58, dim = -1);  attn_weights_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:254, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_11 = torch.nn.functional.dropout(attn_weights_59, p = 0.0, training = False);  attn_weights_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:256, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_55 = torch.bmm(attn_probs_11, value_states_23);  attn_probs_11 = value_states_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:264, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_56 = attn_output_55.view(1, 12, 2048, 64);  attn_output_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = attn_output.transpose(1, 2)
    attn_output_57 = attn_output_56.transpose(1, 2);  attn_output_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:269, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_58 = attn_output_57.reshape(1, 2048, 768);  attn_output_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:271, code: attn_output = self.out_proj(attn_output)
    hidden_states_134 = self.L__mod___model_decoder_layers_11_self_attn_out_proj(attn_output_58);  attn_output_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:337, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_135 = torch.nn.functional.dropout(hidden_states_134, p = 0.1, training = False);  hidden_states_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:338, code: hidden_states = residual + hidden_states
    hidden_states_136 = residual_22 + hidden_states_135;  residual_22 = hidden_states_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:346, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
    residual_23 = hidden_states_136.reshape(-1, 768);  hidden_states_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:351, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_138 = self.L__mod___model_decoder_layers_11_final_layer_norm(residual_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:353, code: hidden_states = self.fc1(hidden_states)
    hidden_states_139 = self.L__mod___model_decoder_layers_11_fc1(hidden_states_138);  hidden_states_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:354, code: hidden_states = self.activation_fn(hidden_states)
    hidden_states_140 = self.L__mod___model_decoder_layers_11_activation_fn(hidden_states_139);  hidden_states_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:356, code: hidden_states = self.fc2(hidden_states)
    hidden_states_141 = self.L__mod___model_decoder_layers_11_fc2(hidden_states_140);  hidden_states_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:357, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_142 = torch.nn.functional.dropout(hidden_states_141, p = 0.1, training = False);  hidden_states_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:359, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
    add_39 = residual_23 + hidden_states_142;  residual_23 = hidden_states_142 = None
    hidden_states_144 = add_39.view((1, 2048, 768));  add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:728, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_145 = self.L__mod___model_decoder_final_layer_norm(hidden_states_144);  hidden_states_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:956, code: logits = self.lm_head(outputs[0]).contiguous()
    l__mod___lm_head = self.L__mod___lm_head(hidden_states_145);  hidden_states_145 = None
    logits = l__mod___lm_head.contiguous();  l__mod___lm_head = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:961, code: labels = labels.to(logits.device)
    labels = l_inputs_labels_.to(device(type='cuda', index=0));  l_inputs_labels_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:963, code: shift_logits = logits[..., :-1, :].contiguous()
    getitem_3 = logits[(Ellipsis, slice(None, -1, None), slice(None, None, None))]
    shift_logits = getitem_3.contiguous();  getitem_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:964, code: shift_labels = labels[..., 1:].contiguous()
    getitem_4 = labels[(Ellipsis, slice(1, None, None))];  labels = None
    shift_labels = getitem_4.contiguous();  getitem_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:967, code: loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
    view_122 = shift_logits.view(-1, 50272);  shift_logits = None
    view_123 = shift_labels.view(-1);  shift_labels = None
    loss = torch.nn.functional.cross_entropy(view_122, view_123, None, None, -100, None, 'mean', 0.0);  view_122 = view_123 = None
    return (loss, logits, key_states, value_states, key_states_2, value_states_2, key_states_4, value_states_4, key_states_6, value_states_6, key_states_8, value_states_8, key_states_10, value_states_10, key_states_12, value_states_12, key_states_14, value_states_14, key_states_16, value_states_16, key_states_18, value_states_18, key_states_20, value_states_20, key_states_22, value_states_22)
    