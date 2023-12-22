from __future__ import annotations



def forward(self, L_inputs_input_ids_ : torch.Tensor, L_inputs_labels_ : torch.Tensor):
    input_1 = L_inputs_input_ids_
    l_inputs_labels_ = L_inputs_labels_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:635, code: input_ids = input_ids.view(-1, input.shape[-1])
    input_ids = input_1.view(-1, 256);  input_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:646, code: inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
    l__mod___model_decoder_embed_tokens = self.L__mod___model_decoder_embed_tokens(input_ids);  input_ids = None
    inputs_embeds = l__mod___model_decoder_embed_tokens * 1.0;  l__mod___model_decoder_embed_tokens = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:95, code: past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
    l__mod___model_decoder_embed_positions_weight = self.L__mod___model_decoder_embed_positions_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:94, code: positions = torch.arange(
    arange = torch.arange(0, 256, dtype = torch.int64, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:96, code: ).expand(bsz, -1)
    positions = arange.expand(1, -1);  arange = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:98, code: return super().forward(positions + self.offset)
    add = positions + 2;  positions = None
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    embed_pos = torch.nn.functional.embedding(add, l__mod___model_decoder_embed_positions_weight, None, None, 2.0, False, False);  add = l__mod___model_decoder_embed_positions_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:653, code: hidden_states = inputs_embeds + embed_pos
    hidden_states = inputs_embeds + embed_pos;  inputs_embeds = embed_pos = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:656, code: hidden_states = self.layernorm_embedding(hidden_states)
    hidden_states_1 = self.L__mod___model_decoder_layernorm_embedding(hidden_states);  hidden_states = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:658, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    residual = torch.nn.functional.dropout(hidden_states_1, p = 0.1, training = False);  hidden_states_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:53, code: mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask = torch.full((256, 256), -3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:54, code: mask_cond = torch.arange(mask.size(-1), device=device)
    mask_cond = torch.arange(256, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:55, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    add_2 = mask_cond + 1
    view_1 = add_2.view(256, 1);  add_2 = None
    lt = mask_cond < view_1;  mask_cond = view_1 = None
    masked_fill_ = mask.masked_fill_(lt, 0);  lt = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:56, code: mask = mask.to(dtype)
    mask_1 = mask.to(torch.float32);  mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:60, code: return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
    getitem = mask_1[(None, None, slice(None, None, None), slice(None, None, None))];  mask_1 = None
    attention_mask = getitem.expand(1, 1, 256, 256);  getitem = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:219, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_0_self_attn_q_proj = self.L__mod___model_decoder_layers_0_self_attn_q_proj(residual)
    query_states = l__mod___model_decoder_layers_0_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_0_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:237, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_0_self_attn_k_proj = self.L__mod___model_decoder_layers_0_self_attn_k_proj(residual)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:200, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_2 = l__mod___model_decoder_layers_0_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_0_self_attn_k_proj = None
    transpose = view_2.transpose(1, 2);  view_2 = None
    key_states = transpose.contiguous();  transpose = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:238, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_0_self_attn_v_proj = self.L__mod___model_decoder_layers_0_self_attn_v_proj(residual)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:200, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_3 = l__mod___model_decoder_layers_0_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_0_self_attn_v_proj = None
    transpose_1 = view_3.transpose(1, 2);  view_3 = None
    value_states = transpose_1.contiguous();  transpose_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:200, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_4 = query_states.view(1, 256, 16, 64);  query_states = None
    transpose_2 = view_4.transpose(1, 2);  view_4 = None
    contiguous_2 = transpose_2.contiguous();  transpose_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:251, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_1 = contiguous_2.view(16, -1, 64);  contiguous_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:252, code: key_states = key_states.view(*proj_shape)
    key_states_1 = key_states.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:253, code: value_states = value_states.view(*proj_shape)
    value_states_1 = value_states.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:256, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_3 = key_states_1.transpose(1, 2);  key_states_1 = None
    attn_weights = torch.bmm(query_states_1, transpose_3);  query_states_1 = transpose_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:269, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_8 = attn_weights.view(1, 16, 256, 256);  attn_weights = None
    attn_weights_1 = view_8 + attention_mask;  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:270, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_2 = attn_weights_1.view(16, 256, 256);  attn_weights_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:272, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_3 = torch.nn.functional.softmax(attn_weights_2, dim = -1);  attn_weights_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:293, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs = torch.nn.functional.dropout(attn_weights_3, p = 0.0, training = False);  attn_weights_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:295, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output = torch.bmm(attn_probs, value_states_1);  attn_probs = value_states_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:303, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_1 = attn_output.view(1, 16, 256, 64);  attn_output = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:304, code: attn_output = attn_output.transpose(1, 2)
    attn_output_2 = attn_output_1.transpose(1, 2);  attn_output_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:305, code: attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
    attn_output_3 = attn_output_2.reshape(1, 256, 1024);  attn_output_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:307, code: attn_output = self.out_proj(attn_output)
    hidden_states_3 = self.L__mod___model_decoder_layers_0_self_attn_out_proj(attn_output_3);  attn_output_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:391, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_4 = torch.nn.functional.dropout(hidden_states_3, p = 0.1, training = False);  hidden_states_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:392, code: hidden_states = residual + hidden_states
    hidden_states_5 = residual + hidden_states_4;  residual = hidden_states_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:393, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    residual_1 = self.L__mod___model_decoder_layers_0_self_attn_layer_norm(hidden_states_5);  hidden_states_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:422, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_0_fc1 = self.L__mod___model_decoder_layers_0_fc1(residual_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_7 = torch._C._nn.gelu(l__mod___model_decoder_layers_0_fc1);  l__mod___model_decoder_layers_0_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:423, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_8 = torch.nn.functional.dropout(hidden_states_7, p = 0.0, training = False);  hidden_states_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:424, code: hidden_states = self.fc2(hidden_states)
    hidden_states_9 = self.L__mod___model_decoder_layers_0_fc2(hidden_states_8);  hidden_states_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:426, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_10 = torch.nn.functional.dropout(hidden_states_9, p = 0.1, training = False);  hidden_states_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:427, code: hidden_states = residual + hidden_states
    hidden_states_11 = residual_1 + hidden_states_10;  residual_1 = hidden_states_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:428, code: hidden_states = self.final_layer_norm(hidden_states)
    residual_2 = self.L__mod___model_decoder_layers_0_final_layer_norm(hidden_states_11);  hidden_states_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:219, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_1_self_attn_q_proj = self.L__mod___model_decoder_layers_1_self_attn_q_proj(residual_2)
    query_states_2 = l__mod___model_decoder_layers_1_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_1_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:237, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_1_self_attn_k_proj = self.L__mod___model_decoder_layers_1_self_attn_k_proj(residual_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:200, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_11 = l__mod___model_decoder_layers_1_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_1_self_attn_k_proj = None
    transpose_5 = view_11.transpose(1, 2);  view_11 = None
    key_states_2 = transpose_5.contiguous();  transpose_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:238, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_1_self_attn_v_proj = self.L__mod___model_decoder_layers_1_self_attn_v_proj(residual_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:200, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_12 = l__mod___model_decoder_layers_1_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_1_self_attn_v_proj = None
    transpose_6 = view_12.transpose(1, 2);  view_12 = None
    value_states_2 = transpose_6.contiguous();  transpose_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:200, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_13 = query_states_2.view(1, 256, 16, 64);  query_states_2 = None
    transpose_7 = view_13.transpose(1, 2);  view_13 = None
    contiguous_5 = transpose_7.contiguous();  transpose_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:251, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_3 = contiguous_5.view(16, -1, 64);  contiguous_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:252, code: key_states = key_states.view(*proj_shape)
    key_states_3 = key_states_2.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:253, code: value_states = value_states.view(*proj_shape)
    value_states_3 = value_states_2.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:256, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_8 = key_states_3.transpose(1, 2);  key_states_3 = None
    attn_weights_4 = torch.bmm(query_states_3, transpose_8);  query_states_3 = transpose_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:269, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_17 = attn_weights_4.view(1, 16, 256, 256);  attn_weights_4 = None
    attn_weights_5 = view_17 + attention_mask;  view_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:270, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_6 = attn_weights_5.view(16, 256, 256);  attn_weights_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:272, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_7 = torch.nn.functional.softmax(attn_weights_6, dim = -1);  attn_weights_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:293, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_1 = torch.nn.functional.dropout(attn_weights_7, p = 0.0, training = False);  attn_weights_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:295, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_5 = torch.bmm(attn_probs_1, value_states_3);  attn_probs_1 = value_states_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:303, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_6 = attn_output_5.view(1, 16, 256, 64);  attn_output_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:304, code: attn_output = attn_output.transpose(1, 2)
    attn_output_7 = attn_output_6.transpose(1, 2);  attn_output_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:305, code: attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
    attn_output_8 = attn_output_7.reshape(1, 256, 1024);  attn_output_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:307, code: attn_output = self.out_proj(attn_output)
    hidden_states_14 = self.L__mod___model_decoder_layers_1_self_attn_out_proj(attn_output_8);  attn_output_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:391, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_15 = torch.nn.functional.dropout(hidden_states_14, p = 0.1, training = False);  hidden_states_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:392, code: hidden_states = residual + hidden_states
    hidden_states_16 = residual_2 + hidden_states_15;  residual_2 = hidden_states_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:393, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    residual_3 = self.L__mod___model_decoder_layers_1_self_attn_layer_norm(hidden_states_16);  hidden_states_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:422, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_1_fc1 = self.L__mod___model_decoder_layers_1_fc1(residual_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_18 = torch._C._nn.gelu(l__mod___model_decoder_layers_1_fc1);  l__mod___model_decoder_layers_1_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:423, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_19 = torch.nn.functional.dropout(hidden_states_18, p = 0.0, training = False);  hidden_states_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:424, code: hidden_states = self.fc2(hidden_states)
    hidden_states_20 = self.L__mod___model_decoder_layers_1_fc2(hidden_states_19);  hidden_states_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:426, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_21 = torch.nn.functional.dropout(hidden_states_20, p = 0.1, training = False);  hidden_states_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:427, code: hidden_states = residual + hidden_states
    hidden_states_22 = residual_3 + hidden_states_21;  residual_3 = hidden_states_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:428, code: hidden_states = self.final_layer_norm(hidden_states)
    residual_4 = self.L__mod___model_decoder_layers_1_final_layer_norm(hidden_states_22);  hidden_states_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:219, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_2_self_attn_q_proj = self.L__mod___model_decoder_layers_2_self_attn_q_proj(residual_4)
    query_states_4 = l__mod___model_decoder_layers_2_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_2_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:237, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_2_self_attn_k_proj = self.L__mod___model_decoder_layers_2_self_attn_k_proj(residual_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:200, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_20 = l__mod___model_decoder_layers_2_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_2_self_attn_k_proj = None
    transpose_10 = view_20.transpose(1, 2);  view_20 = None
    key_states_4 = transpose_10.contiguous();  transpose_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:238, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_2_self_attn_v_proj = self.L__mod___model_decoder_layers_2_self_attn_v_proj(residual_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:200, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_21 = l__mod___model_decoder_layers_2_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_2_self_attn_v_proj = None
    transpose_11 = view_21.transpose(1, 2);  view_21 = None
    value_states_4 = transpose_11.contiguous();  transpose_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:200, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_22 = query_states_4.view(1, 256, 16, 64);  query_states_4 = None
    transpose_12 = view_22.transpose(1, 2);  view_22 = None
    contiguous_8 = transpose_12.contiguous();  transpose_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:251, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_5 = contiguous_8.view(16, -1, 64);  contiguous_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:252, code: key_states = key_states.view(*proj_shape)
    key_states_5 = key_states_4.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:253, code: value_states = value_states.view(*proj_shape)
    value_states_5 = value_states_4.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:256, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_13 = key_states_5.transpose(1, 2);  key_states_5 = None
    attn_weights_8 = torch.bmm(query_states_5, transpose_13);  query_states_5 = transpose_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:269, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_26 = attn_weights_8.view(1, 16, 256, 256);  attn_weights_8 = None
    attn_weights_9 = view_26 + attention_mask;  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:270, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_10 = attn_weights_9.view(16, 256, 256);  attn_weights_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:272, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_11 = torch.nn.functional.softmax(attn_weights_10, dim = -1);  attn_weights_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:293, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_2 = torch.nn.functional.dropout(attn_weights_11, p = 0.0, training = False);  attn_weights_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:295, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_10 = torch.bmm(attn_probs_2, value_states_5);  attn_probs_2 = value_states_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:303, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_11 = attn_output_10.view(1, 16, 256, 64);  attn_output_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:304, code: attn_output = attn_output.transpose(1, 2)
    attn_output_12 = attn_output_11.transpose(1, 2);  attn_output_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:305, code: attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
    attn_output_13 = attn_output_12.reshape(1, 256, 1024);  attn_output_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:307, code: attn_output = self.out_proj(attn_output)
    hidden_states_25 = self.L__mod___model_decoder_layers_2_self_attn_out_proj(attn_output_13);  attn_output_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:391, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_26 = torch.nn.functional.dropout(hidden_states_25, p = 0.1, training = False);  hidden_states_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:392, code: hidden_states = residual + hidden_states
    hidden_states_27 = residual_4 + hidden_states_26;  residual_4 = hidden_states_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:393, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    residual_5 = self.L__mod___model_decoder_layers_2_self_attn_layer_norm(hidden_states_27);  hidden_states_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:422, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_2_fc1 = self.L__mod___model_decoder_layers_2_fc1(residual_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_29 = torch._C._nn.gelu(l__mod___model_decoder_layers_2_fc1);  l__mod___model_decoder_layers_2_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:423, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_30 = torch.nn.functional.dropout(hidden_states_29, p = 0.0, training = False);  hidden_states_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:424, code: hidden_states = self.fc2(hidden_states)
    hidden_states_31 = self.L__mod___model_decoder_layers_2_fc2(hidden_states_30);  hidden_states_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:426, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_32 = torch.nn.functional.dropout(hidden_states_31, p = 0.1, training = False);  hidden_states_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:427, code: hidden_states = residual + hidden_states
    hidden_states_33 = residual_5 + hidden_states_32;  residual_5 = hidden_states_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:428, code: hidden_states = self.final_layer_norm(hidden_states)
    residual_6 = self.L__mod___model_decoder_layers_2_final_layer_norm(hidden_states_33);  hidden_states_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:219, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_3_self_attn_q_proj = self.L__mod___model_decoder_layers_3_self_attn_q_proj(residual_6)
    query_states_6 = l__mod___model_decoder_layers_3_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_3_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:237, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_3_self_attn_k_proj = self.L__mod___model_decoder_layers_3_self_attn_k_proj(residual_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:200, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_29 = l__mod___model_decoder_layers_3_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_3_self_attn_k_proj = None
    transpose_15 = view_29.transpose(1, 2);  view_29 = None
    key_states_6 = transpose_15.contiguous();  transpose_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:238, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_3_self_attn_v_proj = self.L__mod___model_decoder_layers_3_self_attn_v_proj(residual_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:200, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_30 = l__mod___model_decoder_layers_3_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_3_self_attn_v_proj = None
    transpose_16 = view_30.transpose(1, 2);  view_30 = None
    value_states_6 = transpose_16.contiguous();  transpose_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:200, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_31 = query_states_6.view(1, 256, 16, 64);  query_states_6 = None
    transpose_17 = view_31.transpose(1, 2);  view_31 = None
    contiguous_11 = transpose_17.contiguous();  transpose_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:251, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_7 = contiguous_11.view(16, -1, 64);  contiguous_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:252, code: key_states = key_states.view(*proj_shape)
    key_states_7 = key_states_6.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:253, code: value_states = value_states.view(*proj_shape)
    value_states_7 = value_states_6.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:256, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_18 = key_states_7.transpose(1, 2);  key_states_7 = None
    attn_weights_12 = torch.bmm(query_states_7, transpose_18);  query_states_7 = transpose_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:269, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_35 = attn_weights_12.view(1, 16, 256, 256);  attn_weights_12 = None
    attn_weights_13 = view_35 + attention_mask;  view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:270, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_14 = attn_weights_13.view(16, 256, 256);  attn_weights_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:272, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_15 = torch.nn.functional.softmax(attn_weights_14, dim = -1);  attn_weights_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:293, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_3 = torch.nn.functional.dropout(attn_weights_15, p = 0.0, training = False);  attn_weights_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:295, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_15 = torch.bmm(attn_probs_3, value_states_7);  attn_probs_3 = value_states_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:303, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_16 = attn_output_15.view(1, 16, 256, 64);  attn_output_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:304, code: attn_output = attn_output.transpose(1, 2)
    attn_output_17 = attn_output_16.transpose(1, 2);  attn_output_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:305, code: attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
    attn_output_18 = attn_output_17.reshape(1, 256, 1024);  attn_output_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:307, code: attn_output = self.out_proj(attn_output)
    hidden_states_36 = self.L__mod___model_decoder_layers_3_self_attn_out_proj(attn_output_18);  attn_output_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:391, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_37 = torch.nn.functional.dropout(hidden_states_36, p = 0.1, training = False);  hidden_states_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:392, code: hidden_states = residual + hidden_states
    hidden_states_38 = residual_6 + hidden_states_37;  residual_6 = hidden_states_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:393, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    residual_7 = self.L__mod___model_decoder_layers_3_self_attn_layer_norm(hidden_states_38);  hidden_states_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:422, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_3_fc1 = self.L__mod___model_decoder_layers_3_fc1(residual_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_40 = torch._C._nn.gelu(l__mod___model_decoder_layers_3_fc1);  l__mod___model_decoder_layers_3_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:423, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_41 = torch.nn.functional.dropout(hidden_states_40, p = 0.0, training = False);  hidden_states_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:424, code: hidden_states = self.fc2(hidden_states)
    hidden_states_42 = self.L__mod___model_decoder_layers_3_fc2(hidden_states_41);  hidden_states_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:426, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_43 = torch.nn.functional.dropout(hidden_states_42, p = 0.1, training = False);  hidden_states_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:427, code: hidden_states = residual + hidden_states
    hidden_states_44 = residual_7 + hidden_states_43;  residual_7 = hidden_states_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:428, code: hidden_states = self.final_layer_norm(hidden_states)
    residual_8 = self.L__mod___model_decoder_layers_3_final_layer_norm(hidden_states_44);  hidden_states_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:219, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_4_self_attn_q_proj = self.L__mod___model_decoder_layers_4_self_attn_q_proj(residual_8)
    query_states_8 = l__mod___model_decoder_layers_4_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_4_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:237, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_4_self_attn_k_proj = self.L__mod___model_decoder_layers_4_self_attn_k_proj(residual_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:200, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_38 = l__mod___model_decoder_layers_4_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_4_self_attn_k_proj = None
    transpose_20 = view_38.transpose(1, 2);  view_38 = None
    key_states_8 = transpose_20.contiguous();  transpose_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:238, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_4_self_attn_v_proj = self.L__mod___model_decoder_layers_4_self_attn_v_proj(residual_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:200, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_39 = l__mod___model_decoder_layers_4_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_4_self_attn_v_proj = None
    transpose_21 = view_39.transpose(1, 2);  view_39 = None
    value_states_8 = transpose_21.contiguous();  transpose_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:200, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_40 = query_states_8.view(1, 256, 16, 64);  query_states_8 = None
    transpose_22 = view_40.transpose(1, 2);  view_40 = None
    contiguous_14 = transpose_22.contiguous();  transpose_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:251, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_9 = contiguous_14.view(16, -1, 64);  contiguous_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:252, code: key_states = key_states.view(*proj_shape)
    key_states_9 = key_states_8.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:253, code: value_states = value_states.view(*proj_shape)
    value_states_9 = value_states_8.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:256, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_23 = key_states_9.transpose(1, 2);  key_states_9 = None
    attn_weights_16 = torch.bmm(query_states_9, transpose_23);  query_states_9 = transpose_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:269, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_44 = attn_weights_16.view(1, 16, 256, 256);  attn_weights_16 = None
    attn_weights_17 = view_44 + attention_mask;  view_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:270, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_18 = attn_weights_17.view(16, 256, 256);  attn_weights_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:272, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_19 = torch.nn.functional.softmax(attn_weights_18, dim = -1);  attn_weights_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:293, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_4 = torch.nn.functional.dropout(attn_weights_19, p = 0.0, training = False);  attn_weights_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:295, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_20 = torch.bmm(attn_probs_4, value_states_9);  attn_probs_4 = value_states_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:303, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_21 = attn_output_20.view(1, 16, 256, 64);  attn_output_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:304, code: attn_output = attn_output.transpose(1, 2)
    attn_output_22 = attn_output_21.transpose(1, 2);  attn_output_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:305, code: attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
    attn_output_23 = attn_output_22.reshape(1, 256, 1024);  attn_output_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:307, code: attn_output = self.out_proj(attn_output)
    hidden_states_47 = self.L__mod___model_decoder_layers_4_self_attn_out_proj(attn_output_23);  attn_output_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:391, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_48 = torch.nn.functional.dropout(hidden_states_47, p = 0.1, training = False);  hidden_states_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:392, code: hidden_states = residual + hidden_states
    hidden_states_49 = residual_8 + hidden_states_48;  residual_8 = hidden_states_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:393, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    residual_9 = self.L__mod___model_decoder_layers_4_self_attn_layer_norm(hidden_states_49);  hidden_states_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:422, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_4_fc1 = self.L__mod___model_decoder_layers_4_fc1(residual_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_51 = torch._C._nn.gelu(l__mod___model_decoder_layers_4_fc1);  l__mod___model_decoder_layers_4_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:423, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_52 = torch.nn.functional.dropout(hidden_states_51, p = 0.0, training = False);  hidden_states_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:424, code: hidden_states = self.fc2(hidden_states)
    hidden_states_53 = self.L__mod___model_decoder_layers_4_fc2(hidden_states_52);  hidden_states_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:426, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_54 = torch.nn.functional.dropout(hidden_states_53, p = 0.1, training = False);  hidden_states_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:427, code: hidden_states = residual + hidden_states
    hidden_states_55 = residual_9 + hidden_states_54;  residual_9 = hidden_states_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:428, code: hidden_states = self.final_layer_norm(hidden_states)
    residual_10 = self.L__mod___model_decoder_layers_4_final_layer_norm(hidden_states_55);  hidden_states_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:219, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_5_self_attn_q_proj = self.L__mod___model_decoder_layers_5_self_attn_q_proj(residual_10)
    query_states_10 = l__mod___model_decoder_layers_5_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_5_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:237, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_5_self_attn_k_proj = self.L__mod___model_decoder_layers_5_self_attn_k_proj(residual_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:200, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_47 = l__mod___model_decoder_layers_5_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_5_self_attn_k_proj = None
    transpose_25 = view_47.transpose(1, 2);  view_47 = None
    key_states_10 = transpose_25.contiguous();  transpose_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:238, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_5_self_attn_v_proj = self.L__mod___model_decoder_layers_5_self_attn_v_proj(residual_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:200, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_48 = l__mod___model_decoder_layers_5_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_5_self_attn_v_proj = None
    transpose_26 = view_48.transpose(1, 2);  view_48 = None
    value_states_10 = transpose_26.contiguous();  transpose_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:200, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_49 = query_states_10.view(1, 256, 16, 64);  query_states_10 = None
    transpose_27 = view_49.transpose(1, 2);  view_49 = None
    contiguous_17 = transpose_27.contiguous();  transpose_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:251, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_11 = contiguous_17.view(16, -1, 64);  contiguous_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:252, code: key_states = key_states.view(*proj_shape)
    key_states_11 = key_states_10.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:253, code: value_states = value_states.view(*proj_shape)
    value_states_11 = value_states_10.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:256, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_28 = key_states_11.transpose(1, 2);  key_states_11 = None
    attn_weights_20 = torch.bmm(query_states_11, transpose_28);  query_states_11 = transpose_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:269, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_53 = attn_weights_20.view(1, 16, 256, 256);  attn_weights_20 = None
    attn_weights_21 = view_53 + attention_mask;  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:270, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_22 = attn_weights_21.view(16, 256, 256);  attn_weights_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:272, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_23 = torch.nn.functional.softmax(attn_weights_22, dim = -1);  attn_weights_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:293, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_5 = torch.nn.functional.dropout(attn_weights_23, p = 0.0, training = False);  attn_weights_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:295, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_25 = torch.bmm(attn_probs_5, value_states_11);  attn_probs_5 = value_states_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:303, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_26 = attn_output_25.view(1, 16, 256, 64);  attn_output_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:304, code: attn_output = attn_output.transpose(1, 2)
    attn_output_27 = attn_output_26.transpose(1, 2);  attn_output_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:305, code: attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
    attn_output_28 = attn_output_27.reshape(1, 256, 1024);  attn_output_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:307, code: attn_output = self.out_proj(attn_output)
    hidden_states_58 = self.L__mod___model_decoder_layers_5_self_attn_out_proj(attn_output_28);  attn_output_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:391, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_59 = torch.nn.functional.dropout(hidden_states_58, p = 0.1, training = False);  hidden_states_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:392, code: hidden_states = residual + hidden_states
    hidden_states_60 = residual_10 + hidden_states_59;  residual_10 = hidden_states_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:393, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    residual_11 = self.L__mod___model_decoder_layers_5_self_attn_layer_norm(hidden_states_60);  hidden_states_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:422, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_5_fc1 = self.L__mod___model_decoder_layers_5_fc1(residual_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_62 = torch._C._nn.gelu(l__mod___model_decoder_layers_5_fc1);  l__mod___model_decoder_layers_5_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:423, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_63 = torch.nn.functional.dropout(hidden_states_62, p = 0.0, training = False);  hidden_states_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:424, code: hidden_states = self.fc2(hidden_states)
    hidden_states_64 = self.L__mod___model_decoder_layers_5_fc2(hidden_states_63);  hidden_states_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:426, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_65 = torch.nn.functional.dropout(hidden_states_64, p = 0.1, training = False);  hidden_states_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:427, code: hidden_states = residual + hidden_states
    hidden_states_66 = residual_11 + hidden_states_65;  residual_11 = hidden_states_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:428, code: hidden_states = self.final_layer_norm(hidden_states)
    residual_12 = self.L__mod___model_decoder_layers_5_final_layer_norm(hidden_states_66);  hidden_states_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:219, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_6_self_attn_q_proj = self.L__mod___model_decoder_layers_6_self_attn_q_proj(residual_12)
    query_states_12 = l__mod___model_decoder_layers_6_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_6_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:237, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_6_self_attn_k_proj = self.L__mod___model_decoder_layers_6_self_attn_k_proj(residual_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:200, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_56 = l__mod___model_decoder_layers_6_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_6_self_attn_k_proj = None
    transpose_30 = view_56.transpose(1, 2);  view_56 = None
    key_states_12 = transpose_30.contiguous();  transpose_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:238, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_6_self_attn_v_proj = self.L__mod___model_decoder_layers_6_self_attn_v_proj(residual_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:200, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_57 = l__mod___model_decoder_layers_6_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_6_self_attn_v_proj = None
    transpose_31 = view_57.transpose(1, 2);  view_57 = None
    value_states_12 = transpose_31.contiguous();  transpose_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:200, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_58 = query_states_12.view(1, 256, 16, 64);  query_states_12 = None
    transpose_32 = view_58.transpose(1, 2);  view_58 = None
    contiguous_20 = transpose_32.contiguous();  transpose_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:251, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_13 = contiguous_20.view(16, -1, 64);  contiguous_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:252, code: key_states = key_states.view(*proj_shape)
    key_states_13 = key_states_12.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:253, code: value_states = value_states.view(*proj_shape)
    value_states_13 = value_states_12.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:256, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_33 = key_states_13.transpose(1, 2);  key_states_13 = None
    attn_weights_24 = torch.bmm(query_states_13, transpose_33);  query_states_13 = transpose_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:269, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_62 = attn_weights_24.view(1, 16, 256, 256);  attn_weights_24 = None
    attn_weights_25 = view_62 + attention_mask;  view_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:270, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_26 = attn_weights_25.view(16, 256, 256);  attn_weights_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:272, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_27 = torch.nn.functional.softmax(attn_weights_26, dim = -1);  attn_weights_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:293, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_6 = torch.nn.functional.dropout(attn_weights_27, p = 0.0, training = False);  attn_weights_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:295, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_30 = torch.bmm(attn_probs_6, value_states_13);  attn_probs_6 = value_states_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:303, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_31 = attn_output_30.view(1, 16, 256, 64);  attn_output_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:304, code: attn_output = attn_output.transpose(1, 2)
    attn_output_32 = attn_output_31.transpose(1, 2);  attn_output_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:305, code: attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
    attn_output_33 = attn_output_32.reshape(1, 256, 1024);  attn_output_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:307, code: attn_output = self.out_proj(attn_output)
    hidden_states_69 = self.L__mod___model_decoder_layers_6_self_attn_out_proj(attn_output_33);  attn_output_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:391, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_70 = torch.nn.functional.dropout(hidden_states_69, p = 0.1, training = False);  hidden_states_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:392, code: hidden_states = residual + hidden_states
    hidden_states_71 = residual_12 + hidden_states_70;  residual_12 = hidden_states_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:393, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    residual_13 = self.L__mod___model_decoder_layers_6_self_attn_layer_norm(hidden_states_71);  hidden_states_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:422, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_6_fc1 = self.L__mod___model_decoder_layers_6_fc1(residual_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_73 = torch._C._nn.gelu(l__mod___model_decoder_layers_6_fc1);  l__mod___model_decoder_layers_6_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:423, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_74 = torch.nn.functional.dropout(hidden_states_73, p = 0.0, training = False);  hidden_states_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:424, code: hidden_states = self.fc2(hidden_states)
    hidden_states_75 = self.L__mod___model_decoder_layers_6_fc2(hidden_states_74);  hidden_states_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:426, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_76 = torch.nn.functional.dropout(hidden_states_75, p = 0.1, training = False);  hidden_states_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:427, code: hidden_states = residual + hidden_states
    hidden_states_77 = residual_13 + hidden_states_76;  residual_13 = hidden_states_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:428, code: hidden_states = self.final_layer_norm(hidden_states)
    residual_14 = self.L__mod___model_decoder_layers_6_final_layer_norm(hidden_states_77);  hidden_states_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:219, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_7_self_attn_q_proj = self.L__mod___model_decoder_layers_7_self_attn_q_proj(residual_14)
    query_states_14 = l__mod___model_decoder_layers_7_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_7_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:237, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_7_self_attn_k_proj = self.L__mod___model_decoder_layers_7_self_attn_k_proj(residual_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:200, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_65 = l__mod___model_decoder_layers_7_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_7_self_attn_k_proj = None
    transpose_35 = view_65.transpose(1, 2);  view_65 = None
    key_states_14 = transpose_35.contiguous();  transpose_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:238, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_7_self_attn_v_proj = self.L__mod___model_decoder_layers_7_self_attn_v_proj(residual_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:200, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_66 = l__mod___model_decoder_layers_7_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_7_self_attn_v_proj = None
    transpose_36 = view_66.transpose(1, 2);  view_66 = None
    value_states_14 = transpose_36.contiguous();  transpose_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:200, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_67 = query_states_14.view(1, 256, 16, 64);  query_states_14 = None
    transpose_37 = view_67.transpose(1, 2);  view_67 = None
    contiguous_23 = transpose_37.contiguous();  transpose_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:251, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_15 = contiguous_23.view(16, -1, 64);  contiguous_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:252, code: key_states = key_states.view(*proj_shape)
    key_states_15 = key_states_14.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:253, code: value_states = value_states.view(*proj_shape)
    value_states_15 = value_states_14.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:256, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_38 = key_states_15.transpose(1, 2);  key_states_15 = None
    attn_weights_28 = torch.bmm(query_states_15, transpose_38);  query_states_15 = transpose_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:269, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_71 = attn_weights_28.view(1, 16, 256, 256);  attn_weights_28 = None
    attn_weights_29 = view_71 + attention_mask;  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:270, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_30 = attn_weights_29.view(16, 256, 256);  attn_weights_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:272, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_31 = torch.nn.functional.softmax(attn_weights_30, dim = -1);  attn_weights_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:293, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_7 = torch.nn.functional.dropout(attn_weights_31, p = 0.0, training = False);  attn_weights_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:295, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_35 = torch.bmm(attn_probs_7, value_states_15);  attn_probs_7 = value_states_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:303, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_36 = attn_output_35.view(1, 16, 256, 64);  attn_output_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:304, code: attn_output = attn_output.transpose(1, 2)
    attn_output_37 = attn_output_36.transpose(1, 2);  attn_output_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:305, code: attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
    attn_output_38 = attn_output_37.reshape(1, 256, 1024);  attn_output_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:307, code: attn_output = self.out_proj(attn_output)
    hidden_states_80 = self.L__mod___model_decoder_layers_7_self_attn_out_proj(attn_output_38);  attn_output_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:391, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_81 = torch.nn.functional.dropout(hidden_states_80, p = 0.1, training = False);  hidden_states_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:392, code: hidden_states = residual + hidden_states
    hidden_states_82 = residual_14 + hidden_states_81;  residual_14 = hidden_states_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:393, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    residual_15 = self.L__mod___model_decoder_layers_7_self_attn_layer_norm(hidden_states_82);  hidden_states_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:422, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_7_fc1 = self.L__mod___model_decoder_layers_7_fc1(residual_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_84 = torch._C._nn.gelu(l__mod___model_decoder_layers_7_fc1);  l__mod___model_decoder_layers_7_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:423, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_85 = torch.nn.functional.dropout(hidden_states_84, p = 0.0, training = False);  hidden_states_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:424, code: hidden_states = self.fc2(hidden_states)
    hidden_states_86 = self.L__mod___model_decoder_layers_7_fc2(hidden_states_85);  hidden_states_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:426, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_87 = torch.nn.functional.dropout(hidden_states_86, p = 0.1, training = False);  hidden_states_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:427, code: hidden_states = residual + hidden_states
    hidden_states_88 = residual_15 + hidden_states_87;  residual_15 = hidden_states_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:428, code: hidden_states = self.final_layer_norm(hidden_states)
    residual_16 = self.L__mod___model_decoder_layers_7_final_layer_norm(hidden_states_88);  hidden_states_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:219, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_8_self_attn_q_proj = self.L__mod___model_decoder_layers_8_self_attn_q_proj(residual_16)
    query_states_16 = l__mod___model_decoder_layers_8_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_8_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:237, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_8_self_attn_k_proj = self.L__mod___model_decoder_layers_8_self_attn_k_proj(residual_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:200, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_74 = l__mod___model_decoder_layers_8_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_8_self_attn_k_proj = None
    transpose_40 = view_74.transpose(1, 2);  view_74 = None
    key_states_16 = transpose_40.contiguous();  transpose_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:238, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_8_self_attn_v_proj = self.L__mod___model_decoder_layers_8_self_attn_v_proj(residual_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:200, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_75 = l__mod___model_decoder_layers_8_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_8_self_attn_v_proj = None
    transpose_41 = view_75.transpose(1, 2);  view_75 = None
    value_states_16 = transpose_41.contiguous();  transpose_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:200, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_76 = query_states_16.view(1, 256, 16, 64);  query_states_16 = None
    transpose_42 = view_76.transpose(1, 2);  view_76 = None
    contiguous_26 = transpose_42.contiguous();  transpose_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:251, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_17 = contiguous_26.view(16, -1, 64);  contiguous_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:252, code: key_states = key_states.view(*proj_shape)
    key_states_17 = key_states_16.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:253, code: value_states = value_states.view(*proj_shape)
    value_states_17 = value_states_16.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:256, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_43 = key_states_17.transpose(1, 2);  key_states_17 = None
    attn_weights_32 = torch.bmm(query_states_17, transpose_43);  query_states_17 = transpose_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:269, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_80 = attn_weights_32.view(1, 16, 256, 256);  attn_weights_32 = None
    attn_weights_33 = view_80 + attention_mask;  view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:270, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_34 = attn_weights_33.view(16, 256, 256);  attn_weights_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:272, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_35 = torch.nn.functional.softmax(attn_weights_34, dim = -1);  attn_weights_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:293, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_8 = torch.nn.functional.dropout(attn_weights_35, p = 0.0, training = False);  attn_weights_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:295, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_40 = torch.bmm(attn_probs_8, value_states_17);  attn_probs_8 = value_states_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:303, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_41 = attn_output_40.view(1, 16, 256, 64);  attn_output_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:304, code: attn_output = attn_output.transpose(1, 2)
    attn_output_42 = attn_output_41.transpose(1, 2);  attn_output_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:305, code: attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
    attn_output_43 = attn_output_42.reshape(1, 256, 1024);  attn_output_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:307, code: attn_output = self.out_proj(attn_output)
    hidden_states_91 = self.L__mod___model_decoder_layers_8_self_attn_out_proj(attn_output_43);  attn_output_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:391, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_92 = torch.nn.functional.dropout(hidden_states_91, p = 0.1, training = False);  hidden_states_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:392, code: hidden_states = residual + hidden_states
    hidden_states_93 = residual_16 + hidden_states_92;  residual_16 = hidden_states_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:393, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    residual_17 = self.L__mod___model_decoder_layers_8_self_attn_layer_norm(hidden_states_93);  hidden_states_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:422, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_8_fc1 = self.L__mod___model_decoder_layers_8_fc1(residual_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_95 = torch._C._nn.gelu(l__mod___model_decoder_layers_8_fc1);  l__mod___model_decoder_layers_8_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:423, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_96 = torch.nn.functional.dropout(hidden_states_95, p = 0.0, training = False);  hidden_states_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:424, code: hidden_states = self.fc2(hidden_states)
    hidden_states_97 = self.L__mod___model_decoder_layers_8_fc2(hidden_states_96);  hidden_states_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:426, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_98 = torch.nn.functional.dropout(hidden_states_97, p = 0.1, training = False);  hidden_states_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:427, code: hidden_states = residual + hidden_states
    hidden_states_99 = residual_17 + hidden_states_98;  residual_17 = hidden_states_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:428, code: hidden_states = self.final_layer_norm(hidden_states)
    residual_18 = self.L__mod___model_decoder_layers_8_final_layer_norm(hidden_states_99);  hidden_states_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:219, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_9_self_attn_q_proj = self.L__mod___model_decoder_layers_9_self_attn_q_proj(residual_18)
    query_states_18 = l__mod___model_decoder_layers_9_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_9_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:237, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_9_self_attn_k_proj = self.L__mod___model_decoder_layers_9_self_attn_k_proj(residual_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:200, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_83 = l__mod___model_decoder_layers_9_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_9_self_attn_k_proj = None
    transpose_45 = view_83.transpose(1, 2);  view_83 = None
    key_states_18 = transpose_45.contiguous();  transpose_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:238, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_9_self_attn_v_proj = self.L__mod___model_decoder_layers_9_self_attn_v_proj(residual_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:200, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_84 = l__mod___model_decoder_layers_9_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_9_self_attn_v_proj = None
    transpose_46 = view_84.transpose(1, 2);  view_84 = None
    value_states_18 = transpose_46.contiguous();  transpose_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:200, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_85 = query_states_18.view(1, 256, 16, 64);  query_states_18 = None
    transpose_47 = view_85.transpose(1, 2);  view_85 = None
    contiguous_29 = transpose_47.contiguous();  transpose_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:251, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_19 = contiguous_29.view(16, -1, 64);  contiguous_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:252, code: key_states = key_states.view(*proj_shape)
    key_states_19 = key_states_18.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:253, code: value_states = value_states.view(*proj_shape)
    value_states_19 = value_states_18.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:256, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_48 = key_states_19.transpose(1, 2);  key_states_19 = None
    attn_weights_36 = torch.bmm(query_states_19, transpose_48);  query_states_19 = transpose_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:269, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_89 = attn_weights_36.view(1, 16, 256, 256);  attn_weights_36 = None
    attn_weights_37 = view_89 + attention_mask;  view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:270, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_38 = attn_weights_37.view(16, 256, 256);  attn_weights_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:272, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_39 = torch.nn.functional.softmax(attn_weights_38, dim = -1);  attn_weights_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:293, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_9 = torch.nn.functional.dropout(attn_weights_39, p = 0.0, training = False);  attn_weights_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:295, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_45 = torch.bmm(attn_probs_9, value_states_19);  attn_probs_9 = value_states_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:303, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_46 = attn_output_45.view(1, 16, 256, 64);  attn_output_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:304, code: attn_output = attn_output.transpose(1, 2)
    attn_output_47 = attn_output_46.transpose(1, 2);  attn_output_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:305, code: attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
    attn_output_48 = attn_output_47.reshape(1, 256, 1024);  attn_output_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:307, code: attn_output = self.out_proj(attn_output)
    hidden_states_102 = self.L__mod___model_decoder_layers_9_self_attn_out_proj(attn_output_48);  attn_output_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:391, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_103 = torch.nn.functional.dropout(hidden_states_102, p = 0.1, training = False);  hidden_states_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:392, code: hidden_states = residual + hidden_states
    hidden_states_104 = residual_18 + hidden_states_103;  residual_18 = hidden_states_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:393, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    residual_19 = self.L__mod___model_decoder_layers_9_self_attn_layer_norm(hidden_states_104);  hidden_states_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:422, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_9_fc1 = self.L__mod___model_decoder_layers_9_fc1(residual_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_106 = torch._C._nn.gelu(l__mod___model_decoder_layers_9_fc1);  l__mod___model_decoder_layers_9_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:423, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_107 = torch.nn.functional.dropout(hidden_states_106, p = 0.0, training = False);  hidden_states_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:424, code: hidden_states = self.fc2(hidden_states)
    hidden_states_108 = self.L__mod___model_decoder_layers_9_fc2(hidden_states_107);  hidden_states_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:426, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_109 = torch.nn.functional.dropout(hidden_states_108, p = 0.1, training = False);  hidden_states_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:427, code: hidden_states = residual + hidden_states
    hidden_states_110 = residual_19 + hidden_states_109;  residual_19 = hidden_states_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:428, code: hidden_states = self.final_layer_norm(hidden_states)
    residual_20 = self.L__mod___model_decoder_layers_9_final_layer_norm(hidden_states_110);  hidden_states_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:219, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_10_self_attn_q_proj = self.L__mod___model_decoder_layers_10_self_attn_q_proj(residual_20)
    query_states_20 = l__mod___model_decoder_layers_10_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_10_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:237, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_10_self_attn_k_proj = self.L__mod___model_decoder_layers_10_self_attn_k_proj(residual_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:200, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_92 = l__mod___model_decoder_layers_10_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_10_self_attn_k_proj = None
    transpose_50 = view_92.transpose(1, 2);  view_92 = None
    key_states_20 = transpose_50.contiguous();  transpose_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:238, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_10_self_attn_v_proj = self.L__mod___model_decoder_layers_10_self_attn_v_proj(residual_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:200, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_93 = l__mod___model_decoder_layers_10_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_10_self_attn_v_proj = None
    transpose_51 = view_93.transpose(1, 2);  view_93 = None
    value_states_20 = transpose_51.contiguous();  transpose_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:200, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_94 = query_states_20.view(1, 256, 16, 64);  query_states_20 = None
    transpose_52 = view_94.transpose(1, 2);  view_94 = None
    contiguous_32 = transpose_52.contiguous();  transpose_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:251, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_21 = contiguous_32.view(16, -1, 64);  contiguous_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:252, code: key_states = key_states.view(*proj_shape)
    key_states_21 = key_states_20.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:253, code: value_states = value_states.view(*proj_shape)
    value_states_21 = value_states_20.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:256, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_53 = key_states_21.transpose(1, 2);  key_states_21 = None
    attn_weights_40 = torch.bmm(query_states_21, transpose_53);  query_states_21 = transpose_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:269, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_98 = attn_weights_40.view(1, 16, 256, 256);  attn_weights_40 = None
    attn_weights_41 = view_98 + attention_mask;  view_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:270, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_42 = attn_weights_41.view(16, 256, 256);  attn_weights_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:272, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_43 = torch.nn.functional.softmax(attn_weights_42, dim = -1);  attn_weights_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:293, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_10 = torch.nn.functional.dropout(attn_weights_43, p = 0.0, training = False);  attn_weights_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:295, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_50 = torch.bmm(attn_probs_10, value_states_21);  attn_probs_10 = value_states_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:303, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_51 = attn_output_50.view(1, 16, 256, 64);  attn_output_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:304, code: attn_output = attn_output.transpose(1, 2)
    attn_output_52 = attn_output_51.transpose(1, 2);  attn_output_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:305, code: attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
    attn_output_53 = attn_output_52.reshape(1, 256, 1024);  attn_output_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:307, code: attn_output = self.out_proj(attn_output)
    hidden_states_113 = self.L__mod___model_decoder_layers_10_self_attn_out_proj(attn_output_53);  attn_output_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:391, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_114 = torch.nn.functional.dropout(hidden_states_113, p = 0.1, training = False);  hidden_states_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:392, code: hidden_states = residual + hidden_states
    hidden_states_115 = residual_20 + hidden_states_114;  residual_20 = hidden_states_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:393, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    residual_21 = self.L__mod___model_decoder_layers_10_self_attn_layer_norm(hidden_states_115);  hidden_states_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:422, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_10_fc1 = self.L__mod___model_decoder_layers_10_fc1(residual_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_117 = torch._C._nn.gelu(l__mod___model_decoder_layers_10_fc1);  l__mod___model_decoder_layers_10_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:423, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_118 = torch.nn.functional.dropout(hidden_states_117, p = 0.0, training = False);  hidden_states_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:424, code: hidden_states = self.fc2(hidden_states)
    hidden_states_119 = self.L__mod___model_decoder_layers_10_fc2(hidden_states_118);  hidden_states_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:426, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_120 = torch.nn.functional.dropout(hidden_states_119, p = 0.1, training = False);  hidden_states_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:427, code: hidden_states = residual + hidden_states
    hidden_states_121 = residual_21 + hidden_states_120;  residual_21 = hidden_states_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:428, code: hidden_states = self.final_layer_norm(hidden_states)
    residual_22 = self.L__mod___model_decoder_layers_10_final_layer_norm(hidden_states_121);  hidden_states_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:219, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_11_self_attn_q_proj = self.L__mod___model_decoder_layers_11_self_attn_q_proj(residual_22)
    query_states_22 = l__mod___model_decoder_layers_11_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_11_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:237, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_11_self_attn_k_proj = self.L__mod___model_decoder_layers_11_self_attn_k_proj(residual_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:200, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_101 = l__mod___model_decoder_layers_11_self_attn_k_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_11_self_attn_k_proj = None
    transpose_55 = view_101.transpose(1, 2);  view_101 = None
    key_states_22 = transpose_55.contiguous();  transpose_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:238, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_11_self_attn_v_proj = self.L__mod___model_decoder_layers_11_self_attn_v_proj(residual_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:200, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_102 = l__mod___model_decoder_layers_11_self_attn_v_proj.view(1, -1, 16, 64);  l__mod___model_decoder_layers_11_self_attn_v_proj = None
    transpose_56 = view_102.transpose(1, 2);  view_102 = None
    value_states_22 = transpose_56.contiguous();  transpose_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:200, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_103 = query_states_22.view(1, 256, 16, 64);  query_states_22 = None
    transpose_57 = view_103.transpose(1, 2);  view_103 = None
    contiguous_35 = transpose_57.contiguous();  transpose_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:251, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_23 = contiguous_35.view(16, -1, 64);  contiguous_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:252, code: key_states = key_states.view(*proj_shape)
    key_states_23 = key_states_22.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:253, code: value_states = value_states.view(*proj_shape)
    value_states_23 = value_states_22.view(16, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:256, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_58 = key_states_23.transpose(1, 2);  key_states_23 = None
    attn_weights_44 = torch.bmm(query_states_23, transpose_58);  query_states_23 = transpose_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:269, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_107 = attn_weights_44.view(1, 16, 256, 256);  attn_weights_44 = None
    attn_weights_45 = view_107 + attention_mask;  view_107 = attention_mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:270, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_46 = attn_weights_45.view(16, 256, 256);  attn_weights_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:272, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_47 = torch.nn.functional.softmax(attn_weights_46, dim = -1);  attn_weights_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:293, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_11 = torch.nn.functional.dropout(attn_weights_47, p = 0.0, training = False);  attn_weights_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:295, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_55 = torch.bmm(attn_probs_11, value_states_23);  attn_probs_11 = value_states_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:303, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_56 = attn_output_55.view(1, 16, 256, 64);  attn_output_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:304, code: attn_output = attn_output.transpose(1, 2)
    attn_output_57 = attn_output_56.transpose(1, 2);  attn_output_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:305, code: attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
    attn_output_58 = attn_output_57.reshape(1, 256, 1024);  attn_output_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:307, code: attn_output = self.out_proj(attn_output)
    hidden_states_124 = self.L__mod___model_decoder_layers_11_self_attn_out_proj(attn_output_58);  attn_output_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:391, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_125 = torch.nn.functional.dropout(hidden_states_124, p = 0.1, training = False);  hidden_states_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:392, code: hidden_states = residual + hidden_states
    hidden_states_126 = residual_22 + hidden_states_125;  residual_22 = hidden_states_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:393, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    residual_23 = self.L__mod___model_decoder_layers_11_self_attn_layer_norm(hidden_states_126);  hidden_states_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:422, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_11_fc1 = self.L__mod___model_decoder_layers_11_fc1(residual_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_128 = torch._C._nn.gelu(l__mod___model_decoder_layers_11_fc1);  l__mod___model_decoder_layers_11_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:423, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_129 = torch.nn.functional.dropout(hidden_states_128, p = 0.0, training = False);  hidden_states_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:424, code: hidden_states = self.fc2(hidden_states)
    hidden_states_130 = self.L__mod___model_decoder_layers_11_fc2(hidden_states_129);  hidden_states_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:426, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_131 = torch.nn.functional.dropout(hidden_states_130, p = 0.1, training = False);  hidden_states_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:427, code: hidden_states = residual + hidden_states
    hidden_states_132 = residual_23 + hidden_states_131;  residual_23 = hidden_states_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:428, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_134 = self.L__mod___model_decoder_layers_11_final_layer_norm(hidden_states_132);  hidden_states_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:978, code: logits = self.output_projection(outputs[0])
    logits = self.L__mod___output_projection(hidden_states_134);  hidden_states_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:983, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
    view_110 = logits.view(-1, 50265)
    view_111 = l_inputs_labels_.view(-1);  l_inputs_labels_ = None
    loss = torch.nn.functional.cross_entropy(view_110, view_111, None, None, -100, None, 'mean', 0.0);  view_110 = view_111 = None
    return (loss, logits, key_states, value_states, key_states_2, value_states_2, key_states_4, value_states_4, key_states_6, value_states_6, key_states_8, value_states_8, key_states_10, value_states_10, key_states_12, value_states_12, key_states_14, value_states_14, key_states_16, value_states_16, key_states_18, value_states_18, key_states_20, value_states_20, key_states_22, value_states_22)
    