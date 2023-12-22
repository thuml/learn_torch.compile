from __future__ import annotations



def forward(self, L_inputs_input_ids_ : torch.Tensor, L_inputs_labels_ : torch.Tensor):
    l_inputs_input_ids_ = L_inputs_input_ids_
    l_inputs_labels_ = L_inputs_labels_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:612, code: input_ids = input_ids.view(-1, input_shape[-1])
    input_ids = l_inputs_input_ids_.view(-1, 128);  l_inputs_input_ids_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:622, code: inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
    l__mod___model_decoder_embed_tokens = self.L__mod___model_decoder_embed_tokens(input_ids)
    inputs_embeds = l__mod___model_decoder_embed_tokens * 16.0;  l__mod___model_decoder_embed_tokens = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:53, code: mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask = torch.full((128, 128), -3.4028234663852886e+38, device = device(type='cpu'))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:54, code: mask_cond = torch.arange(mask.size(-1), device=device)
    mask_cond = torch.arange(128, device = device(type='cpu'))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:55, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    add = mask_cond + 1
    view_1 = add.view(128, 1);  add = None
    lt = mask_cond < view_1;  mask_cond = view_1 = None
    masked_fill_ = mask.masked_fill_(lt, 0);  lt = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:56, code: mask = mask.to(dtype)
    mask_1 = mask.to(torch.float32);  mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:60, code: return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
    getitem = mask_1[(None, None, slice(None, None, None), slice(None, None, None))];  mask_1 = None
    attention_mask = getitem.expand(1, 1, 128, 128);  getitem = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:144, code: mask = input_ids.ne(padding_idx).int()
    ne = input_ids.ne(1);  input_ids = None
    mask_2 = ne.int();  ne = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:145, code: incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    cumsum = torch.cumsum(mask_2, dim = 1)
    type_as = cumsum.type_as(mask_2);  cumsum = None
    add_1 = type_as + 0;  type_as = None
    incremental_indices = add_1 * mask_2;  add_1 = mask_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:146, code: return incremental_indices.long() + padding_idx
    long = incremental_indices.long();  incremental_indices = None
    add_2 = long + 1;  long = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:121, code: position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length).to(
    position_ids = add_2.to(device(type='cpu'));  add_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:127, code: if max_pos > self.weights.size(0):
    l__mod___model_decoder_embed_positions_weights = self.L__mod___model_decoder_embed_positions_weights
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:130, code: return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, -1).detach()
    view_2 = position_ids.view(-1);  position_ids = None
    index_select = l__mod___model_decoder_embed_positions_weights.index_select(0, view_2);  l__mod___model_decoder_embed_positions_weights = view_2 = None
    view_3 = index_select.view(1, 128, -1);  index_select = None
    positions = view_3.detach();  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:636, code: hidden_states = inputs_embeds + positions
    hidden_states = inputs_embeds + positions;  inputs_embeds = positions = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:637, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    residual = torch.nn.functional.dropout(hidden_states, p = 0.1, training = False);  hidden_states = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:201, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_0_self_attn_q_proj = self.L__mod___model_decoder_layers_0_self_attn_q_proj(residual)
    query_states = l__mod___model_decoder_layers_0_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_0_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:226, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_0_self_attn_k_proj = self.L__mod___model_decoder_layers_0_self_attn_k_proj(residual)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:181, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_4 = l__mod___model_decoder_layers_0_self_attn_k_proj.view(1, -1, 4, 64);  l__mod___model_decoder_layers_0_self_attn_k_proj = None
    transpose = view_4.transpose(1, 2);  view_4 = None
    key_states = transpose.contiguous();  transpose = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:227, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_0_self_attn_v_proj = self.L__mod___model_decoder_layers_0_self_attn_v_proj(residual)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:181, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_5 = l__mod___model_decoder_layers_0_self_attn_v_proj.view(1, -1, 4, 64);  l__mod___model_decoder_layers_0_self_attn_v_proj = None
    transpose_1 = view_5.transpose(1, 2);  view_5 = None
    value_states = transpose_1.contiguous();  transpose_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:181, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_6 = query_states.view(1, 128, 4, 64);  query_states = None
    transpose_2 = view_6.transpose(1, 2);  view_6 = None
    contiguous_2 = transpose_2.contiguous();  transpose_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:240, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_1 = contiguous_2.view(4, -1, 64);  contiguous_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:241, code: key_states = key_states.reshape(*proj_shape)
    key_states_1 = key_states.reshape(4, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:242, code: value_states = value_states.reshape(*proj_shape)
    value_states_1 = value_states.reshape(4, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:245, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_3 = key_states_1.transpose(1, 2);  key_states_1 = None
    attn_weights = torch.bmm(query_states_1, transpose_3);  query_states_1 = transpose_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:258, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_8 = attn_weights.view(1, 4, 128, 128);  attn_weights = None
    attn_weights_1 = view_8 + attention_mask;  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:259, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_2 = attn_weights_1.view(4, 128, 128);  attn_weights_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:261, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_3 = torch.nn.functional.softmax(attn_weights_2, dim = -1);  attn_weights_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:282, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs = torch.nn.functional.dropout(attn_weights_3, p = 0.0, training = False);  attn_weights_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:284, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output = torch.bmm(attn_probs, value_states_1);  attn_probs = value_states_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:292, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_1 = attn_output.view(1, 4, 128, 64);  attn_output = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:293, code: attn_output = attn_output.transpose(1, 2)
    attn_output_2 = attn_output_1.transpose(1, 2);  attn_output_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:297, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_3 = attn_output_2.reshape(1, 128, 256);  attn_output_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:299, code: attn_output = self.out_proj(attn_output)
    hidden_states_2 = self.L__mod___model_decoder_layers_0_self_attn_out_proj(attn_output_3);  attn_output_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:377, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_3 = torch.nn.functional.dropout(hidden_states_2, p = 0.1, training = False);  hidden_states_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:378, code: hidden_states = residual + hidden_states
    hidden_states_4 = residual + hidden_states_3;  residual = hidden_states_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:379, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    residual_1 = self.L__mod___model_decoder_layers_0_self_attn_layer_norm(hidden_states_4);  hidden_states_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:406, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_0_fc1 = self.L__mod___model_decoder_layers_0_fc1(residual_1)
    hidden_states_6 = self.L__mod___model_decoder_layers_0_activation_fn(l__mod___model_decoder_layers_0_fc1);  l__mod___model_decoder_layers_0_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:407, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_7 = torch.nn.functional.dropout(hidden_states_6, p = 0.0, training = False);  hidden_states_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:408, code: hidden_states = self.fc2(hidden_states)
    hidden_states_8 = self.L__mod___model_decoder_layers_0_fc2(hidden_states_7);  hidden_states_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:409, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_9 = torch.nn.functional.dropout(hidden_states_8, p = 0.1, training = False);  hidden_states_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:410, code: hidden_states = residual + hidden_states
    hidden_states_10 = residual_1 + hidden_states_9;  residual_1 = hidden_states_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:411, code: hidden_states = self.final_layer_norm(hidden_states)
    residual_2 = self.L__mod___model_decoder_layers_0_final_layer_norm(hidden_states_10);  hidden_states_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:201, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_1_self_attn_q_proj = self.L__mod___model_decoder_layers_1_self_attn_q_proj(residual_2)
    query_states_2 = l__mod___model_decoder_layers_1_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_1_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:226, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_1_self_attn_k_proj = self.L__mod___model_decoder_layers_1_self_attn_k_proj(residual_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:181, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_11 = l__mod___model_decoder_layers_1_self_attn_k_proj.view(1, -1, 4, 64);  l__mod___model_decoder_layers_1_self_attn_k_proj = None
    transpose_5 = view_11.transpose(1, 2);  view_11 = None
    key_states_2 = transpose_5.contiguous();  transpose_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:227, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_1_self_attn_v_proj = self.L__mod___model_decoder_layers_1_self_attn_v_proj(residual_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:181, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_12 = l__mod___model_decoder_layers_1_self_attn_v_proj.view(1, -1, 4, 64);  l__mod___model_decoder_layers_1_self_attn_v_proj = None
    transpose_6 = view_12.transpose(1, 2);  view_12 = None
    value_states_2 = transpose_6.contiguous();  transpose_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:181, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_13 = query_states_2.view(1, 128, 4, 64);  query_states_2 = None
    transpose_7 = view_13.transpose(1, 2);  view_13 = None
    contiguous_5 = transpose_7.contiguous();  transpose_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:240, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_3 = contiguous_5.view(4, -1, 64);  contiguous_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:241, code: key_states = key_states.reshape(*proj_shape)
    key_states_3 = key_states_2.reshape(4, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:242, code: value_states = value_states.reshape(*proj_shape)
    value_states_3 = value_states_2.reshape(4, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:245, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_8 = key_states_3.transpose(1, 2);  key_states_3 = None
    attn_weights_4 = torch.bmm(query_states_3, transpose_8);  query_states_3 = transpose_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:258, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_15 = attn_weights_4.view(1, 4, 128, 128);  attn_weights_4 = None
    attn_weights_5 = view_15 + attention_mask;  view_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:259, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_6 = attn_weights_5.view(4, 128, 128);  attn_weights_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:261, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_7 = torch.nn.functional.softmax(attn_weights_6, dim = -1);  attn_weights_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:282, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_1 = torch.nn.functional.dropout(attn_weights_7, p = 0.0, training = False);  attn_weights_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:284, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_5 = torch.bmm(attn_probs_1, value_states_3);  attn_probs_1 = value_states_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:292, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_6 = attn_output_5.view(1, 4, 128, 64);  attn_output_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:293, code: attn_output = attn_output.transpose(1, 2)
    attn_output_7 = attn_output_6.transpose(1, 2);  attn_output_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:297, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_8 = attn_output_7.reshape(1, 128, 256);  attn_output_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:299, code: attn_output = self.out_proj(attn_output)
    hidden_states_13 = self.L__mod___model_decoder_layers_1_self_attn_out_proj(attn_output_8);  attn_output_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:377, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_14 = torch.nn.functional.dropout(hidden_states_13, p = 0.1, training = False);  hidden_states_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:378, code: hidden_states = residual + hidden_states
    hidden_states_15 = residual_2 + hidden_states_14;  residual_2 = hidden_states_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:379, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    residual_3 = self.L__mod___model_decoder_layers_1_self_attn_layer_norm(hidden_states_15);  hidden_states_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:406, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_1_fc1 = self.L__mod___model_decoder_layers_1_fc1(residual_3)
    hidden_states_17 = self.L__mod___model_decoder_layers_1_activation_fn(l__mod___model_decoder_layers_1_fc1);  l__mod___model_decoder_layers_1_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:407, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_18 = torch.nn.functional.dropout(hidden_states_17, p = 0.0, training = False);  hidden_states_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:408, code: hidden_states = self.fc2(hidden_states)
    hidden_states_19 = self.L__mod___model_decoder_layers_1_fc2(hidden_states_18);  hidden_states_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:409, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_20 = torch.nn.functional.dropout(hidden_states_19, p = 0.1, training = False);  hidden_states_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:410, code: hidden_states = residual + hidden_states
    hidden_states_21 = residual_3 + hidden_states_20;  residual_3 = hidden_states_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:411, code: hidden_states = self.final_layer_norm(hidden_states)
    residual_4 = self.L__mod___model_decoder_layers_1_final_layer_norm(hidden_states_21);  hidden_states_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:201, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_2_self_attn_q_proj = self.L__mod___model_decoder_layers_2_self_attn_q_proj(residual_4)
    query_states_4 = l__mod___model_decoder_layers_2_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_2_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:226, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_2_self_attn_k_proj = self.L__mod___model_decoder_layers_2_self_attn_k_proj(residual_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:181, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_18 = l__mod___model_decoder_layers_2_self_attn_k_proj.view(1, -1, 4, 64);  l__mod___model_decoder_layers_2_self_attn_k_proj = None
    transpose_10 = view_18.transpose(1, 2);  view_18 = None
    key_states_4 = transpose_10.contiguous();  transpose_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:227, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_2_self_attn_v_proj = self.L__mod___model_decoder_layers_2_self_attn_v_proj(residual_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:181, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_19 = l__mod___model_decoder_layers_2_self_attn_v_proj.view(1, -1, 4, 64);  l__mod___model_decoder_layers_2_self_attn_v_proj = None
    transpose_11 = view_19.transpose(1, 2);  view_19 = None
    value_states_4 = transpose_11.contiguous();  transpose_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:181, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_20 = query_states_4.view(1, 128, 4, 64);  query_states_4 = None
    transpose_12 = view_20.transpose(1, 2);  view_20 = None
    contiguous_8 = transpose_12.contiguous();  transpose_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:240, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_5 = contiguous_8.view(4, -1, 64);  contiguous_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:241, code: key_states = key_states.reshape(*proj_shape)
    key_states_5 = key_states_4.reshape(4, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:242, code: value_states = value_states.reshape(*proj_shape)
    value_states_5 = value_states_4.reshape(4, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:245, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_13 = key_states_5.transpose(1, 2);  key_states_5 = None
    attn_weights_8 = torch.bmm(query_states_5, transpose_13);  query_states_5 = transpose_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:258, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_22 = attn_weights_8.view(1, 4, 128, 128);  attn_weights_8 = None
    attn_weights_9 = view_22 + attention_mask;  view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:259, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_10 = attn_weights_9.view(4, 128, 128);  attn_weights_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:261, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_11 = torch.nn.functional.softmax(attn_weights_10, dim = -1);  attn_weights_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:282, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_2 = torch.nn.functional.dropout(attn_weights_11, p = 0.0, training = False);  attn_weights_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:284, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_10 = torch.bmm(attn_probs_2, value_states_5);  attn_probs_2 = value_states_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:292, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_11 = attn_output_10.view(1, 4, 128, 64);  attn_output_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:293, code: attn_output = attn_output.transpose(1, 2)
    attn_output_12 = attn_output_11.transpose(1, 2);  attn_output_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:297, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_13 = attn_output_12.reshape(1, 128, 256);  attn_output_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:299, code: attn_output = self.out_proj(attn_output)
    hidden_states_24 = self.L__mod___model_decoder_layers_2_self_attn_out_proj(attn_output_13);  attn_output_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:377, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_25 = torch.nn.functional.dropout(hidden_states_24, p = 0.1, training = False);  hidden_states_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:378, code: hidden_states = residual + hidden_states
    hidden_states_26 = residual_4 + hidden_states_25;  residual_4 = hidden_states_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:379, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    residual_5 = self.L__mod___model_decoder_layers_2_self_attn_layer_norm(hidden_states_26);  hidden_states_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:406, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_2_fc1 = self.L__mod___model_decoder_layers_2_fc1(residual_5)
    hidden_states_28 = self.L__mod___model_decoder_layers_2_activation_fn(l__mod___model_decoder_layers_2_fc1);  l__mod___model_decoder_layers_2_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:407, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_29 = torch.nn.functional.dropout(hidden_states_28, p = 0.0, training = False);  hidden_states_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:408, code: hidden_states = self.fc2(hidden_states)
    hidden_states_30 = self.L__mod___model_decoder_layers_2_fc2(hidden_states_29);  hidden_states_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:409, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_31 = torch.nn.functional.dropout(hidden_states_30, p = 0.1, training = False);  hidden_states_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:410, code: hidden_states = residual + hidden_states
    hidden_states_32 = residual_5 + hidden_states_31;  residual_5 = hidden_states_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:411, code: hidden_states = self.final_layer_norm(hidden_states)
    residual_6 = self.L__mod___model_decoder_layers_2_final_layer_norm(hidden_states_32);  hidden_states_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:201, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_3_self_attn_q_proj = self.L__mod___model_decoder_layers_3_self_attn_q_proj(residual_6)
    query_states_6 = l__mod___model_decoder_layers_3_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_3_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:226, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_3_self_attn_k_proj = self.L__mod___model_decoder_layers_3_self_attn_k_proj(residual_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:181, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_25 = l__mod___model_decoder_layers_3_self_attn_k_proj.view(1, -1, 4, 64);  l__mod___model_decoder_layers_3_self_attn_k_proj = None
    transpose_15 = view_25.transpose(1, 2);  view_25 = None
    key_states_6 = transpose_15.contiguous();  transpose_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:227, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_3_self_attn_v_proj = self.L__mod___model_decoder_layers_3_self_attn_v_proj(residual_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:181, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_26 = l__mod___model_decoder_layers_3_self_attn_v_proj.view(1, -1, 4, 64);  l__mod___model_decoder_layers_3_self_attn_v_proj = None
    transpose_16 = view_26.transpose(1, 2);  view_26 = None
    value_states_6 = transpose_16.contiguous();  transpose_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:181, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_27 = query_states_6.view(1, 128, 4, 64);  query_states_6 = None
    transpose_17 = view_27.transpose(1, 2);  view_27 = None
    contiguous_11 = transpose_17.contiguous();  transpose_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:240, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_7 = contiguous_11.view(4, -1, 64);  contiguous_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:241, code: key_states = key_states.reshape(*proj_shape)
    key_states_7 = key_states_6.reshape(4, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:242, code: value_states = value_states.reshape(*proj_shape)
    value_states_7 = value_states_6.reshape(4, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:245, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_18 = key_states_7.transpose(1, 2);  key_states_7 = None
    attn_weights_12 = torch.bmm(query_states_7, transpose_18);  query_states_7 = transpose_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:258, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_29 = attn_weights_12.view(1, 4, 128, 128);  attn_weights_12 = None
    attn_weights_13 = view_29 + attention_mask;  view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:259, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_14 = attn_weights_13.view(4, 128, 128);  attn_weights_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:261, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_15 = torch.nn.functional.softmax(attn_weights_14, dim = -1);  attn_weights_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:282, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_3 = torch.nn.functional.dropout(attn_weights_15, p = 0.0, training = False);  attn_weights_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:284, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_15 = torch.bmm(attn_probs_3, value_states_7);  attn_probs_3 = value_states_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:292, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_16 = attn_output_15.view(1, 4, 128, 64);  attn_output_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:293, code: attn_output = attn_output.transpose(1, 2)
    attn_output_17 = attn_output_16.transpose(1, 2);  attn_output_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:297, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_18 = attn_output_17.reshape(1, 128, 256);  attn_output_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:299, code: attn_output = self.out_proj(attn_output)
    hidden_states_35 = self.L__mod___model_decoder_layers_3_self_attn_out_proj(attn_output_18);  attn_output_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:377, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_36 = torch.nn.functional.dropout(hidden_states_35, p = 0.1, training = False);  hidden_states_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:378, code: hidden_states = residual + hidden_states
    hidden_states_37 = residual_6 + hidden_states_36;  residual_6 = hidden_states_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:379, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    residual_7 = self.L__mod___model_decoder_layers_3_self_attn_layer_norm(hidden_states_37);  hidden_states_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:406, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_3_fc1 = self.L__mod___model_decoder_layers_3_fc1(residual_7)
    hidden_states_39 = self.L__mod___model_decoder_layers_3_activation_fn(l__mod___model_decoder_layers_3_fc1);  l__mod___model_decoder_layers_3_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:407, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_40 = torch.nn.functional.dropout(hidden_states_39, p = 0.0, training = False);  hidden_states_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:408, code: hidden_states = self.fc2(hidden_states)
    hidden_states_41 = self.L__mod___model_decoder_layers_3_fc2(hidden_states_40);  hidden_states_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:409, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_42 = torch.nn.functional.dropout(hidden_states_41, p = 0.1, training = False);  hidden_states_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:410, code: hidden_states = residual + hidden_states
    hidden_states_43 = residual_7 + hidden_states_42;  residual_7 = hidden_states_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:411, code: hidden_states = self.final_layer_norm(hidden_states)
    residual_8 = self.L__mod___model_decoder_layers_3_final_layer_norm(hidden_states_43);  hidden_states_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:201, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_4_self_attn_q_proj = self.L__mod___model_decoder_layers_4_self_attn_q_proj(residual_8)
    query_states_8 = l__mod___model_decoder_layers_4_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_4_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:226, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_4_self_attn_k_proj = self.L__mod___model_decoder_layers_4_self_attn_k_proj(residual_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:181, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_32 = l__mod___model_decoder_layers_4_self_attn_k_proj.view(1, -1, 4, 64);  l__mod___model_decoder_layers_4_self_attn_k_proj = None
    transpose_20 = view_32.transpose(1, 2);  view_32 = None
    key_states_8 = transpose_20.contiguous();  transpose_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:227, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_4_self_attn_v_proj = self.L__mod___model_decoder_layers_4_self_attn_v_proj(residual_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:181, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_33 = l__mod___model_decoder_layers_4_self_attn_v_proj.view(1, -1, 4, 64);  l__mod___model_decoder_layers_4_self_attn_v_proj = None
    transpose_21 = view_33.transpose(1, 2);  view_33 = None
    value_states_8 = transpose_21.contiguous();  transpose_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:181, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_34 = query_states_8.view(1, 128, 4, 64);  query_states_8 = None
    transpose_22 = view_34.transpose(1, 2);  view_34 = None
    contiguous_14 = transpose_22.contiguous();  transpose_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:240, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_9 = contiguous_14.view(4, -1, 64);  contiguous_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:241, code: key_states = key_states.reshape(*proj_shape)
    key_states_9 = key_states_8.reshape(4, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:242, code: value_states = value_states.reshape(*proj_shape)
    value_states_9 = value_states_8.reshape(4, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:245, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_23 = key_states_9.transpose(1, 2);  key_states_9 = None
    attn_weights_16 = torch.bmm(query_states_9, transpose_23);  query_states_9 = transpose_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:258, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_36 = attn_weights_16.view(1, 4, 128, 128);  attn_weights_16 = None
    attn_weights_17 = view_36 + attention_mask;  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:259, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_18 = attn_weights_17.view(4, 128, 128);  attn_weights_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:261, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_19 = torch.nn.functional.softmax(attn_weights_18, dim = -1);  attn_weights_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:282, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_4 = torch.nn.functional.dropout(attn_weights_19, p = 0.0, training = False);  attn_weights_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:284, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_20 = torch.bmm(attn_probs_4, value_states_9);  attn_probs_4 = value_states_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:292, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_21 = attn_output_20.view(1, 4, 128, 64);  attn_output_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:293, code: attn_output = attn_output.transpose(1, 2)
    attn_output_22 = attn_output_21.transpose(1, 2);  attn_output_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:297, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_23 = attn_output_22.reshape(1, 128, 256);  attn_output_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:299, code: attn_output = self.out_proj(attn_output)
    hidden_states_46 = self.L__mod___model_decoder_layers_4_self_attn_out_proj(attn_output_23);  attn_output_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:377, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_47 = torch.nn.functional.dropout(hidden_states_46, p = 0.1, training = False);  hidden_states_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:378, code: hidden_states = residual + hidden_states
    hidden_states_48 = residual_8 + hidden_states_47;  residual_8 = hidden_states_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:379, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    residual_9 = self.L__mod___model_decoder_layers_4_self_attn_layer_norm(hidden_states_48);  hidden_states_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:406, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_4_fc1 = self.L__mod___model_decoder_layers_4_fc1(residual_9)
    hidden_states_50 = self.L__mod___model_decoder_layers_4_activation_fn(l__mod___model_decoder_layers_4_fc1);  l__mod___model_decoder_layers_4_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:407, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_51 = torch.nn.functional.dropout(hidden_states_50, p = 0.0, training = False);  hidden_states_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:408, code: hidden_states = self.fc2(hidden_states)
    hidden_states_52 = self.L__mod___model_decoder_layers_4_fc2(hidden_states_51);  hidden_states_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:409, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_53 = torch.nn.functional.dropout(hidden_states_52, p = 0.1, training = False);  hidden_states_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:410, code: hidden_states = residual + hidden_states
    hidden_states_54 = residual_9 + hidden_states_53;  residual_9 = hidden_states_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:411, code: hidden_states = self.final_layer_norm(hidden_states)
    residual_10 = self.L__mod___model_decoder_layers_4_final_layer_norm(hidden_states_54);  hidden_states_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:201, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_5_self_attn_q_proj = self.L__mod___model_decoder_layers_5_self_attn_q_proj(residual_10)
    query_states_10 = l__mod___model_decoder_layers_5_self_attn_q_proj * 0.125;  l__mod___model_decoder_layers_5_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:226, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_5_self_attn_k_proj = self.L__mod___model_decoder_layers_5_self_attn_k_proj(residual_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:181, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_39 = l__mod___model_decoder_layers_5_self_attn_k_proj.view(1, -1, 4, 64);  l__mod___model_decoder_layers_5_self_attn_k_proj = None
    transpose_25 = view_39.transpose(1, 2);  view_39 = None
    key_states_10 = transpose_25.contiguous();  transpose_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:227, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_5_self_attn_v_proj = self.L__mod___model_decoder_layers_5_self_attn_v_proj(residual_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:181, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_40 = l__mod___model_decoder_layers_5_self_attn_v_proj.view(1, -1, 4, 64);  l__mod___model_decoder_layers_5_self_attn_v_proj = None
    transpose_26 = view_40.transpose(1, 2);  view_40 = None
    value_states_10 = transpose_26.contiguous();  transpose_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:181, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_41 = query_states_10.view(1, 128, 4, 64);  query_states_10 = None
    transpose_27 = view_41.transpose(1, 2);  view_41 = None
    contiguous_17 = transpose_27.contiguous();  transpose_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:240, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_11 = contiguous_17.view(4, -1, 64);  contiguous_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:241, code: key_states = key_states.reshape(*proj_shape)
    key_states_11 = key_states_10.reshape(4, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:242, code: value_states = value_states.reshape(*proj_shape)
    value_states_11 = value_states_10.reshape(4, -1, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:245, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_28 = key_states_11.transpose(1, 2);  key_states_11 = None
    attn_weights_20 = torch.bmm(query_states_11, transpose_28);  query_states_11 = transpose_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:258, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_43 = attn_weights_20.view(1, 4, 128, 128);  attn_weights_20 = None
    attn_weights_21 = view_43 + attention_mask;  view_43 = attention_mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:259, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_22 = attn_weights_21.view(4, 128, 128);  attn_weights_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:261, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_23 = torch.nn.functional.softmax(attn_weights_22, dim = -1);  attn_weights_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:282, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_5 = torch.nn.functional.dropout(attn_weights_23, p = 0.0, training = False);  attn_weights_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:284, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_25 = torch.bmm(attn_probs_5, value_states_11);  attn_probs_5 = value_states_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:292, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_26 = attn_output_25.view(1, 4, 128, 64);  attn_output_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:293, code: attn_output = attn_output.transpose(1, 2)
    attn_output_27 = attn_output_26.transpose(1, 2);  attn_output_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:297, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_28 = attn_output_27.reshape(1, 128, 256);  attn_output_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:299, code: attn_output = self.out_proj(attn_output)
    hidden_states_57 = self.L__mod___model_decoder_layers_5_self_attn_out_proj(attn_output_28);  attn_output_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:377, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_58 = torch.nn.functional.dropout(hidden_states_57, p = 0.1, training = False);  hidden_states_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:378, code: hidden_states = residual + hidden_states
    hidden_states_59 = residual_10 + hidden_states_58;  residual_10 = hidden_states_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:379, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    residual_11 = self.L__mod___model_decoder_layers_5_self_attn_layer_norm(hidden_states_59);  hidden_states_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:406, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_5_fc1 = self.L__mod___model_decoder_layers_5_fc1(residual_11)
    hidden_states_61 = self.L__mod___model_decoder_layers_5_activation_fn(l__mod___model_decoder_layers_5_fc1);  l__mod___model_decoder_layers_5_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:407, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_62 = torch.nn.functional.dropout(hidden_states_61, p = 0.0, training = False);  hidden_states_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:408, code: hidden_states = self.fc2(hidden_states)
    hidden_states_63 = self.L__mod___model_decoder_layers_5_fc2(hidden_states_62);  hidden_states_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:409, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_64 = torch.nn.functional.dropout(hidden_states_63, p = 0.1, training = False);  hidden_states_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:410, code: hidden_states = residual + hidden_states
    hidden_states_65 = residual_11 + hidden_states_64;  residual_11 = hidden_states_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:411, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_67 = self.L__mod___model_decoder_layers_5_final_layer_norm(hidden_states_65);  hidden_states_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:938, code: logits = self.lm_head(outputs[0])
    logits = self.L__mod___lm_head(hidden_states_67);  hidden_states_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:943, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
    view_46 = logits.view(-1, 10000)
    view_47 = l_inputs_labels_.view(-1);  l_inputs_labels_ = None
    loss = torch.nn.functional.cross_entropy(view_46, view_47, None, None, -100, None, 'mean', 0.0);  view_46 = view_47 = None
    return (loss, logits, key_states, value_states, key_states_2, value_states_2, key_states_4, value_states_4, key_states_6, value_states_6, key_states_8, value_states_8, key_states_10, value_states_10)
    