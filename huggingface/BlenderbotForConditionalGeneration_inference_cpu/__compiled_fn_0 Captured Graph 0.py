from __future__ import annotations



def forward(self, L_inputs_labels_ : torch.Tensor, L_inputs_decoder_input_ids_ : torch.Tensor, L_inputs_input_ids_ : torch.Tensor):
    l_inputs_labels_ = L_inputs_labels_
    l_inputs_decoder_input_ids_ = L_inputs_decoder_input_ids_
    l_inputs_input_ids_ = L_inputs_input_ids_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:737, code: input_ids = input_ids.view(-1, input_shape[-1])
    input_ids = l_inputs_input_ids_.view(-1, 128);  l_inputs_input_ids_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:744, code: inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
    l__mod___model_encoder_embed_tokens = self.L__mod___model_encoder_embed_tokens(input_ids);  input_ids = None
    inputs_embeds = l__mod___model_encoder_embed_tokens * 1.0;  l__mod___model_encoder_embed_tokens = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:123, code: past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
    l__mod___model_encoder_embed_positions_weight = self.L__mod___model_encoder_embed_positions_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:122, code: positions = torch.arange(
    positions = torch.arange(0, 128, dtype = torch.int64, device = device(type='cpu'))
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    embed_pos = torch.nn.functional.embedding(positions, l__mod___model_encoder_embed_positions_weight, None, None, 2.0, False, False);  positions = l__mod___model_encoder_embed_positions_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:748, code: hidden_states = inputs_embeds + embed_pos
    hidden_states = inputs_embeds + embed_pos;  inputs_embeds = embed_pos = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:749, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    residual = torch.nn.functional.dropout(hidden_states, p = 0.1, training = False);  hidden_states = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:320, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_2 = self.L__mod___model_encoder_layers_0_self_attn_layer_norm(residual)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_encoder_layers_0_self_attn_q_proj = self.L__mod___model_encoder_layers_0_self_attn_q_proj(hidden_states_2)
    query_states = l__mod___model_encoder_layers_0_self_attn_q_proj * 0.11180339887498948;  l__mod___model_encoder_layers_0_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_0_self_attn_k_proj = self.L__mod___model_encoder_layers_0_self_attn_k_proj(hidden_states_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_1 = l__mod___model_encoder_layers_0_self_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_encoder_layers_0_self_attn_k_proj = None
    transpose = view_1.transpose(1, 2);  view_1 = None
    key_states = transpose.contiguous();  transpose = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_0_self_attn_v_proj = self.L__mod___model_encoder_layers_0_self_attn_v_proj(hidden_states_2);  hidden_states_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_2 = l__mod___model_encoder_layers_0_self_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_encoder_layers_0_self_attn_v_proj = None
    transpose_1 = view_2.transpose(1, 2);  view_2 = None
    value_states = transpose_1.contiguous();  transpose_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_3 = query_states.view(1, 128, 32, 80);  query_states = None
    transpose_2 = view_3.transpose(1, 2);  view_3 = None
    contiguous_2 = transpose_2.contiguous();  transpose_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_1 = contiguous_2.view(32, -1, 80);  contiguous_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_1 = key_states.reshape(32, -1, 80);  key_states = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_1 = value_states.reshape(32, -1, 80);  value_states = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_3 = key_states_1.transpose(1, 2);  key_states_1 = None
    attn_weights = torch.bmm(query_states_1, transpose_3);  query_states_1 = transpose_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_1 = torch.nn.functional.softmax(attn_weights, dim = -1);  attn_weights = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs = torch.nn.functional.dropout(attn_weights_1, p = 0.0, training = False);  attn_weights_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output = torch.bmm(attn_probs, value_states_1);  attn_probs = value_states_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_1 = attn_output.view(1, 32, 128, 80);  attn_output = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_2 = attn_output_1.transpose(1, 2);  attn_output_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_3 = attn_output_2.reshape(1, 128, 2560);  attn_output_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_3 = self.L__mod___model_encoder_layers_0_self_attn_out_proj(attn_output_3);  attn_output_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:327, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_4 = torch.nn.functional.dropout(hidden_states_3, p = 0.1, training = False);  hidden_states_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:328, code: hidden_states = residual + hidden_states
    residual_1 = residual + hidden_states_4;  residual = hidden_states_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:331, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_6 = self.L__mod___model_encoder_layers_0_final_layer_norm(residual_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:332, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_encoder_layers_0_fc1 = self.L__mod___model_encoder_layers_0_fc1(hidden_states_6);  hidden_states_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_7 = torch._C._nn.gelu(l__mod___model_encoder_layers_0_fc1);  l__mod___model_encoder_layers_0_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:333, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_8 = torch.nn.functional.dropout(hidden_states_7, p = 0.0, training = False);  hidden_states_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:334, code: hidden_states = self.fc2(hidden_states)
    hidden_states_9 = self.L__mod___model_encoder_layers_0_fc2(hidden_states_8);  hidden_states_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:335, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_10 = torch.nn.functional.dropout(hidden_states_9, p = 0.1, training = False);  hidden_states_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:336, code: hidden_states = residual + hidden_states
    residual_2 = residual_1 + hidden_states_10;  residual_1 = hidden_states_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:320, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_13 = self.L__mod___model_encoder_layers_1_self_attn_layer_norm(residual_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_encoder_layers_1_self_attn_q_proj = self.L__mod___model_encoder_layers_1_self_attn_q_proj(hidden_states_13)
    query_states_2 = l__mod___model_encoder_layers_1_self_attn_q_proj * 0.11180339887498948;  l__mod___model_encoder_layers_1_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_1_self_attn_k_proj = self.L__mod___model_encoder_layers_1_self_attn_k_proj(hidden_states_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_6 = l__mod___model_encoder_layers_1_self_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_encoder_layers_1_self_attn_k_proj = None
    transpose_5 = view_6.transpose(1, 2);  view_6 = None
    key_states_2 = transpose_5.contiguous();  transpose_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_encoder_layers_1_self_attn_v_proj = self.L__mod___model_encoder_layers_1_self_attn_v_proj(hidden_states_13);  hidden_states_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_7 = l__mod___model_encoder_layers_1_self_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_encoder_layers_1_self_attn_v_proj = None
    transpose_6 = view_7.transpose(1, 2);  view_7 = None
    value_states_2 = transpose_6.contiguous();  transpose_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_8 = query_states_2.view(1, 128, 32, 80);  query_states_2 = None
    transpose_7 = view_8.transpose(1, 2);  view_8 = None
    contiguous_5 = transpose_7.contiguous();  transpose_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_3 = contiguous_5.view(32, -1, 80);  contiguous_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_3 = key_states_2.reshape(32, -1, 80);  key_states_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_3 = value_states_2.reshape(32, -1, 80);  value_states_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_8 = key_states_3.transpose(1, 2);  key_states_3 = None
    attn_weights_2 = torch.bmm(query_states_3, transpose_8);  query_states_3 = transpose_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_3 = torch.nn.functional.softmax(attn_weights_2, dim = -1);  attn_weights_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_1 = torch.nn.functional.dropout(attn_weights_3, p = 0.0, training = False);  attn_weights_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_5 = torch.bmm(attn_probs_1, value_states_3);  attn_probs_1 = value_states_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_6 = attn_output_5.view(1, 32, 128, 80);  attn_output_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_7 = attn_output_6.transpose(1, 2);  attn_output_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_8 = attn_output_7.reshape(1, 128, 2560);  attn_output_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_14 = self.L__mod___model_encoder_layers_1_self_attn_out_proj(attn_output_8);  attn_output_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:327, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_15 = torch.nn.functional.dropout(hidden_states_14, p = 0.1, training = False);  hidden_states_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:328, code: hidden_states = residual + hidden_states
    residual_3 = residual_2 + hidden_states_15;  residual_2 = hidden_states_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:331, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_17 = self.L__mod___model_encoder_layers_1_final_layer_norm(residual_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:332, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_encoder_layers_1_fc1 = self.L__mod___model_encoder_layers_1_fc1(hidden_states_17);  hidden_states_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_18 = torch._C._nn.gelu(l__mod___model_encoder_layers_1_fc1);  l__mod___model_encoder_layers_1_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:333, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_19 = torch.nn.functional.dropout(hidden_states_18, p = 0.0, training = False);  hidden_states_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:334, code: hidden_states = self.fc2(hidden_states)
    hidden_states_20 = self.L__mod___model_encoder_layers_1_fc2(hidden_states_19);  hidden_states_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:335, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_21 = torch.nn.functional.dropout(hidden_states_20, p = 0.1, training = False);  hidden_states_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:336, code: hidden_states = residual + hidden_states
    hidden_states_23 = residual_3 + hidden_states_21;  residual_3 = hidden_states_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:807, code: hidden_states = self.layer_norm(hidden_states)
    hidden_states_24 = self.L__mod___model_encoder_layer_norm(hidden_states_23);  hidden_states_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:975, code: input_ids = input_ids.view(-1, input_shape[-1])
    input_ids_1 = l_inputs_decoder_input_ids_.view(-1, 128);  l_inputs_decoder_input_ids_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:985, code: inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
    l__mod___model_encoder_embed_tokens_1 = self.L__mod___model_encoder_embed_tokens(input_ids_1);  input_ids_1 = None
    inputs_embeds_1 = l__mod___model_encoder_embed_tokens_1 * 1.0;  l__mod___model_encoder_embed_tokens_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:86, code: mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask = torch.full((128, 128), -3.4028234663852886e+38, device = device(type='cpu'))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:87, code: mask_cond = torch.arange(mask.size(-1), device=device)
    mask_cond = torch.arange(128, device = device(type='cpu'))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:88, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    add_5 = mask_cond + 1
    view_12 = add_5.view(128, 1);  add_5 = None
    lt = mask_cond < view_12;  mask_cond = view_12 = None
    masked_fill_ = mask.masked_fill_(lt, 0);  lt = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:89, code: mask = mask.to(dtype)
    mask_1 = mask.to(torch.float32);  mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:93, code: return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
    getitem = mask_1[(None, None, slice(None, None, None), slice(None, None, None))];  mask_1 = None
    attention_mask = getitem.expand(1, 1, 128, 128);  getitem = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:123, code: past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
    l__mod___model_decoder_embed_positions_weight = self.L__mod___model_decoder_embed_positions_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:122, code: positions = torch.arange(
    positions_1 = torch.arange(0, 128, dtype = torch.int64, device = device(type='cpu'))
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    positions_2 = torch.nn.functional.embedding(positions_1, l__mod___model_decoder_embed_positions_weight, None, None, 2.0, False, False);  positions_1 = l__mod___model_decoder_embed_positions_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:999, code: hidden_states = inputs_embeds + positions
    hidden_states_25 = inputs_embeds_1 + positions_2;  inputs_embeds_1 = positions_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:1001, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    residual_4 = torch.nn.functional.dropout(hidden_states_25, p = 0.1, training = False);  hidden_states_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_27 = self.L__mod___model_decoder_layers_0_self_attn_layer_norm(residual_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_0_self_attn_q_proj = self.L__mod___model_decoder_layers_0_self_attn_q_proj(hidden_states_27)
    query_states_4 = l__mod___model_decoder_layers_0_self_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_0_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_0_self_attn_k_proj = self.L__mod___model_decoder_layers_0_self_attn_k_proj(hidden_states_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_13 = l__mod___model_decoder_layers_0_self_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_0_self_attn_k_proj = None
    transpose_10 = view_13.transpose(1, 2);  view_13 = None
    key_states_4 = transpose_10.contiguous();  transpose_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_0_self_attn_v_proj = self.L__mod___model_decoder_layers_0_self_attn_v_proj(hidden_states_27);  hidden_states_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_14 = l__mod___model_decoder_layers_0_self_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_0_self_attn_v_proj = None
    transpose_11 = view_14.transpose(1, 2);  view_14 = None
    value_states_4 = transpose_11.contiguous();  transpose_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_15 = query_states_4.view(1, 128, 32, 80);  query_states_4 = None
    transpose_12 = view_15.transpose(1, 2);  view_15 = None
    contiguous_8 = transpose_12.contiguous();  transpose_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_5 = contiguous_8.view(32, -1, 80);  contiguous_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_5 = key_states_4.reshape(32, -1, 80);  key_states_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_5 = value_states_4.reshape(32, -1, 80);  value_states_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_13 = key_states_5.transpose(1, 2);  key_states_5 = None
    attn_weights_4 = torch.bmm(query_states_5, transpose_13);  query_states_5 = transpose_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_17 = attn_weights_4.view(1, 32, 128, 128);  attn_weights_4 = None
    attn_weights_5 = view_17 + attention_mask;  view_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_6 = attn_weights_5.view(32, 128, 128);  attn_weights_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_7 = torch.nn.functional.softmax(attn_weights_6, dim = -1);  attn_weights_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_2 = torch.nn.functional.dropout(attn_weights_7, p = 0.0, training = False);  attn_weights_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_10 = torch.bmm(attn_probs_2, value_states_5);  attn_probs_2 = value_states_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_11 = attn_output_10.view(1, 32, 128, 80);  attn_output_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_12 = attn_output_11.transpose(1, 2);  attn_output_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_13 = attn_output_12.reshape(1, 128, 2560);  attn_output_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_28 = self.L__mod___model_decoder_layers_0_self_attn_out_proj(attn_output_13);  attn_output_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:424, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_29 = torch.nn.functional.dropout(hidden_states_28, p = 0.1, training = False);  hidden_states_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    residual_5 = residual_4 + hidden_states_29;  residual_4 = hidden_states_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_31 = self.L__mod___model_decoder_layers_0_encoder_attn_layer_norm(residual_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_0_encoder_attn_q_proj = self.L__mod___model_decoder_layers_0_encoder_attn_q_proj(hidden_states_31);  hidden_states_31 = None
    query_states_6 = l__mod___model_decoder_layers_0_encoder_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_0_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_0_encoder_attn_k_proj = self.L__mod___model_decoder_layers_0_encoder_attn_k_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_20 = l__mod___model_decoder_layers_0_encoder_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_0_encoder_attn_k_proj = None
    transpose_15 = view_20.transpose(1, 2);  view_20 = None
    key_states_6 = transpose_15.contiguous();  transpose_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_0_encoder_attn_v_proj = self.L__mod___model_decoder_layers_0_encoder_attn_v_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_21 = l__mod___model_decoder_layers_0_encoder_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_0_encoder_attn_v_proj = None
    transpose_16 = view_21.transpose(1, 2);  view_21 = None
    value_states_6 = transpose_16.contiguous();  transpose_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_22 = query_states_6.view(1, 128, 32, 80);  query_states_6 = None
    transpose_17 = view_22.transpose(1, 2);  view_22 = None
    contiguous_11 = transpose_17.contiguous();  transpose_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_7 = contiguous_11.view(32, -1, 80);  contiguous_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_7 = key_states_6.reshape(32, -1, 80);  key_states_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_7 = value_states_6.reshape(32, -1, 80);  value_states_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_18 = key_states_7.transpose(1, 2);  key_states_7 = None
    attn_weights_8 = torch.bmm(query_states_7, transpose_18);  query_states_7 = transpose_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_9 = torch.nn.functional.softmax(attn_weights_8, dim = -1);  attn_weights_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_3 = torch.nn.functional.dropout(attn_weights_9, p = 0.0, training = False);  attn_weights_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_15 = torch.bmm(attn_probs_3, value_states_7);  attn_probs_3 = value_states_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_16 = attn_output_15.view(1, 32, 128, 80);  attn_output_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_17 = attn_output_16.transpose(1, 2);  attn_output_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_18 = attn_output_17.reshape(1, 128, 2560);  attn_output_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_32 = self.L__mod___model_decoder_layers_0_encoder_attn_out_proj(attn_output_18);  attn_output_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:444, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_33 = torch.nn.functional.dropout(hidden_states_32, p = 0.1, training = False);  hidden_states_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    residual_6 = residual_5 + hidden_states_33;  residual_5 = hidden_states_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_35 = self.L__mod___model_decoder_layers_0_final_layer_norm(residual_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_0_fc1 = self.L__mod___model_decoder_layers_0_fc1(hidden_states_35);  hidden_states_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_36 = torch._C._nn.gelu(l__mod___model_decoder_layers_0_fc1);  l__mod___model_decoder_layers_0_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:454, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_37 = torch.nn.functional.dropout(hidden_states_36, p = 0.0, training = False);  hidden_states_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    hidden_states_38 = self.L__mod___model_decoder_layers_0_fc2(hidden_states_37);  hidden_states_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:456, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_39 = torch.nn.functional.dropout(hidden_states_38, p = 0.1, training = False);  hidden_states_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    residual_7 = residual_6 + hidden_states_39;  residual_6 = hidden_states_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_42 = self.L__mod___model_decoder_layers_1_self_attn_layer_norm(residual_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_1_self_attn_q_proj = self.L__mod___model_decoder_layers_1_self_attn_q_proj(hidden_states_42)
    query_states_8 = l__mod___model_decoder_layers_1_self_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_1_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_1_self_attn_k_proj = self.L__mod___model_decoder_layers_1_self_attn_k_proj(hidden_states_42)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_25 = l__mod___model_decoder_layers_1_self_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_1_self_attn_k_proj = None
    transpose_20 = view_25.transpose(1, 2);  view_25 = None
    key_states_8 = transpose_20.contiguous();  transpose_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_1_self_attn_v_proj = self.L__mod___model_decoder_layers_1_self_attn_v_proj(hidden_states_42);  hidden_states_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_26 = l__mod___model_decoder_layers_1_self_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_1_self_attn_v_proj = None
    transpose_21 = view_26.transpose(1, 2);  view_26 = None
    value_states_8 = transpose_21.contiguous();  transpose_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_27 = query_states_8.view(1, 128, 32, 80);  query_states_8 = None
    transpose_22 = view_27.transpose(1, 2);  view_27 = None
    contiguous_14 = transpose_22.contiguous();  transpose_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_9 = contiguous_14.view(32, -1, 80);  contiguous_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_9 = key_states_8.reshape(32, -1, 80);  key_states_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_9 = value_states_8.reshape(32, -1, 80);  value_states_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_23 = key_states_9.transpose(1, 2);  key_states_9 = None
    attn_weights_10 = torch.bmm(query_states_9, transpose_23);  query_states_9 = transpose_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_29 = attn_weights_10.view(1, 32, 128, 128);  attn_weights_10 = None
    attn_weights_11 = view_29 + attention_mask;  view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_12 = attn_weights_11.view(32, 128, 128);  attn_weights_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_13 = torch.nn.functional.softmax(attn_weights_12, dim = -1);  attn_weights_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_4 = torch.nn.functional.dropout(attn_weights_13, p = 0.0, training = False);  attn_weights_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_20 = torch.bmm(attn_probs_4, value_states_9);  attn_probs_4 = value_states_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_21 = attn_output_20.view(1, 32, 128, 80);  attn_output_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_22 = attn_output_21.transpose(1, 2);  attn_output_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_23 = attn_output_22.reshape(1, 128, 2560);  attn_output_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_43 = self.L__mod___model_decoder_layers_1_self_attn_out_proj(attn_output_23);  attn_output_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:424, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_44 = torch.nn.functional.dropout(hidden_states_43, p = 0.1, training = False);  hidden_states_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    residual_8 = residual_7 + hidden_states_44;  residual_7 = hidden_states_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_46 = self.L__mod___model_decoder_layers_1_encoder_attn_layer_norm(residual_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_1_encoder_attn_q_proj = self.L__mod___model_decoder_layers_1_encoder_attn_q_proj(hidden_states_46);  hidden_states_46 = None
    query_states_10 = l__mod___model_decoder_layers_1_encoder_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_1_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_1_encoder_attn_k_proj = self.L__mod___model_decoder_layers_1_encoder_attn_k_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_32 = l__mod___model_decoder_layers_1_encoder_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_1_encoder_attn_k_proj = None
    transpose_25 = view_32.transpose(1, 2);  view_32 = None
    key_states_10 = transpose_25.contiguous();  transpose_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_1_encoder_attn_v_proj = self.L__mod___model_decoder_layers_1_encoder_attn_v_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_33 = l__mod___model_decoder_layers_1_encoder_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_1_encoder_attn_v_proj = None
    transpose_26 = view_33.transpose(1, 2);  view_33 = None
    value_states_10 = transpose_26.contiguous();  transpose_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_34 = query_states_10.view(1, 128, 32, 80);  query_states_10 = None
    transpose_27 = view_34.transpose(1, 2);  view_34 = None
    contiguous_17 = transpose_27.contiguous();  transpose_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_11 = contiguous_17.view(32, -1, 80);  contiguous_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_11 = key_states_10.reshape(32, -1, 80);  key_states_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_11 = value_states_10.reshape(32, -1, 80);  value_states_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_28 = key_states_11.transpose(1, 2);  key_states_11 = None
    attn_weights_14 = torch.bmm(query_states_11, transpose_28);  query_states_11 = transpose_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_15 = torch.nn.functional.softmax(attn_weights_14, dim = -1);  attn_weights_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_5 = torch.nn.functional.dropout(attn_weights_15, p = 0.0, training = False);  attn_weights_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_25 = torch.bmm(attn_probs_5, value_states_11);  attn_probs_5 = value_states_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_26 = attn_output_25.view(1, 32, 128, 80);  attn_output_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_27 = attn_output_26.transpose(1, 2);  attn_output_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_28 = attn_output_27.reshape(1, 128, 2560);  attn_output_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_47 = self.L__mod___model_decoder_layers_1_encoder_attn_out_proj(attn_output_28);  attn_output_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:444, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_48 = torch.nn.functional.dropout(hidden_states_47, p = 0.1, training = False);  hidden_states_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    residual_9 = residual_8 + hidden_states_48;  residual_8 = hidden_states_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_50 = self.L__mod___model_decoder_layers_1_final_layer_norm(residual_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_1_fc1 = self.L__mod___model_decoder_layers_1_fc1(hidden_states_50);  hidden_states_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_51 = torch._C._nn.gelu(l__mod___model_decoder_layers_1_fc1);  l__mod___model_decoder_layers_1_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:454, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_52 = torch.nn.functional.dropout(hidden_states_51, p = 0.0, training = False);  hidden_states_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    hidden_states_53 = self.L__mod___model_decoder_layers_1_fc2(hidden_states_52);  hidden_states_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:456, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_54 = torch.nn.functional.dropout(hidden_states_53, p = 0.1, training = False);  hidden_states_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    residual_10 = residual_9 + hidden_states_54;  residual_9 = hidden_states_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_57 = self.L__mod___model_decoder_layers_2_self_attn_layer_norm(residual_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_2_self_attn_q_proj = self.L__mod___model_decoder_layers_2_self_attn_q_proj(hidden_states_57)
    query_states_12 = l__mod___model_decoder_layers_2_self_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_2_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_2_self_attn_k_proj = self.L__mod___model_decoder_layers_2_self_attn_k_proj(hidden_states_57)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_37 = l__mod___model_decoder_layers_2_self_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_2_self_attn_k_proj = None
    transpose_30 = view_37.transpose(1, 2);  view_37 = None
    key_states_12 = transpose_30.contiguous();  transpose_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_2_self_attn_v_proj = self.L__mod___model_decoder_layers_2_self_attn_v_proj(hidden_states_57);  hidden_states_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_38 = l__mod___model_decoder_layers_2_self_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_2_self_attn_v_proj = None
    transpose_31 = view_38.transpose(1, 2);  view_38 = None
    value_states_12 = transpose_31.contiguous();  transpose_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_39 = query_states_12.view(1, 128, 32, 80);  query_states_12 = None
    transpose_32 = view_39.transpose(1, 2);  view_39 = None
    contiguous_20 = transpose_32.contiguous();  transpose_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_13 = contiguous_20.view(32, -1, 80);  contiguous_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_13 = key_states_12.reshape(32, -1, 80);  key_states_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_13 = value_states_12.reshape(32, -1, 80);  value_states_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_33 = key_states_13.transpose(1, 2);  key_states_13 = None
    attn_weights_16 = torch.bmm(query_states_13, transpose_33);  query_states_13 = transpose_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_41 = attn_weights_16.view(1, 32, 128, 128);  attn_weights_16 = None
    attn_weights_17 = view_41 + attention_mask;  view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_18 = attn_weights_17.view(32, 128, 128);  attn_weights_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_19 = torch.nn.functional.softmax(attn_weights_18, dim = -1);  attn_weights_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_6 = torch.nn.functional.dropout(attn_weights_19, p = 0.0, training = False);  attn_weights_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_30 = torch.bmm(attn_probs_6, value_states_13);  attn_probs_6 = value_states_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_31 = attn_output_30.view(1, 32, 128, 80);  attn_output_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_32 = attn_output_31.transpose(1, 2);  attn_output_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_33 = attn_output_32.reshape(1, 128, 2560);  attn_output_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_58 = self.L__mod___model_decoder_layers_2_self_attn_out_proj(attn_output_33);  attn_output_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:424, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_59 = torch.nn.functional.dropout(hidden_states_58, p = 0.1, training = False);  hidden_states_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    residual_11 = residual_10 + hidden_states_59;  residual_10 = hidden_states_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_61 = self.L__mod___model_decoder_layers_2_encoder_attn_layer_norm(residual_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_2_encoder_attn_q_proj = self.L__mod___model_decoder_layers_2_encoder_attn_q_proj(hidden_states_61);  hidden_states_61 = None
    query_states_14 = l__mod___model_decoder_layers_2_encoder_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_2_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_2_encoder_attn_k_proj = self.L__mod___model_decoder_layers_2_encoder_attn_k_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_44 = l__mod___model_decoder_layers_2_encoder_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_2_encoder_attn_k_proj = None
    transpose_35 = view_44.transpose(1, 2);  view_44 = None
    key_states_14 = transpose_35.contiguous();  transpose_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_2_encoder_attn_v_proj = self.L__mod___model_decoder_layers_2_encoder_attn_v_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_45 = l__mod___model_decoder_layers_2_encoder_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_2_encoder_attn_v_proj = None
    transpose_36 = view_45.transpose(1, 2);  view_45 = None
    value_states_14 = transpose_36.contiguous();  transpose_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_46 = query_states_14.view(1, 128, 32, 80);  query_states_14 = None
    transpose_37 = view_46.transpose(1, 2);  view_46 = None
    contiguous_23 = transpose_37.contiguous();  transpose_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_15 = contiguous_23.view(32, -1, 80);  contiguous_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_15 = key_states_14.reshape(32, -1, 80);  key_states_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_15 = value_states_14.reshape(32, -1, 80);  value_states_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_38 = key_states_15.transpose(1, 2);  key_states_15 = None
    attn_weights_20 = torch.bmm(query_states_15, transpose_38);  query_states_15 = transpose_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_21 = torch.nn.functional.softmax(attn_weights_20, dim = -1);  attn_weights_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_7 = torch.nn.functional.dropout(attn_weights_21, p = 0.0, training = False);  attn_weights_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_35 = torch.bmm(attn_probs_7, value_states_15);  attn_probs_7 = value_states_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_36 = attn_output_35.view(1, 32, 128, 80);  attn_output_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_37 = attn_output_36.transpose(1, 2);  attn_output_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_38 = attn_output_37.reshape(1, 128, 2560);  attn_output_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_62 = self.L__mod___model_decoder_layers_2_encoder_attn_out_proj(attn_output_38);  attn_output_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:444, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_63 = torch.nn.functional.dropout(hidden_states_62, p = 0.1, training = False);  hidden_states_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    residual_12 = residual_11 + hidden_states_63;  residual_11 = hidden_states_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_65 = self.L__mod___model_decoder_layers_2_final_layer_norm(residual_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_2_fc1 = self.L__mod___model_decoder_layers_2_fc1(hidden_states_65);  hidden_states_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_66 = torch._C._nn.gelu(l__mod___model_decoder_layers_2_fc1);  l__mod___model_decoder_layers_2_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:454, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_67 = torch.nn.functional.dropout(hidden_states_66, p = 0.0, training = False);  hidden_states_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    hidden_states_68 = self.L__mod___model_decoder_layers_2_fc2(hidden_states_67);  hidden_states_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:456, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_69 = torch.nn.functional.dropout(hidden_states_68, p = 0.1, training = False);  hidden_states_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    residual_13 = residual_12 + hidden_states_69;  residual_12 = hidden_states_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_72 = self.L__mod___model_decoder_layers_3_self_attn_layer_norm(residual_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_3_self_attn_q_proj = self.L__mod___model_decoder_layers_3_self_attn_q_proj(hidden_states_72)
    query_states_16 = l__mod___model_decoder_layers_3_self_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_3_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_3_self_attn_k_proj = self.L__mod___model_decoder_layers_3_self_attn_k_proj(hidden_states_72)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_49 = l__mod___model_decoder_layers_3_self_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_3_self_attn_k_proj = None
    transpose_40 = view_49.transpose(1, 2);  view_49 = None
    key_states_16 = transpose_40.contiguous();  transpose_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_3_self_attn_v_proj = self.L__mod___model_decoder_layers_3_self_attn_v_proj(hidden_states_72);  hidden_states_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_50 = l__mod___model_decoder_layers_3_self_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_3_self_attn_v_proj = None
    transpose_41 = view_50.transpose(1, 2);  view_50 = None
    value_states_16 = transpose_41.contiguous();  transpose_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_51 = query_states_16.view(1, 128, 32, 80);  query_states_16 = None
    transpose_42 = view_51.transpose(1, 2);  view_51 = None
    contiguous_26 = transpose_42.contiguous();  transpose_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_17 = contiguous_26.view(32, -1, 80);  contiguous_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_17 = key_states_16.reshape(32, -1, 80);  key_states_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_17 = value_states_16.reshape(32, -1, 80);  value_states_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_43 = key_states_17.transpose(1, 2);  key_states_17 = None
    attn_weights_22 = torch.bmm(query_states_17, transpose_43);  query_states_17 = transpose_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_53 = attn_weights_22.view(1, 32, 128, 128);  attn_weights_22 = None
    attn_weights_23 = view_53 + attention_mask;  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_24 = attn_weights_23.view(32, 128, 128);  attn_weights_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_25 = torch.nn.functional.softmax(attn_weights_24, dim = -1);  attn_weights_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_8 = torch.nn.functional.dropout(attn_weights_25, p = 0.0, training = False);  attn_weights_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_40 = torch.bmm(attn_probs_8, value_states_17);  attn_probs_8 = value_states_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_41 = attn_output_40.view(1, 32, 128, 80);  attn_output_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_42 = attn_output_41.transpose(1, 2);  attn_output_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_43 = attn_output_42.reshape(1, 128, 2560);  attn_output_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_73 = self.L__mod___model_decoder_layers_3_self_attn_out_proj(attn_output_43);  attn_output_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:424, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_74 = torch.nn.functional.dropout(hidden_states_73, p = 0.1, training = False);  hidden_states_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    residual_14 = residual_13 + hidden_states_74;  residual_13 = hidden_states_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_76 = self.L__mod___model_decoder_layers_3_encoder_attn_layer_norm(residual_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_3_encoder_attn_q_proj = self.L__mod___model_decoder_layers_3_encoder_attn_q_proj(hidden_states_76);  hidden_states_76 = None
    query_states_18 = l__mod___model_decoder_layers_3_encoder_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_3_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_3_encoder_attn_k_proj = self.L__mod___model_decoder_layers_3_encoder_attn_k_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_56 = l__mod___model_decoder_layers_3_encoder_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_3_encoder_attn_k_proj = None
    transpose_45 = view_56.transpose(1, 2);  view_56 = None
    key_states_18 = transpose_45.contiguous();  transpose_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_3_encoder_attn_v_proj = self.L__mod___model_decoder_layers_3_encoder_attn_v_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_57 = l__mod___model_decoder_layers_3_encoder_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_3_encoder_attn_v_proj = None
    transpose_46 = view_57.transpose(1, 2);  view_57 = None
    value_states_18 = transpose_46.contiguous();  transpose_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_58 = query_states_18.view(1, 128, 32, 80);  query_states_18 = None
    transpose_47 = view_58.transpose(1, 2);  view_58 = None
    contiguous_29 = transpose_47.contiguous();  transpose_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_19 = contiguous_29.view(32, -1, 80);  contiguous_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_19 = key_states_18.reshape(32, -1, 80);  key_states_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_19 = value_states_18.reshape(32, -1, 80);  value_states_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_48 = key_states_19.transpose(1, 2);  key_states_19 = None
    attn_weights_26 = torch.bmm(query_states_19, transpose_48);  query_states_19 = transpose_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_27 = torch.nn.functional.softmax(attn_weights_26, dim = -1);  attn_weights_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_9 = torch.nn.functional.dropout(attn_weights_27, p = 0.0, training = False);  attn_weights_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_45 = torch.bmm(attn_probs_9, value_states_19);  attn_probs_9 = value_states_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_46 = attn_output_45.view(1, 32, 128, 80);  attn_output_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_47 = attn_output_46.transpose(1, 2);  attn_output_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_48 = attn_output_47.reshape(1, 128, 2560);  attn_output_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_77 = self.L__mod___model_decoder_layers_3_encoder_attn_out_proj(attn_output_48);  attn_output_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:444, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_78 = torch.nn.functional.dropout(hidden_states_77, p = 0.1, training = False);  hidden_states_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    residual_15 = residual_14 + hidden_states_78;  residual_14 = hidden_states_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_80 = self.L__mod___model_decoder_layers_3_final_layer_norm(residual_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_3_fc1 = self.L__mod___model_decoder_layers_3_fc1(hidden_states_80);  hidden_states_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_81 = torch._C._nn.gelu(l__mod___model_decoder_layers_3_fc1);  l__mod___model_decoder_layers_3_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:454, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_82 = torch.nn.functional.dropout(hidden_states_81, p = 0.0, training = False);  hidden_states_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    hidden_states_83 = self.L__mod___model_decoder_layers_3_fc2(hidden_states_82);  hidden_states_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:456, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_84 = torch.nn.functional.dropout(hidden_states_83, p = 0.1, training = False);  hidden_states_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    residual_16 = residual_15 + hidden_states_84;  residual_15 = hidden_states_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_87 = self.L__mod___model_decoder_layers_4_self_attn_layer_norm(residual_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_4_self_attn_q_proj = self.L__mod___model_decoder_layers_4_self_attn_q_proj(hidden_states_87)
    query_states_20 = l__mod___model_decoder_layers_4_self_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_4_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_4_self_attn_k_proj = self.L__mod___model_decoder_layers_4_self_attn_k_proj(hidden_states_87)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_61 = l__mod___model_decoder_layers_4_self_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_4_self_attn_k_proj = None
    transpose_50 = view_61.transpose(1, 2);  view_61 = None
    key_states_20 = transpose_50.contiguous();  transpose_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_4_self_attn_v_proj = self.L__mod___model_decoder_layers_4_self_attn_v_proj(hidden_states_87);  hidden_states_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_62 = l__mod___model_decoder_layers_4_self_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_4_self_attn_v_proj = None
    transpose_51 = view_62.transpose(1, 2);  view_62 = None
    value_states_20 = transpose_51.contiguous();  transpose_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_63 = query_states_20.view(1, 128, 32, 80);  query_states_20 = None
    transpose_52 = view_63.transpose(1, 2);  view_63 = None
    contiguous_32 = transpose_52.contiguous();  transpose_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_21 = contiguous_32.view(32, -1, 80);  contiguous_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_21 = key_states_20.reshape(32, -1, 80);  key_states_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_21 = value_states_20.reshape(32, -1, 80);  value_states_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_53 = key_states_21.transpose(1, 2);  key_states_21 = None
    attn_weights_28 = torch.bmm(query_states_21, transpose_53);  query_states_21 = transpose_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_65 = attn_weights_28.view(1, 32, 128, 128);  attn_weights_28 = None
    attn_weights_29 = view_65 + attention_mask;  view_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_30 = attn_weights_29.view(32, 128, 128);  attn_weights_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_31 = torch.nn.functional.softmax(attn_weights_30, dim = -1);  attn_weights_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_10 = torch.nn.functional.dropout(attn_weights_31, p = 0.0, training = False);  attn_weights_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_50 = torch.bmm(attn_probs_10, value_states_21);  attn_probs_10 = value_states_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_51 = attn_output_50.view(1, 32, 128, 80);  attn_output_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_52 = attn_output_51.transpose(1, 2);  attn_output_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_53 = attn_output_52.reshape(1, 128, 2560);  attn_output_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_88 = self.L__mod___model_decoder_layers_4_self_attn_out_proj(attn_output_53);  attn_output_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:424, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_89 = torch.nn.functional.dropout(hidden_states_88, p = 0.1, training = False);  hidden_states_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    residual_17 = residual_16 + hidden_states_89;  residual_16 = hidden_states_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_91 = self.L__mod___model_decoder_layers_4_encoder_attn_layer_norm(residual_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_4_encoder_attn_q_proj = self.L__mod___model_decoder_layers_4_encoder_attn_q_proj(hidden_states_91);  hidden_states_91 = None
    query_states_22 = l__mod___model_decoder_layers_4_encoder_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_4_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_4_encoder_attn_k_proj = self.L__mod___model_decoder_layers_4_encoder_attn_k_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_68 = l__mod___model_decoder_layers_4_encoder_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_4_encoder_attn_k_proj = None
    transpose_55 = view_68.transpose(1, 2);  view_68 = None
    key_states_22 = transpose_55.contiguous();  transpose_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_4_encoder_attn_v_proj = self.L__mod___model_decoder_layers_4_encoder_attn_v_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_69 = l__mod___model_decoder_layers_4_encoder_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_4_encoder_attn_v_proj = None
    transpose_56 = view_69.transpose(1, 2);  view_69 = None
    value_states_22 = transpose_56.contiguous();  transpose_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_70 = query_states_22.view(1, 128, 32, 80);  query_states_22 = None
    transpose_57 = view_70.transpose(1, 2);  view_70 = None
    contiguous_35 = transpose_57.contiguous();  transpose_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_23 = contiguous_35.view(32, -1, 80);  contiguous_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_23 = key_states_22.reshape(32, -1, 80);  key_states_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_23 = value_states_22.reshape(32, -1, 80);  value_states_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_58 = key_states_23.transpose(1, 2);  key_states_23 = None
    attn_weights_32 = torch.bmm(query_states_23, transpose_58);  query_states_23 = transpose_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_33 = torch.nn.functional.softmax(attn_weights_32, dim = -1);  attn_weights_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_11 = torch.nn.functional.dropout(attn_weights_33, p = 0.0, training = False);  attn_weights_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_55 = torch.bmm(attn_probs_11, value_states_23);  attn_probs_11 = value_states_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_56 = attn_output_55.view(1, 32, 128, 80);  attn_output_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_57 = attn_output_56.transpose(1, 2);  attn_output_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_58 = attn_output_57.reshape(1, 128, 2560);  attn_output_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_92 = self.L__mod___model_decoder_layers_4_encoder_attn_out_proj(attn_output_58);  attn_output_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:444, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_93 = torch.nn.functional.dropout(hidden_states_92, p = 0.1, training = False);  hidden_states_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    residual_18 = residual_17 + hidden_states_93;  residual_17 = hidden_states_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_95 = self.L__mod___model_decoder_layers_4_final_layer_norm(residual_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_4_fc1 = self.L__mod___model_decoder_layers_4_fc1(hidden_states_95);  hidden_states_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_96 = torch._C._nn.gelu(l__mod___model_decoder_layers_4_fc1);  l__mod___model_decoder_layers_4_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:454, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_97 = torch.nn.functional.dropout(hidden_states_96, p = 0.0, training = False);  hidden_states_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    hidden_states_98 = self.L__mod___model_decoder_layers_4_fc2(hidden_states_97);  hidden_states_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:456, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_99 = torch.nn.functional.dropout(hidden_states_98, p = 0.1, training = False);  hidden_states_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    residual_19 = residual_18 + hidden_states_99;  residual_18 = hidden_states_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_102 = self.L__mod___model_decoder_layers_5_self_attn_layer_norm(residual_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_5_self_attn_q_proj = self.L__mod___model_decoder_layers_5_self_attn_q_proj(hidden_states_102)
    query_states_24 = l__mod___model_decoder_layers_5_self_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_5_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_5_self_attn_k_proj = self.L__mod___model_decoder_layers_5_self_attn_k_proj(hidden_states_102)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_73 = l__mod___model_decoder_layers_5_self_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_5_self_attn_k_proj = None
    transpose_60 = view_73.transpose(1, 2);  view_73 = None
    key_states_24 = transpose_60.contiguous();  transpose_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_5_self_attn_v_proj = self.L__mod___model_decoder_layers_5_self_attn_v_proj(hidden_states_102);  hidden_states_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_74 = l__mod___model_decoder_layers_5_self_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_5_self_attn_v_proj = None
    transpose_61 = view_74.transpose(1, 2);  view_74 = None
    value_states_24 = transpose_61.contiguous();  transpose_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_75 = query_states_24.view(1, 128, 32, 80);  query_states_24 = None
    transpose_62 = view_75.transpose(1, 2);  view_75 = None
    contiguous_38 = transpose_62.contiguous();  transpose_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_25 = contiguous_38.view(32, -1, 80);  contiguous_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_25 = key_states_24.reshape(32, -1, 80);  key_states_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_25 = value_states_24.reshape(32, -1, 80);  value_states_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_63 = key_states_25.transpose(1, 2);  key_states_25 = None
    attn_weights_34 = torch.bmm(query_states_25, transpose_63);  query_states_25 = transpose_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_77 = attn_weights_34.view(1, 32, 128, 128);  attn_weights_34 = None
    attn_weights_35 = view_77 + attention_mask;  view_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_36 = attn_weights_35.view(32, 128, 128);  attn_weights_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_37 = torch.nn.functional.softmax(attn_weights_36, dim = -1);  attn_weights_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_12 = torch.nn.functional.dropout(attn_weights_37, p = 0.0, training = False);  attn_weights_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_60 = torch.bmm(attn_probs_12, value_states_25);  attn_probs_12 = value_states_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_61 = attn_output_60.view(1, 32, 128, 80);  attn_output_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_62 = attn_output_61.transpose(1, 2);  attn_output_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_63 = attn_output_62.reshape(1, 128, 2560);  attn_output_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_103 = self.L__mod___model_decoder_layers_5_self_attn_out_proj(attn_output_63);  attn_output_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:424, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_104 = torch.nn.functional.dropout(hidden_states_103, p = 0.1, training = False);  hidden_states_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    residual_20 = residual_19 + hidden_states_104;  residual_19 = hidden_states_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_106 = self.L__mod___model_decoder_layers_5_encoder_attn_layer_norm(residual_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_5_encoder_attn_q_proj = self.L__mod___model_decoder_layers_5_encoder_attn_q_proj(hidden_states_106);  hidden_states_106 = None
    query_states_26 = l__mod___model_decoder_layers_5_encoder_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_5_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_5_encoder_attn_k_proj = self.L__mod___model_decoder_layers_5_encoder_attn_k_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_80 = l__mod___model_decoder_layers_5_encoder_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_5_encoder_attn_k_proj = None
    transpose_65 = view_80.transpose(1, 2);  view_80 = None
    key_states_26 = transpose_65.contiguous();  transpose_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_5_encoder_attn_v_proj = self.L__mod___model_decoder_layers_5_encoder_attn_v_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_81 = l__mod___model_decoder_layers_5_encoder_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_5_encoder_attn_v_proj = None
    transpose_66 = view_81.transpose(1, 2);  view_81 = None
    value_states_26 = transpose_66.contiguous();  transpose_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_82 = query_states_26.view(1, 128, 32, 80);  query_states_26 = None
    transpose_67 = view_82.transpose(1, 2);  view_82 = None
    contiguous_41 = transpose_67.contiguous();  transpose_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_27 = contiguous_41.view(32, -1, 80);  contiguous_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_27 = key_states_26.reshape(32, -1, 80);  key_states_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_27 = value_states_26.reshape(32, -1, 80);  value_states_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_68 = key_states_27.transpose(1, 2);  key_states_27 = None
    attn_weights_38 = torch.bmm(query_states_27, transpose_68);  query_states_27 = transpose_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_39 = torch.nn.functional.softmax(attn_weights_38, dim = -1);  attn_weights_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_13 = torch.nn.functional.dropout(attn_weights_39, p = 0.0, training = False);  attn_weights_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_65 = torch.bmm(attn_probs_13, value_states_27);  attn_probs_13 = value_states_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_66 = attn_output_65.view(1, 32, 128, 80);  attn_output_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_67 = attn_output_66.transpose(1, 2);  attn_output_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_68 = attn_output_67.reshape(1, 128, 2560);  attn_output_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_107 = self.L__mod___model_decoder_layers_5_encoder_attn_out_proj(attn_output_68);  attn_output_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:444, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_108 = torch.nn.functional.dropout(hidden_states_107, p = 0.1, training = False);  hidden_states_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    residual_21 = residual_20 + hidden_states_108;  residual_20 = hidden_states_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_110 = self.L__mod___model_decoder_layers_5_final_layer_norm(residual_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_5_fc1 = self.L__mod___model_decoder_layers_5_fc1(hidden_states_110);  hidden_states_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_111 = torch._C._nn.gelu(l__mod___model_decoder_layers_5_fc1);  l__mod___model_decoder_layers_5_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:454, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_112 = torch.nn.functional.dropout(hidden_states_111, p = 0.0, training = False);  hidden_states_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    hidden_states_113 = self.L__mod___model_decoder_layers_5_fc2(hidden_states_112);  hidden_states_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:456, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_114 = torch.nn.functional.dropout(hidden_states_113, p = 0.1, training = False);  hidden_states_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    residual_22 = residual_21 + hidden_states_114;  residual_21 = hidden_states_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_117 = self.L__mod___model_decoder_layers_6_self_attn_layer_norm(residual_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_6_self_attn_q_proj = self.L__mod___model_decoder_layers_6_self_attn_q_proj(hidden_states_117)
    query_states_28 = l__mod___model_decoder_layers_6_self_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_6_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_6_self_attn_k_proj = self.L__mod___model_decoder_layers_6_self_attn_k_proj(hidden_states_117)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_85 = l__mod___model_decoder_layers_6_self_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_6_self_attn_k_proj = None
    transpose_70 = view_85.transpose(1, 2);  view_85 = None
    key_states_28 = transpose_70.contiguous();  transpose_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_6_self_attn_v_proj = self.L__mod___model_decoder_layers_6_self_attn_v_proj(hidden_states_117);  hidden_states_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_86 = l__mod___model_decoder_layers_6_self_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_6_self_attn_v_proj = None
    transpose_71 = view_86.transpose(1, 2);  view_86 = None
    value_states_28 = transpose_71.contiguous();  transpose_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_87 = query_states_28.view(1, 128, 32, 80);  query_states_28 = None
    transpose_72 = view_87.transpose(1, 2);  view_87 = None
    contiguous_44 = transpose_72.contiguous();  transpose_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_29 = contiguous_44.view(32, -1, 80);  contiguous_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_29 = key_states_28.reshape(32, -1, 80);  key_states_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_29 = value_states_28.reshape(32, -1, 80);  value_states_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_73 = key_states_29.transpose(1, 2);  key_states_29 = None
    attn_weights_40 = torch.bmm(query_states_29, transpose_73);  query_states_29 = transpose_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_89 = attn_weights_40.view(1, 32, 128, 128);  attn_weights_40 = None
    attn_weights_41 = view_89 + attention_mask;  view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_42 = attn_weights_41.view(32, 128, 128);  attn_weights_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_43 = torch.nn.functional.softmax(attn_weights_42, dim = -1);  attn_weights_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_14 = torch.nn.functional.dropout(attn_weights_43, p = 0.0, training = False);  attn_weights_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_70 = torch.bmm(attn_probs_14, value_states_29);  attn_probs_14 = value_states_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_71 = attn_output_70.view(1, 32, 128, 80);  attn_output_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_72 = attn_output_71.transpose(1, 2);  attn_output_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_73 = attn_output_72.reshape(1, 128, 2560);  attn_output_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_118 = self.L__mod___model_decoder_layers_6_self_attn_out_proj(attn_output_73);  attn_output_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:424, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_119 = torch.nn.functional.dropout(hidden_states_118, p = 0.1, training = False);  hidden_states_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    residual_23 = residual_22 + hidden_states_119;  residual_22 = hidden_states_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_121 = self.L__mod___model_decoder_layers_6_encoder_attn_layer_norm(residual_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_6_encoder_attn_q_proj = self.L__mod___model_decoder_layers_6_encoder_attn_q_proj(hidden_states_121);  hidden_states_121 = None
    query_states_30 = l__mod___model_decoder_layers_6_encoder_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_6_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_6_encoder_attn_k_proj = self.L__mod___model_decoder_layers_6_encoder_attn_k_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_92 = l__mod___model_decoder_layers_6_encoder_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_6_encoder_attn_k_proj = None
    transpose_75 = view_92.transpose(1, 2);  view_92 = None
    key_states_30 = transpose_75.contiguous();  transpose_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_6_encoder_attn_v_proj = self.L__mod___model_decoder_layers_6_encoder_attn_v_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_93 = l__mod___model_decoder_layers_6_encoder_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_6_encoder_attn_v_proj = None
    transpose_76 = view_93.transpose(1, 2);  view_93 = None
    value_states_30 = transpose_76.contiguous();  transpose_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_94 = query_states_30.view(1, 128, 32, 80);  query_states_30 = None
    transpose_77 = view_94.transpose(1, 2);  view_94 = None
    contiguous_47 = transpose_77.contiguous();  transpose_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_31 = contiguous_47.view(32, -1, 80);  contiguous_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_31 = key_states_30.reshape(32, -1, 80);  key_states_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_31 = value_states_30.reshape(32, -1, 80);  value_states_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_78 = key_states_31.transpose(1, 2);  key_states_31 = None
    attn_weights_44 = torch.bmm(query_states_31, transpose_78);  query_states_31 = transpose_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_45 = torch.nn.functional.softmax(attn_weights_44, dim = -1);  attn_weights_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_15 = torch.nn.functional.dropout(attn_weights_45, p = 0.0, training = False);  attn_weights_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_75 = torch.bmm(attn_probs_15, value_states_31);  attn_probs_15 = value_states_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_76 = attn_output_75.view(1, 32, 128, 80);  attn_output_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_77 = attn_output_76.transpose(1, 2);  attn_output_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_78 = attn_output_77.reshape(1, 128, 2560);  attn_output_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_122 = self.L__mod___model_decoder_layers_6_encoder_attn_out_proj(attn_output_78);  attn_output_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:444, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_123 = torch.nn.functional.dropout(hidden_states_122, p = 0.1, training = False);  hidden_states_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    residual_24 = residual_23 + hidden_states_123;  residual_23 = hidden_states_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_125 = self.L__mod___model_decoder_layers_6_final_layer_norm(residual_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_6_fc1 = self.L__mod___model_decoder_layers_6_fc1(hidden_states_125);  hidden_states_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_126 = torch._C._nn.gelu(l__mod___model_decoder_layers_6_fc1);  l__mod___model_decoder_layers_6_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:454, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_127 = torch.nn.functional.dropout(hidden_states_126, p = 0.0, training = False);  hidden_states_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    hidden_states_128 = self.L__mod___model_decoder_layers_6_fc2(hidden_states_127);  hidden_states_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:456, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_129 = torch.nn.functional.dropout(hidden_states_128, p = 0.1, training = False);  hidden_states_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    residual_25 = residual_24 + hidden_states_129;  residual_24 = hidden_states_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_132 = self.L__mod___model_decoder_layers_7_self_attn_layer_norm(residual_25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_7_self_attn_q_proj = self.L__mod___model_decoder_layers_7_self_attn_q_proj(hidden_states_132)
    query_states_32 = l__mod___model_decoder_layers_7_self_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_7_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_7_self_attn_k_proj = self.L__mod___model_decoder_layers_7_self_attn_k_proj(hidden_states_132)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_97 = l__mod___model_decoder_layers_7_self_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_7_self_attn_k_proj = None
    transpose_80 = view_97.transpose(1, 2);  view_97 = None
    key_states_32 = transpose_80.contiguous();  transpose_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_7_self_attn_v_proj = self.L__mod___model_decoder_layers_7_self_attn_v_proj(hidden_states_132);  hidden_states_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_98 = l__mod___model_decoder_layers_7_self_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_7_self_attn_v_proj = None
    transpose_81 = view_98.transpose(1, 2);  view_98 = None
    value_states_32 = transpose_81.contiguous();  transpose_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_99 = query_states_32.view(1, 128, 32, 80);  query_states_32 = None
    transpose_82 = view_99.transpose(1, 2);  view_99 = None
    contiguous_50 = transpose_82.contiguous();  transpose_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_33 = contiguous_50.view(32, -1, 80);  contiguous_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_33 = key_states_32.reshape(32, -1, 80);  key_states_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_33 = value_states_32.reshape(32, -1, 80);  value_states_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_83 = key_states_33.transpose(1, 2);  key_states_33 = None
    attn_weights_46 = torch.bmm(query_states_33, transpose_83);  query_states_33 = transpose_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_101 = attn_weights_46.view(1, 32, 128, 128);  attn_weights_46 = None
    attn_weights_47 = view_101 + attention_mask;  view_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_48 = attn_weights_47.view(32, 128, 128);  attn_weights_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_49 = torch.nn.functional.softmax(attn_weights_48, dim = -1);  attn_weights_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_16 = torch.nn.functional.dropout(attn_weights_49, p = 0.0, training = False);  attn_weights_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_80 = torch.bmm(attn_probs_16, value_states_33);  attn_probs_16 = value_states_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_81 = attn_output_80.view(1, 32, 128, 80);  attn_output_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_82 = attn_output_81.transpose(1, 2);  attn_output_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_83 = attn_output_82.reshape(1, 128, 2560);  attn_output_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_133 = self.L__mod___model_decoder_layers_7_self_attn_out_proj(attn_output_83);  attn_output_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:424, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_134 = torch.nn.functional.dropout(hidden_states_133, p = 0.1, training = False);  hidden_states_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    residual_26 = residual_25 + hidden_states_134;  residual_25 = hidden_states_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_136 = self.L__mod___model_decoder_layers_7_encoder_attn_layer_norm(residual_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_7_encoder_attn_q_proj = self.L__mod___model_decoder_layers_7_encoder_attn_q_proj(hidden_states_136);  hidden_states_136 = None
    query_states_34 = l__mod___model_decoder_layers_7_encoder_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_7_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_7_encoder_attn_k_proj = self.L__mod___model_decoder_layers_7_encoder_attn_k_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_104 = l__mod___model_decoder_layers_7_encoder_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_7_encoder_attn_k_proj = None
    transpose_85 = view_104.transpose(1, 2);  view_104 = None
    key_states_34 = transpose_85.contiguous();  transpose_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_7_encoder_attn_v_proj = self.L__mod___model_decoder_layers_7_encoder_attn_v_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_105 = l__mod___model_decoder_layers_7_encoder_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_7_encoder_attn_v_proj = None
    transpose_86 = view_105.transpose(1, 2);  view_105 = None
    value_states_34 = transpose_86.contiguous();  transpose_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_106 = query_states_34.view(1, 128, 32, 80);  query_states_34 = None
    transpose_87 = view_106.transpose(1, 2);  view_106 = None
    contiguous_53 = transpose_87.contiguous();  transpose_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_35 = contiguous_53.view(32, -1, 80);  contiguous_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_35 = key_states_34.reshape(32, -1, 80);  key_states_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_35 = value_states_34.reshape(32, -1, 80);  value_states_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_88 = key_states_35.transpose(1, 2);  key_states_35 = None
    attn_weights_50 = torch.bmm(query_states_35, transpose_88);  query_states_35 = transpose_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_51 = torch.nn.functional.softmax(attn_weights_50, dim = -1);  attn_weights_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_17 = torch.nn.functional.dropout(attn_weights_51, p = 0.0, training = False);  attn_weights_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_85 = torch.bmm(attn_probs_17, value_states_35);  attn_probs_17 = value_states_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_86 = attn_output_85.view(1, 32, 128, 80);  attn_output_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_87 = attn_output_86.transpose(1, 2);  attn_output_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_88 = attn_output_87.reshape(1, 128, 2560);  attn_output_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_137 = self.L__mod___model_decoder_layers_7_encoder_attn_out_proj(attn_output_88);  attn_output_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:444, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_138 = torch.nn.functional.dropout(hidden_states_137, p = 0.1, training = False);  hidden_states_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    residual_27 = residual_26 + hidden_states_138;  residual_26 = hidden_states_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_140 = self.L__mod___model_decoder_layers_7_final_layer_norm(residual_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_7_fc1 = self.L__mod___model_decoder_layers_7_fc1(hidden_states_140);  hidden_states_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_141 = torch._C._nn.gelu(l__mod___model_decoder_layers_7_fc1);  l__mod___model_decoder_layers_7_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:454, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_142 = torch.nn.functional.dropout(hidden_states_141, p = 0.0, training = False);  hidden_states_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    hidden_states_143 = self.L__mod___model_decoder_layers_7_fc2(hidden_states_142);  hidden_states_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:456, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_144 = torch.nn.functional.dropout(hidden_states_143, p = 0.1, training = False);  hidden_states_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    residual_28 = residual_27 + hidden_states_144;  residual_27 = hidden_states_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_147 = self.L__mod___model_decoder_layers_8_self_attn_layer_norm(residual_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_8_self_attn_q_proj = self.L__mod___model_decoder_layers_8_self_attn_q_proj(hidden_states_147)
    query_states_36 = l__mod___model_decoder_layers_8_self_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_8_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_8_self_attn_k_proj = self.L__mod___model_decoder_layers_8_self_attn_k_proj(hidden_states_147)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_109 = l__mod___model_decoder_layers_8_self_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_8_self_attn_k_proj = None
    transpose_90 = view_109.transpose(1, 2);  view_109 = None
    key_states_36 = transpose_90.contiguous();  transpose_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_8_self_attn_v_proj = self.L__mod___model_decoder_layers_8_self_attn_v_proj(hidden_states_147);  hidden_states_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_110 = l__mod___model_decoder_layers_8_self_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_8_self_attn_v_proj = None
    transpose_91 = view_110.transpose(1, 2);  view_110 = None
    value_states_36 = transpose_91.contiguous();  transpose_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_111 = query_states_36.view(1, 128, 32, 80);  query_states_36 = None
    transpose_92 = view_111.transpose(1, 2);  view_111 = None
    contiguous_56 = transpose_92.contiguous();  transpose_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_37 = contiguous_56.view(32, -1, 80);  contiguous_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_37 = key_states_36.reshape(32, -1, 80);  key_states_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_37 = value_states_36.reshape(32, -1, 80);  value_states_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_93 = key_states_37.transpose(1, 2);  key_states_37 = None
    attn_weights_52 = torch.bmm(query_states_37, transpose_93);  query_states_37 = transpose_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_113 = attn_weights_52.view(1, 32, 128, 128);  attn_weights_52 = None
    attn_weights_53 = view_113 + attention_mask;  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_54 = attn_weights_53.view(32, 128, 128);  attn_weights_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_55 = torch.nn.functional.softmax(attn_weights_54, dim = -1);  attn_weights_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_18 = torch.nn.functional.dropout(attn_weights_55, p = 0.0, training = False);  attn_weights_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_90 = torch.bmm(attn_probs_18, value_states_37);  attn_probs_18 = value_states_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_91 = attn_output_90.view(1, 32, 128, 80);  attn_output_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_92 = attn_output_91.transpose(1, 2);  attn_output_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_93 = attn_output_92.reshape(1, 128, 2560);  attn_output_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_148 = self.L__mod___model_decoder_layers_8_self_attn_out_proj(attn_output_93);  attn_output_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:424, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_149 = torch.nn.functional.dropout(hidden_states_148, p = 0.1, training = False);  hidden_states_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    residual_29 = residual_28 + hidden_states_149;  residual_28 = hidden_states_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_151 = self.L__mod___model_decoder_layers_8_encoder_attn_layer_norm(residual_29)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_8_encoder_attn_q_proj = self.L__mod___model_decoder_layers_8_encoder_attn_q_proj(hidden_states_151);  hidden_states_151 = None
    query_states_38 = l__mod___model_decoder_layers_8_encoder_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_8_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_8_encoder_attn_k_proj = self.L__mod___model_decoder_layers_8_encoder_attn_k_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_116 = l__mod___model_decoder_layers_8_encoder_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_8_encoder_attn_k_proj = None
    transpose_95 = view_116.transpose(1, 2);  view_116 = None
    key_states_38 = transpose_95.contiguous();  transpose_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_8_encoder_attn_v_proj = self.L__mod___model_decoder_layers_8_encoder_attn_v_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_117 = l__mod___model_decoder_layers_8_encoder_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_8_encoder_attn_v_proj = None
    transpose_96 = view_117.transpose(1, 2);  view_117 = None
    value_states_38 = transpose_96.contiguous();  transpose_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_118 = query_states_38.view(1, 128, 32, 80);  query_states_38 = None
    transpose_97 = view_118.transpose(1, 2);  view_118 = None
    contiguous_59 = transpose_97.contiguous();  transpose_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_39 = contiguous_59.view(32, -1, 80);  contiguous_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_39 = key_states_38.reshape(32, -1, 80);  key_states_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_39 = value_states_38.reshape(32, -1, 80);  value_states_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_98 = key_states_39.transpose(1, 2);  key_states_39 = None
    attn_weights_56 = torch.bmm(query_states_39, transpose_98);  query_states_39 = transpose_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_57 = torch.nn.functional.softmax(attn_weights_56, dim = -1);  attn_weights_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_19 = torch.nn.functional.dropout(attn_weights_57, p = 0.0, training = False);  attn_weights_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_95 = torch.bmm(attn_probs_19, value_states_39);  attn_probs_19 = value_states_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_96 = attn_output_95.view(1, 32, 128, 80);  attn_output_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_97 = attn_output_96.transpose(1, 2);  attn_output_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_98 = attn_output_97.reshape(1, 128, 2560);  attn_output_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_152 = self.L__mod___model_decoder_layers_8_encoder_attn_out_proj(attn_output_98);  attn_output_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:444, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_153 = torch.nn.functional.dropout(hidden_states_152, p = 0.1, training = False);  hidden_states_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    residual_30 = residual_29 + hidden_states_153;  residual_29 = hidden_states_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_155 = self.L__mod___model_decoder_layers_8_final_layer_norm(residual_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_8_fc1 = self.L__mod___model_decoder_layers_8_fc1(hidden_states_155);  hidden_states_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_156 = torch._C._nn.gelu(l__mod___model_decoder_layers_8_fc1);  l__mod___model_decoder_layers_8_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:454, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_157 = torch.nn.functional.dropout(hidden_states_156, p = 0.0, training = False);  hidden_states_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    hidden_states_158 = self.L__mod___model_decoder_layers_8_fc2(hidden_states_157);  hidden_states_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:456, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_159 = torch.nn.functional.dropout(hidden_states_158, p = 0.1, training = False);  hidden_states_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    residual_31 = residual_30 + hidden_states_159;  residual_30 = hidden_states_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_162 = self.L__mod___model_decoder_layers_9_self_attn_layer_norm(residual_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_9_self_attn_q_proj = self.L__mod___model_decoder_layers_9_self_attn_q_proj(hidden_states_162)
    query_states_40 = l__mod___model_decoder_layers_9_self_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_9_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_9_self_attn_k_proj = self.L__mod___model_decoder_layers_9_self_attn_k_proj(hidden_states_162)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_121 = l__mod___model_decoder_layers_9_self_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_9_self_attn_k_proj = None
    transpose_100 = view_121.transpose(1, 2);  view_121 = None
    key_states_40 = transpose_100.contiguous();  transpose_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_9_self_attn_v_proj = self.L__mod___model_decoder_layers_9_self_attn_v_proj(hidden_states_162);  hidden_states_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_122 = l__mod___model_decoder_layers_9_self_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_9_self_attn_v_proj = None
    transpose_101 = view_122.transpose(1, 2);  view_122 = None
    value_states_40 = transpose_101.contiguous();  transpose_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_123 = query_states_40.view(1, 128, 32, 80);  query_states_40 = None
    transpose_102 = view_123.transpose(1, 2);  view_123 = None
    contiguous_62 = transpose_102.contiguous();  transpose_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_41 = contiguous_62.view(32, -1, 80);  contiguous_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_41 = key_states_40.reshape(32, -1, 80);  key_states_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_41 = value_states_40.reshape(32, -1, 80);  value_states_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_103 = key_states_41.transpose(1, 2);  key_states_41 = None
    attn_weights_58 = torch.bmm(query_states_41, transpose_103);  query_states_41 = transpose_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_125 = attn_weights_58.view(1, 32, 128, 128);  attn_weights_58 = None
    attn_weights_59 = view_125 + attention_mask;  view_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_60 = attn_weights_59.view(32, 128, 128);  attn_weights_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_61 = torch.nn.functional.softmax(attn_weights_60, dim = -1);  attn_weights_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_20 = torch.nn.functional.dropout(attn_weights_61, p = 0.0, training = False);  attn_weights_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_100 = torch.bmm(attn_probs_20, value_states_41);  attn_probs_20 = value_states_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_101 = attn_output_100.view(1, 32, 128, 80);  attn_output_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_102 = attn_output_101.transpose(1, 2);  attn_output_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_103 = attn_output_102.reshape(1, 128, 2560);  attn_output_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_163 = self.L__mod___model_decoder_layers_9_self_attn_out_proj(attn_output_103);  attn_output_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:424, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_164 = torch.nn.functional.dropout(hidden_states_163, p = 0.1, training = False);  hidden_states_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    residual_32 = residual_31 + hidden_states_164;  residual_31 = hidden_states_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_166 = self.L__mod___model_decoder_layers_9_encoder_attn_layer_norm(residual_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_9_encoder_attn_q_proj = self.L__mod___model_decoder_layers_9_encoder_attn_q_proj(hidden_states_166);  hidden_states_166 = None
    query_states_42 = l__mod___model_decoder_layers_9_encoder_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_9_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_9_encoder_attn_k_proj = self.L__mod___model_decoder_layers_9_encoder_attn_k_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_128 = l__mod___model_decoder_layers_9_encoder_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_9_encoder_attn_k_proj = None
    transpose_105 = view_128.transpose(1, 2);  view_128 = None
    key_states_42 = transpose_105.contiguous();  transpose_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_9_encoder_attn_v_proj = self.L__mod___model_decoder_layers_9_encoder_attn_v_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_129 = l__mod___model_decoder_layers_9_encoder_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_9_encoder_attn_v_proj = None
    transpose_106 = view_129.transpose(1, 2);  view_129 = None
    value_states_42 = transpose_106.contiguous();  transpose_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_130 = query_states_42.view(1, 128, 32, 80);  query_states_42 = None
    transpose_107 = view_130.transpose(1, 2);  view_130 = None
    contiguous_65 = transpose_107.contiguous();  transpose_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_43 = contiguous_65.view(32, -1, 80);  contiguous_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_43 = key_states_42.reshape(32, -1, 80);  key_states_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_43 = value_states_42.reshape(32, -1, 80);  value_states_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_108 = key_states_43.transpose(1, 2);  key_states_43 = None
    attn_weights_62 = torch.bmm(query_states_43, transpose_108);  query_states_43 = transpose_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_63 = torch.nn.functional.softmax(attn_weights_62, dim = -1);  attn_weights_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_21 = torch.nn.functional.dropout(attn_weights_63, p = 0.0, training = False);  attn_weights_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_105 = torch.bmm(attn_probs_21, value_states_43);  attn_probs_21 = value_states_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_106 = attn_output_105.view(1, 32, 128, 80);  attn_output_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_107 = attn_output_106.transpose(1, 2);  attn_output_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_108 = attn_output_107.reshape(1, 128, 2560);  attn_output_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_167 = self.L__mod___model_decoder_layers_9_encoder_attn_out_proj(attn_output_108);  attn_output_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:444, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_168 = torch.nn.functional.dropout(hidden_states_167, p = 0.1, training = False);  hidden_states_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    residual_33 = residual_32 + hidden_states_168;  residual_32 = hidden_states_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_170 = self.L__mod___model_decoder_layers_9_final_layer_norm(residual_33)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_9_fc1 = self.L__mod___model_decoder_layers_9_fc1(hidden_states_170);  hidden_states_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_171 = torch._C._nn.gelu(l__mod___model_decoder_layers_9_fc1);  l__mod___model_decoder_layers_9_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:454, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_172 = torch.nn.functional.dropout(hidden_states_171, p = 0.0, training = False);  hidden_states_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    hidden_states_173 = self.L__mod___model_decoder_layers_9_fc2(hidden_states_172);  hidden_states_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:456, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_174 = torch.nn.functional.dropout(hidden_states_173, p = 0.1, training = False);  hidden_states_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    residual_34 = residual_33 + hidden_states_174;  residual_33 = hidden_states_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_177 = self.L__mod___model_decoder_layers_10_self_attn_layer_norm(residual_34)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_10_self_attn_q_proj = self.L__mod___model_decoder_layers_10_self_attn_q_proj(hidden_states_177)
    query_states_44 = l__mod___model_decoder_layers_10_self_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_10_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_10_self_attn_k_proj = self.L__mod___model_decoder_layers_10_self_attn_k_proj(hidden_states_177)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_133 = l__mod___model_decoder_layers_10_self_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_10_self_attn_k_proj = None
    transpose_110 = view_133.transpose(1, 2);  view_133 = None
    key_states_44 = transpose_110.contiguous();  transpose_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_10_self_attn_v_proj = self.L__mod___model_decoder_layers_10_self_attn_v_proj(hidden_states_177);  hidden_states_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_134 = l__mod___model_decoder_layers_10_self_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_10_self_attn_v_proj = None
    transpose_111 = view_134.transpose(1, 2);  view_134 = None
    value_states_44 = transpose_111.contiguous();  transpose_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_135 = query_states_44.view(1, 128, 32, 80);  query_states_44 = None
    transpose_112 = view_135.transpose(1, 2);  view_135 = None
    contiguous_68 = transpose_112.contiguous();  transpose_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_45 = contiguous_68.view(32, -1, 80);  contiguous_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_45 = key_states_44.reshape(32, -1, 80);  key_states_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_45 = value_states_44.reshape(32, -1, 80);  value_states_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_113 = key_states_45.transpose(1, 2);  key_states_45 = None
    attn_weights_64 = torch.bmm(query_states_45, transpose_113);  query_states_45 = transpose_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_137 = attn_weights_64.view(1, 32, 128, 128);  attn_weights_64 = None
    attn_weights_65 = view_137 + attention_mask;  view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_66 = attn_weights_65.view(32, 128, 128);  attn_weights_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_67 = torch.nn.functional.softmax(attn_weights_66, dim = -1);  attn_weights_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_22 = torch.nn.functional.dropout(attn_weights_67, p = 0.0, training = False);  attn_weights_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_110 = torch.bmm(attn_probs_22, value_states_45);  attn_probs_22 = value_states_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_111 = attn_output_110.view(1, 32, 128, 80);  attn_output_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_112 = attn_output_111.transpose(1, 2);  attn_output_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_113 = attn_output_112.reshape(1, 128, 2560);  attn_output_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_178 = self.L__mod___model_decoder_layers_10_self_attn_out_proj(attn_output_113);  attn_output_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:424, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_179 = torch.nn.functional.dropout(hidden_states_178, p = 0.1, training = False);  hidden_states_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    residual_35 = residual_34 + hidden_states_179;  residual_34 = hidden_states_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_181 = self.L__mod___model_decoder_layers_10_encoder_attn_layer_norm(residual_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_10_encoder_attn_q_proj = self.L__mod___model_decoder_layers_10_encoder_attn_q_proj(hidden_states_181);  hidden_states_181 = None
    query_states_46 = l__mod___model_decoder_layers_10_encoder_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_10_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_10_encoder_attn_k_proj = self.L__mod___model_decoder_layers_10_encoder_attn_k_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_140 = l__mod___model_decoder_layers_10_encoder_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_10_encoder_attn_k_proj = None
    transpose_115 = view_140.transpose(1, 2);  view_140 = None
    key_states_46 = transpose_115.contiguous();  transpose_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_10_encoder_attn_v_proj = self.L__mod___model_decoder_layers_10_encoder_attn_v_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_141 = l__mod___model_decoder_layers_10_encoder_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_10_encoder_attn_v_proj = None
    transpose_116 = view_141.transpose(1, 2);  view_141 = None
    value_states_46 = transpose_116.contiguous();  transpose_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_142 = query_states_46.view(1, 128, 32, 80);  query_states_46 = None
    transpose_117 = view_142.transpose(1, 2);  view_142 = None
    contiguous_71 = transpose_117.contiguous();  transpose_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_47 = contiguous_71.view(32, -1, 80);  contiguous_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_47 = key_states_46.reshape(32, -1, 80);  key_states_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_47 = value_states_46.reshape(32, -1, 80);  value_states_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_118 = key_states_47.transpose(1, 2);  key_states_47 = None
    attn_weights_68 = torch.bmm(query_states_47, transpose_118);  query_states_47 = transpose_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_69 = torch.nn.functional.softmax(attn_weights_68, dim = -1);  attn_weights_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_23 = torch.nn.functional.dropout(attn_weights_69, p = 0.0, training = False);  attn_weights_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_115 = torch.bmm(attn_probs_23, value_states_47);  attn_probs_23 = value_states_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_116 = attn_output_115.view(1, 32, 128, 80);  attn_output_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_117 = attn_output_116.transpose(1, 2);  attn_output_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_118 = attn_output_117.reshape(1, 128, 2560);  attn_output_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_182 = self.L__mod___model_decoder_layers_10_encoder_attn_out_proj(attn_output_118);  attn_output_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:444, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_183 = torch.nn.functional.dropout(hidden_states_182, p = 0.1, training = False);  hidden_states_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    residual_36 = residual_35 + hidden_states_183;  residual_35 = hidden_states_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_185 = self.L__mod___model_decoder_layers_10_final_layer_norm(residual_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_10_fc1 = self.L__mod___model_decoder_layers_10_fc1(hidden_states_185);  hidden_states_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_186 = torch._C._nn.gelu(l__mod___model_decoder_layers_10_fc1);  l__mod___model_decoder_layers_10_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:454, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_187 = torch.nn.functional.dropout(hidden_states_186, p = 0.0, training = False);  hidden_states_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    hidden_states_188 = self.L__mod___model_decoder_layers_10_fc2(hidden_states_187);  hidden_states_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:456, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_189 = torch.nn.functional.dropout(hidden_states_188, p = 0.1, training = False);  hidden_states_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    residual_37 = residual_36 + hidden_states_189;  residual_36 = hidden_states_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_192 = self.L__mod___model_decoder_layers_11_self_attn_layer_norm(residual_37)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_11_self_attn_q_proj = self.L__mod___model_decoder_layers_11_self_attn_q_proj(hidden_states_192)
    query_states_48 = l__mod___model_decoder_layers_11_self_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_11_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_11_self_attn_k_proj = self.L__mod___model_decoder_layers_11_self_attn_k_proj(hidden_states_192)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_145 = l__mod___model_decoder_layers_11_self_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_11_self_attn_k_proj = None
    transpose_120 = view_145.transpose(1, 2);  view_145 = None
    key_states_48 = transpose_120.contiguous();  transpose_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_11_self_attn_v_proj = self.L__mod___model_decoder_layers_11_self_attn_v_proj(hidden_states_192);  hidden_states_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_146 = l__mod___model_decoder_layers_11_self_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_11_self_attn_v_proj = None
    transpose_121 = view_146.transpose(1, 2);  view_146 = None
    value_states_48 = transpose_121.contiguous();  transpose_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_147 = query_states_48.view(1, 128, 32, 80);  query_states_48 = None
    transpose_122 = view_147.transpose(1, 2);  view_147 = None
    contiguous_74 = transpose_122.contiguous();  transpose_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_49 = contiguous_74.view(32, -1, 80);  contiguous_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_49 = key_states_48.reshape(32, -1, 80);  key_states_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_49 = value_states_48.reshape(32, -1, 80);  value_states_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_123 = key_states_49.transpose(1, 2);  key_states_49 = None
    attn_weights_70 = torch.bmm(query_states_49, transpose_123);  query_states_49 = transpose_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_149 = attn_weights_70.view(1, 32, 128, 128);  attn_weights_70 = None
    attn_weights_71 = view_149 + attention_mask;  view_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_72 = attn_weights_71.view(32, 128, 128);  attn_weights_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_73 = torch.nn.functional.softmax(attn_weights_72, dim = -1);  attn_weights_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_24 = torch.nn.functional.dropout(attn_weights_73, p = 0.0, training = False);  attn_weights_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_120 = torch.bmm(attn_probs_24, value_states_49);  attn_probs_24 = value_states_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_121 = attn_output_120.view(1, 32, 128, 80);  attn_output_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_122 = attn_output_121.transpose(1, 2);  attn_output_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_123 = attn_output_122.reshape(1, 128, 2560);  attn_output_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_193 = self.L__mod___model_decoder_layers_11_self_attn_out_proj(attn_output_123);  attn_output_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:424, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_194 = torch.nn.functional.dropout(hidden_states_193, p = 0.1, training = False);  hidden_states_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    residual_38 = residual_37 + hidden_states_194;  residual_37 = hidden_states_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_196 = self.L__mod___model_decoder_layers_11_encoder_attn_layer_norm(residual_38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_11_encoder_attn_q_proj = self.L__mod___model_decoder_layers_11_encoder_attn_q_proj(hidden_states_196);  hidden_states_196 = None
    query_states_50 = l__mod___model_decoder_layers_11_encoder_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_11_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_11_encoder_attn_k_proj = self.L__mod___model_decoder_layers_11_encoder_attn_k_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_152 = l__mod___model_decoder_layers_11_encoder_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_11_encoder_attn_k_proj = None
    transpose_125 = view_152.transpose(1, 2);  view_152 = None
    key_states_50 = transpose_125.contiguous();  transpose_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_11_encoder_attn_v_proj = self.L__mod___model_decoder_layers_11_encoder_attn_v_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_153 = l__mod___model_decoder_layers_11_encoder_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_11_encoder_attn_v_proj = None
    transpose_126 = view_153.transpose(1, 2);  view_153 = None
    value_states_50 = transpose_126.contiguous();  transpose_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_154 = query_states_50.view(1, 128, 32, 80);  query_states_50 = None
    transpose_127 = view_154.transpose(1, 2);  view_154 = None
    contiguous_77 = transpose_127.contiguous();  transpose_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_51 = contiguous_77.view(32, -1, 80);  contiguous_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_51 = key_states_50.reshape(32, -1, 80);  key_states_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_51 = value_states_50.reshape(32, -1, 80);  value_states_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_128 = key_states_51.transpose(1, 2);  key_states_51 = None
    attn_weights_74 = torch.bmm(query_states_51, transpose_128);  query_states_51 = transpose_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_75 = torch.nn.functional.softmax(attn_weights_74, dim = -1);  attn_weights_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_25 = torch.nn.functional.dropout(attn_weights_75, p = 0.0, training = False);  attn_weights_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_125 = torch.bmm(attn_probs_25, value_states_51);  attn_probs_25 = value_states_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_126 = attn_output_125.view(1, 32, 128, 80);  attn_output_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_127 = attn_output_126.transpose(1, 2);  attn_output_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_128 = attn_output_127.reshape(1, 128, 2560);  attn_output_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_197 = self.L__mod___model_decoder_layers_11_encoder_attn_out_proj(attn_output_128);  attn_output_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:444, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_198 = torch.nn.functional.dropout(hidden_states_197, p = 0.1, training = False);  hidden_states_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    residual_39 = residual_38 + hidden_states_198;  residual_38 = hidden_states_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_200 = self.L__mod___model_decoder_layers_11_final_layer_norm(residual_39)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_11_fc1 = self.L__mod___model_decoder_layers_11_fc1(hidden_states_200);  hidden_states_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_201 = torch._C._nn.gelu(l__mod___model_decoder_layers_11_fc1);  l__mod___model_decoder_layers_11_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:454, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_202 = torch.nn.functional.dropout(hidden_states_201, p = 0.0, training = False);  hidden_states_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    hidden_states_203 = self.L__mod___model_decoder_layers_11_fc2(hidden_states_202);  hidden_states_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:456, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_204 = torch.nn.functional.dropout(hidden_states_203, p = 0.1, training = False);  hidden_states_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    residual_40 = residual_39 + hidden_states_204;  residual_39 = hidden_states_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_207 = self.L__mod___model_decoder_layers_12_self_attn_layer_norm(residual_40)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_12_self_attn_q_proj = self.L__mod___model_decoder_layers_12_self_attn_q_proj(hidden_states_207)
    query_states_52 = l__mod___model_decoder_layers_12_self_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_12_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_12_self_attn_k_proj = self.L__mod___model_decoder_layers_12_self_attn_k_proj(hidden_states_207)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_157 = l__mod___model_decoder_layers_12_self_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_12_self_attn_k_proj = None
    transpose_130 = view_157.transpose(1, 2);  view_157 = None
    key_states_52 = transpose_130.contiguous();  transpose_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_12_self_attn_v_proj = self.L__mod___model_decoder_layers_12_self_attn_v_proj(hidden_states_207);  hidden_states_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_158 = l__mod___model_decoder_layers_12_self_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_12_self_attn_v_proj = None
    transpose_131 = view_158.transpose(1, 2);  view_158 = None
    value_states_52 = transpose_131.contiguous();  transpose_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_159 = query_states_52.view(1, 128, 32, 80);  query_states_52 = None
    transpose_132 = view_159.transpose(1, 2);  view_159 = None
    contiguous_80 = transpose_132.contiguous();  transpose_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_53 = contiguous_80.view(32, -1, 80);  contiguous_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_53 = key_states_52.reshape(32, -1, 80);  key_states_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_53 = value_states_52.reshape(32, -1, 80);  value_states_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_133 = key_states_53.transpose(1, 2);  key_states_53 = None
    attn_weights_76 = torch.bmm(query_states_53, transpose_133);  query_states_53 = transpose_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_161 = attn_weights_76.view(1, 32, 128, 128);  attn_weights_76 = None
    attn_weights_77 = view_161 + attention_mask;  view_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_78 = attn_weights_77.view(32, 128, 128);  attn_weights_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_79 = torch.nn.functional.softmax(attn_weights_78, dim = -1);  attn_weights_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_26 = torch.nn.functional.dropout(attn_weights_79, p = 0.0, training = False);  attn_weights_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_130 = torch.bmm(attn_probs_26, value_states_53);  attn_probs_26 = value_states_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_131 = attn_output_130.view(1, 32, 128, 80);  attn_output_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_132 = attn_output_131.transpose(1, 2);  attn_output_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_133 = attn_output_132.reshape(1, 128, 2560);  attn_output_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_208 = self.L__mod___model_decoder_layers_12_self_attn_out_proj(attn_output_133);  attn_output_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:424, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_209 = torch.nn.functional.dropout(hidden_states_208, p = 0.1, training = False);  hidden_states_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    residual_41 = residual_40 + hidden_states_209;  residual_40 = hidden_states_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_211 = self.L__mod___model_decoder_layers_12_encoder_attn_layer_norm(residual_41)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_12_encoder_attn_q_proj = self.L__mod___model_decoder_layers_12_encoder_attn_q_proj(hidden_states_211);  hidden_states_211 = None
    query_states_54 = l__mod___model_decoder_layers_12_encoder_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_12_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_12_encoder_attn_k_proj = self.L__mod___model_decoder_layers_12_encoder_attn_k_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_164 = l__mod___model_decoder_layers_12_encoder_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_12_encoder_attn_k_proj = None
    transpose_135 = view_164.transpose(1, 2);  view_164 = None
    key_states_54 = transpose_135.contiguous();  transpose_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_12_encoder_attn_v_proj = self.L__mod___model_decoder_layers_12_encoder_attn_v_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_165 = l__mod___model_decoder_layers_12_encoder_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_12_encoder_attn_v_proj = None
    transpose_136 = view_165.transpose(1, 2);  view_165 = None
    value_states_54 = transpose_136.contiguous();  transpose_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_166 = query_states_54.view(1, 128, 32, 80);  query_states_54 = None
    transpose_137 = view_166.transpose(1, 2);  view_166 = None
    contiguous_83 = transpose_137.contiguous();  transpose_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_55 = contiguous_83.view(32, -1, 80);  contiguous_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_55 = key_states_54.reshape(32, -1, 80);  key_states_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_55 = value_states_54.reshape(32, -1, 80);  value_states_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_138 = key_states_55.transpose(1, 2);  key_states_55 = None
    attn_weights_80 = torch.bmm(query_states_55, transpose_138);  query_states_55 = transpose_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_81 = torch.nn.functional.softmax(attn_weights_80, dim = -1);  attn_weights_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_27 = torch.nn.functional.dropout(attn_weights_81, p = 0.0, training = False);  attn_weights_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_135 = torch.bmm(attn_probs_27, value_states_55);  attn_probs_27 = value_states_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_136 = attn_output_135.view(1, 32, 128, 80);  attn_output_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_137 = attn_output_136.transpose(1, 2);  attn_output_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_138 = attn_output_137.reshape(1, 128, 2560);  attn_output_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_212 = self.L__mod___model_decoder_layers_12_encoder_attn_out_proj(attn_output_138);  attn_output_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:444, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_213 = torch.nn.functional.dropout(hidden_states_212, p = 0.1, training = False);  hidden_states_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    residual_42 = residual_41 + hidden_states_213;  residual_41 = hidden_states_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_215 = self.L__mod___model_decoder_layers_12_final_layer_norm(residual_42)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_12_fc1 = self.L__mod___model_decoder_layers_12_fc1(hidden_states_215);  hidden_states_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_216 = torch._C._nn.gelu(l__mod___model_decoder_layers_12_fc1);  l__mod___model_decoder_layers_12_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:454, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_217 = torch.nn.functional.dropout(hidden_states_216, p = 0.0, training = False);  hidden_states_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    hidden_states_218 = self.L__mod___model_decoder_layers_12_fc2(hidden_states_217);  hidden_states_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:456, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_219 = torch.nn.functional.dropout(hidden_states_218, p = 0.1, training = False);  hidden_states_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    residual_43 = residual_42 + hidden_states_219;  residual_42 = hidden_states_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_222 = self.L__mod___model_decoder_layers_13_self_attn_layer_norm(residual_43)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_13_self_attn_q_proj = self.L__mod___model_decoder_layers_13_self_attn_q_proj(hidden_states_222)
    query_states_56 = l__mod___model_decoder_layers_13_self_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_13_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_13_self_attn_k_proj = self.L__mod___model_decoder_layers_13_self_attn_k_proj(hidden_states_222)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_169 = l__mod___model_decoder_layers_13_self_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_13_self_attn_k_proj = None
    transpose_140 = view_169.transpose(1, 2);  view_169 = None
    key_states_56 = transpose_140.contiguous();  transpose_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_13_self_attn_v_proj = self.L__mod___model_decoder_layers_13_self_attn_v_proj(hidden_states_222);  hidden_states_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_170 = l__mod___model_decoder_layers_13_self_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_13_self_attn_v_proj = None
    transpose_141 = view_170.transpose(1, 2);  view_170 = None
    value_states_56 = transpose_141.contiguous();  transpose_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_171 = query_states_56.view(1, 128, 32, 80);  query_states_56 = None
    transpose_142 = view_171.transpose(1, 2);  view_171 = None
    contiguous_86 = transpose_142.contiguous();  transpose_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_57 = contiguous_86.view(32, -1, 80);  contiguous_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_57 = key_states_56.reshape(32, -1, 80);  key_states_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_57 = value_states_56.reshape(32, -1, 80);  value_states_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_143 = key_states_57.transpose(1, 2);  key_states_57 = None
    attn_weights_82 = torch.bmm(query_states_57, transpose_143);  query_states_57 = transpose_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_173 = attn_weights_82.view(1, 32, 128, 128);  attn_weights_82 = None
    attn_weights_83 = view_173 + attention_mask;  view_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_84 = attn_weights_83.view(32, 128, 128);  attn_weights_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_85 = torch.nn.functional.softmax(attn_weights_84, dim = -1);  attn_weights_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_28 = torch.nn.functional.dropout(attn_weights_85, p = 0.0, training = False);  attn_weights_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_140 = torch.bmm(attn_probs_28, value_states_57);  attn_probs_28 = value_states_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_141 = attn_output_140.view(1, 32, 128, 80);  attn_output_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_142 = attn_output_141.transpose(1, 2);  attn_output_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_143 = attn_output_142.reshape(1, 128, 2560);  attn_output_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_223 = self.L__mod___model_decoder_layers_13_self_attn_out_proj(attn_output_143);  attn_output_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:424, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_224 = torch.nn.functional.dropout(hidden_states_223, p = 0.1, training = False);  hidden_states_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    residual_44 = residual_43 + hidden_states_224;  residual_43 = hidden_states_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_226 = self.L__mod___model_decoder_layers_13_encoder_attn_layer_norm(residual_44)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_13_encoder_attn_q_proj = self.L__mod___model_decoder_layers_13_encoder_attn_q_proj(hidden_states_226);  hidden_states_226 = None
    query_states_58 = l__mod___model_decoder_layers_13_encoder_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_13_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_13_encoder_attn_k_proj = self.L__mod___model_decoder_layers_13_encoder_attn_k_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_176 = l__mod___model_decoder_layers_13_encoder_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_13_encoder_attn_k_proj = None
    transpose_145 = view_176.transpose(1, 2);  view_176 = None
    key_states_58 = transpose_145.contiguous();  transpose_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_13_encoder_attn_v_proj = self.L__mod___model_decoder_layers_13_encoder_attn_v_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_177 = l__mod___model_decoder_layers_13_encoder_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_13_encoder_attn_v_proj = None
    transpose_146 = view_177.transpose(1, 2);  view_177 = None
    value_states_58 = transpose_146.contiguous();  transpose_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_178 = query_states_58.view(1, 128, 32, 80);  query_states_58 = None
    transpose_147 = view_178.transpose(1, 2);  view_178 = None
    contiguous_89 = transpose_147.contiguous();  transpose_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_59 = contiguous_89.view(32, -1, 80);  contiguous_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_59 = key_states_58.reshape(32, -1, 80);  key_states_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_59 = value_states_58.reshape(32, -1, 80);  value_states_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_148 = key_states_59.transpose(1, 2);  key_states_59 = None
    attn_weights_86 = torch.bmm(query_states_59, transpose_148);  query_states_59 = transpose_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_87 = torch.nn.functional.softmax(attn_weights_86, dim = -1);  attn_weights_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_29 = torch.nn.functional.dropout(attn_weights_87, p = 0.0, training = False);  attn_weights_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_145 = torch.bmm(attn_probs_29, value_states_59);  attn_probs_29 = value_states_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_146 = attn_output_145.view(1, 32, 128, 80);  attn_output_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_147 = attn_output_146.transpose(1, 2);  attn_output_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_148 = attn_output_147.reshape(1, 128, 2560);  attn_output_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_227 = self.L__mod___model_decoder_layers_13_encoder_attn_out_proj(attn_output_148);  attn_output_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:444, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_228 = torch.nn.functional.dropout(hidden_states_227, p = 0.1, training = False);  hidden_states_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    residual_45 = residual_44 + hidden_states_228;  residual_44 = hidden_states_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_230 = self.L__mod___model_decoder_layers_13_final_layer_norm(residual_45)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_13_fc1 = self.L__mod___model_decoder_layers_13_fc1(hidden_states_230);  hidden_states_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_231 = torch._C._nn.gelu(l__mod___model_decoder_layers_13_fc1);  l__mod___model_decoder_layers_13_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:454, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_232 = torch.nn.functional.dropout(hidden_states_231, p = 0.0, training = False);  hidden_states_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    hidden_states_233 = self.L__mod___model_decoder_layers_13_fc2(hidden_states_232);  hidden_states_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:456, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_234 = torch.nn.functional.dropout(hidden_states_233, p = 0.1, training = False);  hidden_states_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    residual_46 = residual_45 + hidden_states_234;  residual_45 = hidden_states_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_237 = self.L__mod___model_decoder_layers_14_self_attn_layer_norm(residual_46)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_14_self_attn_q_proj = self.L__mod___model_decoder_layers_14_self_attn_q_proj(hidden_states_237)
    query_states_60 = l__mod___model_decoder_layers_14_self_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_14_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_14_self_attn_k_proj = self.L__mod___model_decoder_layers_14_self_attn_k_proj(hidden_states_237)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_181 = l__mod___model_decoder_layers_14_self_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_14_self_attn_k_proj = None
    transpose_150 = view_181.transpose(1, 2);  view_181 = None
    key_states_60 = transpose_150.contiguous();  transpose_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_14_self_attn_v_proj = self.L__mod___model_decoder_layers_14_self_attn_v_proj(hidden_states_237);  hidden_states_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_182 = l__mod___model_decoder_layers_14_self_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_14_self_attn_v_proj = None
    transpose_151 = view_182.transpose(1, 2);  view_182 = None
    value_states_60 = transpose_151.contiguous();  transpose_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_183 = query_states_60.view(1, 128, 32, 80);  query_states_60 = None
    transpose_152 = view_183.transpose(1, 2);  view_183 = None
    contiguous_92 = transpose_152.contiguous();  transpose_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_61 = contiguous_92.view(32, -1, 80);  contiguous_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_61 = key_states_60.reshape(32, -1, 80);  key_states_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_61 = value_states_60.reshape(32, -1, 80);  value_states_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_153 = key_states_61.transpose(1, 2);  key_states_61 = None
    attn_weights_88 = torch.bmm(query_states_61, transpose_153);  query_states_61 = transpose_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_185 = attn_weights_88.view(1, 32, 128, 128);  attn_weights_88 = None
    attn_weights_89 = view_185 + attention_mask;  view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_90 = attn_weights_89.view(32, 128, 128);  attn_weights_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_91 = torch.nn.functional.softmax(attn_weights_90, dim = -1);  attn_weights_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_30 = torch.nn.functional.dropout(attn_weights_91, p = 0.0, training = False);  attn_weights_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_150 = torch.bmm(attn_probs_30, value_states_61);  attn_probs_30 = value_states_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_151 = attn_output_150.view(1, 32, 128, 80);  attn_output_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_152 = attn_output_151.transpose(1, 2);  attn_output_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_153 = attn_output_152.reshape(1, 128, 2560);  attn_output_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_238 = self.L__mod___model_decoder_layers_14_self_attn_out_proj(attn_output_153);  attn_output_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:424, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_239 = torch.nn.functional.dropout(hidden_states_238, p = 0.1, training = False);  hidden_states_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    residual_47 = residual_46 + hidden_states_239;  residual_46 = hidden_states_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_241 = self.L__mod___model_decoder_layers_14_encoder_attn_layer_norm(residual_47)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_14_encoder_attn_q_proj = self.L__mod___model_decoder_layers_14_encoder_attn_q_proj(hidden_states_241);  hidden_states_241 = None
    query_states_62 = l__mod___model_decoder_layers_14_encoder_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_14_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_14_encoder_attn_k_proj = self.L__mod___model_decoder_layers_14_encoder_attn_k_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_188 = l__mod___model_decoder_layers_14_encoder_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_14_encoder_attn_k_proj = None
    transpose_155 = view_188.transpose(1, 2);  view_188 = None
    key_states_62 = transpose_155.contiguous();  transpose_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_14_encoder_attn_v_proj = self.L__mod___model_decoder_layers_14_encoder_attn_v_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_189 = l__mod___model_decoder_layers_14_encoder_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_14_encoder_attn_v_proj = None
    transpose_156 = view_189.transpose(1, 2);  view_189 = None
    value_states_62 = transpose_156.contiguous();  transpose_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_190 = query_states_62.view(1, 128, 32, 80);  query_states_62 = None
    transpose_157 = view_190.transpose(1, 2);  view_190 = None
    contiguous_95 = transpose_157.contiguous();  transpose_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_63 = contiguous_95.view(32, -1, 80);  contiguous_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_63 = key_states_62.reshape(32, -1, 80);  key_states_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_63 = value_states_62.reshape(32, -1, 80);  value_states_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_158 = key_states_63.transpose(1, 2);  key_states_63 = None
    attn_weights_92 = torch.bmm(query_states_63, transpose_158);  query_states_63 = transpose_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_93 = torch.nn.functional.softmax(attn_weights_92, dim = -1);  attn_weights_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_31 = torch.nn.functional.dropout(attn_weights_93, p = 0.0, training = False);  attn_weights_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_155 = torch.bmm(attn_probs_31, value_states_63);  attn_probs_31 = value_states_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_156 = attn_output_155.view(1, 32, 128, 80);  attn_output_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_157 = attn_output_156.transpose(1, 2);  attn_output_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_158 = attn_output_157.reshape(1, 128, 2560);  attn_output_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_242 = self.L__mod___model_decoder_layers_14_encoder_attn_out_proj(attn_output_158);  attn_output_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:444, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_243 = torch.nn.functional.dropout(hidden_states_242, p = 0.1, training = False);  hidden_states_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    residual_48 = residual_47 + hidden_states_243;  residual_47 = hidden_states_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_245 = self.L__mod___model_decoder_layers_14_final_layer_norm(residual_48)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_14_fc1 = self.L__mod___model_decoder_layers_14_fc1(hidden_states_245);  hidden_states_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_246 = torch._C._nn.gelu(l__mod___model_decoder_layers_14_fc1);  l__mod___model_decoder_layers_14_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:454, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_247 = torch.nn.functional.dropout(hidden_states_246, p = 0.0, training = False);  hidden_states_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    hidden_states_248 = self.L__mod___model_decoder_layers_14_fc2(hidden_states_247);  hidden_states_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:456, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_249 = torch.nn.functional.dropout(hidden_states_248, p = 0.1, training = False);  hidden_states_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    residual_49 = residual_48 + hidden_states_249;  residual_48 = hidden_states_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_252 = self.L__mod___model_decoder_layers_15_self_attn_layer_norm(residual_49)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_15_self_attn_q_proj = self.L__mod___model_decoder_layers_15_self_attn_q_proj(hidden_states_252)
    query_states_64 = l__mod___model_decoder_layers_15_self_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_15_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_15_self_attn_k_proj = self.L__mod___model_decoder_layers_15_self_attn_k_proj(hidden_states_252)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_193 = l__mod___model_decoder_layers_15_self_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_15_self_attn_k_proj = None
    transpose_160 = view_193.transpose(1, 2);  view_193 = None
    key_states_64 = transpose_160.contiguous();  transpose_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_15_self_attn_v_proj = self.L__mod___model_decoder_layers_15_self_attn_v_proj(hidden_states_252);  hidden_states_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_194 = l__mod___model_decoder_layers_15_self_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_15_self_attn_v_proj = None
    transpose_161 = view_194.transpose(1, 2);  view_194 = None
    value_states_64 = transpose_161.contiguous();  transpose_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_195 = query_states_64.view(1, 128, 32, 80);  query_states_64 = None
    transpose_162 = view_195.transpose(1, 2);  view_195 = None
    contiguous_98 = transpose_162.contiguous();  transpose_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_65 = contiguous_98.view(32, -1, 80);  contiguous_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_65 = key_states_64.reshape(32, -1, 80);  key_states_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_65 = value_states_64.reshape(32, -1, 80);  value_states_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_163 = key_states_65.transpose(1, 2);  key_states_65 = None
    attn_weights_94 = torch.bmm(query_states_65, transpose_163);  query_states_65 = transpose_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_197 = attn_weights_94.view(1, 32, 128, 128);  attn_weights_94 = None
    attn_weights_95 = view_197 + attention_mask;  view_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_96 = attn_weights_95.view(32, 128, 128);  attn_weights_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_97 = torch.nn.functional.softmax(attn_weights_96, dim = -1);  attn_weights_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_32 = torch.nn.functional.dropout(attn_weights_97, p = 0.0, training = False);  attn_weights_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_160 = torch.bmm(attn_probs_32, value_states_65);  attn_probs_32 = value_states_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_161 = attn_output_160.view(1, 32, 128, 80);  attn_output_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_162 = attn_output_161.transpose(1, 2);  attn_output_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_163 = attn_output_162.reshape(1, 128, 2560);  attn_output_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_253 = self.L__mod___model_decoder_layers_15_self_attn_out_proj(attn_output_163);  attn_output_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:424, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_254 = torch.nn.functional.dropout(hidden_states_253, p = 0.1, training = False);  hidden_states_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    residual_50 = residual_49 + hidden_states_254;  residual_49 = hidden_states_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_256 = self.L__mod___model_decoder_layers_15_encoder_attn_layer_norm(residual_50)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_15_encoder_attn_q_proj = self.L__mod___model_decoder_layers_15_encoder_attn_q_proj(hidden_states_256);  hidden_states_256 = None
    query_states_66 = l__mod___model_decoder_layers_15_encoder_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_15_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_15_encoder_attn_k_proj = self.L__mod___model_decoder_layers_15_encoder_attn_k_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_200 = l__mod___model_decoder_layers_15_encoder_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_15_encoder_attn_k_proj = None
    transpose_165 = view_200.transpose(1, 2);  view_200 = None
    key_states_66 = transpose_165.contiguous();  transpose_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_15_encoder_attn_v_proj = self.L__mod___model_decoder_layers_15_encoder_attn_v_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_201 = l__mod___model_decoder_layers_15_encoder_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_15_encoder_attn_v_proj = None
    transpose_166 = view_201.transpose(1, 2);  view_201 = None
    value_states_66 = transpose_166.contiguous();  transpose_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_202 = query_states_66.view(1, 128, 32, 80);  query_states_66 = None
    transpose_167 = view_202.transpose(1, 2);  view_202 = None
    contiguous_101 = transpose_167.contiguous();  transpose_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_67 = contiguous_101.view(32, -1, 80);  contiguous_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_67 = key_states_66.reshape(32, -1, 80);  key_states_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_67 = value_states_66.reshape(32, -1, 80);  value_states_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_168 = key_states_67.transpose(1, 2);  key_states_67 = None
    attn_weights_98 = torch.bmm(query_states_67, transpose_168);  query_states_67 = transpose_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_99 = torch.nn.functional.softmax(attn_weights_98, dim = -1);  attn_weights_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_33 = torch.nn.functional.dropout(attn_weights_99, p = 0.0, training = False);  attn_weights_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_165 = torch.bmm(attn_probs_33, value_states_67);  attn_probs_33 = value_states_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_166 = attn_output_165.view(1, 32, 128, 80);  attn_output_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_167 = attn_output_166.transpose(1, 2);  attn_output_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_168 = attn_output_167.reshape(1, 128, 2560);  attn_output_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_257 = self.L__mod___model_decoder_layers_15_encoder_attn_out_proj(attn_output_168);  attn_output_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:444, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_258 = torch.nn.functional.dropout(hidden_states_257, p = 0.1, training = False);  hidden_states_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    residual_51 = residual_50 + hidden_states_258;  residual_50 = hidden_states_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_260 = self.L__mod___model_decoder_layers_15_final_layer_norm(residual_51)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_15_fc1 = self.L__mod___model_decoder_layers_15_fc1(hidden_states_260);  hidden_states_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_261 = torch._C._nn.gelu(l__mod___model_decoder_layers_15_fc1);  l__mod___model_decoder_layers_15_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:454, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_262 = torch.nn.functional.dropout(hidden_states_261, p = 0.0, training = False);  hidden_states_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    hidden_states_263 = self.L__mod___model_decoder_layers_15_fc2(hidden_states_262);  hidden_states_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:456, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_264 = torch.nn.functional.dropout(hidden_states_263, p = 0.1, training = False);  hidden_states_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    residual_52 = residual_51 + hidden_states_264;  residual_51 = hidden_states_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_267 = self.L__mod___model_decoder_layers_16_self_attn_layer_norm(residual_52)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_16_self_attn_q_proj = self.L__mod___model_decoder_layers_16_self_attn_q_proj(hidden_states_267)
    query_states_68 = l__mod___model_decoder_layers_16_self_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_16_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_16_self_attn_k_proj = self.L__mod___model_decoder_layers_16_self_attn_k_proj(hidden_states_267)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_205 = l__mod___model_decoder_layers_16_self_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_16_self_attn_k_proj = None
    transpose_170 = view_205.transpose(1, 2);  view_205 = None
    key_states_68 = transpose_170.contiguous();  transpose_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_16_self_attn_v_proj = self.L__mod___model_decoder_layers_16_self_attn_v_proj(hidden_states_267);  hidden_states_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_206 = l__mod___model_decoder_layers_16_self_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_16_self_attn_v_proj = None
    transpose_171 = view_206.transpose(1, 2);  view_206 = None
    value_states_68 = transpose_171.contiguous();  transpose_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_207 = query_states_68.view(1, 128, 32, 80);  query_states_68 = None
    transpose_172 = view_207.transpose(1, 2);  view_207 = None
    contiguous_104 = transpose_172.contiguous();  transpose_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_69 = contiguous_104.view(32, -1, 80);  contiguous_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_69 = key_states_68.reshape(32, -1, 80);  key_states_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_69 = value_states_68.reshape(32, -1, 80);  value_states_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_173 = key_states_69.transpose(1, 2);  key_states_69 = None
    attn_weights_100 = torch.bmm(query_states_69, transpose_173);  query_states_69 = transpose_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_209 = attn_weights_100.view(1, 32, 128, 128);  attn_weights_100 = None
    attn_weights_101 = view_209 + attention_mask;  view_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_102 = attn_weights_101.view(32, 128, 128);  attn_weights_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_103 = torch.nn.functional.softmax(attn_weights_102, dim = -1);  attn_weights_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_34 = torch.nn.functional.dropout(attn_weights_103, p = 0.0, training = False);  attn_weights_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_170 = torch.bmm(attn_probs_34, value_states_69);  attn_probs_34 = value_states_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_171 = attn_output_170.view(1, 32, 128, 80);  attn_output_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_172 = attn_output_171.transpose(1, 2);  attn_output_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_173 = attn_output_172.reshape(1, 128, 2560);  attn_output_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_268 = self.L__mod___model_decoder_layers_16_self_attn_out_proj(attn_output_173);  attn_output_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:424, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_269 = torch.nn.functional.dropout(hidden_states_268, p = 0.1, training = False);  hidden_states_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    residual_53 = residual_52 + hidden_states_269;  residual_52 = hidden_states_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_271 = self.L__mod___model_decoder_layers_16_encoder_attn_layer_norm(residual_53)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_16_encoder_attn_q_proj = self.L__mod___model_decoder_layers_16_encoder_attn_q_proj(hidden_states_271);  hidden_states_271 = None
    query_states_70 = l__mod___model_decoder_layers_16_encoder_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_16_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_16_encoder_attn_k_proj = self.L__mod___model_decoder_layers_16_encoder_attn_k_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_212 = l__mod___model_decoder_layers_16_encoder_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_16_encoder_attn_k_proj = None
    transpose_175 = view_212.transpose(1, 2);  view_212 = None
    key_states_70 = transpose_175.contiguous();  transpose_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_16_encoder_attn_v_proj = self.L__mod___model_decoder_layers_16_encoder_attn_v_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_213 = l__mod___model_decoder_layers_16_encoder_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_16_encoder_attn_v_proj = None
    transpose_176 = view_213.transpose(1, 2);  view_213 = None
    value_states_70 = transpose_176.contiguous();  transpose_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_214 = query_states_70.view(1, 128, 32, 80);  query_states_70 = None
    transpose_177 = view_214.transpose(1, 2);  view_214 = None
    contiguous_107 = transpose_177.contiguous();  transpose_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_71 = contiguous_107.view(32, -1, 80);  contiguous_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_71 = key_states_70.reshape(32, -1, 80);  key_states_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_71 = value_states_70.reshape(32, -1, 80);  value_states_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_178 = key_states_71.transpose(1, 2);  key_states_71 = None
    attn_weights_104 = torch.bmm(query_states_71, transpose_178);  query_states_71 = transpose_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_105 = torch.nn.functional.softmax(attn_weights_104, dim = -1);  attn_weights_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_35 = torch.nn.functional.dropout(attn_weights_105, p = 0.0, training = False);  attn_weights_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_175 = torch.bmm(attn_probs_35, value_states_71);  attn_probs_35 = value_states_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_176 = attn_output_175.view(1, 32, 128, 80);  attn_output_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_177 = attn_output_176.transpose(1, 2);  attn_output_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_178 = attn_output_177.reshape(1, 128, 2560);  attn_output_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_272 = self.L__mod___model_decoder_layers_16_encoder_attn_out_proj(attn_output_178);  attn_output_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:444, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_273 = torch.nn.functional.dropout(hidden_states_272, p = 0.1, training = False);  hidden_states_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    residual_54 = residual_53 + hidden_states_273;  residual_53 = hidden_states_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_275 = self.L__mod___model_decoder_layers_16_final_layer_norm(residual_54)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_16_fc1 = self.L__mod___model_decoder_layers_16_fc1(hidden_states_275);  hidden_states_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_276 = torch._C._nn.gelu(l__mod___model_decoder_layers_16_fc1);  l__mod___model_decoder_layers_16_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:454, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_277 = torch.nn.functional.dropout(hidden_states_276, p = 0.0, training = False);  hidden_states_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    hidden_states_278 = self.L__mod___model_decoder_layers_16_fc2(hidden_states_277);  hidden_states_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:456, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_279 = torch.nn.functional.dropout(hidden_states_278, p = 0.1, training = False);  hidden_states_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    residual_55 = residual_54 + hidden_states_279;  residual_54 = hidden_states_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_282 = self.L__mod___model_decoder_layers_17_self_attn_layer_norm(residual_55)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_17_self_attn_q_proj = self.L__mod___model_decoder_layers_17_self_attn_q_proj(hidden_states_282)
    query_states_72 = l__mod___model_decoder_layers_17_self_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_17_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_17_self_attn_k_proj = self.L__mod___model_decoder_layers_17_self_attn_k_proj(hidden_states_282)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_217 = l__mod___model_decoder_layers_17_self_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_17_self_attn_k_proj = None
    transpose_180 = view_217.transpose(1, 2);  view_217 = None
    key_states_72 = transpose_180.contiguous();  transpose_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_17_self_attn_v_proj = self.L__mod___model_decoder_layers_17_self_attn_v_proj(hidden_states_282);  hidden_states_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_218 = l__mod___model_decoder_layers_17_self_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_17_self_attn_v_proj = None
    transpose_181 = view_218.transpose(1, 2);  view_218 = None
    value_states_72 = transpose_181.contiguous();  transpose_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_219 = query_states_72.view(1, 128, 32, 80);  query_states_72 = None
    transpose_182 = view_219.transpose(1, 2);  view_219 = None
    contiguous_110 = transpose_182.contiguous();  transpose_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_73 = contiguous_110.view(32, -1, 80);  contiguous_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_73 = key_states_72.reshape(32, -1, 80);  key_states_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_73 = value_states_72.reshape(32, -1, 80);  value_states_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_183 = key_states_73.transpose(1, 2);  key_states_73 = None
    attn_weights_106 = torch.bmm(query_states_73, transpose_183);  query_states_73 = transpose_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_221 = attn_weights_106.view(1, 32, 128, 128);  attn_weights_106 = None
    attn_weights_107 = view_221 + attention_mask;  view_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_108 = attn_weights_107.view(32, 128, 128);  attn_weights_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_109 = torch.nn.functional.softmax(attn_weights_108, dim = -1);  attn_weights_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_36 = torch.nn.functional.dropout(attn_weights_109, p = 0.0, training = False);  attn_weights_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_180 = torch.bmm(attn_probs_36, value_states_73);  attn_probs_36 = value_states_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_181 = attn_output_180.view(1, 32, 128, 80);  attn_output_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_182 = attn_output_181.transpose(1, 2);  attn_output_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_183 = attn_output_182.reshape(1, 128, 2560);  attn_output_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_283 = self.L__mod___model_decoder_layers_17_self_attn_out_proj(attn_output_183);  attn_output_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:424, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_284 = torch.nn.functional.dropout(hidden_states_283, p = 0.1, training = False);  hidden_states_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    residual_56 = residual_55 + hidden_states_284;  residual_55 = hidden_states_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_286 = self.L__mod___model_decoder_layers_17_encoder_attn_layer_norm(residual_56)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_17_encoder_attn_q_proj = self.L__mod___model_decoder_layers_17_encoder_attn_q_proj(hidden_states_286);  hidden_states_286 = None
    query_states_74 = l__mod___model_decoder_layers_17_encoder_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_17_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_17_encoder_attn_k_proj = self.L__mod___model_decoder_layers_17_encoder_attn_k_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_224 = l__mod___model_decoder_layers_17_encoder_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_17_encoder_attn_k_proj = None
    transpose_185 = view_224.transpose(1, 2);  view_224 = None
    key_states_74 = transpose_185.contiguous();  transpose_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_17_encoder_attn_v_proj = self.L__mod___model_decoder_layers_17_encoder_attn_v_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_225 = l__mod___model_decoder_layers_17_encoder_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_17_encoder_attn_v_proj = None
    transpose_186 = view_225.transpose(1, 2);  view_225 = None
    value_states_74 = transpose_186.contiguous();  transpose_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_226 = query_states_74.view(1, 128, 32, 80);  query_states_74 = None
    transpose_187 = view_226.transpose(1, 2);  view_226 = None
    contiguous_113 = transpose_187.contiguous();  transpose_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_75 = contiguous_113.view(32, -1, 80);  contiguous_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_75 = key_states_74.reshape(32, -1, 80);  key_states_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_75 = value_states_74.reshape(32, -1, 80);  value_states_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_188 = key_states_75.transpose(1, 2);  key_states_75 = None
    attn_weights_110 = torch.bmm(query_states_75, transpose_188);  query_states_75 = transpose_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_111 = torch.nn.functional.softmax(attn_weights_110, dim = -1);  attn_weights_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_37 = torch.nn.functional.dropout(attn_weights_111, p = 0.0, training = False);  attn_weights_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_185 = torch.bmm(attn_probs_37, value_states_75);  attn_probs_37 = value_states_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_186 = attn_output_185.view(1, 32, 128, 80);  attn_output_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_187 = attn_output_186.transpose(1, 2);  attn_output_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_188 = attn_output_187.reshape(1, 128, 2560);  attn_output_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_287 = self.L__mod___model_decoder_layers_17_encoder_attn_out_proj(attn_output_188);  attn_output_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:444, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_288 = torch.nn.functional.dropout(hidden_states_287, p = 0.1, training = False);  hidden_states_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    residual_57 = residual_56 + hidden_states_288;  residual_56 = hidden_states_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_290 = self.L__mod___model_decoder_layers_17_final_layer_norm(residual_57)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_17_fc1 = self.L__mod___model_decoder_layers_17_fc1(hidden_states_290);  hidden_states_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_291 = torch._C._nn.gelu(l__mod___model_decoder_layers_17_fc1);  l__mod___model_decoder_layers_17_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:454, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_292 = torch.nn.functional.dropout(hidden_states_291, p = 0.0, training = False);  hidden_states_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    hidden_states_293 = self.L__mod___model_decoder_layers_17_fc2(hidden_states_292);  hidden_states_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:456, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_294 = torch.nn.functional.dropout(hidden_states_293, p = 0.1, training = False);  hidden_states_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    residual_58 = residual_57 + hidden_states_294;  residual_57 = hidden_states_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_297 = self.L__mod___model_decoder_layers_18_self_attn_layer_norm(residual_58)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_18_self_attn_q_proj = self.L__mod___model_decoder_layers_18_self_attn_q_proj(hidden_states_297)
    query_states_76 = l__mod___model_decoder_layers_18_self_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_18_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_18_self_attn_k_proj = self.L__mod___model_decoder_layers_18_self_attn_k_proj(hidden_states_297)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_229 = l__mod___model_decoder_layers_18_self_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_18_self_attn_k_proj = None
    transpose_190 = view_229.transpose(1, 2);  view_229 = None
    key_states_76 = transpose_190.contiguous();  transpose_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_18_self_attn_v_proj = self.L__mod___model_decoder_layers_18_self_attn_v_proj(hidden_states_297);  hidden_states_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_230 = l__mod___model_decoder_layers_18_self_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_18_self_attn_v_proj = None
    transpose_191 = view_230.transpose(1, 2);  view_230 = None
    value_states_76 = transpose_191.contiguous();  transpose_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_231 = query_states_76.view(1, 128, 32, 80);  query_states_76 = None
    transpose_192 = view_231.transpose(1, 2);  view_231 = None
    contiguous_116 = transpose_192.contiguous();  transpose_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_77 = contiguous_116.view(32, -1, 80);  contiguous_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_77 = key_states_76.reshape(32, -1, 80);  key_states_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_77 = value_states_76.reshape(32, -1, 80);  value_states_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_193 = key_states_77.transpose(1, 2);  key_states_77 = None
    attn_weights_112 = torch.bmm(query_states_77, transpose_193);  query_states_77 = transpose_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_233 = attn_weights_112.view(1, 32, 128, 128);  attn_weights_112 = None
    attn_weights_113 = view_233 + attention_mask;  view_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_114 = attn_weights_113.view(32, 128, 128);  attn_weights_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_115 = torch.nn.functional.softmax(attn_weights_114, dim = -1);  attn_weights_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_38 = torch.nn.functional.dropout(attn_weights_115, p = 0.0, training = False);  attn_weights_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_190 = torch.bmm(attn_probs_38, value_states_77);  attn_probs_38 = value_states_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_191 = attn_output_190.view(1, 32, 128, 80);  attn_output_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_192 = attn_output_191.transpose(1, 2);  attn_output_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_193 = attn_output_192.reshape(1, 128, 2560);  attn_output_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_298 = self.L__mod___model_decoder_layers_18_self_attn_out_proj(attn_output_193);  attn_output_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:424, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_299 = torch.nn.functional.dropout(hidden_states_298, p = 0.1, training = False);  hidden_states_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    residual_59 = residual_58 + hidden_states_299;  residual_58 = hidden_states_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_301 = self.L__mod___model_decoder_layers_18_encoder_attn_layer_norm(residual_59)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_18_encoder_attn_q_proj = self.L__mod___model_decoder_layers_18_encoder_attn_q_proj(hidden_states_301);  hidden_states_301 = None
    query_states_78 = l__mod___model_decoder_layers_18_encoder_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_18_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_18_encoder_attn_k_proj = self.L__mod___model_decoder_layers_18_encoder_attn_k_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_236 = l__mod___model_decoder_layers_18_encoder_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_18_encoder_attn_k_proj = None
    transpose_195 = view_236.transpose(1, 2);  view_236 = None
    key_states_78 = transpose_195.contiguous();  transpose_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_18_encoder_attn_v_proj = self.L__mod___model_decoder_layers_18_encoder_attn_v_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_237 = l__mod___model_decoder_layers_18_encoder_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_18_encoder_attn_v_proj = None
    transpose_196 = view_237.transpose(1, 2);  view_237 = None
    value_states_78 = transpose_196.contiguous();  transpose_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_238 = query_states_78.view(1, 128, 32, 80);  query_states_78 = None
    transpose_197 = view_238.transpose(1, 2);  view_238 = None
    contiguous_119 = transpose_197.contiguous();  transpose_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_79 = contiguous_119.view(32, -1, 80);  contiguous_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_79 = key_states_78.reshape(32, -1, 80);  key_states_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_79 = value_states_78.reshape(32, -1, 80);  value_states_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_198 = key_states_79.transpose(1, 2);  key_states_79 = None
    attn_weights_116 = torch.bmm(query_states_79, transpose_198);  query_states_79 = transpose_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_117 = torch.nn.functional.softmax(attn_weights_116, dim = -1);  attn_weights_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_39 = torch.nn.functional.dropout(attn_weights_117, p = 0.0, training = False);  attn_weights_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_195 = torch.bmm(attn_probs_39, value_states_79);  attn_probs_39 = value_states_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_196 = attn_output_195.view(1, 32, 128, 80);  attn_output_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_197 = attn_output_196.transpose(1, 2);  attn_output_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_198 = attn_output_197.reshape(1, 128, 2560);  attn_output_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_302 = self.L__mod___model_decoder_layers_18_encoder_attn_out_proj(attn_output_198);  attn_output_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:444, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_303 = torch.nn.functional.dropout(hidden_states_302, p = 0.1, training = False);  hidden_states_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    residual_60 = residual_59 + hidden_states_303;  residual_59 = hidden_states_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_305 = self.L__mod___model_decoder_layers_18_final_layer_norm(residual_60)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_18_fc1 = self.L__mod___model_decoder_layers_18_fc1(hidden_states_305);  hidden_states_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_306 = torch._C._nn.gelu(l__mod___model_decoder_layers_18_fc1);  l__mod___model_decoder_layers_18_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:454, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_307 = torch.nn.functional.dropout(hidden_states_306, p = 0.0, training = False);  hidden_states_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    hidden_states_308 = self.L__mod___model_decoder_layers_18_fc2(hidden_states_307);  hidden_states_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:456, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_309 = torch.nn.functional.dropout(hidden_states_308, p = 0.1, training = False);  hidden_states_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    residual_61 = residual_60 + hidden_states_309;  residual_60 = hidden_states_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_312 = self.L__mod___model_decoder_layers_19_self_attn_layer_norm(residual_61)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_19_self_attn_q_proj = self.L__mod___model_decoder_layers_19_self_attn_q_proj(hidden_states_312)
    query_states_80 = l__mod___model_decoder_layers_19_self_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_19_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_19_self_attn_k_proj = self.L__mod___model_decoder_layers_19_self_attn_k_proj(hidden_states_312)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_241 = l__mod___model_decoder_layers_19_self_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_19_self_attn_k_proj = None
    transpose_200 = view_241.transpose(1, 2);  view_241 = None
    key_states_80 = transpose_200.contiguous();  transpose_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_19_self_attn_v_proj = self.L__mod___model_decoder_layers_19_self_attn_v_proj(hidden_states_312);  hidden_states_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_242 = l__mod___model_decoder_layers_19_self_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_19_self_attn_v_proj = None
    transpose_201 = view_242.transpose(1, 2);  view_242 = None
    value_states_80 = transpose_201.contiguous();  transpose_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_243 = query_states_80.view(1, 128, 32, 80);  query_states_80 = None
    transpose_202 = view_243.transpose(1, 2);  view_243 = None
    contiguous_122 = transpose_202.contiguous();  transpose_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_81 = contiguous_122.view(32, -1, 80);  contiguous_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_81 = key_states_80.reshape(32, -1, 80);  key_states_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_81 = value_states_80.reshape(32, -1, 80);  value_states_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_203 = key_states_81.transpose(1, 2);  key_states_81 = None
    attn_weights_118 = torch.bmm(query_states_81, transpose_203);  query_states_81 = transpose_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_245 = attn_weights_118.view(1, 32, 128, 128);  attn_weights_118 = None
    attn_weights_119 = view_245 + attention_mask;  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_120 = attn_weights_119.view(32, 128, 128);  attn_weights_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_121 = torch.nn.functional.softmax(attn_weights_120, dim = -1);  attn_weights_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_40 = torch.nn.functional.dropout(attn_weights_121, p = 0.0, training = False);  attn_weights_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_200 = torch.bmm(attn_probs_40, value_states_81);  attn_probs_40 = value_states_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_201 = attn_output_200.view(1, 32, 128, 80);  attn_output_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_202 = attn_output_201.transpose(1, 2);  attn_output_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_203 = attn_output_202.reshape(1, 128, 2560);  attn_output_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_313 = self.L__mod___model_decoder_layers_19_self_attn_out_proj(attn_output_203);  attn_output_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:424, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_314 = torch.nn.functional.dropout(hidden_states_313, p = 0.1, training = False);  hidden_states_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    residual_62 = residual_61 + hidden_states_314;  residual_61 = hidden_states_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_316 = self.L__mod___model_decoder_layers_19_encoder_attn_layer_norm(residual_62)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_19_encoder_attn_q_proj = self.L__mod___model_decoder_layers_19_encoder_attn_q_proj(hidden_states_316);  hidden_states_316 = None
    query_states_82 = l__mod___model_decoder_layers_19_encoder_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_19_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_19_encoder_attn_k_proj = self.L__mod___model_decoder_layers_19_encoder_attn_k_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_248 = l__mod___model_decoder_layers_19_encoder_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_19_encoder_attn_k_proj = None
    transpose_205 = view_248.transpose(1, 2);  view_248 = None
    key_states_82 = transpose_205.contiguous();  transpose_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_19_encoder_attn_v_proj = self.L__mod___model_decoder_layers_19_encoder_attn_v_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_249 = l__mod___model_decoder_layers_19_encoder_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_19_encoder_attn_v_proj = None
    transpose_206 = view_249.transpose(1, 2);  view_249 = None
    value_states_82 = transpose_206.contiguous();  transpose_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_250 = query_states_82.view(1, 128, 32, 80);  query_states_82 = None
    transpose_207 = view_250.transpose(1, 2);  view_250 = None
    contiguous_125 = transpose_207.contiguous();  transpose_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_83 = contiguous_125.view(32, -1, 80);  contiguous_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_83 = key_states_82.reshape(32, -1, 80);  key_states_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_83 = value_states_82.reshape(32, -1, 80);  value_states_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_208 = key_states_83.transpose(1, 2);  key_states_83 = None
    attn_weights_122 = torch.bmm(query_states_83, transpose_208);  query_states_83 = transpose_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_123 = torch.nn.functional.softmax(attn_weights_122, dim = -1);  attn_weights_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_41 = torch.nn.functional.dropout(attn_weights_123, p = 0.0, training = False);  attn_weights_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_205 = torch.bmm(attn_probs_41, value_states_83);  attn_probs_41 = value_states_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_206 = attn_output_205.view(1, 32, 128, 80);  attn_output_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_207 = attn_output_206.transpose(1, 2);  attn_output_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_208 = attn_output_207.reshape(1, 128, 2560);  attn_output_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_317 = self.L__mod___model_decoder_layers_19_encoder_attn_out_proj(attn_output_208);  attn_output_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:444, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_318 = torch.nn.functional.dropout(hidden_states_317, p = 0.1, training = False);  hidden_states_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    residual_63 = residual_62 + hidden_states_318;  residual_62 = hidden_states_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_320 = self.L__mod___model_decoder_layers_19_final_layer_norm(residual_63)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_19_fc1 = self.L__mod___model_decoder_layers_19_fc1(hidden_states_320);  hidden_states_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_321 = torch._C._nn.gelu(l__mod___model_decoder_layers_19_fc1);  l__mod___model_decoder_layers_19_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:454, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_322 = torch.nn.functional.dropout(hidden_states_321, p = 0.0, training = False);  hidden_states_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    hidden_states_323 = self.L__mod___model_decoder_layers_19_fc2(hidden_states_322);  hidden_states_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:456, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_324 = torch.nn.functional.dropout(hidden_states_323, p = 0.1, training = False);  hidden_states_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    residual_64 = residual_63 + hidden_states_324;  residual_63 = hidden_states_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_327 = self.L__mod___model_decoder_layers_20_self_attn_layer_norm(residual_64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_20_self_attn_q_proj = self.L__mod___model_decoder_layers_20_self_attn_q_proj(hidden_states_327)
    query_states_84 = l__mod___model_decoder_layers_20_self_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_20_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_20_self_attn_k_proj = self.L__mod___model_decoder_layers_20_self_attn_k_proj(hidden_states_327)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_253 = l__mod___model_decoder_layers_20_self_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_20_self_attn_k_proj = None
    transpose_210 = view_253.transpose(1, 2);  view_253 = None
    key_states_84 = transpose_210.contiguous();  transpose_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_20_self_attn_v_proj = self.L__mod___model_decoder_layers_20_self_attn_v_proj(hidden_states_327);  hidden_states_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_254 = l__mod___model_decoder_layers_20_self_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_20_self_attn_v_proj = None
    transpose_211 = view_254.transpose(1, 2);  view_254 = None
    value_states_84 = transpose_211.contiguous();  transpose_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_255 = query_states_84.view(1, 128, 32, 80);  query_states_84 = None
    transpose_212 = view_255.transpose(1, 2);  view_255 = None
    contiguous_128 = transpose_212.contiguous();  transpose_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_85 = contiguous_128.view(32, -1, 80);  contiguous_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_85 = key_states_84.reshape(32, -1, 80);  key_states_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_85 = value_states_84.reshape(32, -1, 80);  value_states_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_213 = key_states_85.transpose(1, 2);  key_states_85 = None
    attn_weights_124 = torch.bmm(query_states_85, transpose_213);  query_states_85 = transpose_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_257 = attn_weights_124.view(1, 32, 128, 128);  attn_weights_124 = None
    attn_weights_125 = view_257 + attention_mask;  view_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_126 = attn_weights_125.view(32, 128, 128);  attn_weights_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_127 = torch.nn.functional.softmax(attn_weights_126, dim = -1);  attn_weights_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_42 = torch.nn.functional.dropout(attn_weights_127, p = 0.0, training = False);  attn_weights_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_210 = torch.bmm(attn_probs_42, value_states_85);  attn_probs_42 = value_states_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_211 = attn_output_210.view(1, 32, 128, 80);  attn_output_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_212 = attn_output_211.transpose(1, 2);  attn_output_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_213 = attn_output_212.reshape(1, 128, 2560);  attn_output_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_328 = self.L__mod___model_decoder_layers_20_self_attn_out_proj(attn_output_213);  attn_output_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:424, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_329 = torch.nn.functional.dropout(hidden_states_328, p = 0.1, training = False);  hidden_states_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    residual_65 = residual_64 + hidden_states_329;  residual_64 = hidden_states_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_331 = self.L__mod___model_decoder_layers_20_encoder_attn_layer_norm(residual_65)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_20_encoder_attn_q_proj = self.L__mod___model_decoder_layers_20_encoder_attn_q_proj(hidden_states_331);  hidden_states_331 = None
    query_states_86 = l__mod___model_decoder_layers_20_encoder_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_20_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_20_encoder_attn_k_proj = self.L__mod___model_decoder_layers_20_encoder_attn_k_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_260 = l__mod___model_decoder_layers_20_encoder_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_20_encoder_attn_k_proj = None
    transpose_215 = view_260.transpose(1, 2);  view_260 = None
    key_states_86 = transpose_215.contiguous();  transpose_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_20_encoder_attn_v_proj = self.L__mod___model_decoder_layers_20_encoder_attn_v_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_261 = l__mod___model_decoder_layers_20_encoder_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_20_encoder_attn_v_proj = None
    transpose_216 = view_261.transpose(1, 2);  view_261 = None
    value_states_86 = transpose_216.contiguous();  transpose_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_262 = query_states_86.view(1, 128, 32, 80);  query_states_86 = None
    transpose_217 = view_262.transpose(1, 2);  view_262 = None
    contiguous_131 = transpose_217.contiguous();  transpose_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_87 = contiguous_131.view(32, -1, 80);  contiguous_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_87 = key_states_86.reshape(32, -1, 80);  key_states_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_87 = value_states_86.reshape(32, -1, 80);  value_states_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_218 = key_states_87.transpose(1, 2);  key_states_87 = None
    attn_weights_128 = torch.bmm(query_states_87, transpose_218);  query_states_87 = transpose_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_129 = torch.nn.functional.softmax(attn_weights_128, dim = -1);  attn_weights_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_43 = torch.nn.functional.dropout(attn_weights_129, p = 0.0, training = False);  attn_weights_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_215 = torch.bmm(attn_probs_43, value_states_87);  attn_probs_43 = value_states_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_216 = attn_output_215.view(1, 32, 128, 80);  attn_output_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_217 = attn_output_216.transpose(1, 2);  attn_output_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_218 = attn_output_217.reshape(1, 128, 2560);  attn_output_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_332 = self.L__mod___model_decoder_layers_20_encoder_attn_out_proj(attn_output_218);  attn_output_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:444, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_333 = torch.nn.functional.dropout(hidden_states_332, p = 0.1, training = False);  hidden_states_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    residual_66 = residual_65 + hidden_states_333;  residual_65 = hidden_states_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_335 = self.L__mod___model_decoder_layers_20_final_layer_norm(residual_66)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_20_fc1 = self.L__mod___model_decoder_layers_20_fc1(hidden_states_335);  hidden_states_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_336 = torch._C._nn.gelu(l__mod___model_decoder_layers_20_fc1);  l__mod___model_decoder_layers_20_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:454, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_337 = torch.nn.functional.dropout(hidden_states_336, p = 0.0, training = False);  hidden_states_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    hidden_states_338 = self.L__mod___model_decoder_layers_20_fc2(hidden_states_337);  hidden_states_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:456, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_339 = torch.nn.functional.dropout(hidden_states_338, p = 0.1, training = False);  hidden_states_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    residual_67 = residual_66 + hidden_states_339;  residual_66 = hidden_states_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_342 = self.L__mod___model_decoder_layers_21_self_attn_layer_norm(residual_67)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_21_self_attn_q_proj = self.L__mod___model_decoder_layers_21_self_attn_q_proj(hidden_states_342)
    query_states_88 = l__mod___model_decoder_layers_21_self_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_21_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_21_self_attn_k_proj = self.L__mod___model_decoder_layers_21_self_attn_k_proj(hidden_states_342)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_265 = l__mod___model_decoder_layers_21_self_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_21_self_attn_k_proj = None
    transpose_220 = view_265.transpose(1, 2);  view_265 = None
    key_states_88 = transpose_220.contiguous();  transpose_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_21_self_attn_v_proj = self.L__mod___model_decoder_layers_21_self_attn_v_proj(hidden_states_342);  hidden_states_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_266 = l__mod___model_decoder_layers_21_self_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_21_self_attn_v_proj = None
    transpose_221 = view_266.transpose(1, 2);  view_266 = None
    value_states_88 = transpose_221.contiguous();  transpose_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_267 = query_states_88.view(1, 128, 32, 80);  query_states_88 = None
    transpose_222 = view_267.transpose(1, 2);  view_267 = None
    contiguous_134 = transpose_222.contiguous();  transpose_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_89 = contiguous_134.view(32, -1, 80);  contiguous_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_89 = key_states_88.reshape(32, -1, 80);  key_states_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_89 = value_states_88.reshape(32, -1, 80);  value_states_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_223 = key_states_89.transpose(1, 2);  key_states_89 = None
    attn_weights_130 = torch.bmm(query_states_89, transpose_223);  query_states_89 = transpose_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_269 = attn_weights_130.view(1, 32, 128, 128);  attn_weights_130 = None
    attn_weights_131 = view_269 + attention_mask;  view_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_132 = attn_weights_131.view(32, 128, 128);  attn_weights_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_133 = torch.nn.functional.softmax(attn_weights_132, dim = -1);  attn_weights_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_44 = torch.nn.functional.dropout(attn_weights_133, p = 0.0, training = False);  attn_weights_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_220 = torch.bmm(attn_probs_44, value_states_89);  attn_probs_44 = value_states_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_221 = attn_output_220.view(1, 32, 128, 80);  attn_output_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_222 = attn_output_221.transpose(1, 2);  attn_output_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_223 = attn_output_222.reshape(1, 128, 2560);  attn_output_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_343 = self.L__mod___model_decoder_layers_21_self_attn_out_proj(attn_output_223);  attn_output_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:424, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_344 = torch.nn.functional.dropout(hidden_states_343, p = 0.1, training = False);  hidden_states_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    residual_68 = residual_67 + hidden_states_344;  residual_67 = hidden_states_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_346 = self.L__mod___model_decoder_layers_21_encoder_attn_layer_norm(residual_68)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_21_encoder_attn_q_proj = self.L__mod___model_decoder_layers_21_encoder_attn_q_proj(hidden_states_346);  hidden_states_346 = None
    query_states_90 = l__mod___model_decoder_layers_21_encoder_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_21_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_21_encoder_attn_k_proj = self.L__mod___model_decoder_layers_21_encoder_attn_k_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_272 = l__mod___model_decoder_layers_21_encoder_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_21_encoder_attn_k_proj = None
    transpose_225 = view_272.transpose(1, 2);  view_272 = None
    key_states_90 = transpose_225.contiguous();  transpose_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_21_encoder_attn_v_proj = self.L__mod___model_decoder_layers_21_encoder_attn_v_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_273 = l__mod___model_decoder_layers_21_encoder_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_21_encoder_attn_v_proj = None
    transpose_226 = view_273.transpose(1, 2);  view_273 = None
    value_states_90 = transpose_226.contiguous();  transpose_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_274 = query_states_90.view(1, 128, 32, 80);  query_states_90 = None
    transpose_227 = view_274.transpose(1, 2);  view_274 = None
    contiguous_137 = transpose_227.contiguous();  transpose_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_91 = contiguous_137.view(32, -1, 80);  contiguous_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_91 = key_states_90.reshape(32, -1, 80);  key_states_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_91 = value_states_90.reshape(32, -1, 80);  value_states_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_228 = key_states_91.transpose(1, 2);  key_states_91 = None
    attn_weights_134 = torch.bmm(query_states_91, transpose_228);  query_states_91 = transpose_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_135 = torch.nn.functional.softmax(attn_weights_134, dim = -1);  attn_weights_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_45 = torch.nn.functional.dropout(attn_weights_135, p = 0.0, training = False);  attn_weights_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_225 = torch.bmm(attn_probs_45, value_states_91);  attn_probs_45 = value_states_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_226 = attn_output_225.view(1, 32, 128, 80);  attn_output_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_227 = attn_output_226.transpose(1, 2);  attn_output_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_228 = attn_output_227.reshape(1, 128, 2560);  attn_output_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_347 = self.L__mod___model_decoder_layers_21_encoder_attn_out_proj(attn_output_228);  attn_output_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:444, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_348 = torch.nn.functional.dropout(hidden_states_347, p = 0.1, training = False);  hidden_states_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    residual_69 = residual_68 + hidden_states_348;  residual_68 = hidden_states_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_350 = self.L__mod___model_decoder_layers_21_final_layer_norm(residual_69)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_21_fc1 = self.L__mod___model_decoder_layers_21_fc1(hidden_states_350);  hidden_states_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_351 = torch._C._nn.gelu(l__mod___model_decoder_layers_21_fc1);  l__mod___model_decoder_layers_21_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:454, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_352 = torch.nn.functional.dropout(hidden_states_351, p = 0.0, training = False);  hidden_states_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    hidden_states_353 = self.L__mod___model_decoder_layers_21_fc2(hidden_states_352);  hidden_states_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:456, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_354 = torch.nn.functional.dropout(hidden_states_353, p = 0.1, training = False);  hidden_states_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    residual_70 = residual_69 + hidden_states_354;  residual_69 = hidden_states_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_357 = self.L__mod___model_decoder_layers_22_self_attn_layer_norm(residual_70)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_22_self_attn_q_proj = self.L__mod___model_decoder_layers_22_self_attn_q_proj(hidden_states_357)
    query_states_92 = l__mod___model_decoder_layers_22_self_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_22_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_22_self_attn_k_proj = self.L__mod___model_decoder_layers_22_self_attn_k_proj(hidden_states_357)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_277 = l__mod___model_decoder_layers_22_self_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_22_self_attn_k_proj = None
    transpose_230 = view_277.transpose(1, 2);  view_277 = None
    key_states_92 = transpose_230.contiguous();  transpose_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_22_self_attn_v_proj = self.L__mod___model_decoder_layers_22_self_attn_v_proj(hidden_states_357);  hidden_states_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_278 = l__mod___model_decoder_layers_22_self_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_22_self_attn_v_proj = None
    transpose_231 = view_278.transpose(1, 2);  view_278 = None
    value_states_92 = transpose_231.contiguous();  transpose_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_279 = query_states_92.view(1, 128, 32, 80);  query_states_92 = None
    transpose_232 = view_279.transpose(1, 2);  view_279 = None
    contiguous_140 = transpose_232.contiguous();  transpose_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_93 = contiguous_140.view(32, -1, 80);  contiguous_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_93 = key_states_92.reshape(32, -1, 80);  key_states_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_93 = value_states_92.reshape(32, -1, 80);  value_states_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_233 = key_states_93.transpose(1, 2);  key_states_93 = None
    attn_weights_136 = torch.bmm(query_states_93, transpose_233);  query_states_93 = transpose_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_281 = attn_weights_136.view(1, 32, 128, 128);  attn_weights_136 = None
    attn_weights_137 = view_281 + attention_mask;  view_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_138 = attn_weights_137.view(32, 128, 128);  attn_weights_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_139 = torch.nn.functional.softmax(attn_weights_138, dim = -1);  attn_weights_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_46 = torch.nn.functional.dropout(attn_weights_139, p = 0.0, training = False);  attn_weights_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_230 = torch.bmm(attn_probs_46, value_states_93);  attn_probs_46 = value_states_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_231 = attn_output_230.view(1, 32, 128, 80);  attn_output_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_232 = attn_output_231.transpose(1, 2);  attn_output_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_233 = attn_output_232.reshape(1, 128, 2560);  attn_output_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_358 = self.L__mod___model_decoder_layers_22_self_attn_out_proj(attn_output_233);  attn_output_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:424, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_359 = torch.nn.functional.dropout(hidden_states_358, p = 0.1, training = False);  hidden_states_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    residual_71 = residual_70 + hidden_states_359;  residual_70 = hidden_states_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_361 = self.L__mod___model_decoder_layers_22_encoder_attn_layer_norm(residual_71)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_22_encoder_attn_q_proj = self.L__mod___model_decoder_layers_22_encoder_attn_q_proj(hidden_states_361);  hidden_states_361 = None
    query_states_94 = l__mod___model_decoder_layers_22_encoder_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_22_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_22_encoder_attn_k_proj = self.L__mod___model_decoder_layers_22_encoder_attn_k_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_284 = l__mod___model_decoder_layers_22_encoder_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_22_encoder_attn_k_proj = None
    transpose_235 = view_284.transpose(1, 2);  view_284 = None
    key_states_94 = transpose_235.contiguous();  transpose_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_22_encoder_attn_v_proj = self.L__mod___model_decoder_layers_22_encoder_attn_v_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_285 = l__mod___model_decoder_layers_22_encoder_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_22_encoder_attn_v_proj = None
    transpose_236 = view_285.transpose(1, 2);  view_285 = None
    value_states_94 = transpose_236.contiguous();  transpose_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_286 = query_states_94.view(1, 128, 32, 80);  query_states_94 = None
    transpose_237 = view_286.transpose(1, 2);  view_286 = None
    contiguous_143 = transpose_237.contiguous();  transpose_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_95 = contiguous_143.view(32, -1, 80);  contiguous_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_95 = key_states_94.reshape(32, -1, 80);  key_states_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_95 = value_states_94.reshape(32, -1, 80);  value_states_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_238 = key_states_95.transpose(1, 2);  key_states_95 = None
    attn_weights_140 = torch.bmm(query_states_95, transpose_238);  query_states_95 = transpose_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_141 = torch.nn.functional.softmax(attn_weights_140, dim = -1);  attn_weights_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_47 = torch.nn.functional.dropout(attn_weights_141, p = 0.0, training = False);  attn_weights_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_235 = torch.bmm(attn_probs_47, value_states_95);  attn_probs_47 = value_states_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_236 = attn_output_235.view(1, 32, 128, 80);  attn_output_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_237 = attn_output_236.transpose(1, 2);  attn_output_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_238 = attn_output_237.reshape(1, 128, 2560);  attn_output_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_362 = self.L__mod___model_decoder_layers_22_encoder_attn_out_proj(attn_output_238);  attn_output_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:444, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_363 = torch.nn.functional.dropout(hidden_states_362, p = 0.1, training = False);  hidden_states_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    residual_72 = residual_71 + hidden_states_363;  residual_71 = hidden_states_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_365 = self.L__mod___model_decoder_layers_22_final_layer_norm(residual_72)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_22_fc1 = self.L__mod___model_decoder_layers_22_fc1(hidden_states_365);  hidden_states_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_366 = torch._C._nn.gelu(l__mod___model_decoder_layers_22_fc1);  l__mod___model_decoder_layers_22_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:454, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_367 = torch.nn.functional.dropout(hidden_states_366, p = 0.0, training = False);  hidden_states_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    hidden_states_368 = self.L__mod___model_decoder_layers_22_fc2(hidden_states_367);  hidden_states_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:456, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_369 = torch.nn.functional.dropout(hidden_states_368, p = 0.1, training = False);  hidden_states_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    residual_73 = residual_72 + hidden_states_369;  residual_72 = hidden_states_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:411, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states_372 = self.L__mod___model_decoder_layers_23_self_attn_layer_norm(residual_73)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_23_self_attn_q_proj = self.L__mod___model_decoder_layers_23_self_attn_q_proj(hidden_states_372)
    query_states_96 = l__mod___model_decoder_layers_23_self_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_23_self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_23_self_attn_k_proj = self.L__mod___model_decoder_layers_23_self_attn_k_proj(hidden_states_372)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_289 = l__mod___model_decoder_layers_23_self_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_23_self_attn_k_proj = None
    transpose_240 = view_289.transpose(1, 2);  view_289 = None
    key_states_96 = transpose_240.contiguous();  transpose_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__mod___model_decoder_layers_23_self_attn_v_proj = self.L__mod___model_decoder_layers_23_self_attn_v_proj(hidden_states_372);  hidden_states_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_290 = l__mod___model_decoder_layers_23_self_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_23_self_attn_v_proj = None
    transpose_241 = view_290.transpose(1, 2);  view_290 = None
    value_states_96 = transpose_241.contiguous();  transpose_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_291 = query_states_96.view(1, 128, 32, 80);  query_states_96 = None
    transpose_242 = view_291.transpose(1, 2);  view_291 = None
    contiguous_146 = transpose_242.contiguous();  transpose_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_97 = contiguous_146.view(32, -1, 80);  contiguous_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_97 = key_states_96.reshape(32, -1, 80);  key_states_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_97 = value_states_96.reshape(32, -1, 80);  value_states_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_243 = key_states_97.transpose(1, 2);  key_states_97 = None
    attn_weights_142 = torch.bmm(query_states_97, transpose_243);  query_states_97 = transpose_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_293 = attn_weights_142.view(1, 32, 128, 128);  attn_weights_142 = None
    attn_weights_143 = view_293 + attention_mask;  view_293 = attention_mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:238, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_144 = attn_weights_143.view(32, 128, 128);  attn_weights_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_145 = torch.nn.functional.softmax(attn_weights_144, dim = -1);  attn_weights_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_48 = torch.nn.functional.dropout(attn_weights_145, p = 0.0, training = False);  attn_weights_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_240 = torch.bmm(attn_probs_48, value_states_97);  attn_probs_48 = value_states_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_241 = attn_output_240.view(1, 32, 128, 80);  attn_output_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_242 = attn_output_241.transpose(1, 2);  attn_output_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_243 = attn_output_242.reshape(1, 128, 2560);  attn_output_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_373 = self.L__mod___model_decoder_layers_23_self_attn_out_proj(attn_output_243);  attn_output_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:424, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_374 = torch.nn.functional.dropout(hidden_states_373, p = 0.1, training = False);  hidden_states_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:425, code: hidden_states = residual + hidden_states
    residual_74 = residual_73 + hidden_states_374;  residual_73 = hidden_states_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:432, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states_376 = self.L__mod___model_decoder_layers_23_encoder_attn_layer_norm(residual_74)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__mod___model_decoder_layers_23_encoder_attn_q_proj = self.L__mod___model_decoder_layers_23_encoder_attn_q_proj(hidden_states_376);  hidden_states_376 = None
    query_states_98 = l__mod___model_decoder_layers_23_encoder_attn_q_proj * 0.11180339887498948;  l__mod___model_decoder_layers_23_encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:195, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_23_encoder_attn_k_proj = self.L__mod___model_decoder_layers_23_encoder_attn_k_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_296 = l__mod___model_decoder_layers_23_encoder_attn_k_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_23_encoder_attn_k_proj = None
    transpose_245 = view_296.transpose(1, 2);  view_296 = None
    key_states_98 = transpose_245.contiguous();  transpose_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:196, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__mod___model_decoder_layers_23_encoder_attn_v_proj = self.L__mod___model_decoder_layers_23_encoder_attn_v_proj(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_297 = l__mod___model_decoder_layers_23_encoder_attn_v_proj.view(1, -1, 32, 80);  l__mod___model_decoder_layers_23_encoder_attn_v_proj = None
    transpose_246 = view_297.transpose(1, 2);  view_297 = None
    value_states_98 = transpose_246.contiguous();  transpose_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_298 = query_states_98.view(1, 128, 32, 80);  query_states_98 = None
    transpose_247 = view_298.transpose(1, 2);  view_298 = None
    contiguous_149 = transpose_247.contiguous();  transpose_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_99 = contiguous_149.view(32, -1, 80);  contiguous_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    key_states_99 = key_states_98.reshape(32, -1, 80);  key_states_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    value_states_99 = value_states_98.reshape(32, -1, 80);  value_states_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_248 = key_states_99.transpose(1, 2);  key_states_99 = None
    attn_weights_146 = torch.bmm(query_states_99, transpose_248);  query_states_99 = transpose_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_147 = torch.nn.functional.softmax(attn_weights_146, dim = -1);  attn_weights_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_49 = torch.nn.functional.dropout(attn_weights_147, p = 0.0, training = False);  attn_weights_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_245 = torch.bmm(attn_probs_49, value_states_99);  attn_probs_49 = value_states_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_246 = attn_output_245.view(1, 32, 128, 80);  attn_output_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_247 = attn_output_246.transpose(1, 2);  attn_output_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_248 = attn_output_247.reshape(1, 128, 2560);  attn_output_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_377 = self.L__mod___model_decoder_layers_23_encoder_attn_out_proj(attn_output_248);  attn_output_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:444, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_378 = torch.nn.functional.dropout(hidden_states_377, p = 0.1, training = False);  hidden_states_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:445, code: hidden_states = residual + hidden_states
    residual_75 = residual_74 + hidden_states_378;  residual_74 = hidden_states_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:452, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_380 = self.L__mod___model_decoder_layers_23_final_layer_norm(residual_75)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:453, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__mod___model_decoder_layers_23_fc1 = self.L__mod___model_decoder_layers_23_fc1(hidden_states_380);  hidden_states_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_381 = torch._C._nn.gelu(l__mod___model_decoder_layers_23_fc1);  l__mod___model_decoder_layers_23_fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:454, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_382 = torch.nn.functional.dropout(hidden_states_381, p = 0.0, training = False);  hidden_states_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:455, code: hidden_states = self.fc2(hidden_states)
    hidden_states_383 = self.L__mod___model_decoder_layers_23_fc2(hidden_states_382);  hidden_states_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:456, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_384 = torch.nn.functional.dropout(hidden_states_383, p = 0.1, training = False);  hidden_states_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:457, code: hidden_states = residual + hidden_states
    hidden_states_386 = residual_75 + hidden_states_384;  residual_75 = hidden_states_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:1079, code: hidden_states = self.layer_norm(hidden_states)
    hidden_states_387 = self.L__mod___model_decoder_layer_norm(hidden_states_386);  hidden_states_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:1358, code: lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
    l__mod___lm_head = self.L__mod___lm_head(hidden_states_387);  hidden_states_387 = None
    l__mod___final_logits_bias = self.L__mod___final_logits_bias
    lm_logits = l__mod___lm_head + l__mod___final_logits_bias;  l__mod___lm_head = l__mod___final_logits_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:1363, code: masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
    view_301 = lm_logits.view(-1, 8008)
    view_302 = l_inputs_labels_.view(-1);  l_inputs_labels_ = None
    masked_lm_loss = torch.nn.functional.cross_entropy(view_301, view_302, None, None, -100, None, 'mean', 0.0);  view_301 = view_302 = None
    return (masked_lm_loss, lm_logits, hidden_states_24)
    