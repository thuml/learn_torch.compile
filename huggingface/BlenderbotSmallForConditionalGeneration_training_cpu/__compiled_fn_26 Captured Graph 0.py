from __future__ import annotations



def forward(self, L_hidden_states_ : torch.Tensor, L_attention_mask_ : torch.Tensor, L_encoder_hidden_states_ : torch.Tensor):
    residual = L_hidden_states_
    l_attention_mask_ = L_attention_mask_
    l_encoder_hidden_states_ = L_encoder_hidden_states_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__self___self_attn_q_proj = self.L__self___self_attn_q_proj(residual)
    query_states = l__self___self_attn_q_proj * 0.1767766952966369;  l__self___self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__self___self_attn_k_proj = self.L__self___self_attn_k_proj(residual)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view = l__self___self_attn_k_proj.view(1, -1, 16, 32);  l__self___self_attn_k_proj = None
    transpose = view.transpose(1, 2);  view = None
    key_states = transpose.contiguous();  transpose = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__self___self_attn_v_proj = self.L__self___self_attn_v_proj(residual)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_1 = l__self___self_attn_v_proj.view(1, -1, 16, 32);  l__self___self_attn_v_proj = None
    transpose_1 = view_1.transpose(1, 2);  view_1 = None
    value_states = transpose_1.contiguous();  transpose_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_2 = query_states.view(1, 128, 16, 32);  query_states = None
    transpose_2 = view_2.transpose(1, 2);  view_2 = None
    contiguous_2 = transpose_2.contiguous();  transpose_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_1 = contiguous_2.view(16, -1, 32);  contiguous_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    key_states_1 = key_states.reshape(16, -1, 32);  key_states = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    value_states_1 = value_states.reshape(16, -1, 32);  value_states = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:221, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_3 = key_states_1.transpose(1, 2);  key_states_1 = None
    attn_weights = torch.bmm(query_states_1, transpose_3);  query_states_1 = transpose_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:234, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_4 = attn_weights.view(1, 16, 128, 128);  attn_weights = None
    attn_weights_1 = view_4 + l_attention_mask_;  view_4 = l_attention_mask_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:235, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights_2 = attn_weights_1.view(16, 128, 128);  attn_weights_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:237, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_3 = torch.nn.functional.softmax(attn_weights_2, dim = -1);  attn_weights_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:258, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs = torch.nn.functional.dropout(attn_weights_3, p = 0.0, training = True);  attn_weights_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:260, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output = torch.bmm(attn_probs, value_states_1);  attn_probs = value_states_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_1 = attn_output.view(1, 16, 128, 32);  attn_output = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    attn_output_2 = attn_output_1.transpose(1, 2);  attn_output_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_3 = attn_output_2.reshape(1, 128, 512);  attn_output_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    hidden_states = self.L__self___self_attn_out_proj(attn_output_3);  attn_output_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:420, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_1 = torch.nn.functional.dropout(hidden_states, p = 0.1, training = True);  hidden_states = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:421, code: hidden_states = residual + hidden_states
    hidden_states_2 = residual + hidden_states_1;  residual = hidden_states_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:422, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    residual_1 = self.L__self___self_attn_layer_norm(hidden_states_2);  hidden_states_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__self___encoder_attn_q_proj = self.L__self___encoder_attn_q_proj(residual_1)
    query_states_2 = l__self___encoder_attn_q_proj * 0.1767766952966369;  l__self___encoder_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:192, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    l__self___encoder_attn_k_proj = self.L__self___encoder_attn_k_proj(l_encoder_hidden_states_)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_7 = l__self___encoder_attn_k_proj.view(1, -1, 16, 32);  l__self___encoder_attn_k_proj = None
    transpose_5 = view_7.transpose(1, 2);  view_7 = None
    key_states_2 = transpose_5.contiguous();  transpose_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:193, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    l__self___encoder_attn_v_proj = self.L__self___encoder_attn_v_proj(l_encoder_hidden_states_);  l_encoder_hidden_states_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_8 = l__self___encoder_attn_v_proj.view(1, -1, 16, 32);  l__self___encoder_attn_v_proj = None
    transpose_6 = view_8.transpose(1, 2);  view_8 = None
    value_states_2 = transpose_6.contiguous();  transpose_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_9 = query_states_2.view(1, 128, 16, 32);  query_states_2 = None
    transpose_7 = view_9.transpose(1, 2);  view_9 = None
    contiguous_5 = transpose_7.contiguous();  transpose_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    query_states_3 = contiguous_5.view(16, -1, 32);  contiguous_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    key_states_3 = key_states_2.reshape(16, -1, 32);  key_states_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    value_states_3 = value_states_2.reshape(16, -1, 32);  value_states_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:221, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    transpose_8 = key_states_3.transpose(1, 2);  key_states_3 = None
    attn_weights_4 = torch.bmm(query_states_3, transpose_8);  query_states_3 = transpose_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:237, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_5 = torch.nn.functional.softmax(attn_weights_4, dim = -1);  attn_weights_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:258, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_probs_1 = torch.nn.functional.dropout(attn_weights_5, p = 0.0, training = True);  attn_weights_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:260, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output_5 = torch.bmm(attn_probs_1, value_states_3);  attn_probs_1 = value_states_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_6 = attn_output_5.view(1, 16, 128, 32);  attn_output_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    attn_output_7 = attn_output_6.transpose(1, 2);  attn_output_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_8 = attn_output_7.reshape(1, 128, 512);  attn_output_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    hidden_states_4 = self.L__self___encoder_attn_out_proj(attn_output_8);  attn_output_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:440, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_5 = torch.nn.functional.dropout(hidden_states_4, p = 0.1, training = True);  hidden_states_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:441, code: hidden_states = residual + hidden_states
    hidden_states_6 = residual_1 + hidden_states_5;  residual_1 = hidden_states_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:442, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    residual_2 = self.L__self___encoder_attn_layer_norm(hidden_states_6);  hidden_states_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:449, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__self___fc1 = self.L__self___fc1(residual_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_8 = torch._C._nn.gelu(l__self___fc1);  l__self___fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:450, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_9 = torch.nn.functional.dropout(hidden_states_8, p = 0.0, training = True);  hidden_states_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:451, code: hidden_states = self.fc2(hidden_states)
    hidden_states_10 = self.L__self___fc2(hidden_states_9);  hidden_states_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:452, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_11 = torch.nn.functional.dropout(hidden_states_10, p = 0.1, training = True);  hidden_states_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:453, code: hidden_states = residual + hidden_states
    hidden_states_12 = residual_2 + hidden_states_11;  residual_2 = hidden_states_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:454, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_13 = self.L__self___final_layer_norm(hidden_states_12);  hidden_states_12 = None
    return (hidden_states_13,)
    