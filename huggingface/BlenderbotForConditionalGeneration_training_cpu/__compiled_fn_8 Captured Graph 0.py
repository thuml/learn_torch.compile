from __future__ import annotations



def forward(self, L_hidden_states_ : torch.Tensor):
    residual = L_hidden_states_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:320, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states = self.L__self___self_attn_layer_norm(residual)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    l__self___self_attn_q_proj = self.L__self___self_attn_q_proj(hidden_states)
    query_states = l__self___self_attn_q_proj * 0.11180339887498948;  l__self___self_attn_q_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    l__self___self_attn_k_proj = self.L__self___self_attn_k_proj(hidden_states)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view = l__self___self_attn_k_proj.view(1, -1, 32, 80);  l__self___self_attn_k_proj = None
    transpose = view.transpose(1, 2);  view = None
    key_states = transpose.contiguous();  transpose = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    l__self___self_attn_v_proj = self.L__self___self_attn_v_proj(hidden_states);  hidden_states = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_1 = l__self___self_attn_v_proj.view(1, -1, 32, 80);  l__self___self_attn_v_proj = None
    transpose_1 = view_1.transpose(1, 2);  view_1 = None
    value_states = transpose_1.contiguous();  transpose_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_2 = query_states.view(1, 128, 32, 80);  query_states = None
    transpose_2 = view_2.transpose(1, 2);  view_2 = None
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
    attn_probs = torch.nn.functional.dropout(attn_weights_1, p = 0.0, training = True);  attn_weights_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    attn_output = torch.bmm(attn_probs, value_states_1);  attn_probs = value_states_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_1 = attn_output.view(1, 32, 128, 80);  attn_output = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    attn_output_2 = attn_output_1.transpose(1, 2);  attn_output_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output_3 = attn_output_2.reshape(1, 128, 2560);  attn_output_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    hidden_states_1 = self.L__self___self_attn_out_proj(attn_output_3);  attn_output_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:327, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_2 = torch.nn.functional.dropout(hidden_states_1, p = 0.1, training = True);  hidden_states_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:328, code: hidden_states = residual + hidden_states
    residual_1 = residual + hidden_states_2;  residual = hidden_states_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:331, code: hidden_states = self.final_layer_norm(hidden_states)
    hidden_states_4 = self.L__self___final_layer_norm(residual_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:332, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    l__self___fc1 = self.L__self___fc1(hidden_states_4);  hidden_states_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_5 = torch._C._nn.gelu(l__self___fc1);  l__self___fc1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:333, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states_6 = torch.nn.functional.dropout(hidden_states_5, p = 0.0, training = True);  hidden_states_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:334, code: hidden_states = self.fc2(hidden_states)
    hidden_states_7 = self.L__self___fc2(hidden_states_6);  hidden_states_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:335, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states_8 = torch.nn.functional.dropout(hidden_states_7, p = 0.1, training = True);  hidden_states_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:336, code: hidden_states = residual + hidden_states
    hidden_states_9 = residual_1 + hidden_states_8;  residual_1 = hidden_states_8 = None
    return (hidden_states_9,)
    