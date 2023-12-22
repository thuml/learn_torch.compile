from __future__ import annotations



def forward(self, L_inputs_input_ids_ : torch.Tensor, L_inputs_labels_ : torch.Tensor):
    l_inputs_input_ids_ = L_inputs_input_ids_
    l_inputs_labels_ = L_inputs_labels_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:530, code: input_ids = input_ids.view(-1, input_shape[-1])
    input_ids = l_inputs_input_ids_.view(-1, 128);  l_inputs_input_ids_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:552, code: position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
    position_ids = torch.arange(0, 128, dtype = torch.int64, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:553, code: position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
    unsqueeze = position_ids.unsqueeze(0);  position_ids = None
    position_ids_1 = unsqueeze.view(-1, 128);  unsqueeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:582, code: inputs_embeds = self.wte(input_ids)
    inputs_embeds = self.L__mod___transformer_wte(input_ids);  input_ids = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:583, code: position_embeds = self.wpe(position_ids)
    position_embeds = self.L__mod___transformer_wpe(position_ids_1);  position_ids_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:584, code: hidden_states = inputs_embeds + position_embeds
    hidden_states = inputs_embeds + position_embeds;  inputs_embeds = position_embeds = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:590, code: hidden_states = self.drop(hidden_states)
    residual = self.L__mod___transformer_drop(hidden_states);  hidden_states = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_2 = self.L__mod___transformer_h_0_ln_1(residual)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    query = self.L__mod___transformer_h_0_attn_attention_q_proj(hidden_states_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    key = self.L__mod___transformer_h_0_attn_attention_k_proj(hidden_states_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    value = self.L__mod___transformer_h_0_attn_attention_v_proj(hidden_states_2);  hidden_states_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor = query.view((1, 128, 16, 128));  query = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_1 = tensor.permute(0, 2, 1, 3);  tensor = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_1 = key.view((1, 128, 16, 128));  key = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_1 = tensor_1.permute(0, 2, 1, 3);  tensor_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_2 = value.view((1, 128, 16, 128));  value = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_1 = tensor_2.permute(0, 2, 1, 3);  tensor_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:186, code: query = query.to(torch.float32)
    query_2 = query_1.to(torch.float32);  query_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:187, code: key = key.to(torch.float32)
    key_2 = key_1.to(torch.float32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose = key_2.transpose(-1, -2);  key_2 = None
    attn_weights = torch.matmul(query_2, transpose);  query_2 = transpose = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_0_attn_attention_bias = self.L__mod___transformer_h_0_attn_attention_bias
    causal_mask = l__mod___transformer_h_0_attn_attention_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_0_attn_attention_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_3 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value = tensor_3.to(device(type='cuda', index=0));  tensor_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_1 = torch.where(causal_mask, attn_weights, mask_value);  causal_mask = attn_weights = mask_value = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_2 = torch.nn.functional.softmax(attn_weights_1, dim = -1);  attn_weights_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:204, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_3 = attn_weights_2.to(torch.float32);  attn_weights_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_5 = self.L__mod___transformer_h_0_attn_attention_attn_dropout(attn_weights_3);  attn_weights_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    attn_output = torch.matmul(attn_weights_5, value_1);  attn_weights_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_3 = attn_output.permute(0, 2, 1, 3);  attn_output = None
    tensor_4 = permute_3.contiguous();  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    attn_output_1 = tensor_4.view((1, 128, 2048));  tensor_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    attn_output_2 = self.L__mod___transformer_h_0_attn_attention_out_proj(attn_output_1);  attn_output_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    attn_output_4 = self.L__mod___transformer_h_0_attn_attention_resid_dropout(attn_output_2);  attn_output_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    residual_1 = attn_output_4 + residual;  attn_output_4 = residual = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    hidden_states_4 = self.L__mod___transformer_h_0_ln_2(residual_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    hidden_states_5 = self.L__mod___transformer_h_0_mlp_c_fc(hidden_states_4);  hidden_states_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul = 0.5 * hidden_states_5
    pow_1 = torch.pow(hidden_states_5, 3.0)
    mul_1 = 0.044715 * pow_1;  pow_1 = None
    add_2 = hidden_states_5 + mul_1;  hidden_states_5 = mul_1 = None
    mul_2 = 0.7978845608028654 * add_2;  add_2 = None
    tanh = torch.tanh(mul_2);  mul_2 = None
    add_3 = 1.0 + tanh;  tanh = None
    hidden_states_6 = mul * add_3;  mul = add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    hidden_states_7 = self.L__mod___transformer_h_0_mlp_c_proj(hidden_states_6);  hidden_states_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states = self.L__mod___transformer_h_0_mlp_dropout(hidden_states_7);  hidden_states_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    residual_2 = residual_1 + feed_forward_hidden_states;  residual_1 = feed_forward_hidden_states = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_11 = self.L__mod___transformer_h_1_ln_1(residual_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    query_3 = self.L__mod___transformer_h_1_attn_attention_q_proj(hidden_states_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    key_3 = self.L__mod___transformer_h_1_attn_attention_k_proj(hidden_states_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    value_2 = self.L__mod___transformer_h_1_attn_attention_v_proj(hidden_states_11);  hidden_states_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_5 = query_3.view((1, 128, 16, 128));  query_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_4 = tensor_5.permute(0, 2, 1, 3);  tensor_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_6 = key_3.view((1, 128, 16, 128));  key_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_4 = tensor_6.permute(0, 2, 1, 3);  tensor_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_7 = value_2.view((1, 128, 16, 128));  value_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_3 = tensor_7.permute(0, 2, 1, 3);  tensor_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:186, code: query = query.to(torch.float32)
    query_5 = query_4.to(torch.float32);  query_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:187, code: key = key.to(torch.float32)
    key_5 = key_4.to(torch.float32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_1 = key_5.transpose(-1, -2);  key_5 = None
    attn_weights_6 = torch.matmul(query_5, transpose_1);  query_5 = transpose_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_1_attn_attention_bias = self.L__mod___transformer_h_1_attn_attention_bias
    causal_mask_1 = l__mod___transformer_h_1_attn_attention_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_1_attn_attention_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_8 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_1 = tensor_8.to(device(type='cuda', index=0));  tensor_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_7 = torch.where(causal_mask_1, attn_weights_6, mask_value_1);  causal_mask_1 = attn_weights_6 = mask_value_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_8 = torch.nn.functional.softmax(attn_weights_7, dim = -1);  attn_weights_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:204, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_9 = attn_weights_8.to(torch.float32);  attn_weights_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_11 = self.L__mod___transformer_h_1_attn_attention_attn_dropout(attn_weights_9);  attn_weights_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_6 = torch.matmul(attn_weights_11, value_3);  attn_weights_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_7 = attn_output_6.permute(0, 2, 1, 3);  attn_output_6 = None
    tensor_9 = permute_7.contiguous();  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    attn_output_7 = tensor_9.view((1, 128, 2048));  tensor_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    attn_output_8 = self.L__mod___transformer_h_1_attn_attention_out_proj(attn_output_7);  attn_output_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    attn_output_10 = self.L__mod___transformer_h_1_attn_attention_resid_dropout(attn_output_8);  attn_output_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    residual_3 = attn_output_10 + residual_2;  attn_output_10 = residual_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    hidden_states_13 = self.L__mod___transformer_h_1_ln_2(residual_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    hidden_states_14 = self.L__mod___transformer_h_1_mlp_c_fc(hidden_states_13);  hidden_states_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_4 = 0.5 * hidden_states_14
    pow_2 = torch.pow(hidden_states_14, 3.0)
    mul_5 = 0.044715 * pow_2;  pow_2 = None
    add_6 = hidden_states_14 + mul_5;  hidden_states_14 = mul_5 = None
    mul_6 = 0.7978845608028654 * add_6;  add_6 = None
    tanh_1 = torch.tanh(mul_6);  mul_6 = None
    add_7 = 1.0 + tanh_1;  tanh_1 = None
    hidden_states_15 = mul_4 * add_7;  mul_4 = add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    hidden_states_16 = self.L__mod___transformer_h_1_mlp_c_proj(hidden_states_15);  hidden_states_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_1 = self.L__mod___transformer_h_1_mlp_dropout(hidden_states_16);  hidden_states_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    residual_4 = residual_3 + feed_forward_hidden_states_1;  residual_3 = feed_forward_hidden_states_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_20 = self.L__mod___transformer_h_2_ln_1(residual_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    query_6 = self.L__mod___transformer_h_2_attn_attention_q_proj(hidden_states_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    key_6 = self.L__mod___transformer_h_2_attn_attention_k_proj(hidden_states_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    value_4 = self.L__mod___transformer_h_2_attn_attention_v_proj(hidden_states_20);  hidden_states_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_10 = query_6.view((1, 128, 16, 128));  query_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_7 = tensor_10.permute(0, 2, 1, 3);  tensor_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_11 = key_6.view((1, 128, 16, 128));  key_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_7 = tensor_11.permute(0, 2, 1, 3);  tensor_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_12 = value_4.view((1, 128, 16, 128));  value_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_5 = tensor_12.permute(0, 2, 1, 3);  tensor_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:186, code: query = query.to(torch.float32)
    query_8 = query_7.to(torch.float32);  query_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:187, code: key = key.to(torch.float32)
    key_8 = key_7.to(torch.float32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_2 = key_8.transpose(-1, -2);  key_8 = None
    attn_weights_12 = torch.matmul(query_8, transpose_2);  query_8 = transpose_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_2_attn_attention_bias = self.L__mod___transformer_h_2_attn_attention_bias
    causal_mask_2 = l__mod___transformer_h_2_attn_attention_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_2_attn_attention_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_13 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_2 = tensor_13.to(device(type='cuda', index=0));  tensor_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_13 = torch.where(causal_mask_2, attn_weights_12, mask_value_2);  causal_mask_2 = attn_weights_12 = mask_value_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_14 = torch.nn.functional.softmax(attn_weights_13, dim = -1);  attn_weights_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:204, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_15 = attn_weights_14.to(torch.float32);  attn_weights_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_17 = self.L__mod___transformer_h_2_attn_attention_attn_dropout(attn_weights_15);  attn_weights_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_12 = torch.matmul(attn_weights_17, value_5);  attn_weights_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_11 = attn_output_12.permute(0, 2, 1, 3);  attn_output_12 = None
    tensor_14 = permute_11.contiguous();  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    attn_output_13 = tensor_14.view((1, 128, 2048));  tensor_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    attn_output_14 = self.L__mod___transformer_h_2_attn_attention_out_proj(attn_output_13);  attn_output_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    attn_output_16 = self.L__mod___transformer_h_2_attn_attention_resid_dropout(attn_output_14);  attn_output_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    residual_5 = attn_output_16 + residual_4;  attn_output_16 = residual_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    hidden_states_22 = self.L__mod___transformer_h_2_ln_2(residual_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    hidden_states_23 = self.L__mod___transformer_h_2_mlp_c_fc(hidden_states_22);  hidden_states_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_8 = 0.5 * hidden_states_23
    pow_3 = torch.pow(hidden_states_23, 3.0)
    mul_9 = 0.044715 * pow_3;  pow_3 = None
    add_10 = hidden_states_23 + mul_9;  hidden_states_23 = mul_9 = None
    mul_10 = 0.7978845608028654 * add_10;  add_10 = None
    tanh_2 = torch.tanh(mul_10);  mul_10 = None
    add_11 = 1.0 + tanh_2;  tanh_2 = None
    hidden_states_24 = mul_8 * add_11;  mul_8 = add_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    hidden_states_25 = self.L__mod___transformer_h_2_mlp_c_proj(hidden_states_24);  hidden_states_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_2 = self.L__mod___transformer_h_2_mlp_dropout(hidden_states_25);  hidden_states_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    residual_6 = residual_5 + feed_forward_hidden_states_2;  residual_5 = feed_forward_hidden_states_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_29 = self.L__mod___transformer_h_3_ln_1(residual_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    query_9 = self.L__mod___transformer_h_3_attn_attention_q_proj(hidden_states_29)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    key_9 = self.L__mod___transformer_h_3_attn_attention_k_proj(hidden_states_29)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    value_6 = self.L__mod___transformer_h_3_attn_attention_v_proj(hidden_states_29);  hidden_states_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_15 = query_9.view((1, 128, 16, 128));  query_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_10 = tensor_15.permute(0, 2, 1, 3);  tensor_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_16 = key_9.view((1, 128, 16, 128));  key_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_10 = tensor_16.permute(0, 2, 1, 3);  tensor_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_17 = value_6.view((1, 128, 16, 128));  value_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_7 = tensor_17.permute(0, 2, 1, 3);  tensor_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:186, code: query = query.to(torch.float32)
    query_11 = query_10.to(torch.float32);  query_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:187, code: key = key.to(torch.float32)
    key_11 = key_10.to(torch.float32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_3 = key_11.transpose(-1, -2);  key_11 = None
    attn_weights_18 = torch.matmul(query_11, transpose_3);  query_11 = transpose_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_3_attn_attention_bias = self.L__mod___transformer_h_3_attn_attention_bias
    causal_mask_3 = l__mod___transformer_h_3_attn_attention_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_3_attn_attention_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_18 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_3 = tensor_18.to(device(type='cuda', index=0));  tensor_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_19 = torch.where(causal_mask_3, attn_weights_18, mask_value_3);  causal_mask_3 = attn_weights_18 = mask_value_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_20 = torch.nn.functional.softmax(attn_weights_19, dim = -1);  attn_weights_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:204, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_21 = attn_weights_20.to(torch.float32);  attn_weights_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_23 = self.L__mod___transformer_h_3_attn_attention_attn_dropout(attn_weights_21);  attn_weights_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_18 = torch.matmul(attn_weights_23, value_7);  attn_weights_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_15 = attn_output_18.permute(0, 2, 1, 3);  attn_output_18 = None
    tensor_19 = permute_15.contiguous();  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    attn_output_19 = tensor_19.view((1, 128, 2048));  tensor_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    attn_output_20 = self.L__mod___transformer_h_3_attn_attention_out_proj(attn_output_19);  attn_output_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    attn_output_22 = self.L__mod___transformer_h_3_attn_attention_resid_dropout(attn_output_20);  attn_output_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    residual_7 = attn_output_22 + residual_6;  attn_output_22 = residual_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    hidden_states_31 = self.L__mod___transformer_h_3_ln_2(residual_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    hidden_states_32 = self.L__mod___transformer_h_3_mlp_c_fc(hidden_states_31);  hidden_states_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_12 = 0.5 * hidden_states_32
    pow_4 = torch.pow(hidden_states_32, 3.0)
    mul_13 = 0.044715 * pow_4;  pow_4 = None
    add_14 = hidden_states_32 + mul_13;  hidden_states_32 = mul_13 = None
    mul_14 = 0.7978845608028654 * add_14;  add_14 = None
    tanh_3 = torch.tanh(mul_14);  mul_14 = None
    add_15 = 1.0 + tanh_3;  tanh_3 = None
    hidden_states_33 = mul_12 * add_15;  mul_12 = add_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    hidden_states_34 = self.L__mod___transformer_h_3_mlp_c_proj(hidden_states_33);  hidden_states_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_3 = self.L__mod___transformer_h_3_mlp_dropout(hidden_states_34);  hidden_states_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    residual_8 = residual_7 + feed_forward_hidden_states_3;  residual_7 = feed_forward_hidden_states_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_38 = self.L__mod___transformer_h_4_ln_1(residual_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    query_12 = self.L__mod___transformer_h_4_attn_attention_q_proj(hidden_states_38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    key_12 = self.L__mod___transformer_h_4_attn_attention_k_proj(hidden_states_38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    value_8 = self.L__mod___transformer_h_4_attn_attention_v_proj(hidden_states_38);  hidden_states_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_20 = query_12.view((1, 128, 16, 128));  query_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_13 = tensor_20.permute(0, 2, 1, 3);  tensor_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_21 = key_12.view((1, 128, 16, 128));  key_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_13 = tensor_21.permute(0, 2, 1, 3);  tensor_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_22 = value_8.view((1, 128, 16, 128));  value_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_9 = tensor_22.permute(0, 2, 1, 3);  tensor_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:186, code: query = query.to(torch.float32)
    query_14 = query_13.to(torch.float32);  query_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:187, code: key = key.to(torch.float32)
    key_14 = key_13.to(torch.float32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_4 = key_14.transpose(-1, -2);  key_14 = None
    attn_weights_24 = torch.matmul(query_14, transpose_4);  query_14 = transpose_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_4_attn_attention_bias = self.L__mod___transformer_h_4_attn_attention_bias
    causal_mask_4 = l__mod___transformer_h_4_attn_attention_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_4_attn_attention_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_23 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_4 = tensor_23.to(device(type='cuda', index=0));  tensor_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_25 = torch.where(causal_mask_4, attn_weights_24, mask_value_4);  causal_mask_4 = attn_weights_24 = mask_value_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_26 = torch.nn.functional.softmax(attn_weights_25, dim = -1);  attn_weights_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:204, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_27 = attn_weights_26.to(torch.float32);  attn_weights_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_29 = self.L__mod___transformer_h_4_attn_attention_attn_dropout(attn_weights_27);  attn_weights_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_24 = torch.matmul(attn_weights_29, value_9);  attn_weights_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_19 = attn_output_24.permute(0, 2, 1, 3);  attn_output_24 = None
    tensor_24 = permute_19.contiguous();  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    attn_output_25 = tensor_24.view((1, 128, 2048));  tensor_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    attn_output_26 = self.L__mod___transformer_h_4_attn_attention_out_proj(attn_output_25);  attn_output_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    attn_output_28 = self.L__mod___transformer_h_4_attn_attention_resid_dropout(attn_output_26);  attn_output_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    residual_9 = attn_output_28 + residual_8;  attn_output_28 = residual_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    hidden_states_40 = self.L__mod___transformer_h_4_ln_2(residual_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    hidden_states_41 = self.L__mod___transformer_h_4_mlp_c_fc(hidden_states_40);  hidden_states_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_16 = 0.5 * hidden_states_41
    pow_5 = torch.pow(hidden_states_41, 3.0)
    mul_17 = 0.044715 * pow_5;  pow_5 = None
    add_18 = hidden_states_41 + mul_17;  hidden_states_41 = mul_17 = None
    mul_18 = 0.7978845608028654 * add_18;  add_18 = None
    tanh_4 = torch.tanh(mul_18);  mul_18 = None
    add_19 = 1.0 + tanh_4;  tanh_4 = None
    hidden_states_42 = mul_16 * add_19;  mul_16 = add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    hidden_states_43 = self.L__mod___transformer_h_4_mlp_c_proj(hidden_states_42);  hidden_states_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_4 = self.L__mod___transformer_h_4_mlp_dropout(hidden_states_43);  hidden_states_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    residual_10 = residual_9 + feed_forward_hidden_states_4;  residual_9 = feed_forward_hidden_states_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_47 = self.L__mod___transformer_h_5_ln_1(residual_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    query_15 = self.L__mod___transformer_h_5_attn_attention_q_proj(hidden_states_47)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    key_15 = self.L__mod___transformer_h_5_attn_attention_k_proj(hidden_states_47)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    value_10 = self.L__mod___transformer_h_5_attn_attention_v_proj(hidden_states_47);  hidden_states_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_25 = query_15.view((1, 128, 16, 128));  query_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_16 = tensor_25.permute(0, 2, 1, 3);  tensor_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_26 = key_15.view((1, 128, 16, 128));  key_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_16 = tensor_26.permute(0, 2, 1, 3);  tensor_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_27 = value_10.view((1, 128, 16, 128));  value_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_11 = tensor_27.permute(0, 2, 1, 3);  tensor_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:186, code: query = query.to(torch.float32)
    query_17 = query_16.to(torch.float32);  query_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:187, code: key = key.to(torch.float32)
    key_17 = key_16.to(torch.float32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_5 = key_17.transpose(-1, -2);  key_17 = None
    attn_weights_30 = torch.matmul(query_17, transpose_5);  query_17 = transpose_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_5_attn_attention_bias = self.L__mod___transformer_h_5_attn_attention_bias
    causal_mask_5 = l__mod___transformer_h_5_attn_attention_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_5_attn_attention_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_28 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_5 = tensor_28.to(device(type='cuda', index=0));  tensor_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_31 = torch.where(causal_mask_5, attn_weights_30, mask_value_5);  causal_mask_5 = attn_weights_30 = mask_value_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_32 = torch.nn.functional.softmax(attn_weights_31, dim = -1);  attn_weights_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:204, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_33 = attn_weights_32.to(torch.float32);  attn_weights_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_35 = self.L__mod___transformer_h_5_attn_attention_attn_dropout(attn_weights_33);  attn_weights_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_30 = torch.matmul(attn_weights_35, value_11);  attn_weights_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_23 = attn_output_30.permute(0, 2, 1, 3);  attn_output_30 = None
    tensor_29 = permute_23.contiguous();  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    attn_output_31 = tensor_29.view((1, 128, 2048));  tensor_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    attn_output_32 = self.L__mod___transformer_h_5_attn_attention_out_proj(attn_output_31);  attn_output_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    attn_output_34 = self.L__mod___transformer_h_5_attn_attention_resid_dropout(attn_output_32);  attn_output_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    residual_11 = attn_output_34 + residual_10;  attn_output_34 = residual_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    hidden_states_49 = self.L__mod___transformer_h_5_ln_2(residual_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    hidden_states_50 = self.L__mod___transformer_h_5_mlp_c_fc(hidden_states_49);  hidden_states_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_20 = 0.5 * hidden_states_50
    pow_6 = torch.pow(hidden_states_50, 3.0)
    mul_21 = 0.044715 * pow_6;  pow_6 = None
    add_22 = hidden_states_50 + mul_21;  hidden_states_50 = mul_21 = None
    mul_22 = 0.7978845608028654 * add_22;  add_22 = None
    tanh_5 = torch.tanh(mul_22);  mul_22 = None
    add_23 = 1.0 + tanh_5;  tanh_5 = None
    hidden_states_51 = mul_20 * add_23;  mul_20 = add_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    hidden_states_52 = self.L__mod___transformer_h_5_mlp_c_proj(hidden_states_51);  hidden_states_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_5 = self.L__mod___transformer_h_5_mlp_dropout(hidden_states_52);  hidden_states_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    residual_12 = residual_11 + feed_forward_hidden_states_5;  residual_11 = feed_forward_hidden_states_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_56 = self.L__mod___transformer_h_6_ln_1(residual_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    query_18 = self.L__mod___transformer_h_6_attn_attention_q_proj(hidden_states_56)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    key_18 = self.L__mod___transformer_h_6_attn_attention_k_proj(hidden_states_56)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    value_12 = self.L__mod___transformer_h_6_attn_attention_v_proj(hidden_states_56);  hidden_states_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_30 = query_18.view((1, 128, 16, 128));  query_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_19 = tensor_30.permute(0, 2, 1, 3);  tensor_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_31 = key_18.view((1, 128, 16, 128));  key_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_19 = tensor_31.permute(0, 2, 1, 3);  tensor_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_32 = value_12.view((1, 128, 16, 128));  value_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_13 = tensor_32.permute(0, 2, 1, 3);  tensor_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:186, code: query = query.to(torch.float32)
    query_20 = query_19.to(torch.float32);  query_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:187, code: key = key.to(torch.float32)
    key_20 = key_19.to(torch.float32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_6 = key_20.transpose(-1, -2);  key_20 = None
    attn_weights_36 = torch.matmul(query_20, transpose_6);  query_20 = transpose_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_6_attn_attention_bias = self.L__mod___transformer_h_6_attn_attention_bias
    causal_mask_6 = l__mod___transformer_h_6_attn_attention_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_6_attn_attention_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_33 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_6 = tensor_33.to(device(type='cuda', index=0));  tensor_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_37 = torch.where(causal_mask_6, attn_weights_36, mask_value_6);  causal_mask_6 = attn_weights_36 = mask_value_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_38 = torch.nn.functional.softmax(attn_weights_37, dim = -1);  attn_weights_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:204, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_39 = attn_weights_38.to(torch.float32);  attn_weights_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_41 = self.L__mod___transformer_h_6_attn_attention_attn_dropout(attn_weights_39);  attn_weights_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_36 = torch.matmul(attn_weights_41, value_13);  attn_weights_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_27 = attn_output_36.permute(0, 2, 1, 3);  attn_output_36 = None
    tensor_34 = permute_27.contiguous();  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    attn_output_37 = tensor_34.view((1, 128, 2048));  tensor_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    attn_output_38 = self.L__mod___transformer_h_6_attn_attention_out_proj(attn_output_37);  attn_output_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    attn_output_40 = self.L__mod___transformer_h_6_attn_attention_resid_dropout(attn_output_38);  attn_output_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    residual_13 = attn_output_40 + residual_12;  attn_output_40 = residual_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    hidden_states_58 = self.L__mod___transformer_h_6_ln_2(residual_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    hidden_states_59 = self.L__mod___transformer_h_6_mlp_c_fc(hidden_states_58);  hidden_states_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_24 = 0.5 * hidden_states_59
    pow_7 = torch.pow(hidden_states_59, 3.0)
    mul_25 = 0.044715 * pow_7;  pow_7 = None
    add_26 = hidden_states_59 + mul_25;  hidden_states_59 = mul_25 = None
    mul_26 = 0.7978845608028654 * add_26;  add_26 = None
    tanh_6 = torch.tanh(mul_26);  mul_26 = None
    add_27 = 1.0 + tanh_6;  tanh_6 = None
    hidden_states_60 = mul_24 * add_27;  mul_24 = add_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    hidden_states_61 = self.L__mod___transformer_h_6_mlp_c_proj(hidden_states_60);  hidden_states_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_6 = self.L__mod___transformer_h_6_mlp_dropout(hidden_states_61);  hidden_states_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    residual_14 = residual_13 + feed_forward_hidden_states_6;  residual_13 = feed_forward_hidden_states_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_65 = self.L__mod___transformer_h_7_ln_1(residual_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    query_21 = self.L__mod___transformer_h_7_attn_attention_q_proj(hidden_states_65)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    key_21 = self.L__mod___transformer_h_7_attn_attention_k_proj(hidden_states_65)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    value_14 = self.L__mod___transformer_h_7_attn_attention_v_proj(hidden_states_65);  hidden_states_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_35 = query_21.view((1, 128, 16, 128));  query_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_22 = tensor_35.permute(0, 2, 1, 3);  tensor_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_36 = key_21.view((1, 128, 16, 128));  key_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_22 = tensor_36.permute(0, 2, 1, 3);  tensor_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_37 = value_14.view((1, 128, 16, 128));  value_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_15 = tensor_37.permute(0, 2, 1, 3);  tensor_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:186, code: query = query.to(torch.float32)
    query_23 = query_22.to(torch.float32);  query_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:187, code: key = key.to(torch.float32)
    key_23 = key_22.to(torch.float32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_7 = key_23.transpose(-1, -2);  key_23 = None
    attn_weights_42 = torch.matmul(query_23, transpose_7);  query_23 = transpose_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_7_attn_attention_bias = self.L__mod___transformer_h_7_attn_attention_bias
    causal_mask_7 = l__mod___transformer_h_7_attn_attention_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_7_attn_attention_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_38 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_7 = tensor_38.to(device(type='cuda', index=0));  tensor_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_43 = torch.where(causal_mask_7, attn_weights_42, mask_value_7);  causal_mask_7 = attn_weights_42 = mask_value_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_44 = torch.nn.functional.softmax(attn_weights_43, dim = -1);  attn_weights_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:204, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_45 = attn_weights_44.to(torch.float32);  attn_weights_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_47 = self.L__mod___transformer_h_7_attn_attention_attn_dropout(attn_weights_45);  attn_weights_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_42 = torch.matmul(attn_weights_47, value_15);  attn_weights_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_31 = attn_output_42.permute(0, 2, 1, 3);  attn_output_42 = None
    tensor_39 = permute_31.contiguous();  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    attn_output_43 = tensor_39.view((1, 128, 2048));  tensor_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    attn_output_44 = self.L__mod___transformer_h_7_attn_attention_out_proj(attn_output_43);  attn_output_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    attn_output_46 = self.L__mod___transformer_h_7_attn_attention_resid_dropout(attn_output_44);  attn_output_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    residual_15 = attn_output_46 + residual_14;  attn_output_46 = residual_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    hidden_states_67 = self.L__mod___transformer_h_7_ln_2(residual_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    hidden_states_68 = self.L__mod___transformer_h_7_mlp_c_fc(hidden_states_67);  hidden_states_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_28 = 0.5 * hidden_states_68
    pow_8 = torch.pow(hidden_states_68, 3.0)
    mul_29 = 0.044715 * pow_8;  pow_8 = None
    add_30 = hidden_states_68 + mul_29;  hidden_states_68 = mul_29 = None
    mul_30 = 0.7978845608028654 * add_30;  add_30 = None
    tanh_7 = torch.tanh(mul_30);  mul_30 = None
    add_31 = 1.0 + tanh_7;  tanh_7 = None
    hidden_states_69 = mul_28 * add_31;  mul_28 = add_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    hidden_states_70 = self.L__mod___transformer_h_7_mlp_c_proj(hidden_states_69);  hidden_states_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_7 = self.L__mod___transformer_h_7_mlp_dropout(hidden_states_70);  hidden_states_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    residual_16 = residual_15 + feed_forward_hidden_states_7;  residual_15 = feed_forward_hidden_states_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_74 = self.L__mod___transformer_h_8_ln_1(residual_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    query_24 = self.L__mod___transformer_h_8_attn_attention_q_proj(hidden_states_74)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    key_24 = self.L__mod___transformer_h_8_attn_attention_k_proj(hidden_states_74)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    value_16 = self.L__mod___transformer_h_8_attn_attention_v_proj(hidden_states_74);  hidden_states_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_40 = query_24.view((1, 128, 16, 128));  query_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_25 = tensor_40.permute(0, 2, 1, 3);  tensor_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_41 = key_24.view((1, 128, 16, 128));  key_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_25 = tensor_41.permute(0, 2, 1, 3);  tensor_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_42 = value_16.view((1, 128, 16, 128));  value_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_17 = tensor_42.permute(0, 2, 1, 3);  tensor_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:186, code: query = query.to(torch.float32)
    query_26 = query_25.to(torch.float32);  query_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:187, code: key = key.to(torch.float32)
    key_26 = key_25.to(torch.float32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_8 = key_26.transpose(-1, -2);  key_26 = None
    attn_weights_48 = torch.matmul(query_26, transpose_8);  query_26 = transpose_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_8_attn_attention_bias = self.L__mod___transformer_h_8_attn_attention_bias
    causal_mask_8 = l__mod___transformer_h_8_attn_attention_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_8_attn_attention_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_43 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_8 = tensor_43.to(device(type='cuda', index=0));  tensor_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_49 = torch.where(causal_mask_8, attn_weights_48, mask_value_8);  causal_mask_8 = attn_weights_48 = mask_value_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_50 = torch.nn.functional.softmax(attn_weights_49, dim = -1);  attn_weights_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:204, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_51 = attn_weights_50.to(torch.float32);  attn_weights_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_53 = self.L__mod___transformer_h_8_attn_attention_attn_dropout(attn_weights_51);  attn_weights_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_48 = torch.matmul(attn_weights_53, value_17);  attn_weights_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_35 = attn_output_48.permute(0, 2, 1, 3);  attn_output_48 = None
    tensor_44 = permute_35.contiguous();  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    attn_output_49 = tensor_44.view((1, 128, 2048));  tensor_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    attn_output_50 = self.L__mod___transformer_h_8_attn_attention_out_proj(attn_output_49);  attn_output_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    attn_output_52 = self.L__mod___transformer_h_8_attn_attention_resid_dropout(attn_output_50);  attn_output_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    residual_17 = attn_output_52 + residual_16;  attn_output_52 = residual_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    hidden_states_76 = self.L__mod___transformer_h_8_ln_2(residual_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    hidden_states_77 = self.L__mod___transformer_h_8_mlp_c_fc(hidden_states_76);  hidden_states_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_32 = 0.5 * hidden_states_77
    pow_9 = torch.pow(hidden_states_77, 3.0)
    mul_33 = 0.044715 * pow_9;  pow_9 = None
    add_34 = hidden_states_77 + mul_33;  hidden_states_77 = mul_33 = None
    mul_34 = 0.7978845608028654 * add_34;  add_34 = None
    tanh_8 = torch.tanh(mul_34);  mul_34 = None
    add_35 = 1.0 + tanh_8;  tanh_8 = None
    hidden_states_78 = mul_32 * add_35;  mul_32 = add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    hidden_states_79 = self.L__mod___transformer_h_8_mlp_c_proj(hidden_states_78);  hidden_states_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_8 = self.L__mod___transformer_h_8_mlp_dropout(hidden_states_79);  hidden_states_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    residual_18 = residual_17 + feed_forward_hidden_states_8;  residual_17 = feed_forward_hidden_states_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_83 = self.L__mod___transformer_h_9_ln_1(residual_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    query_27 = self.L__mod___transformer_h_9_attn_attention_q_proj(hidden_states_83)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    key_27 = self.L__mod___transformer_h_9_attn_attention_k_proj(hidden_states_83)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    value_18 = self.L__mod___transformer_h_9_attn_attention_v_proj(hidden_states_83);  hidden_states_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_45 = query_27.view((1, 128, 16, 128));  query_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_28 = tensor_45.permute(0, 2, 1, 3);  tensor_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_46 = key_27.view((1, 128, 16, 128));  key_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_28 = tensor_46.permute(0, 2, 1, 3);  tensor_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_47 = value_18.view((1, 128, 16, 128));  value_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_19 = tensor_47.permute(0, 2, 1, 3);  tensor_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:186, code: query = query.to(torch.float32)
    query_29 = query_28.to(torch.float32);  query_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:187, code: key = key.to(torch.float32)
    key_29 = key_28.to(torch.float32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_9 = key_29.transpose(-1, -2);  key_29 = None
    attn_weights_54 = torch.matmul(query_29, transpose_9);  query_29 = transpose_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_9_attn_attention_bias = self.L__mod___transformer_h_9_attn_attention_bias
    causal_mask_9 = l__mod___transformer_h_9_attn_attention_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_9_attn_attention_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_48 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_9 = tensor_48.to(device(type='cuda', index=0));  tensor_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_55 = torch.where(causal_mask_9, attn_weights_54, mask_value_9);  causal_mask_9 = attn_weights_54 = mask_value_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_56 = torch.nn.functional.softmax(attn_weights_55, dim = -1);  attn_weights_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:204, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_57 = attn_weights_56.to(torch.float32);  attn_weights_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_59 = self.L__mod___transformer_h_9_attn_attention_attn_dropout(attn_weights_57);  attn_weights_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_54 = torch.matmul(attn_weights_59, value_19);  attn_weights_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_39 = attn_output_54.permute(0, 2, 1, 3);  attn_output_54 = None
    tensor_49 = permute_39.contiguous();  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    attn_output_55 = tensor_49.view((1, 128, 2048));  tensor_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    attn_output_56 = self.L__mod___transformer_h_9_attn_attention_out_proj(attn_output_55);  attn_output_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    attn_output_58 = self.L__mod___transformer_h_9_attn_attention_resid_dropout(attn_output_56);  attn_output_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    residual_19 = attn_output_58 + residual_18;  attn_output_58 = residual_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    hidden_states_85 = self.L__mod___transformer_h_9_ln_2(residual_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    hidden_states_86 = self.L__mod___transformer_h_9_mlp_c_fc(hidden_states_85);  hidden_states_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_36 = 0.5 * hidden_states_86
    pow_10 = torch.pow(hidden_states_86, 3.0)
    mul_37 = 0.044715 * pow_10;  pow_10 = None
    add_38 = hidden_states_86 + mul_37;  hidden_states_86 = mul_37 = None
    mul_38 = 0.7978845608028654 * add_38;  add_38 = None
    tanh_9 = torch.tanh(mul_38);  mul_38 = None
    add_39 = 1.0 + tanh_9;  tanh_9 = None
    hidden_states_87 = mul_36 * add_39;  mul_36 = add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    hidden_states_88 = self.L__mod___transformer_h_9_mlp_c_proj(hidden_states_87);  hidden_states_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_9 = self.L__mod___transformer_h_9_mlp_dropout(hidden_states_88);  hidden_states_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    residual_20 = residual_19 + feed_forward_hidden_states_9;  residual_19 = feed_forward_hidden_states_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_92 = self.L__mod___transformer_h_10_ln_1(residual_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    query_30 = self.L__mod___transformer_h_10_attn_attention_q_proj(hidden_states_92)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    key_30 = self.L__mod___transformer_h_10_attn_attention_k_proj(hidden_states_92)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    value_20 = self.L__mod___transformer_h_10_attn_attention_v_proj(hidden_states_92);  hidden_states_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_50 = query_30.view((1, 128, 16, 128));  query_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_31 = tensor_50.permute(0, 2, 1, 3);  tensor_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_51 = key_30.view((1, 128, 16, 128));  key_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_31 = tensor_51.permute(0, 2, 1, 3);  tensor_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_52 = value_20.view((1, 128, 16, 128));  value_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_21 = tensor_52.permute(0, 2, 1, 3);  tensor_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:186, code: query = query.to(torch.float32)
    query_32 = query_31.to(torch.float32);  query_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:187, code: key = key.to(torch.float32)
    key_32 = key_31.to(torch.float32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_10 = key_32.transpose(-1, -2);  key_32 = None
    attn_weights_60 = torch.matmul(query_32, transpose_10);  query_32 = transpose_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_10_attn_attention_bias = self.L__mod___transformer_h_10_attn_attention_bias
    causal_mask_10 = l__mod___transformer_h_10_attn_attention_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_10_attn_attention_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_53 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_10 = tensor_53.to(device(type='cuda', index=0));  tensor_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_61 = torch.where(causal_mask_10, attn_weights_60, mask_value_10);  causal_mask_10 = attn_weights_60 = mask_value_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_62 = torch.nn.functional.softmax(attn_weights_61, dim = -1);  attn_weights_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:204, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_63 = attn_weights_62.to(torch.float32);  attn_weights_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_65 = self.L__mod___transformer_h_10_attn_attention_attn_dropout(attn_weights_63);  attn_weights_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_60 = torch.matmul(attn_weights_65, value_21);  attn_weights_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_43 = attn_output_60.permute(0, 2, 1, 3);  attn_output_60 = None
    tensor_54 = permute_43.contiguous();  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    attn_output_61 = tensor_54.view((1, 128, 2048));  tensor_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    attn_output_62 = self.L__mod___transformer_h_10_attn_attention_out_proj(attn_output_61);  attn_output_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    attn_output_64 = self.L__mod___transformer_h_10_attn_attention_resid_dropout(attn_output_62);  attn_output_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    residual_21 = attn_output_64 + residual_20;  attn_output_64 = residual_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    hidden_states_94 = self.L__mod___transformer_h_10_ln_2(residual_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    hidden_states_95 = self.L__mod___transformer_h_10_mlp_c_fc(hidden_states_94);  hidden_states_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_40 = 0.5 * hidden_states_95
    pow_11 = torch.pow(hidden_states_95, 3.0)
    mul_41 = 0.044715 * pow_11;  pow_11 = None
    add_42 = hidden_states_95 + mul_41;  hidden_states_95 = mul_41 = None
    mul_42 = 0.7978845608028654 * add_42;  add_42 = None
    tanh_10 = torch.tanh(mul_42);  mul_42 = None
    add_43 = 1.0 + tanh_10;  tanh_10 = None
    hidden_states_96 = mul_40 * add_43;  mul_40 = add_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    hidden_states_97 = self.L__mod___transformer_h_10_mlp_c_proj(hidden_states_96);  hidden_states_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_10 = self.L__mod___transformer_h_10_mlp_dropout(hidden_states_97);  hidden_states_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    residual_22 = residual_21 + feed_forward_hidden_states_10;  residual_21 = feed_forward_hidden_states_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_101 = self.L__mod___transformer_h_11_ln_1(residual_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    query_33 = self.L__mod___transformer_h_11_attn_attention_q_proj(hidden_states_101)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    key_33 = self.L__mod___transformer_h_11_attn_attention_k_proj(hidden_states_101)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    value_22 = self.L__mod___transformer_h_11_attn_attention_v_proj(hidden_states_101);  hidden_states_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_55 = query_33.view((1, 128, 16, 128));  query_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_34 = tensor_55.permute(0, 2, 1, 3);  tensor_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_56 = key_33.view((1, 128, 16, 128));  key_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_34 = tensor_56.permute(0, 2, 1, 3);  tensor_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_57 = value_22.view((1, 128, 16, 128));  value_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_23 = tensor_57.permute(0, 2, 1, 3);  tensor_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:186, code: query = query.to(torch.float32)
    query_35 = query_34.to(torch.float32);  query_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:187, code: key = key.to(torch.float32)
    key_35 = key_34.to(torch.float32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_11 = key_35.transpose(-1, -2);  key_35 = None
    attn_weights_66 = torch.matmul(query_35, transpose_11);  query_35 = transpose_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_11_attn_attention_bias = self.L__mod___transformer_h_11_attn_attention_bias
    causal_mask_11 = l__mod___transformer_h_11_attn_attention_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_11_attn_attention_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_58 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_11 = tensor_58.to(device(type='cuda', index=0));  tensor_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_67 = torch.where(causal_mask_11, attn_weights_66, mask_value_11);  causal_mask_11 = attn_weights_66 = mask_value_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_68 = torch.nn.functional.softmax(attn_weights_67, dim = -1);  attn_weights_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:204, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_69 = attn_weights_68.to(torch.float32);  attn_weights_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_71 = self.L__mod___transformer_h_11_attn_attention_attn_dropout(attn_weights_69);  attn_weights_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_66 = torch.matmul(attn_weights_71, value_23);  attn_weights_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_47 = attn_output_66.permute(0, 2, 1, 3);  attn_output_66 = None
    tensor_59 = permute_47.contiguous();  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    attn_output_67 = tensor_59.view((1, 128, 2048));  tensor_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    attn_output_68 = self.L__mod___transformer_h_11_attn_attention_out_proj(attn_output_67);  attn_output_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    attn_output_70 = self.L__mod___transformer_h_11_attn_attention_resid_dropout(attn_output_68);  attn_output_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    residual_23 = attn_output_70 + residual_22;  attn_output_70 = residual_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    hidden_states_103 = self.L__mod___transformer_h_11_ln_2(residual_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    hidden_states_104 = self.L__mod___transformer_h_11_mlp_c_fc(hidden_states_103);  hidden_states_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_44 = 0.5 * hidden_states_104
    pow_12 = torch.pow(hidden_states_104, 3.0)
    mul_45 = 0.044715 * pow_12;  pow_12 = None
    add_46 = hidden_states_104 + mul_45;  hidden_states_104 = mul_45 = None
    mul_46 = 0.7978845608028654 * add_46;  add_46 = None
    tanh_11 = torch.tanh(mul_46);  mul_46 = None
    add_47 = 1.0 + tanh_11;  tanh_11 = None
    hidden_states_105 = mul_44 * add_47;  mul_44 = add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    hidden_states_106 = self.L__mod___transformer_h_11_mlp_c_proj(hidden_states_105);  hidden_states_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_11 = self.L__mod___transformer_h_11_mlp_dropout(hidden_states_106);  hidden_states_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    residual_24 = residual_23 + feed_forward_hidden_states_11;  residual_23 = feed_forward_hidden_states_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_110 = self.L__mod___transformer_h_12_ln_1(residual_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    query_36 = self.L__mod___transformer_h_12_attn_attention_q_proj(hidden_states_110)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    key_36 = self.L__mod___transformer_h_12_attn_attention_k_proj(hidden_states_110)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    value_24 = self.L__mod___transformer_h_12_attn_attention_v_proj(hidden_states_110);  hidden_states_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_60 = query_36.view((1, 128, 16, 128));  query_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_37 = tensor_60.permute(0, 2, 1, 3);  tensor_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_61 = key_36.view((1, 128, 16, 128));  key_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_37 = tensor_61.permute(0, 2, 1, 3);  tensor_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_62 = value_24.view((1, 128, 16, 128));  value_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_25 = tensor_62.permute(0, 2, 1, 3);  tensor_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:186, code: query = query.to(torch.float32)
    query_38 = query_37.to(torch.float32);  query_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:187, code: key = key.to(torch.float32)
    key_38 = key_37.to(torch.float32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_12 = key_38.transpose(-1, -2);  key_38 = None
    attn_weights_72 = torch.matmul(query_38, transpose_12);  query_38 = transpose_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_12_attn_attention_bias = self.L__mod___transformer_h_12_attn_attention_bias
    causal_mask_12 = l__mod___transformer_h_12_attn_attention_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_12_attn_attention_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_63 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_12 = tensor_63.to(device(type='cuda', index=0));  tensor_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_73 = torch.where(causal_mask_12, attn_weights_72, mask_value_12);  causal_mask_12 = attn_weights_72 = mask_value_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_74 = torch.nn.functional.softmax(attn_weights_73, dim = -1);  attn_weights_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:204, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_75 = attn_weights_74.to(torch.float32);  attn_weights_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_77 = self.L__mod___transformer_h_12_attn_attention_attn_dropout(attn_weights_75);  attn_weights_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_72 = torch.matmul(attn_weights_77, value_25);  attn_weights_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_51 = attn_output_72.permute(0, 2, 1, 3);  attn_output_72 = None
    tensor_64 = permute_51.contiguous();  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    attn_output_73 = tensor_64.view((1, 128, 2048));  tensor_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    attn_output_74 = self.L__mod___transformer_h_12_attn_attention_out_proj(attn_output_73);  attn_output_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    attn_output_76 = self.L__mod___transformer_h_12_attn_attention_resid_dropout(attn_output_74);  attn_output_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    residual_25 = attn_output_76 + residual_24;  attn_output_76 = residual_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    hidden_states_112 = self.L__mod___transformer_h_12_ln_2(residual_25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    hidden_states_113 = self.L__mod___transformer_h_12_mlp_c_fc(hidden_states_112);  hidden_states_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_48 = 0.5 * hidden_states_113
    pow_13 = torch.pow(hidden_states_113, 3.0)
    mul_49 = 0.044715 * pow_13;  pow_13 = None
    add_50 = hidden_states_113 + mul_49;  hidden_states_113 = mul_49 = None
    mul_50 = 0.7978845608028654 * add_50;  add_50 = None
    tanh_12 = torch.tanh(mul_50);  mul_50 = None
    add_51 = 1.0 + tanh_12;  tanh_12 = None
    hidden_states_114 = mul_48 * add_51;  mul_48 = add_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    hidden_states_115 = self.L__mod___transformer_h_12_mlp_c_proj(hidden_states_114);  hidden_states_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_12 = self.L__mod___transformer_h_12_mlp_dropout(hidden_states_115);  hidden_states_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    residual_26 = residual_25 + feed_forward_hidden_states_12;  residual_25 = feed_forward_hidden_states_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_119 = self.L__mod___transformer_h_13_ln_1(residual_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    query_39 = self.L__mod___transformer_h_13_attn_attention_q_proj(hidden_states_119)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    key_39 = self.L__mod___transformer_h_13_attn_attention_k_proj(hidden_states_119)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    value_26 = self.L__mod___transformer_h_13_attn_attention_v_proj(hidden_states_119);  hidden_states_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_65 = query_39.view((1, 128, 16, 128));  query_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_40 = tensor_65.permute(0, 2, 1, 3);  tensor_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_66 = key_39.view((1, 128, 16, 128));  key_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_40 = tensor_66.permute(0, 2, 1, 3);  tensor_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_67 = value_26.view((1, 128, 16, 128));  value_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_27 = tensor_67.permute(0, 2, 1, 3);  tensor_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:186, code: query = query.to(torch.float32)
    query_41 = query_40.to(torch.float32);  query_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:187, code: key = key.to(torch.float32)
    key_41 = key_40.to(torch.float32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_13 = key_41.transpose(-1, -2);  key_41 = None
    attn_weights_78 = torch.matmul(query_41, transpose_13);  query_41 = transpose_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_13_attn_attention_bias = self.L__mod___transformer_h_13_attn_attention_bias
    causal_mask_13 = l__mod___transformer_h_13_attn_attention_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_13_attn_attention_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_68 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_13 = tensor_68.to(device(type='cuda', index=0));  tensor_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_79 = torch.where(causal_mask_13, attn_weights_78, mask_value_13);  causal_mask_13 = attn_weights_78 = mask_value_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_80 = torch.nn.functional.softmax(attn_weights_79, dim = -1);  attn_weights_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:204, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_81 = attn_weights_80.to(torch.float32);  attn_weights_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_83 = self.L__mod___transformer_h_13_attn_attention_attn_dropout(attn_weights_81);  attn_weights_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_78 = torch.matmul(attn_weights_83, value_27);  attn_weights_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_55 = attn_output_78.permute(0, 2, 1, 3);  attn_output_78 = None
    tensor_69 = permute_55.contiguous();  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    attn_output_79 = tensor_69.view((1, 128, 2048));  tensor_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    attn_output_80 = self.L__mod___transformer_h_13_attn_attention_out_proj(attn_output_79);  attn_output_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    attn_output_82 = self.L__mod___transformer_h_13_attn_attention_resid_dropout(attn_output_80);  attn_output_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    residual_27 = attn_output_82 + residual_26;  attn_output_82 = residual_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    hidden_states_121 = self.L__mod___transformer_h_13_ln_2(residual_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    hidden_states_122 = self.L__mod___transformer_h_13_mlp_c_fc(hidden_states_121);  hidden_states_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_52 = 0.5 * hidden_states_122
    pow_14 = torch.pow(hidden_states_122, 3.0)
    mul_53 = 0.044715 * pow_14;  pow_14 = None
    add_54 = hidden_states_122 + mul_53;  hidden_states_122 = mul_53 = None
    mul_54 = 0.7978845608028654 * add_54;  add_54 = None
    tanh_13 = torch.tanh(mul_54);  mul_54 = None
    add_55 = 1.0 + tanh_13;  tanh_13 = None
    hidden_states_123 = mul_52 * add_55;  mul_52 = add_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    hidden_states_124 = self.L__mod___transformer_h_13_mlp_c_proj(hidden_states_123);  hidden_states_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_13 = self.L__mod___transformer_h_13_mlp_dropout(hidden_states_124);  hidden_states_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    residual_28 = residual_27 + feed_forward_hidden_states_13;  residual_27 = feed_forward_hidden_states_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_128 = self.L__mod___transformer_h_14_ln_1(residual_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    query_42 = self.L__mod___transformer_h_14_attn_attention_q_proj(hidden_states_128)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    key_42 = self.L__mod___transformer_h_14_attn_attention_k_proj(hidden_states_128)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    value_28 = self.L__mod___transformer_h_14_attn_attention_v_proj(hidden_states_128);  hidden_states_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_70 = query_42.view((1, 128, 16, 128));  query_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_43 = tensor_70.permute(0, 2, 1, 3);  tensor_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_71 = key_42.view((1, 128, 16, 128));  key_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_43 = tensor_71.permute(0, 2, 1, 3);  tensor_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_72 = value_28.view((1, 128, 16, 128));  value_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_29 = tensor_72.permute(0, 2, 1, 3);  tensor_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:186, code: query = query.to(torch.float32)
    query_44 = query_43.to(torch.float32);  query_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:187, code: key = key.to(torch.float32)
    key_44 = key_43.to(torch.float32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_14 = key_44.transpose(-1, -2);  key_44 = None
    attn_weights_84 = torch.matmul(query_44, transpose_14);  query_44 = transpose_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_14_attn_attention_bias = self.L__mod___transformer_h_14_attn_attention_bias
    causal_mask_14 = l__mod___transformer_h_14_attn_attention_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_14_attn_attention_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_73 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_14 = tensor_73.to(device(type='cuda', index=0));  tensor_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_85 = torch.where(causal_mask_14, attn_weights_84, mask_value_14);  causal_mask_14 = attn_weights_84 = mask_value_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_86 = torch.nn.functional.softmax(attn_weights_85, dim = -1);  attn_weights_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:204, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_87 = attn_weights_86.to(torch.float32);  attn_weights_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_89 = self.L__mod___transformer_h_14_attn_attention_attn_dropout(attn_weights_87);  attn_weights_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_84 = torch.matmul(attn_weights_89, value_29);  attn_weights_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_59 = attn_output_84.permute(0, 2, 1, 3);  attn_output_84 = None
    tensor_74 = permute_59.contiguous();  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    attn_output_85 = tensor_74.view((1, 128, 2048));  tensor_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    attn_output_86 = self.L__mod___transformer_h_14_attn_attention_out_proj(attn_output_85);  attn_output_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    attn_output_88 = self.L__mod___transformer_h_14_attn_attention_resid_dropout(attn_output_86);  attn_output_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    residual_29 = attn_output_88 + residual_28;  attn_output_88 = residual_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    hidden_states_130 = self.L__mod___transformer_h_14_ln_2(residual_29)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    hidden_states_131 = self.L__mod___transformer_h_14_mlp_c_fc(hidden_states_130);  hidden_states_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_56 = 0.5 * hidden_states_131
    pow_15 = torch.pow(hidden_states_131, 3.0)
    mul_57 = 0.044715 * pow_15;  pow_15 = None
    add_58 = hidden_states_131 + mul_57;  hidden_states_131 = mul_57 = None
    mul_58 = 0.7978845608028654 * add_58;  add_58 = None
    tanh_14 = torch.tanh(mul_58);  mul_58 = None
    add_59 = 1.0 + tanh_14;  tanh_14 = None
    hidden_states_132 = mul_56 * add_59;  mul_56 = add_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    hidden_states_133 = self.L__mod___transformer_h_14_mlp_c_proj(hidden_states_132);  hidden_states_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_14 = self.L__mod___transformer_h_14_mlp_dropout(hidden_states_133);  hidden_states_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    residual_30 = residual_29 + feed_forward_hidden_states_14;  residual_29 = feed_forward_hidden_states_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_137 = self.L__mod___transformer_h_15_ln_1(residual_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    query_45 = self.L__mod___transformer_h_15_attn_attention_q_proj(hidden_states_137)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    key_45 = self.L__mod___transformer_h_15_attn_attention_k_proj(hidden_states_137)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    value_30 = self.L__mod___transformer_h_15_attn_attention_v_proj(hidden_states_137);  hidden_states_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_75 = query_45.view((1, 128, 16, 128));  query_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_46 = tensor_75.permute(0, 2, 1, 3);  tensor_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_76 = key_45.view((1, 128, 16, 128));  key_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_46 = tensor_76.permute(0, 2, 1, 3);  tensor_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_77 = value_30.view((1, 128, 16, 128));  value_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_31 = tensor_77.permute(0, 2, 1, 3);  tensor_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:186, code: query = query.to(torch.float32)
    query_47 = query_46.to(torch.float32);  query_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:187, code: key = key.to(torch.float32)
    key_47 = key_46.to(torch.float32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_15 = key_47.transpose(-1, -2);  key_47 = None
    attn_weights_90 = torch.matmul(query_47, transpose_15);  query_47 = transpose_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_15_attn_attention_bias = self.L__mod___transformer_h_15_attn_attention_bias
    causal_mask_15 = l__mod___transformer_h_15_attn_attention_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_15_attn_attention_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_78 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_15 = tensor_78.to(device(type='cuda', index=0));  tensor_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_91 = torch.where(causal_mask_15, attn_weights_90, mask_value_15);  causal_mask_15 = attn_weights_90 = mask_value_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_92 = torch.nn.functional.softmax(attn_weights_91, dim = -1);  attn_weights_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:204, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_93 = attn_weights_92.to(torch.float32);  attn_weights_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_95 = self.L__mod___transformer_h_15_attn_attention_attn_dropout(attn_weights_93);  attn_weights_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_90 = torch.matmul(attn_weights_95, value_31);  attn_weights_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_63 = attn_output_90.permute(0, 2, 1, 3);  attn_output_90 = None
    tensor_79 = permute_63.contiguous();  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    attn_output_91 = tensor_79.view((1, 128, 2048));  tensor_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    attn_output_92 = self.L__mod___transformer_h_15_attn_attention_out_proj(attn_output_91);  attn_output_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    attn_output_94 = self.L__mod___transformer_h_15_attn_attention_resid_dropout(attn_output_92);  attn_output_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    residual_31 = attn_output_94 + residual_30;  attn_output_94 = residual_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    hidden_states_139 = self.L__mod___transformer_h_15_ln_2(residual_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    hidden_states_140 = self.L__mod___transformer_h_15_mlp_c_fc(hidden_states_139);  hidden_states_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_60 = 0.5 * hidden_states_140
    pow_16 = torch.pow(hidden_states_140, 3.0)
    mul_61 = 0.044715 * pow_16;  pow_16 = None
    add_62 = hidden_states_140 + mul_61;  hidden_states_140 = mul_61 = None
    mul_62 = 0.7978845608028654 * add_62;  add_62 = None
    tanh_15 = torch.tanh(mul_62);  mul_62 = None
    add_63 = 1.0 + tanh_15;  tanh_15 = None
    hidden_states_141 = mul_60 * add_63;  mul_60 = add_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    hidden_states_142 = self.L__mod___transformer_h_15_mlp_c_proj(hidden_states_141);  hidden_states_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_15 = self.L__mod___transformer_h_15_mlp_dropout(hidden_states_142);  hidden_states_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    residual_32 = residual_31 + feed_forward_hidden_states_15;  residual_31 = feed_forward_hidden_states_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_146 = self.L__mod___transformer_h_16_ln_1(residual_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    query_48 = self.L__mod___transformer_h_16_attn_attention_q_proj(hidden_states_146)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    key_48 = self.L__mod___transformer_h_16_attn_attention_k_proj(hidden_states_146)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    value_32 = self.L__mod___transformer_h_16_attn_attention_v_proj(hidden_states_146);  hidden_states_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_80 = query_48.view((1, 128, 16, 128));  query_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_49 = tensor_80.permute(0, 2, 1, 3);  tensor_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_81 = key_48.view((1, 128, 16, 128));  key_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_49 = tensor_81.permute(0, 2, 1, 3);  tensor_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_82 = value_32.view((1, 128, 16, 128));  value_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_33 = tensor_82.permute(0, 2, 1, 3);  tensor_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:186, code: query = query.to(torch.float32)
    query_50 = query_49.to(torch.float32);  query_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:187, code: key = key.to(torch.float32)
    key_50 = key_49.to(torch.float32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_16 = key_50.transpose(-1, -2);  key_50 = None
    attn_weights_96 = torch.matmul(query_50, transpose_16);  query_50 = transpose_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_16_attn_attention_bias = self.L__mod___transformer_h_16_attn_attention_bias
    causal_mask_16 = l__mod___transformer_h_16_attn_attention_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_16_attn_attention_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_83 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_16 = tensor_83.to(device(type='cuda', index=0));  tensor_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_97 = torch.where(causal_mask_16, attn_weights_96, mask_value_16);  causal_mask_16 = attn_weights_96 = mask_value_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_98 = torch.nn.functional.softmax(attn_weights_97, dim = -1);  attn_weights_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:204, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_99 = attn_weights_98.to(torch.float32);  attn_weights_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_101 = self.L__mod___transformer_h_16_attn_attention_attn_dropout(attn_weights_99);  attn_weights_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_96 = torch.matmul(attn_weights_101, value_33);  attn_weights_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_67 = attn_output_96.permute(0, 2, 1, 3);  attn_output_96 = None
    tensor_84 = permute_67.contiguous();  permute_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    attn_output_97 = tensor_84.view((1, 128, 2048));  tensor_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    attn_output_98 = self.L__mod___transformer_h_16_attn_attention_out_proj(attn_output_97);  attn_output_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    attn_output_100 = self.L__mod___transformer_h_16_attn_attention_resid_dropout(attn_output_98);  attn_output_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    residual_33 = attn_output_100 + residual_32;  attn_output_100 = residual_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    hidden_states_148 = self.L__mod___transformer_h_16_ln_2(residual_33)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    hidden_states_149 = self.L__mod___transformer_h_16_mlp_c_fc(hidden_states_148);  hidden_states_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_64 = 0.5 * hidden_states_149
    pow_17 = torch.pow(hidden_states_149, 3.0)
    mul_65 = 0.044715 * pow_17;  pow_17 = None
    add_66 = hidden_states_149 + mul_65;  hidden_states_149 = mul_65 = None
    mul_66 = 0.7978845608028654 * add_66;  add_66 = None
    tanh_16 = torch.tanh(mul_66);  mul_66 = None
    add_67 = 1.0 + tanh_16;  tanh_16 = None
    hidden_states_150 = mul_64 * add_67;  mul_64 = add_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    hidden_states_151 = self.L__mod___transformer_h_16_mlp_c_proj(hidden_states_150);  hidden_states_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_16 = self.L__mod___transformer_h_16_mlp_dropout(hidden_states_151);  hidden_states_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    residual_34 = residual_33 + feed_forward_hidden_states_16;  residual_33 = feed_forward_hidden_states_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_155 = self.L__mod___transformer_h_17_ln_1(residual_34)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    query_51 = self.L__mod___transformer_h_17_attn_attention_q_proj(hidden_states_155)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    key_51 = self.L__mod___transformer_h_17_attn_attention_k_proj(hidden_states_155)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    value_34 = self.L__mod___transformer_h_17_attn_attention_v_proj(hidden_states_155);  hidden_states_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_85 = query_51.view((1, 128, 16, 128));  query_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_52 = tensor_85.permute(0, 2, 1, 3);  tensor_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_86 = key_51.view((1, 128, 16, 128));  key_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_52 = tensor_86.permute(0, 2, 1, 3);  tensor_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_87 = value_34.view((1, 128, 16, 128));  value_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_35 = tensor_87.permute(0, 2, 1, 3);  tensor_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:186, code: query = query.to(torch.float32)
    query_53 = query_52.to(torch.float32);  query_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:187, code: key = key.to(torch.float32)
    key_53 = key_52.to(torch.float32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_17 = key_53.transpose(-1, -2);  key_53 = None
    attn_weights_102 = torch.matmul(query_53, transpose_17);  query_53 = transpose_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_17_attn_attention_bias = self.L__mod___transformer_h_17_attn_attention_bias
    causal_mask_17 = l__mod___transformer_h_17_attn_attention_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_17_attn_attention_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_88 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_17 = tensor_88.to(device(type='cuda', index=0));  tensor_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_103 = torch.where(causal_mask_17, attn_weights_102, mask_value_17);  causal_mask_17 = attn_weights_102 = mask_value_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_104 = torch.nn.functional.softmax(attn_weights_103, dim = -1);  attn_weights_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:204, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_105 = attn_weights_104.to(torch.float32);  attn_weights_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_107 = self.L__mod___transformer_h_17_attn_attention_attn_dropout(attn_weights_105);  attn_weights_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_102 = torch.matmul(attn_weights_107, value_35);  attn_weights_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_71 = attn_output_102.permute(0, 2, 1, 3);  attn_output_102 = None
    tensor_89 = permute_71.contiguous();  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    attn_output_103 = tensor_89.view((1, 128, 2048));  tensor_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    attn_output_104 = self.L__mod___transformer_h_17_attn_attention_out_proj(attn_output_103);  attn_output_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    attn_output_106 = self.L__mod___transformer_h_17_attn_attention_resid_dropout(attn_output_104);  attn_output_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    residual_35 = attn_output_106 + residual_34;  attn_output_106 = residual_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    hidden_states_157 = self.L__mod___transformer_h_17_ln_2(residual_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    hidden_states_158 = self.L__mod___transformer_h_17_mlp_c_fc(hidden_states_157);  hidden_states_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_68 = 0.5 * hidden_states_158
    pow_18 = torch.pow(hidden_states_158, 3.0)
    mul_69 = 0.044715 * pow_18;  pow_18 = None
    add_70 = hidden_states_158 + mul_69;  hidden_states_158 = mul_69 = None
    mul_70 = 0.7978845608028654 * add_70;  add_70 = None
    tanh_17 = torch.tanh(mul_70);  mul_70 = None
    add_71 = 1.0 + tanh_17;  tanh_17 = None
    hidden_states_159 = mul_68 * add_71;  mul_68 = add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    hidden_states_160 = self.L__mod___transformer_h_17_mlp_c_proj(hidden_states_159);  hidden_states_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_17 = self.L__mod___transformer_h_17_mlp_dropout(hidden_states_160);  hidden_states_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    residual_36 = residual_35 + feed_forward_hidden_states_17;  residual_35 = feed_forward_hidden_states_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_164 = self.L__mod___transformer_h_18_ln_1(residual_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    query_54 = self.L__mod___transformer_h_18_attn_attention_q_proj(hidden_states_164)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    key_54 = self.L__mod___transformer_h_18_attn_attention_k_proj(hidden_states_164)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    value_36 = self.L__mod___transformer_h_18_attn_attention_v_proj(hidden_states_164);  hidden_states_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_90 = query_54.view((1, 128, 16, 128));  query_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_55 = tensor_90.permute(0, 2, 1, 3);  tensor_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_91 = key_54.view((1, 128, 16, 128));  key_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_55 = tensor_91.permute(0, 2, 1, 3);  tensor_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_92 = value_36.view((1, 128, 16, 128));  value_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_37 = tensor_92.permute(0, 2, 1, 3);  tensor_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:186, code: query = query.to(torch.float32)
    query_56 = query_55.to(torch.float32);  query_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:187, code: key = key.to(torch.float32)
    key_56 = key_55.to(torch.float32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_18 = key_56.transpose(-1, -2);  key_56 = None
    attn_weights_108 = torch.matmul(query_56, transpose_18);  query_56 = transpose_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_18_attn_attention_bias = self.L__mod___transformer_h_18_attn_attention_bias
    causal_mask_18 = l__mod___transformer_h_18_attn_attention_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_18_attn_attention_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_93 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_18 = tensor_93.to(device(type='cuda', index=0));  tensor_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_109 = torch.where(causal_mask_18, attn_weights_108, mask_value_18);  causal_mask_18 = attn_weights_108 = mask_value_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_110 = torch.nn.functional.softmax(attn_weights_109, dim = -1);  attn_weights_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:204, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_111 = attn_weights_110.to(torch.float32);  attn_weights_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_113 = self.L__mod___transformer_h_18_attn_attention_attn_dropout(attn_weights_111);  attn_weights_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_108 = torch.matmul(attn_weights_113, value_37);  attn_weights_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_75 = attn_output_108.permute(0, 2, 1, 3);  attn_output_108 = None
    tensor_94 = permute_75.contiguous();  permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    attn_output_109 = tensor_94.view((1, 128, 2048));  tensor_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    attn_output_110 = self.L__mod___transformer_h_18_attn_attention_out_proj(attn_output_109);  attn_output_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    attn_output_112 = self.L__mod___transformer_h_18_attn_attention_resid_dropout(attn_output_110);  attn_output_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    residual_37 = attn_output_112 + residual_36;  attn_output_112 = residual_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    hidden_states_166 = self.L__mod___transformer_h_18_ln_2(residual_37)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    hidden_states_167 = self.L__mod___transformer_h_18_mlp_c_fc(hidden_states_166);  hidden_states_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_72 = 0.5 * hidden_states_167
    pow_19 = torch.pow(hidden_states_167, 3.0)
    mul_73 = 0.044715 * pow_19;  pow_19 = None
    add_74 = hidden_states_167 + mul_73;  hidden_states_167 = mul_73 = None
    mul_74 = 0.7978845608028654 * add_74;  add_74 = None
    tanh_18 = torch.tanh(mul_74);  mul_74 = None
    add_75 = 1.0 + tanh_18;  tanh_18 = None
    hidden_states_168 = mul_72 * add_75;  mul_72 = add_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    hidden_states_169 = self.L__mod___transformer_h_18_mlp_c_proj(hidden_states_168);  hidden_states_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_18 = self.L__mod___transformer_h_18_mlp_dropout(hidden_states_169);  hidden_states_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    residual_38 = residual_37 + feed_forward_hidden_states_18;  residual_37 = feed_forward_hidden_states_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_173 = self.L__mod___transformer_h_19_ln_1(residual_38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    query_57 = self.L__mod___transformer_h_19_attn_attention_q_proj(hidden_states_173)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    key_57 = self.L__mod___transformer_h_19_attn_attention_k_proj(hidden_states_173)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    value_38 = self.L__mod___transformer_h_19_attn_attention_v_proj(hidden_states_173);  hidden_states_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_95 = query_57.view((1, 128, 16, 128));  query_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_58 = tensor_95.permute(0, 2, 1, 3);  tensor_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_96 = key_57.view((1, 128, 16, 128));  key_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_58 = tensor_96.permute(0, 2, 1, 3);  tensor_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_97 = value_38.view((1, 128, 16, 128));  value_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_39 = tensor_97.permute(0, 2, 1, 3);  tensor_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:186, code: query = query.to(torch.float32)
    query_59 = query_58.to(torch.float32);  query_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:187, code: key = key.to(torch.float32)
    key_59 = key_58.to(torch.float32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_19 = key_59.transpose(-1, -2);  key_59 = None
    attn_weights_114 = torch.matmul(query_59, transpose_19);  query_59 = transpose_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_19_attn_attention_bias = self.L__mod___transformer_h_19_attn_attention_bias
    causal_mask_19 = l__mod___transformer_h_19_attn_attention_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_19_attn_attention_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_98 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_19 = tensor_98.to(device(type='cuda', index=0));  tensor_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_115 = torch.where(causal_mask_19, attn_weights_114, mask_value_19);  causal_mask_19 = attn_weights_114 = mask_value_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_116 = torch.nn.functional.softmax(attn_weights_115, dim = -1);  attn_weights_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:204, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_117 = attn_weights_116.to(torch.float32);  attn_weights_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_119 = self.L__mod___transformer_h_19_attn_attention_attn_dropout(attn_weights_117);  attn_weights_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_114 = torch.matmul(attn_weights_119, value_39);  attn_weights_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_79 = attn_output_114.permute(0, 2, 1, 3);  attn_output_114 = None
    tensor_99 = permute_79.contiguous();  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    attn_output_115 = tensor_99.view((1, 128, 2048));  tensor_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    attn_output_116 = self.L__mod___transformer_h_19_attn_attention_out_proj(attn_output_115);  attn_output_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    attn_output_118 = self.L__mod___transformer_h_19_attn_attention_resid_dropout(attn_output_116);  attn_output_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    residual_39 = attn_output_118 + residual_38;  attn_output_118 = residual_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    hidden_states_175 = self.L__mod___transformer_h_19_ln_2(residual_39)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    hidden_states_176 = self.L__mod___transformer_h_19_mlp_c_fc(hidden_states_175);  hidden_states_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_76 = 0.5 * hidden_states_176
    pow_20 = torch.pow(hidden_states_176, 3.0)
    mul_77 = 0.044715 * pow_20;  pow_20 = None
    add_78 = hidden_states_176 + mul_77;  hidden_states_176 = mul_77 = None
    mul_78 = 0.7978845608028654 * add_78;  add_78 = None
    tanh_19 = torch.tanh(mul_78);  mul_78 = None
    add_79 = 1.0 + tanh_19;  tanh_19 = None
    hidden_states_177 = mul_76 * add_79;  mul_76 = add_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    hidden_states_178 = self.L__mod___transformer_h_19_mlp_c_proj(hidden_states_177);  hidden_states_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_19 = self.L__mod___transformer_h_19_mlp_dropout(hidden_states_178);  hidden_states_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    residual_40 = residual_39 + feed_forward_hidden_states_19;  residual_39 = feed_forward_hidden_states_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_182 = self.L__mod___transformer_h_20_ln_1(residual_40)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    query_60 = self.L__mod___transformer_h_20_attn_attention_q_proj(hidden_states_182)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    key_60 = self.L__mod___transformer_h_20_attn_attention_k_proj(hidden_states_182)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    value_40 = self.L__mod___transformer_h_20_attn_attention_v_proj(hidden_states_182);  hidden_states_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_100 = query_60.view((1, 128, 16, 128));  query_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_61 = tensor_100.permute(0, 2, 1, 3);  tensor_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_101 = key_60.view((1, 128, 16, 128));  key_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_61 = tensor_101.permute(0, 2, 1, 3);  tensor_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_102 = value_40.view((1, 128, 16, 128));  value_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_41 = tensor_102.permute(0, 2, 1, 3);  tensor_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:186, code: query = query.to(torch.float32)
    query_62 = query_61.to(torch.float32);  query_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:187, code: key = key.to(torch.float32)
    key_62 = key_61.to(torch.float32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_20 = key_62.transpose(-1, -2);  key_62 = None
    attn_weights_120 = torch.matmul(query_62, transpose_20);  query_62 = transpose_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_20_attn_attention_bias = self.L__mod___transformer_h_20_attn_attention_bias
    causal_mask_20 = l__mod___transformer_h_20_attn_attention_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_20_attn_attention_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_103 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_20 = tensor_103.to(device(type='cuda', index=0));  tensor_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_121 = torch.where(causal_mask_20, attn_weights_120, mask_value_20);  causal_mask_20 = attn_weights_120 = mask_value_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_122 = torch.nn.functional.softmax(attn_weights_121, dim = -1);  attn_weights_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:204, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_123 = attn_weights_122.to(torch.float32);  attn_weights_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_125 = self.L__mod___transformer_h_20_attn_attention_attn_dropout(attn_weights_123);  attn_weights_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_120 = torch.matmul(attn_weights_125, value_41);  attn_weights_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_83 = attn_output_120.permute(0, 2, 1, 3);  attn_output_120 = None
    tensor_104 = permute_83.contiguous();  permute_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    attn_output_121 = tensor_104.view((1, 128, 2048));  tensor_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    attn_output_122 = self.L__mod___transformer_h_20_attn_attention_out_proj(attn_output_121);  attn_output_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    attn_output_124 = self.L__mod___transformer_h_20_attn_attention_resid_dropout(attn_output_122);  attn_output_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    residual_41 = attn_output_124 + residual_40;  attn_output_124 = residual_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    hidden_states_184 = self.L__mod___transformer_h_20_ln_2(residual_41)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    hidden_states_185 = self.L__mod___transformer_h_20_mlp_c_fc(hidden_states_184);  hidden_states_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_80 = 0.5 * hidden_states_185
    pow_21 = torch.pow(hidden_states_185, 3.0)
    mul_81 = 0.044715 * pow_21;  pow_21 = None
    add_82 = hidden_states_185 + mul_81;  hidden_states_185 = mul_81 = None
    mul_82 = 0.7978845608028654 * add_82;  add_82 = None
    tanh_20 = torch.tanh(mul_82);  mul_82 = None
    add_83 = 1.0 + tanh_20;  tanh_20 = None
    hidden_states_186 = mul_80 * add_83;  mul_80 = add_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    hidden_states_187 = self.L__mod___transformer_h_20_mlp_c_proj(hidden_states_186);  hidden_states_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_20 = self.L__mod___transformer_h_20_mlp_dropout(hidden_states_187);  hidden_states_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    residual_42 = residual_41 + feed_forward_hidden_states_20;  residual_41 = feed_forward_hidden_states_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_191 = self.L__mod___transformer_h_21_ln_1(residual_42)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    query_63 = self.L__mod___transformer_h_21_attn_attention_q_proj(hidden_states_191)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    key_63 = self.L__mod___transformer_h_21_attn_attention_k_proj(hidden_states_191)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    value_42 = self.L__mod___transformer_h_21_attn_attention_v_proj(hidden_states_191);  hidden_states_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_105 = query_63.view((1, 128, 16, 128));  query_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_64 = tensor_105.permute(0, 2, 1, 3);  tensor_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_106 = key_63.view((1, 128, 16, 128));  key_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_64 = tensor_106.permute(0, 2, 1, 3);  tensor_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_107 = value_42.view((1, 128, 16, 128));  value_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_43 = tensor_107.permute(0, 2, 1, 3);  tensor_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:186, code: query = query.to(torch.float32)
    query_65 = query_64.to(torch.float32);  query_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:187, code: key = key.to(torch.float32)
    key_65 = key_64.to(torch.float32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_21 = key_65.transpose(-1, -2);  key_65 = None
    attn_weights_126 = torch.matmul(query_65, transpose_21);  query_65 = transpose_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_21_attn_attention_bias = self.L__mod___transformer_h_21_attn_attention_bias
    causal_mask_21 = l__mod___transformer_h_21_attn_attention_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_21_attn_attention_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_108 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_21 = tensor_108.to(device(type='cuda', index=0));  tensor_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_127 = torch.where(causal_mask_21, attn_weights_126, mask_value_21);  causal_mask_21 = attn_weights_126 = mask_value_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_128 = torch.nn.functional.softmax(attn_weights_127, dim = -1);  attn_weights_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:204, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_129 = attn_weights_128.to(torch.float32);  attn_weights_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_131 = self.L__mod___transformer_h_21_attn_attention_attn_dropout(attn_weights_129);  attn_weights_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_126 = torch.matmul(attn_weights_131, value_43);  attn_weights_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_87 = attn_output_126.permute(0, 2, 1, 3);  attn_output_126 = None
    tensor_109 = permute_87.contiguous();  permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    attn_output_127 = tensor_109.view((1, 128, 2048));  tensor_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    attn_output_128 = self.L__mod___transformer_h_21_attn_attention_out_proj(attn_output_127);  attn_output_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    attn_output_130 = self.L__mod___transformer_h_21_attn_attention_resid_dropout(attn_output_128);  attn_output_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    residual_43 = attn_output_130 + residual_42;  attn_output_130 = residual_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    hidden_states_193 = self.L__mod___transformer_h_21_ln_2(residual_43)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    hidden_states_194 = self.L__mod___transformer_h_21_mlp_c_fc(hidden_states_193);  hidden_states_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_84 = 0.5 * hidden_states_194
    pow_22 = torch.pow(hidden_states_194, 3.0)
    mul_85 = 0.044715 * pow_22;  pow_22 = None
    add_86 = hidden_states_194 + mul_85;  hidden_states_194 = mul_85 = None
    mul_86 = 0.7978845608028654 * add_86;  add_86 = None
    tanh_21 = torch.tanh(mul_86);  mul_86 = None
    add_87 = 1.0 + tanh_21;  tanh_21 = None
    hidden_states_195 = mul_84 * add_87;  mul_84 = add_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    hidden_states_196 = self.L__mod___transformer_h_21_mlp_c_proj(hidden_states_195);  hidden_states_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_21 = self.L__mod___transformer_h_21_mlp_dropout(hidden_states_196);  hidden_states_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    residual_44 = residual_43 + feed_forward_hidden_states_21;  residual_43 = feed_forward_hidden_states_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_200 = self.L__mod___transformer_h_22_ln_1(residual_44)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    query_66 = self.L__mod___transformer_h_22_attn_attention_q_proj(hidden_states_200)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    key_66 = self.L__mod___transformer_h_22_attn_attention_k_proj(hidden_states_200)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    value_44 = self.L__mod___transformer_h_22_attn_attention_v_proj(hidden_states_200);  hidden_states_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_110 = query_66.view((1, 128, 16, 128));  query_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_67 = tensor_110.permute(0, 2, 1, 3);  tensor_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_111 = key_66.view((1, 128, 16, 128));  key_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_67 = tensor_111.permute(0, 2, 1, 3);  tensor_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_112 = value_44.view((1, 128, 16, 128));  value_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_45 = tensor_112.permute(0, 2, 1, 3);  tensor_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:186, code: query = query.to(torch.float32)
    query_68 = query_67.to(torch.float32);  query_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:187, code: key = key.to(torch.float32)
    key_68 = key_67.to(torch.float32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_22 = key_68.transpose(-1, -2);  key_68 = None
    attn_weights_132 = torch.matmul(query_68, transpose_22);  query_68 = transpose_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_22_attn_attention_bias = self.L__mod___transformer_h_22_attn_attention_bias
    causal_mask_22 = l__mod___transformer_h_22_attn_attention_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_22_attn_attention_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_113 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_22 = tensor_113.to(device(type='cuda', index=0));  tensor_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_133 = torch.where(causal_mask_22, attn_weights_132, mask_value_22);  causal_mask_22 = attn_weights_132 = mask_value_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_134 = torch.nn.functional.softmax(attn_weights_133, dim = -1);  attn_weights_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:204, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_135 = attn_weights_134.to(torch.float32);  attn_weights_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_137 = self.L__mod___transformer_h_22_attn_attention_attn_dropout(attn_weights_135);  attn_weights_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_132 = torch.matmul(attn_weights_137, value_45);  attn_weights_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_91 = attn_output_132.permute(0, 2, 1, 3);  attn_output_132 = None
    tensor_114 = permute_91.contiguous();  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    attn_output_133 = tensor_114.view((1, 128, 2048));  tensor_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    attn_output_134 = self.L__mod___transformer_h_22_attn_attention_out_proj(attn_output_133);  attn_output_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    attn_output_136 = self.L__mod___transformer_h_22_attn_attention_resid_dropout(attn_output_134);  attn_output_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    residual_45 = attn_output_136 + residual_44;  attn_output_136 = residual_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    hidden_states_202 = self.L__mod___transformer_h_22_ln_2(residual_45)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    hidden_states_203 = self.L__mod___transformer_h_22_mlp_c_fc(hidden_states_202);  hidden_states_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_88 = 0.5 * hidden_states_203
    pow_23 = torch.pow(hidden_states_203, 3.0)
    mul_89 = 0.044715 * pow_23;  pow_23 = None
    add_90 = hidden_states_203 + mul_89;  hidden_states_203 = mul_89 = None
    mul_90 = 0.7978845608028654 * add_90;  add_90 = None
    tanh_22 = torch.tanh(mul_90);  mul_90 = None
    add_91 = 1.0 + tanh_22;  tanh_22 = None
    hidden_states_204 = mul_88 * add_91;  mul_88 = add_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    hidden_states_205 = self.L__mod___transformer_h_22_mlp_c_proj(hidden_states_204);  hidden_states_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_22 = self.L__mod___transformer_h_22_mlp_dropout(hidden_states_205);  hidden_states_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    residual_46 = residual_45 + feed_forward_hidden_states_22;  residual_45 = feed_forward_hidden_states_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_209 = self.L__mod___transformer_h_23_ln_1(residual_46)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    query_69 = self.L__mod___transformer_h_23_attn_attention_q_proj(hidden_states_209)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    key_69 = self.L__mod___transformer_h_23_attn_attention_k_proj(hidden_states_209)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    value_46 = self.L__mod___transformer_h_23_attn_attention_v_proj(hidden_states_209);  hidden_states_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_115 = query_69.view((1, 128, 16, 128));  query_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_70 = tensor_115.permute(0, 2, 1, 3);  tensor_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_116 = key_69.view((1, 128, 16, 128));  key_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_70 = tensor_116.permute(0, 2, 1, 3);  tensor_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    tensor_117 = value_46.view((1, 128, 16, 128));  value_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_47 = tensor_117.permute(0, 2, 1, 3);  tensor_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:186, code: query = query.to(torch.float32)
    query_71 = query_70.to(torch.float32);  query_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:187, code: key = key.to(torch.float32)
    key_71 = key_70.to(torch.float32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_23 = key_71.transpose(-1, -2);  key_71 = None
    attn_weights_138 = torch.matmul(query_71, transpose_23);  query_71 = transpose_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_23_attn_attention_bias = self.L__mod___transformer_h_23_attn_attention_bias
    causal_mask_23 = l__mod___transformer_h_23_attn_attention_bias[(slice(None, None, None), slice(None, None, None), slice(0, 128, None), slice(None, 128, None))];  l__mod___transformer_h_23_attn_attention_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    tensor_118 = torch.tensor(-3.4028234663852886e+38, dtype = torch.float32)
    mask_value_23 = tensor_118.to(device(type='cuda', index=0));  tensor_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    attn_weights_139 = torch.where(causal_mask_23, attn_weights_138, mask_value_23);  causal_mask_23 = attn_weights_138 = mask_value_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_140 = torch.nn.functional.softmax(attn_weights_139, dim = -1);  attn_weights_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:204, code: attn_weights = attn_weights.to(value.dtype)
    attn_weights_141 = attn_weights_140.to(torch.float32);  attn_weights_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_143 = self.L__mod___transformer_h_23_attn_attention_attn_dropout(attn_weights_141);  attn_weights_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_138 = torch.matmul(attn_weights_143, value_47);  attn_weights_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_95 = attn_output_138.permute(0, 2, 1, 3);  attn_output_138 = None
    tensor_119 = permute_95.contiguous();  permute_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    attn_output_139 = tensor_119.view((1, 128, 2048));  tensor_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    attn_output_140 = self.L__mod___transformer_h_23_attn_attention_out_proj(attn_output_139);  attn_output_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    attn_output_142 = self.L__mod___transformer_h_23_attn_attention_resid_dropout(attn_output_140);  attn_output_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    residual_47 = attn_output_142 + residual_46;  attn_output_142 = residual_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    hidden_states_211 = self.L__mod___transformer_h_23_ln_2(residual_47)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    hidden_states_212 = self.L__mod___transformer_h_23_mlp_c_fc(hidden_states_211);  hidden_states_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_92 = 0.5 * hidden_states_212
    pow_24 = torch.pow(hidden_states_212, 3.0)
    mul_93 = 0.044715 * pow_24;  pow_24 = None
    add_94 = hidden_states_212 + mul_93;  hidden_states_212 = mul_93 = None
    mul_94 = 0.7978845608028654 * add_94;  add_94 = None
    tanh_23 = torch.tanh(mul_94);  mul_94 = None
    add_95 = 1.0 + tanh_23;  tanh_23 = None
    hidden_states_213 = mul_92 * add_95;  mul_92 = add_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    hidden_states_214 = self.L__mod___transformer_h_23_mlp_c_proj(hidden_states_213);  hidden_states_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_23 = self.L__mod___transformer_h_23_mlp_dropout(hidden_states_214);  hidden_states_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    hidden_states_217 = residual_47 + feed_forward_hidden_states_23;  residual_47 = feed_forward_hidden_states_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:641, code: hidden_states = self.ln_f(hidden_states)
    hidden_states_218 = self.L__mod___transformer_ln_f(hidden_states_217);  hidden_states_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:643, code: hidden_states = hidden_states.view(output_shape)
    hidden_states_220 = hidden_states_218.view((-1, 128, 2048));  hidden_states_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:763, code: lm_logits = self.lm_head(hidden_states)
    lm_logits = self.L__mod___lm_head(hidden_states_220);  hidden_states_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:768, code: labels = labels.to(lm_logits.device)
    labels = l_inputs_labels_.to(device(type='cuda', index=0));  l_inputs_labels_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:771, code: lm_logits = lm_logits.to(torch.float32)
    lm_logits_1 = lm_logits.to(torch.float32);  lm_logits = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:774, code: shift_logits = lm_logits[..., :-1, :].contiguous()
    getitem_24 = lm_logits_1[(Ellipsis, slice(None, -1, None), slice(None, None, None))]
    shift_logits = getitem_24.contiguous();  getitem_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:775, code: shift_labels = labels[..., 1:].contiguous()
    getitem_25 = labels[(Ellipsis, slice(1, None, None))];  labels = None
    shift_labels = getitem_25.contiguous();  getitem_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:778, code: loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    view_99 = shift_logits.view(-1, 50257);  shift_logits = None
    view_100 = shift_labels.view(-1);  shift_labels = None
    loss = torch.nn.functional.cross_entropy(view_99, view_100, None, None, -100, None, 'mean', 0.0);  view_99 = view_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:780, code: lm_logits = lm_logits.to(hidden_states.dtype)
    lm_logits_2 = lm_logits_1.to(torch.float32);  lm_logits_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:781, code: loss = loss.to(hidden_states.dtype)
    loss_1 = loss.to(torch.float32);  loss = None
    return (loss_1, lm_logits_2, key_1, value_1, key_4, value_3, key_7, value_5, key_10, value_7, key_13, value_9, key_16, value_11, key_19, value_13, key_22, value_15, key_25, value_17, key_28, value_19, key_31, value_21, key_34, value_23, key_37, value_25, key_40, value_27, key_43, value_29, key_46, value_31, key_49, value_33, key_52, value_35, key_55, value_37, key_58, value_39, key_61, value_41, key_64, value_43, key_67, value_45, key_70, value_47)
    