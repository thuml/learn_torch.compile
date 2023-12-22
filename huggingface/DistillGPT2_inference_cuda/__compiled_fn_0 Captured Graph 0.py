from __future__ import annotations



def forward(self, L_inputs_input_ids_ : torch.Tensor, L_inputs_labels_ : torch.Tensor):
    l_inputs_input_ids_ = L_inputs_input_ids_
    l_inputs_labels_ = L_inputs_labels_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:781, code: input_ids = input_ids.view(-1, input_shape[-1])
    input_ids = l_inputs_input_ids_.view(-1, 512);  l_inputs_input_ids_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:802, code: position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
    position_ids = torch.arange(0, 512, dtype = torch.int64, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:803, code: position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
    unsqueeze = position_ids.unsqueeze(0);  position_ids = None
    position_ids_1 = unsqueeze.view(-1, 512);  unsqueeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:843, code: inputs_embeds = self.wte(input_ids)
    inputs_embeds = self.L__mod___transformer_wte(input_ids);  input_ids = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:844, code: position_embeds = self.wpe(position_ids)
    position_embeds = self.L__mod___transformer_wpe(position_ids_1);  position_ids_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:845, code: hidden_states = inputs_embeds + position_embeds
    add = inputs_embeds + position_embeds;  inputs_embeds = position_embeds = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:851, code: hidden_states = self.drop(hidden_states)
    residual = self.L__mod___transformer_drop(add);  add = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    hidden_states = self.L__mod___transformer_h_0_ln_1(residual)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_0_attn_c_attn_bias = self.L__mod___transformer_h_0_attn_c_attn_bias
    view_2 = hidden_states.view(-1, 768);  hidden_states = None
    l__mod___transformer_h_0_attn_c_attn_weight = self.L__mod___transformer_h_0_attn_c_attn_weight
    x = torch.addmm(l__mod___transformer_h_0_attn_c_attn_bias, view_2, l__mod___transformer_h_0_attn_c_attn_weight);  l__mod___transformer_h_0_attn_c_attn_bias = view_2 = l__mod___transformer_h_0_attn_c_attn_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    x_1 = x.view((1, 512, 2304));  x = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split = x_1.split(768, dim = 2);  x_1 = None
    query = split[0]
    key = split[1]
    value = split[2];  split = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor = query.view((1, 512, 12, 64));  query = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_1 = tensor.permute(0, 2, 1, 3);  tensor = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_1 = key.view((1, 512, 12, 64));  key = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_1 = tensor_1.permute(0, 2, 1, 3);  tensor_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_2 = value.view((1, 512, 12, 64));  value = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_1 = tensor_2.permute(0, 2, 1, 3);  tensor_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose = key_1.transpose(-1, -2)
    attn_weights = torch.matmul(query_1, transpose);  query_1 = transpose = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full = torch.full([], 8.0, dtype = torch.float32, device = device(type='cuda', index=0))
    attn_weights_1 = attn_weights / full;  attn_weights = full = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_0_attn_bias = self.L__mod___transformer_h_0_attn_bias
    causal_mask = l__mod___transformer_h_0_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 512, None), slice(None, 512, None))];  l__mod___transformer_h_0_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_1 = torch.full([], -3.4028234663852886e+38, dtype = torch.float32)
    mask_value = full_1.to(device(type='cuda', index=0));  full_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    to_1 = attn_weights_1.to(torch.float32);  attn_weights_1 = None
    attn_weights_2 = torch.where(causal_mask, to_1, mask_value);  causal_mask = to_1 = mask_value = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_3 = torch.nn.functional.softmax(attn_weights_2, dim = -1);  attn_weights_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:211, code: attn_weights = attn_weights.type(value.dtype)
    attn_weights_4 = attn_weights_3.type(torch.float32);  attn_weights_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_6 = self.L__mod___transformer_h_0_attn_attn_dropout(attn_weights_4);  attn_weights_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    attn_output = torch.matmul(attn_weights_6, value_1);  attn_weights_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_3 = attn_output.permute(0, 2, 1, 3);  attn_output = None
    tensor_3 = permute_3.contiguous();  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    attn_output_1 = tensor_3.view((1, 512, 768));  tensor_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_0_attn_c_proj_bias = self.L__mod___transformer_h_0_attn_c_proj_bias
    view_8 = attn_output_1.view(-1, 768);  attn_output_1 = None
    l__mod___transformer_h_0_attn_c_proj_weight = self.L__mod___transformer_h_0_attn_c_proj_weight
    x_2 = torch.addmm(l__mod___transformer_h_0_attn_c_proj_bias, view_8, l__mod___transformer_h_0_attn_c_proj_weight);  l__mod___transformer_h_0_attn_c_proj_bias = view_8 = l__mod___transformer_h_0_attn_c_proj_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    attn_output_2 = x_2.view((1, 512, 768));  x_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    attn_output_4 = self.L__mod___transformer_h_0_attn_resid_dropout(attn_output_2);  attn_output_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    residual_1 = attn_output_4 + residual;  attn_output_4 = residual = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    hidden_states_2 = self.L__mod___transformer_h_0_ln_2(residual_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_0_mlp_c_fc_bias = self.L__mod___transformer_h_0_mlp_c_fc_bias
    view_10 = hidden_states_2.view(-1, 768);  hidden_states_2 = None
    l__mod___transformer_h_0_mlp_c_fc_weight = self.L__mod___transformer_h_0_mlp_c_fc_weight
    x_4 = torch.addmm(l__mod___transformer_h_0_mlp_c_fc_bias, view_10, l__mod___transformer_h_0_mlp_c_fc_weight);  l__mod___transformer_h_0_mlp_c_fc_bias = view_10 = l__mod___transformer_h_0_mlp_c_fc_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    hidden_states_3 = x_4.view((1, 512, 3072));  x_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul = 0.5 * hidden_states_3
    pow_1 = torch.pow(hidden_states_3, 3.0)
    mul_1 = 0.044715 * pow_1;  pow_1 = None
    add_2 = hidden_states_3 + mul_1;  hidden_states_3 = mul_1 = None
    mul_2 = 0.7978845608028654 * add_2;  add_2 = None
    tanh = torch.tanh(mul_2);  mul_2 = None
    add_3 = 1.0 + tanh;  tanh = None
    hidden_states_4 = mul * add_3;  mul = add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_0_mlp_c_proj_bias = self.L__mod___transformer_h_0_mlp_c_proj_bias
    view_12 = hidden_states_4.view(-1, 3072);  hidden_states_4 = None
    l__mod___transformer_h_0_mlp_c_proj_weight = self.L__mod___transformer_h_0_mlp_c_proj_weight
    x_6 = torch.addmm(l__mod___transformer_h_0_mlp_c_proj_bias, view_12, l__mod___transformer_h_0_mlp_c_proj_weight);  l__mod___transformer_h_0_mlp_c_proj_bias = view_12 = l__mod___transformer_h_0_mlp_c_proj_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    hidden_states_5 = x_6.view((1, 512, 768));  x_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states = self.L__mod___transformer_h_0_mlp_dropout(hidden_states_5);  hidden_states_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    residual_2 = residual_1 + feed_forward_hidden_states;  residual_1 = feed_forward_hidden_states = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_8 = self.L__mod___transformer_h_1_ln_1(residual_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_1_attn_c_attn_bias = self.L__mod___transformer_h_1_attn_c_attn_bias
    view_14 = hidden_states_8.view(-1, 768);  hidden_states_8 = None
    l__mod___transformer_h_1_attn_c_attn_weight = self.L__mod___transformer_h_1_attn_c_attn_weight
    x_8 = torch.addmm(l__mod___transformer_h_1_attn_c_attn_bias, view_14, l__mod___transformer_h_1_attn_c_attn_weight);  l__mod___transformer_h_1_attn_c_attn_bias = view_14 = l__mod___transformer_h_1_attn_c_attn_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    x_9 = x_8.view((1, 512, 2304));  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_1 = x_9.split(768, dim = 2);  x_9 = None
    query_2 = split_1[0]
    key_2 = split_1[1]
    value_2 = split_1[2];  split_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_4 = query_2.view((1, 512, 12, 64));  query_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_3 = tensor_4.permute(0, 2, 1, 3);  tensor_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_5 = key_2.view((1, 512, 12, 64));  key_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_3 = tensor_5.permute(0, 2, 1, 3);  tensor_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_6 = value_2.view((1, 512, 12, 64));  value_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_3 = tensor_6.permute(0, 2, 1, 3);  tensor_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_1 = key_3.transpose(-1, -2)
    attn_weights_7 = torch.matmul(query_3, transpose_1);  query_3 = transpose_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_2 = torch.full([], 8.0, dtype = torch.float32, device = device(type='cuda', index=0))
    attn_weights_8 = attn_weights_7 / full_2;  attn_weights_7 = full_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_1_attn_bias = self.L__mod___transformer_h_1_attn_bias
    causal_mask_1 = l__mod___transformer_h_1_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 512, None), slice(None, 512, None))];  l__mod___transformer_h_1_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_3 = torch.full([], -3.4028234663852886e+38, dtype = torch.float32)
    mask_value_1 = full_3.to(device(type='cuda', index=0));  full_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    to_3 = attn_weights_8.to(torch.float32);  attn_weights_8 = None
    attn_weights_9 = torch.where(causal_mask_1, to_3, mask_value_1);  causal_mask_1 = to_3 = mask_value_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_10 = torch.nn.functional.softmax(attn_weights_9, dim = -1);  attn_weights_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:211, code: attn_weights = attn_weights.type(value.dtype)
    attn_weights_11 = attn_weights_10.type(torch.float32);  attn_weights_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_13 = self.L__mod___transformer_h_1_attn_attn_dropout(attn_weights_11);  attn_weights_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_6 = torch.matmul(attn_weights_13, value_3);  attn_weights_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_7 = attn_output_6.permute(0, 2, 1, 3);  attn_output_6 = None
    tensor_7 = permute_7.contiguous();  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    attn_output_7 = tensor_7.view((1, 512, 768));  tensor_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_1_attn_c_proj_bias = self.L__mod___transformer_h_1_attn_c_proj_bias
    view_20 = attn_output_7.view(-1, 768);  attn_output_7 = None
    l__mod___transformer_h_1_attn_c_proj_weight = self.L__mod___transformer_h_1_attn_c_proj_weight
    x_10 = torch.addmm(l__mod___transformer_h_1_attn_c_proj_bias, view_20, l__mod___transformer_h_1_attn_c_proj_weight);  l__mod___transformer_h_1_attn_c_proj_bias = view_20 = l__mod___transformer_h_1_attn_c_proj_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    attn_output_8 = x_10.view((1, 512, 768));  x_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    attn_output_10 = self.L__mod___transformer_h_1_attn_resid_dropout(attn_output_8);  attn_output_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    residual_3 = attn_output_10 + residual_2;  attn_output_10 = residual_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    hidden_states_10 = self.L__mod___transformer_h_1_ln_2(residual_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_1_mlp_c_fc_bias = self.L__mod___transformer_h_1_mlp_c_fc_bias
    view_22 = hidden_states_10.view(-1, 768);  hidden_states_10 = None
    l__mod___transformer_h_1_mlp_c_fc_weight = self.L__mod___transformer_h_1_mlp_c_fc_weight
    x_12 = torch.addmm(l__mod___transformer_h_1_mlp_c_fc_bias, view_22, l__mod___transformer_h_1_mlp_c_fc_weight);  l__mod___transformer_h_1_mlp_c_fc_bias = view_22 = l__mod___transformer_h_1_mlp_c_fc_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    hidden_states_11 = x_12.view((1, 512, 3072));  x_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_4 = 0.5 * hidden_states_11
    pow_2 = torch.pow(hidden_states_11, 3.0)
    mul_5 = 0.044715 * pow_2;  pow_2 = None
    add_6 = hidden_states_11 + mul_5;  hidden_states_11 = mul_5 = None
    mul_6 = 0.7978845608028654 * add_6;  add_6 = None
    tanh_1 = torch.tanh(mul_6);  mul_6 = None
    add_7 = 1.0 + tanh_1;  tanh_1 = None
    hidden_states_12 = mul_4 * add_7;  mul_4 = add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_1_mlp_c_proj_bias = self.L__mod___transformer_h_1_mlp_c_proj_bias
    view_24 = hidden_states_12.view(-1, 3072);  hidden_states_12 = None
    l__mod___transformer_h_1_mlp_c_proj_weight = self.L__mod___transformer_h_1_mlp_c_proj_weight
    x_14 = torch.addmm(l__mod___transformer_h_1_mlp_c_proj_bias, view_24, l__mod___transformer_h_1_mlp_c_proj_weight);  l__mod___transformer_h_1_mlp_c_proj_bias = view_24 = l__mod___transformer_h_1_mlp_c_proj_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    hidden_states_13 = x_14.view((1, 512, 768));  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_1 = self.L__mod___transformer_h_1_mlp_dropout(hidden_states_13);  hidden_states_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    residual_4 = residual_3 + feed_forward_hidden_states_1;  residual_3 = feed_forward_hidden_states_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_16 = self.L__mod___transformer_h_2_ln_1(residual_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_2_attn_c_attn_bias = self.L__mod___transformer_h_2_attn_c_attn_bias
    view_26 = hidden_states_16.view(-1, 768);  hidden_states_16 = None
    l__mod___transformer_h_2_attn_c_attn_weight = self.L__mod___transformer_h_2_attn_c_attn_weight
    x_16 = torch.addmm(l__mod___transformer_h_2_attn_c_attn_bias, view_26, l__mod___transformer_h_2_attn_c_attn_weight);  l__mod___transformer_h_2_attn_c_attn_bias = view_26 = l__mod___transformer_h_2_attn_c_attn_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    x_17 = x_16.view((1, 512, 2304));  x_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_2 = x_17.split(768, dim = 2);  x_17 = None
    query_4 = split_2[0]
    key_4 = split_2[1]
    value_4 = split_2[2];  split_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_8 = query_4.view((1, 512, 12, 64));  query_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_5 = tensor_8.permute(0, 2, 1, 3);  tensor_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_9 = key_4.view((1, 512, 12, 64));  key_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_5 = tensor_9.permute(0, 2, 1, 3);  tensor_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_10 = value_4.view((1, 512, 12, 64));  value_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_5 = tensor_10.permute(0, 2, 1, 3);  tensor_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_2 = key_5.transpose(-1, -2)
    attn_weights_14 = torch.matmul(query_5, transpose_2);  query_5 = transpose_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_4 = torch.full([], 8.0, dtype = torch.float32, device = device(type='cuda', index=0))
    attn_weights_15 = attn_weights_14 / full_4;  attn_weights_14 = full_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_2_attn_bias = self.L__mod___transformer_h_2_attn_bias
    causal_mask_2 = l__mod___transformer_h_2_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 512, None), slice(None, 512, None))];  l__mod___transformer_h_2_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_5 = torch.full([], -3.4028234663852886e+38, dtype = torch.float32)
    mask_value_2 = full_5.to(device(type='cuda', index=0));  full_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    to_5 = attn_weights_15.to(torch.float32);  attn_weights_15 = None
    attn_weights_16 = torch.where(causal_mask_2, to_5, mask_value_2);  causal_mask_2 = to_5 = mask_value_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_17 = torch.nn.functional.softmax(attn_weights_16, dim = -1);  attn_weights_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:211, code: attn_weights = attn_weights.type(value.dtype)
    attn_weights_18 = attn_weights_17.type(torch.float32);  attn_weights_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_20 = self.L__mod___transformer_h_2_attn_attn_dropout(attn_weights_18);  attn_weights_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_12 = torch.matmul(attn_weights_20, value_5);  attn_weights_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_11 = attn_output_12.permute(0, 2, 1, 3);  attn_output_12 = None
    tensor_11 = permute_11.contiguous();  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    attn_output_13 = tensor_11.view((1, 512, 768));  tensor_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_2_attn_c_proj_bias = self.L__mod___transformer_h_2_attn_c_proj_bias
    view_32 = attn_output_13.view(-1, 768);  attn_output_13 = None
    l__mod___transformer_h_2_attn_c_proj_weight = self.L__mod___transformer_h_2_attn_c_proj_weight
    x_18 = torch.addmm(l__mod___transformer_h_2_attn_c_proj_bias, view_32, l__mod___transformer_h_2_attn_c_proj_weight);  l__mod___transformer_h_2_attn_c_proj_bias = view_32 = l__mod___transformer_h_2_attn_c_proj_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    attn_output_14 = x_18.view((1, 512, 768));  x_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    attn_output_16 = self.L__mod___transformer_h_2_attn_resid_dropout(attn_output_14);  attn_output_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    residual_5 = attn_output_16 + residual_4;  attn_output_16 = residual_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    hidden_states_18 = self.L__mod___transformer_h_2_ln_2(residual_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_2_mlp_c_fc_bias = self.L__mod___transformer_h_2_mlp_c_fc_bias
    view_34 = hidden_states_18.view(-1, 768);  hidden_states_18 = None
    l__mod___transformer_h_2_mlp_c_fc_weight = self.L__mod___transformer_h_2_mlp_c_fc_weight
    x_20 = torch.addmm(l__mod___transformer_h_2_mlp_c_fc_bias, view_34, l__mod___transformer_h_2_mlp_c_fc_weight);  l__mod___transformer_h_2_mlp_c_fc_bias = view_34 = l__mod___transformer_h_2_mlp_c_fc_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    hidden_states_19 = x_20.view((1, 512, 3072));  x_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_8 = 0.5 * hidden_states_19
    pow_3 = torch.pow(hidden_states_19, 3.0)
    mul_9 = 0.044715 * pow_3;  pow_3 = None
    add_10 = hidden_states_19 + mul_9;  hidden_states_19 = mul_9 = None
    mul_10 = 0.7978845608028654 * add_10;  add_10 = None
    tanh_2 = torch.tanh(mul_10);  mul_10 = None
    add_11 = 1.0 + tanh_2;  tanh_2 = None
    hidden_states_20 = mul_8 * add_11;  mul_8 = add_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_2_mlp_c_proj_bias = self.L__mod___transformer_h_2_mlp_c_proj_bias
    view_36 = hidden_states_20.view(-1, 3072);  hidden_states_20 = None
    l__mod___transformer_h_2_mlp_c_proj_weight = self.L__mod___transformer_h_2_mlp_c_proj_weight
    x_22 = torch.addmm(l__mod___transformer_h_2_mlp_c_proj_bias, view_36, l__mod___transformer_h_2_mlp_c_proj_weight);  l__mod___transformer_h_2_mlp_c_proj_bias = view_36 = l__mod___transformer_h_2_mlp_c_proj_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    hidden_states_21 = x_22.view((1, 512, 768));  x_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_2 = self.L__mod___transformer_h_2_mlp_dropout(hidden_states_21);  hidden_states_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    residual_6 = residual_5 + feed_forward_hidden_states_2;  residual_5 = feed_forward_hidden_states_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_24 = self.L__mod___transformer_h_3_ln_1(residual_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_3_attn_c_attn_bias = self.L__mod___transformer_h_3_attn_c_attn_bias
    view_38 = hidden_states_24.view(-1, 768);  hidden_states_24 = None
    l__mod___transformer_h_3_attn_c_attn_weight = self.L__mod___transformer_h_3_attn_c_attn_weight
    x_24 = torch.addmm(l__mod___transformer_h_3_attn_c_attn_bias, view_38, l__mod___transformer_h_3_attn_c_attn_weight);  l__mod___transformer_h_3_attn_c_attn_bias = view_38 = l__mod___transformer_h_3_attn_c_attn_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    x_25 = x_24.view((1, 512, 2304));  x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_3 = x_25.split(768, dim = 2);  x_25 = None
    query_6 = split_3[0]
    key_6 = split_3[1]
    value_6 = split_3[2];  split_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_12 = query_6.view((1, 512, 12, 64));  query_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_7 = tensor_12.permute(0, 2, 1, 3);  tensor_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_13 = key_6.view((1, 512, 12, 64));  key_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_7 = tensor_13.permute(0, 2, 1, 3);  tensor_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_14 = value_6.view((1, 512, 12, 64));  value_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_7 = tensor_14.permute(0, 2, 1, 3);  tensor_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_3 = key_7.transpose(-1, -2)
    attn_weights_21 = torch.matmul(query_7, transpose_3);  query_7 = transpose_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_6 = torch.full([], 8.0, dtype = torch.float32, device = device(type='cuda', index=0))
    attn_weights_22 = attn_weights_21 / full_6;  attn_weights_21 = full_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_3_attn_bias = self.L__mod___transformer_h_3_attn_bias
    causal_mask_3 = l__mod___transformer_h_3_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 512, None), slice(None, 512, None))];  l__mod___transformer_h_3_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_7 = torch.full([], -3.4028234663852886e+38, dtype = torch.float32)
    mask_value_3 = full_7.to(device(type='cuda', index=0));  full_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    to_7 = attn_weights_22.to(torch.float32);  attn_weights_22 = None
    attn_weights_23 = torch.where(causal_mask_3, to_7, mask_value_3);  causal_mask_3 = to_7 = mask_value_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_24 = torch.nn.functional.softmax(attn_weights_23, dim = -1);  attn_weights_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:211, code: attn_weights = attn_weights.type(value.dtype)
    attn_weights_25 = attn_weights_24.type(torch.float32);  attn_weights_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_27 = self.L__mod___transformer_h_3_attn_attn_dropout(attn_weights_25);  attn_weights_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_18 = torch.matmul(attn_weights_27, value_7);  attn_weights_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_15 = attn_output_18.permute(0, 2, 1, 3);  attn_output_18 = None
    tensor_15 = permute_15.contiguous();  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    attn_output_19 = tensor_15.view((1, 512, 768));  tensor_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_3_attn_c_proj_bias = self.L__mod___transformer_h_3_attn_c_proj_bias
    view_44 = attn_output_19.view(-1, 768);  attn_output_19 = None
    l__mod___transformer_h_3_attn_c_proj_weight = self.L__mod___transformer_h_3_attn_c_proj_weight
    x_26 = torch.addmm(l__mod___transformer_h_3_attn_c_proj_bias, view_44, l__mod___transformer_h_3_attn_c_proj_weight);  l__mod___transformer_h_3_attn_c_proj_bias = view_44 = l__mod___transformer_h_3_attn_c_proj_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    attn_output_20 = x_26.view((1, 512, 768));  x_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    attn_output_22 = self.L__mod___transformer_h_3_attn_resid_dropout(attn_output_20);  attn_output_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    residual_7 = attn_output_22 + residual_6;  attn_output_22 = residual_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    hidden_states_26 = self.L__mod___transformer_h_3_ln_2(residual_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_3_mlp_c_fc_bias = self.L__mod___transformer_h_3_mlp_c_fc_bias
    view_46 = hidden_states_26.view(-1, 768);  hidden_states_26 = None
    l__mod___transformer_h_3_mlp_c_fc_weight = self.L__mod___transformer_h_3_mlp_c_fc_weight
    x_28 = torch.addmm(l__mod___transformer_h_3_mlp_c_fc_bias, view_46, l__mod___transformer_h_3_mlp_c_fc_weight);  l__mod___transformer_h_3_mlp_c_fc_bias = view_46 = l__mod___transformer_h_3_mlp_c_fc_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    hidden_states_27 = x_28.view((1, 512, 3072));  x_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_12 = 0.5 * hidden_states_27
    pow_4 = torch.pow(hidden_states_27, 3.0)
    mul_13 = 0.044715 * pow_4;  pow_4 = None
    add_14 = hidden_states_27 + mul_13;  hidden_states_27 = mul_13 = None
    mul_14 = 0.7978845608028654 * add_14;  add_14 = None
    tanh_3 = torch.tanh(mul_14);  mul_14 = None
    add_15 = 1.0 + tanh_3;  tanh_3 = None
    hidden_states_28 = mul_12 * add_15;  mul_12 = add_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_3_mlp_c_proj_bias = self.L__mod___transformer_h_3_mlp_c_proj_bias
    view_48 = hidden_states_28.view(-1, 3072);  hidden_states_28 = None
    l__mod___transformer_h_3_mlp_c_proj_weight = self.L__mod___transformer_h_3_mlp_c_proj_weight
    x_30 = torch.addmm(l__mod___transformer_h_3_mlp_c_proj_bias, view_48, l__mod___transformer_h_3_mlp_c_proj_weight);  l__mod___transformer_h_3_mlp_c_proj_bias = view_48 = l__mod___transformer_h_3_mlp_c_proj_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    hidden_states_29 = x_30.view((1, 512, 768));  x_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_3 = self.L__mod___transformer_h_3_mlp_dropout(hidden_states_29);  hidden_states_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    residual_8 = residual_7 + feed_forward_hidden_states_3;  residual_7 = feed_forward_hidden_states_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_32 = self.L__mod___transformer_h_4_ln_1(residual_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_4_attn_c_attn_bias = self.L__mod___transformer_h_4_attn_c_attn_bias
    view_50 = hidden_states_32.view(-1, 768);  hidden_states_32 = None
    l__mod___transformer_h_4_attn_c_attn_weight = self.L__mod___transformer_h_4_attn_c_attn_weight
    x_32 = torch.addmm(l__mod___transformer_h_4_attn_c_attn_bias, view_50, l__mod___transformer_h_4_attn_c_attn_weight);  l__mod___transformer_h_4_attn_c_attn_bias = view_50 = l__mod___transformer_h_4_attn_c_attn_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    x_33 = x_32.view((1, 512, 2304));  x_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_4 = x_33.split(768, dim = 2);  x_33 = None
    query_8 = split_4[0]
    key_8 = split_4[1]
    value_8 = split_4[2];  split_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_16 = query_8.view((1, 512, 12, 64));  query_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_9 = tensor_16.permute(0, 2, 1, 3);  tensor_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_17 = key_8.view((1, 512, 12, 64));  key_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_9 = tensor_17.permute(0, 2, 1, 3);  tensor_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_18 = value_8.view((1, 512, 12, 64));  value_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_9 = tensor_18.permute(0, 2, 1, 3);  tensor_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_4 = key_9.transpose(-1, -2)
    attn_weights_28 = torch.matmul(query_9, transpose_4);  query_9 = transpose_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_8 = torch.full([], 8.0, dtype = torch.float32, device = device(type='cuda', index=0))
    attn_weights_29 = attn_weights_28 / full_8;  attn_weights_28 = full_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_4_attn_bias = self.L__mod___transformer_h_4_attn_bias
    causal_mask_4 = l__mod___transformer_h_4_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 512, None), slice(None, 512, None))];  l__mod___transformer_h_4_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_9 = torch.full([], -3.4028234663852886e+38, dtype = torch.float32)
    mask_value_4 = full_9.to(device(type='cuda', index=0));  full_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    to_9 = attn_weights_29.to(torch.float32);  attn_weights_29 = None
    attn_weights_30 = torch.where(causal_mask_4, to_9, mask_value_4);  causal_mask_4 = to_9 = mask_value_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_31 = torch.nn.functional.softmax(attn_weights_30, dim = -1);  attn_weights_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:211, code: attn_weights = attn_weights.type(value.dtype)
    attn_weights_32 = attn_weights_31.type(torch.float32);  attn_weights_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_34 = self.L__mod___transformer_h_4_attn_attn_dropout(attn_weights_32);  attn_weights_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_24 = torch.matmul(attn_weights_34, value_9);  attn_weights_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_19 = attn_output_24.permute(0, 2, 1, 3);  attn_output_24 = None
    tensor_19 = permute_19.contiguous();  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    attn_output_25 = tensor_19.view((1, 512, 768));  tensor_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_4_attn_c_proj_bias = self.L__mod___transformer_h_4_attn_c_proj_bias
    view_56 = attn_output_25.view(-1, 768);  attn_output_25 = None
    l__mod___transformer_h_4_attn_c_proj_weight = self.L__mod___transformer_h_4_attn_c_proj_weight
    x_34 = torch.addmm(l__mod___transformer_h_4_attn_c_proj_bias, view_56, l__mod___transformer_h_4_attn_c_proj_weight);  l__mod___transformer_h_4_attn_c_proj_bias = view_56 = l__mod___transformer_h_4_attn_c_proj_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    attn_output_26 = x_34.view((1, 512, 768));  x_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    attn_output_28 = self.L__mod___transformer_h_4_attn_resid_dropout(attn_output_26);  attn_output_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    residual_9 = attn_output_28 + residual_8;  attn_output_28 = residual_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    hidden_states_34 = self.L__mod___transformer_h_4_ln_2(residual_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_4_mlp_c_fc_bias = self.L__mod___transformer_h_4_mlp_c_fc_bias
    view_58 = hidden_states_34.view(-1, 768);  hidden_states_34 = None
    l__mod___transformer_h_4_mlp_c_fc_weight = self.L__mod___transformer_h_4_mlp_c_fc_weight
    x_36 = torch.addmm(l__mod___transformer_h_4_mlp_c_fc_bias, view_58, l__mod___transformer_h_4_mlp_c_fc_weight);  l__mod___transformer_h_4_mlp_c_fc_bias = view_58 = l__mod___transformer_h_4_mlp_c_fc_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    hidden_states_35 = x_36.view((1, 512, 3072));  x_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_16 = 0.5 * hidden_states_35
    pow_5 = torch.pow(hidden_states_35, 3.0)
    mul_17 = 0.044715 * pow_5;  pow_5 = None
    add_18 = hidden_states_35 + mul_17;  hidden_states_35 = mul_17 = None
    mul_18 = 0.7978845608028654 * add_18;  add_18 = None
    tanh_4 = torch.tanh(mul_18);  mul_18 = None
    add_19 = 1.0 + tanh_4;  tanh_4 = None
    hidden_states_36 = mul_16 * add_19;  mul_16 = add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_4_mlp_c_proj_bias = self.L__mod___transformer_h_4_mlp_c_proj_bias
    view_60 = hidden_states_36.view(-1, 3072);  hidden_states_36 = None
    l__mod___transformer_h_4_mlp_c_proj_weight = self.L__mod___transformer_h_4_mlp_c_proj_weight
    x_38 = torch.addmm(l__mod___transformer_h_4_mlp_c_proj_bias, view_60, l__mod___transformer_h_4_mlp_c_proj_weight);  l__mod___transformer_h_4_mlp_c_proj_bias = view_60 = l__mod___transformer_h_4_mlp_c_proj_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    hidden_states_37 = x_38.view((1, 512, 768));  x_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_4 = self.L__mod___transformer_h_4_mlp_dropout(hidden_states_37);  hidden_states_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    residual_10 = residual_9 + feed_forward_hidden_states_4;  residual_9 = feed_forward_hidden_states_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_40 = self.L__mod___transformer_h_5_ln_1(residual_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_5_attn_c_attn_bias = self.L__mod___transformer_h_5_attn_c_attn_bias
    view_62 = hidden_states_40.view(-1, 768);  hidden_states_40 = None
    l__mod___transformer_h_5_attn_c_attn_weight = self.L__mod___transformer_h_5_attn_c_attn_weight
    x_40 = torch.addmm(l__mod___transformer_h_5_attn_c_attn_bias, view_62, l__mod___transformer_h_5_attn_c_attn_weight);  l__mod___transformer_h_5_attn_c_attn_bias = view_62 = l__mod___transformer_h_5_attn_c_attn_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    x_41 = x_40.view((1, 512, 2304));  x_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_5 = x_41.split(768, dim = 2);  x_41 = None
    query_10 = split_5[0]
    key_10 = split_5[1]
    value_10 = split_5[2];  split_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_20 = query_10.view((1, 512, 12, 64));  query_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_11 = tensor_20.permute(0, 2, 1, 3);  tensor_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_21 = key_10.view((1, 512, 12, 64));  key_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_11 = tensor_21.permute(0, 2, 1, 3);  tensor_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_22 = value_10.view((1, 512, 12, 64));  value_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_11 = tensor_22.permute(0, 2, 1, 3);  tensor_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_5 = key_11.transpose(-1, -2)
    attn_weights_35 = torch.matmul(query_11, transpose_5);  query_11 = transpose_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_10 = torch.full([], 8.0, dtype = torch.float32, device = device(type='cuda', index=0))
    attn_weights_36 = attn_weights_35 / full_10;  attn_weights_35 = full_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_5_attn_bias = self.L__mod___transformer_h_5_attn_bias
    causal_mask_5 = l__mod___transformer_h_5_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 512, None), slice(None, 512, None))];  l__mod___transformer_h_5_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_11 = torch.full([], -3.4028234663852886e+38, dtype = torch.float32)
    mask_value_5 = full_11.to(device(type='cuda', index=0));  full_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    to_11 = attn_weights_36.to(torch.float32);  attn_weights_36 = None
    attn_weights_37 = torch.where(causal_mask_5, to_11, mask_value_5);  causal_mask_5 = to_11 = mask_value_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_38 = torch.nn.functional.softmax(attn_weights_37, dim = -1);  attn_weights_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:211, code: attn_weights = attn_weights.type(value.dtype)
    attn_weights_39 = attn_weights_38.type(torch.float32);  attn_weights_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_41 = self.L__mod___transformer_h_5_attn_attn_dropout(attn_weights_39);  attn_weights_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_30 = torch.matmul(attn_weights_41, value_11);  attn_weights_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_23 = attn_output_30.permute(0, 2, 1, 3);  attn_output_30 = None
    tensor_23 = permute_23.contiguous();  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    attn_output_31 = tensor_23.view((1, 512, 768));  tensor_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_5_attn_c_proj_bias = self.L__mod___transformer_h_5_attn_c_proj_bias
    view_68 = attn_output_31.view(-1, 768);  attn_output_31 = None
    l__mod___transformer_h_5_attn_c_proj_weight = self.L__mod___transformer_h_5_attn_c_proj_weight
    x_42 = torch.addmm(l__mod___transformer_h_5_attn_c_proj_bias, view_68, l__mod___transformer_h_5_attn_c_proj_weight);  l__mod___transformer_h_5_attn_c_proj_bias = view_68 = l__mod___transformer_h_5_attn_c_proj_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    attn_output_32 = x_42.view((1, 512, 768));  x_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    attn_output_34 = self.L__mod___transformer_h_5_attn_resid_dropout(attn_output_32);  attn_output_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    residual_11 = attn_output_34 + residual_10;  attn_output_34 = residual_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    hidden_states_42 = self.L__mod___transformer_h_5_ln_2(residual_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_5_mlp_c_fc_bias = self.L__mod___transformer_h_5_mlp_c_fc_bias
    view_70 = hidden_states_42.view(-1, 768);  hidden_states_42 = None
    l__mod___transformer_h_5_mlp_c_fc_weight = self.L__mod___transformer_h_5_mlp_c_fc_weight
    x_44 = torch.addmm(l__mod___transformer_h_5_mlp_c_fc_bias, view_70, l__mod___transformer_h_5_mlp_c_fc_weight);  l__mod___transformer_h_5_mlp_c_fc_bias = view_70 = l__mod___transformer_h_5_mlp_c_fc_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    hidden_states_43 = x_44.view((1, 512, 3072));  x_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_20 = 0.5 * hidden_states_43
    pow_6 = torch.pow(hidden_states_43, 3.0)
    mul_21 = 0.044715 * pow_6;  pow_6 = None
    add_22 = hidden_states_43 + mul_21;  hidden_states_43 = mul_21 = None
    mul_22 = 0.7978845608028654 * add_22;  add_22 = None
    tanh_5 = torch.tanh(mul_22);  mul_22 = None
    add_23 = 1.0 + tanh_5;  tanh_5 = None
    hidden_states_44 = mul_20 * add_23;  mul_20 = add_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_5_mlp_c_proj_bias = self.L__mod___transformer_h_5_mlp_c_proj_bias
    view_72 = hidden_states_44.view(-1, 3072);  hidden_states_44 = None
    l__mod___transformer_h_5_mlp_c_proj_weight = self.L__mod___transformer_h_5_mlp_c_proj_weight
    x_46 = torch.addmm(l__mod___transformer_h_5_mlp_c_proj_bias, view_72, l__mod___transformer_h_5_mlp_c_proj_weight);  l__mod___transformer_h_5_mlp_c_proj_bias = view_72 = l__mod___transformer_h_5_mlp_c_proj_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    hidden_states_45 = x_46.view((1, 512, 768));  x_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_5 = self.L__mod___transformer_h_5_mlp_dropout(hidden_states_45);  hidden_states_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    hidden_states_47 = residual_11 + feed_forward_hidden_states_5;  residual_11 = feed_forward_hidden_states_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:926, code: hidden_states = self.ln_f(hidden_states)
    l__mod___transformer_ln_f = self.L__mod___transformer_ln_f(hidden_states_47);  hidden_states_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:928, code: hidden_states = hidden_states.view(output_shape)
    hidden_states_48 = l__mod___transformer_ln_f.view((-1, 512, 768));  l__mod___transformer_ln_f = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1098, code: lm_logits = self.lm_head(hidden_states)
    lm_logits = self.L__mod___lm_head(hidden_states_48);  hidden_states_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1103, code: labels = labels.to(lm_logits.device)
    labels = l_inputs_labels_.to(device(type='cuda', index=0));  l_inputs_labels_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1105, code: shift_logits = lm_logits[..., :-1, :].contiguous()
    getitem_24 = lm_logits[(Ellipsis, slice(None, -1, None), slice(None, None, None))]
    shift_logits = getitem_24.contiguous();  getitem_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1106, code: shift_labels = labels[..., 1:].contiguous()
    getitem_25 = labels[(Ellipsis, slice(1, None, None))];  labels = None
    shift_labels = getitem_25.contiguous();  getitem_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1109, code: loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    view_75 = shift_logits.view(-1, 50257);  shift_logits = None
    view_76 = shift_labels.view(-1);  shift_labels = None
    loss = torch.nn.functional.cross_entropy(view_75, view_76, None, None, -100, None, 'mean', 0.0);  view_75 = view_76 = None
    return (loss, lm_logits, key_1, value_1, key_3, value_3, key_5, value_5, key_7, value_7, key_9, value_9, key_11, value_11)
    