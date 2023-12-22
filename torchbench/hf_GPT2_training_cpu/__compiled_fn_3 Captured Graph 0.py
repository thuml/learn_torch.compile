from __future__ import annotations



def forward(self, L_cloned_inputs_0_ : torch.Tensor):
    l_cloned_inputs_0_ = L_cloned_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:781, code: input_ids = input_ids.view(-1, input_shape[-1])
    input_ids = l_cloned_inputs_0_.view(-1, 512);  l_cloned_inputs_0_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:802, code: position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
    position_ids = torch.arange(0, 512, dtype = torch.int64, device = device(type='cpu'))
    
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
    x_1 = x.view((2, 512, 2304));  x = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split = x_1.split(768, dim = 2);  x_1 = None
    query = split[0]
    key = split[1]
    value = split[2];  split = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor = query.view((2, 512, 12, 64));  query = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_1 = tensor.permute(0, 2, 1, 3);  tensor = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_1 = key.view((2, 512, 12, 64));  key = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_1 = tensor_1.permute(0, 2, 1, 3);  tensor_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_2 = value.view((2, 512, 12, 64));  value = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_1 = tensor_2.permute(0, 2, 1, 3);  tensor_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose = key_1.transpose(-1, -2)
    attn_weights = torch.matmul(query_1, transpose);  query_1 = transpose = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full = torch.full([], 8.0, dtype = torch.float32, device = device(type='cpu'))
    attn_weights_1 = attn_weights / full;  attn_weights = full = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_0_attn_bias = self.L__mod___transformer_h_0_attn_bias
    causal_mask = l__mod___transformer_h_0_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 512, None), slice(None, 512, None))];  l__mod___transformer_h_0_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_1 = torch.full([], -3.4028234663852886e+38, dtype = torch.float32)
    mask_value = full_1.to(device(type='cpu'));  full_1 = None
    
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
    attn_output_1 = tensor_3.view((2, 512, 768));  tensor_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_0_attn_c_proj_bias = self.L__mod___transformer_h_0_attn_c_proj_bias
    view_8 = attn_output_1.view(-1, 768);  attn_output_1 = None
    l__mod___transformer_h_0_attn_c_proj_weight = self.L__mod___transformer_h_0_attn_c_proj_weight
    x_2 = torch.addmm(l__mod___transformer_h_0_attn_c_proj_bias, view_8, l__mod___transformer_h_0_attn_c_proj_weight);  l__mod___transformer_h_0_attn_c_proj_bias = view_8 = l__mod___transformer_h_0_attn_c_proj_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    attn_output_2 = x_2.view((2, 512, 768));  x_2 = None
    
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
    hidden_states_3 = x_4.view((2, 512, 3072));  x_4 = None
    
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
    hidden_states_5 = x_6.view((2, 512, 768));  x_6 = None
    
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
    x_9 = x_8.view((2, 512, 2304));  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_1 = x_9.split(768, dim = 2);  x_9 = None
    query_2 = split_1[0]
    key_2 = split_1[1]
    value_2 = split_1[2];  split_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_4 = query_2.view((2, 512, 12, 64));  query_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_3 = tensor_4.permute(0, 2, 1, 3);  tensor_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_5 = key_2.view((2, 512, 12, 64));  key_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_3 = tensor_5.permute(0, 2, 1, 3);  tensor_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_6 = value_2.view((2, 512, 12, 64));  value_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_3 = tensor_6.permute(0, 2, 1, 3);  tensor_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_1 = key_3.transpose(-1, -2)
    attn_weights_7 = torch.matmul(query_3, transpose_1);  query_3 = transpose_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_2 = torch.full([], 8.0, dtype = torch.float32, device = device(type='cpu'))
    attn_weights_8 = attn_weights_7 / full_2;  attn_weights_7 = full_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_1_attn_bias = self.L__mod___transformer_h_1_attn_bias
    causal_mask_1 = l__mod___transformer_h_1_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 512, None), slice(None, 512, None))];  l__mod___transformer_h_1_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_3 = torch.full([], -3.4028234663852886e+38, dtype = torch.float32)
    mask_value_1 = full_3.to(device(type='cpu'));  full_3 = None
    
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
    attn_output_7 = tensor_7.view((2, 512, 768));  tensor_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_1_attn_c_proj_bias = self.L__mod___transformer_h_1_attn_c_proj_bias
    view_20 = attn_output_7.view(-1, 768);  attn_output_7 = None
    l__mod___transformer_h_1_attn_c_proj_weight = self.L__mod___transformer_h_1_attn_c_proj_weight
    x_10 = torch.addmm(l__mod___transformer_h_1_attn_c_proj_bias, view_20, l__mod___transformer_h_1_attn_c_proj_weight);  l__mod___transformer_h_1_attn_c_proj_bias = view_20 = l__mod___transformer_h_1_attn_c_proj_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    attn_output_8 = x_10.view((2, 512, 768));  x_10 = None
    
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
    hidden_states_11 = x_12.view((2, 512, 3072));  x_12 = None
    
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
    hidden_states_13 = x_14.view((2, 512, 768));  x_14 = None
    
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
    x_17 = x_16.view((2, 512, 2304));  x_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_2 = x_17.split(768, dim = 2);  x_17 = None
    query_4 = split_2[0]
    key_4 = split_2[1]
    value_4 = split_2[2];  split_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_8 = query_4.view((2, 512, 12, 64));  query_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_5 = tensor_8.permute(0, 2, 1, 3);  tensor_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_9 = key_4.view((2, 512, 12, 64));  key_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_5 = tensor_9.permute(0, 2, 1, 3);  tensor_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_10 = value_4.view((2, 512, 12, 64));  value_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_5 = tensor_10.permute(0, 2, 1, 3);  tensor_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_2 = key_5.transpose(-1, -2)
    attn_weights_14 = torch.matmul(query_5, transpose_2);  query_5 = transpose_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_4 = torch.full([], 8.0, dtype = torch.float32, device = device(type='cpu'))
    attn_weights_15 = attn_weights_14 / full_4;  attn_weights_14 = full_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_2_attn_bias = self.L__mod___transformer_h_2_attn_bias
    causal_mask_2 = l__mod___transformer_h_2_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 512, None), slice(None, 512, None))];  l__mod___transformer_h_2_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_5 = torch.full([], -3.4028234663852886e+38, dtype = torch.float32)
    mask_value_2 = full_5.to(device(type='cpu'));  full_5 = None
    
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
    attn_output_13 = tensor_11.view((2, 512, 768));  tensor_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_2_attn_c_proj_bias = self.L__mod___transformer_h_2_attn_c_proj_bias
    view_32 = attn_output_13.view(-1, 768);  attn_output_13 = None
    l__mod___transformer_h_2_attn_c_proj_weight = self.L__mod___transformer_h_2_attn_c_proj_weight
    x_18 = torch.addmm(l__mod___transformer_h_2_attn_c_proj_bias, view_32, l__mod___transformer_h_2_attn_c_proj_weight);  l__mod___transformer_h_2_attn_c_proj_bias = view_32 = l__mod___transformer_h_2_attn_c_proj_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    attn_output_14 = x_18.view((2, 512, 768));  x_18 = None
    
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
    hidden_states_19 = x_20.view((2, 512, 3072));  x_20 = None
    
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
    hidden_states_21 = x_22.view((2, 512, 768));  x_22 = None
    
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
    x_25 = x_24.view((2, 512, 2304));  x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_3 = x_25.split(768, dim = 2);  x_25 = None
    query_6 = split_3[0]
    key_6 = split_3[1]
    value_6 = split_3[2];  split_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_12 = query_6.view((2, 512, 12, 64));  query_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_7 = tensor_12.permute(0, 2, 1, 3);  tensor_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_13 = key_6.view((2, 512, 12, 64));  key_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_7 = tensor_13.permute(0, 2, 1, 3);  tensor_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_14 = value_6.view((2, 512, 12, 64));  value_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_7 = tensor_14.permute(0, 2, 1, 3);  tensor_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_3 = key_7.transpose(-1, -2)
    attn_weights_21 = torch.matmul(query_7, transpose_3);  query_7 = transpose_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_6 = torch.full([], 8.0, dtype = torch.float32, device = device(type='cpu'))
    attn_weights_22 = attn_weights_21 / full_6;  attn_weights_21 = full_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_3_attn_bias = self.L__mod___transformer_h_3_attn_bias
    causal_mask_3 = l__mod___transformer_h_3_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 512, None), slice(None, 512, None))];  l__mod___transformer_h_3_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_7 = torch.full([], -3.4028234663852886e+38, dtype = torch.float32)
    mask_value_3 = full_7.to(device(type='cpu'));  full_7 = None
    
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
    attn_output_19 = tensor_15.view((2, 512, 768));  tensor_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_3_attn_c_proj_bias = self.L__mod___transformer_h_3_attn_c_proj_bias
    view_44 = attn_output_19.view(-1, 768);  attn_output_19 = None
    l__mod___transformer_h_3_attn_c_proj_weight = self.L__mod___transformer_h_3_attn_c_proj_weight
    x_26 = torch.addmm(l__mod___transformer_h_3_attn_c_proj_bias, view_44, l__mod___transformer_h_3_attn_c_proj_weight);  l__mod___transformer_h_3_attn_c_proj_bias = view_44 = l__mod___transformer_h_3_attn_c_proj_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    attn_output_20 = x_26.view((2, 512, 768));  x_26 = None
    
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
    hidden_states_27 = x_28.view((2, 512, 3072));  x_28 = None
    
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
    hidden_states_29 = x_30.view((2, 512, 768));  x_30 = None
    
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
    x_33 = x_32.view((2, 512, 2304));  x_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_4 = x_33.split(768, dim = 2);  x_33 = None
    query_8 = split_4[0]
    key_8 = split_4[1]
    value_8 = split_4[2];  split_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_16 = query_8.view((2, 512, 12, 64));  query_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_9 = tensor_16.permute(0, 2, 1, 3);  tensor_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_17 = key_8.view((2, 512, 12, 64));  key_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_9 = tensor_17.permute(0, 2, 1, 3);  tensor_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_18 = value_8.view((2, 512, 12, 64));  value_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_9 = tensor_18.permute(0, 2, 1, 3);  tensor_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_4 = key_9.transpose(-1, -2)
    attn_weights_28 = torch.matmul(query_9, transpose_4);  query_9 = transpose_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_8 = torch.full([], 8.0, dtype = torch.float32, device = device(type='cpu'))
    attn_weights_29 = attn_weights_28 / full_8;  attn_weights_28 = full_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_4_attn_bias = self.L__mod___transformer_h_4_attn_bias
    causal_mask_4 = l__mod___transformer_h_4_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 512, None), slice(None, 512, None))];  l__mod___transformer_h_4_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_9 = torch.full([], -3.4028234663852886e+38, dtype = torch.float32)
    mask_value_4 = full_9.to(device(type='cpu'));  full_9 = None
    
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
    attn_output_25 = tensor_19.view((2, 512, 768));  tensor_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_4_attn_c_proj_bias = self.L__mod___transformer_h_4_attn_c_proj_bias
    view_56 = attn_output_25.view(-1, 768);  attn_output_25 = None
    l__mod___transformer_h_4_attn_c_proj_weight = self.L__mod___transformer_h_4_attn_c_proj_weight
    x_34 = torch.addmm(l__mod___transformer_h_4_attn_c_proj_bias, view_56, l__mod___transformer_h_4_attn_c_proj_weight);  l__mod___transformer_h_4_attn_c_proj_bias = view_56 = l__mod___transformer_h_4_attn_c_proj_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    attn_output_26 = x_34.view((2, 512, 768));  x_34 = None
    
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
    hidden_states_35 = x_36.view((2, 512, 3072));  x_36 = None
    
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
    hidden_states_37 = x_38.view((2, 512, 768));  x_38 = None
    
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
    x_41 = x_40.view((2, 512, 2304));  x_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_5 = x_41.split(768, dim = 2);  x_41 = None
    query_10 = split_5[0]
    key_10 = split_5[1]
    value_10 = split_5[2];  split_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_20 = query_10.view((2, 512, 12, 64));  query_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_11 = tensor_20.permute(0, 2, 1, 3);  tensor_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_21 = key_10.view((2, 512, 12, 64));  key_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_11 = tensor_21.permute(0, 2, 1, 3);  tensor_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_22 = value_10.view((2, 512, 12, 64));  value_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_11 = tensor_22.permute(0, 2, 1, 3);  tensor_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_5 = key_11.transpose(-1, -2)
    attn_weights_35 = torch.matmul(query_11, transpose_5);  query_11 = transpose_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_10 = torch.full([], 8.0, dtype = torch.float32, device = device(type='cpu'))
    attn_weights_36 = attn_weights_35 / full_10;  attn_weights_35 = full_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_5_attn_bias = self.L__mod___transformer_h_5_attn_bias
    causal_mask_5 = l__mod___transformer_h_5_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 512, None), slice(None, 512, None))];  l__mod___transformer_h_5_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_11 = torch.full([], -3.4028234663852886e+38, dtype = torch.float32)
    mask_value_5 = full_11.to(device(type='cpu'));  full_11 = None
    
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
    attn_output_31 = tensor_23.view((2, 512, 768));  tensor_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_5_attn_c_proj_bias = self.L__mod___transformer_h_5_attn_c_proj_bias
    view_68 = attn_output_31.view(-1, 768);  attn_output_31 = None
    l__mod___transformer_h_5_attn_c_proj_weight = self.L__mod___transformer_h_5_attn_c_proj_weight
    x_42 = torch.addmm(l__mod___transformer_h_5_attn_c_proj_bias, view_68, l__mod___transformer_h_5_attn_c_proj_weight);  l__mod___transformer_h_5_attn_c_proj_bias = view_68 = l__mod___transformer_h_5_attn_c_proj_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    attn_output_32 = x_42.view((2, 512, 768));  x_42 = None
    
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
    hidden_states_43 = x_44.view((2, 512, 3072));  x_44 = None
    
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
    hidden_states_45 = x_46.view((2, 512, 768));  x_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_5 = self.L__mod___transformer_h_5_mlp_dropout(hidden_states_45);  hidden_states_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    residual_12 = residual_11 + feed_forward_hidden_states_5;  residual_11 = feed_forward_hidden_states_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_48 = self.L__mod___transformer_h_6_ln_1(residual_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_6_attn_c_attn_bias = self.L__mod___transformer_h_6_attn_c_attn_bias
    view_74 = hidden_states_48.view(-1, 768);  hidden_states_48 = None
    l__mod___transformer_h_6_attn_c_attn_weight = self.L__mod___transformer_h_6_attn_c_attn_weight
    x_48 = torch.addmm(l__mod___transformer_h_6_attn_c_attn_bias, view_74, l__mod___transformer_h_6_attn_c_attn_weight);  l__mod___transformer_h_6_attn_c_attn_bias = view_74 = l__mod___transformer_h_6_attn_c_attn_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    x_49 = x_48.view((2, 512, 2304));  x_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_6 = x_49.split(768, dim = 2);  x_49 = None
    query_12 = split_6[0]
    key_12 = split_6[1]
    value_12 = split_6[2];  split_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_24 = query_12.view((2, 512, 12, 64));  query_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_13 = tensor_24.permute(0, 2, 1, 3);  tensor_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_25 = key_12.view((2, 512, 12, 64));  key_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_13 = tensor_25.permute(0, 2, 1, 3);  tensor_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_26 = value_12.view((2, 512, 12, 64));  value_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_13 = tensor_26.permute(0, 2, 1, 3);  tensor_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_6 = key_13.transpose(-1, -2)
    attn_weights_42 = torch.matmul(query_13, transpose_6);  query_13 = transpose_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_12 = torch.full([], 8.0, dtype = torch.float32, device = device(type='cpu'))
    attn_weights_43 = attn_weights_42 / full_12;  attn_weights_42 = full_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_6_attn_bias = self.L__mod___transformer_h_6_attn_bias
    causal_mask_6 = l__mod___transformer_h_6_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 512, None), slice(None, 512, None))];  l__mod___transformer_h_6_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_13 = torch.full([], -3.4028234663852886e+38, dtype = torch.float32)
    mask_value_6 = full_13.to(device(type='cpu'));  full_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    to_13 = attn_weights_43.to(torch.float32);  attn_weights_43 = None
    attn_weights_44 = torch.where(causal_mask_6, to_13, mask_value_6);  causal_mask_6 = to_13 = mask_value_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_45 = torch.nn.functional.softmax(attn_weights_44, dim = -1);  attn_weights_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:211, code: attn_weights = attn_weights.type(value.dtype)
    attn_weights_46 = attn_weights_45.type(torch.float32);  attn_weights_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_48 = self.L__mod___transformer_h_6_attn_attn_dropout(attn_weights_46);  attn_weights_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_36 = torch.matmul(attn_weights_48, value_13);  attn_weights_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_27 = attn_output_36.permute(0, 2, 1, 3);  attn_output_36 = None
    tensor_27 = permute_27.contiguous();  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    attn_output_37 = tensor_27.view((2, 512, 768));  tensor_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_6_attn_c_proj_bias = self.L__mod___transformer_h_6_attn_c_proj_bias
    view_80 = attn_output_37.view(-1, 768);  attn_output_37 = None
    l__mod___transformer_h_6_attn_c_proj_weight = self.L__mod___transformer_h_6_attn_c_proj_weight
    x_50 = torch.addmm(l__mod___transformer_h_6_attn_c_proj_bias, view_80, l__mod___transformer_h_6_attn_c_proj_weight);  l__mod___transformer_h_6_attn_c_proj_bias = view_80 = l__mod___transformer_h_6_attn_c_proj_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    attn_output_38 = x_50.view((2, 512, 768));  x_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    attn_output_40 = self.L__mod___transformer_h_6_attn_resid_dropout(attn_output_38);  attn_output_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    residual_13 = attn_output_40 + residual_12;  attn_output_40 = residual_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    hidden_states_50 = self.L__mod___transformer_h_6_ln_2(residual_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_6_mlp_c_fc_bias = self.L__mod___transformer_h_6_mlp_c_fc_bias
    view_82 = hidden_states_50.view(-1, 768);  hidden_states_50 = None
    l__mod___transformer_h_6_mlp_c_fc_weight = self.L__mod___transformer_h_6_mlp_c_fc_weight
    x_52 = torch.addmm(l__mod___transformer_h_6_mlp_c_fc_bias, view_82, l__mod___transformer_h_6_mlp_c_fc_weight);  l__mod___transformer_h_6_mlp_c_fc_bias = view_82 = l__mod___transformer_h_6_mlp_c_fc_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    hidden_states_51 = x_52.view((2, 512, 3072));  x_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_24 = 0.5 * hidden_states_51
    pow_7 = torch.pow(hidden_states_51, 3.0)
    mul_25 = 0.044715 * pow_7;  pow_7 = None
    add_26 = hidden_states_51 + mul_25;  hidden_states_51 = mul_25 = None
    mul_26 = 0.7978845608028654 * add_26;  add_26 = None
    tanh_6 = torch.tanh(mul_26);  mul_26 = None
    add_27 = 1.0 + tanh_6;  tanh_6 = None
    hidden_states_52 = mul_24 * add_27;  mul_24 = add_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_6_mlp_c_proj_bias = self.L__mod___transformer_h_6_mlp_c_proj_bias
    view_84 = hidden_states_52.view(-1, 3072);  hidden_states_52 = None
    l__mod___transformer_h_6_mlp_c_proj_weight = self.L__mod___transformer_h_6_mlp_c_proj_weight
    x_54 = torch.addmm(l__mod___transformer_h_6_mlp_c_proj_bias, view_84, l__mod___transformer_h_6_mlp_c_proj_weight);  l__mod___transformer_h_6_mlp_c_proj_bias = view_84 = l__mod___transformer_h_6_mlp_c_proj_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    hidden_states_53 = x_54.view((2, 512, 768));  x_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_6 = self.L__mod___transformer_h_6_mlp_dropout(hidden_states_53);  hidden_states_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    residual_14 = residual_13 + feed_forward_hidden_states_6;  residual_13 = feed_forward_hidden_states_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_56 = self.L__mod___transformer_h_7_ln_1(residual_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_7_attn_c_attn_bias = self.L__mod___transformer_h_7_attn_c_attn_bias
    view_86 = hidden_states_56.view(-1, 768);  hidden_states_56 = None
    l__mod___transformer_h_7_attn_c_attn_weight = self.L__mod___transformer_h_7_attn_c_attn_weight
    x_56 = torch.addmm(l__mod___transformer_h_7_attn_c_attn_bias, view_86, l__mod___transformer_h_7_attn_c_attn_weight);  l__mod___transformer_h_7_attn_c_attn_bias = view_86 = l__mod___transformer_h_7_attn_c_attn_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    x_57 = x_56.view((2, 512, 2304));  x_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_7 = x_57.split(768, dim = 2);  x_57 = None
    query_14 = split_7[0]
    key_14 = split_7[1]
    value_14 = split_7[2];  split_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_28 = query_14.view((2, 512, 12, 64));  query_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_15 = tensor_28.permute(0, 2, 1, 3);  tensor_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_29 = key_14.view((2, 512, 12, 64));  key_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_15 = tensor_29.permute(0, 2, 1, 3);  tensor_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_30 = value_14.view((2, 512, 12, 64));  value_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_15 = tensor_30.permute(0, 2, 1, 3);  tensor_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_7 = key_15.transpose(-1, -2)
    attn_weights_49 = torch.matmul(query_15, transpose_7);  query_15 = transpose_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_14 = torch.full([], 8.0, dtype = torch.float32, device = device(type='cpu'))
    attn_weights_50 = attn_weights_49 / full_14;  attn_weights_49 = full_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_7_attn_bias = self.L__mod___transformer_h_7_attn_bias
    causal_mask_7 = l__mod___transformer_h_7_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 512, None), slice(None, 512, None))];  l__mod___transformer_h_7_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_15 = torch.full([], -3.4028234663852886e+38, dtype = torch.float32)
    mask_value_7 = full_15.to(device(type='cpu'));  full_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    to_15 = attn_weights_50.to(torch.float32);  attn_weights_50 = None
    attn_weights_51 = torch.where(causal_mask_7, to_15, mask_value_7);  causal_mask_7 = to_15 = mask_value_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_52 = torch.nn.functional.softmax(attn_weights_51, dim = -1);  attn_weights_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:211, code: attn_weights = attn_weights.type(value.dtype)
    attn_weights_53 = attn_weights_52.type(torch.float32);  attn_weights_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_55 = self.L__mod___transformer_h_7_attn_attn_dropout(attn_weights_53);  attn_weights_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_42 = torch.matmul(attn_weights_55, value_15);  attn_weights_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_31 = attn_output_42.permute(0, 2, 1, 3);  attn_output_42 = None
    tensor_31 = permute_31.contiguous();  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    attn_output_43 = tensor_31.view((2, 512, 768));  tensor_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_7_attn_c_proj_bias = self.L__mod___transformer_h_7_attn_c_proj_bias
    view_92 = attn_output_43.view(-1, 768);  attn_output_43 = None
    l__mod___transformer_h_7_attn_c_proj_weight = self.L__mod___transformer_h_7_attn_c_proj_weight
    x_58 = torch.addmm(l__mod___transformer_h_7_attn_c_proj_bias, view_92, l__mod___transformer_h_7_attn_c_proj_weight);  l__mod___transformer_h_7_attn_c_proj_bias = view_92 = l__mod___transformer_h_7_attn_c_proj_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    attn_output_44 = x_58.view((2, 512, 768));  x_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    attn_output_46 = self.L__mod___transformer_h_7_attn_resid_dropout(attn_output_44);  attn_output_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    residual_15 = attn_output_46 + residual_14;  attn_output_46 = residual_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    hidden_states_58 = self.L__mod___transformer_h_7_ln_2(residual_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_7_mlp_c_fc_bias = self.L__mod___transformer_h_7_mlp_c_fc_bias
    view_94 = hidden_states_58.view(-1, 768);  hidden_states_58 = None
    l__mod___transformer_h_7_mlp_c_fc_weight = self.L__mod___transformer_h_7_mlp_c_fc_weight
    x_60 = torch.addmm(l__mod___transformer_h_7_mlp_c_fc_bias, view_94, l__mod___transformer_h_7_mlp_c_fc_weight);  l__mod___transformer_h_7_mlp_c_fc_bias = view_94 = l__mod___transformer_h_7_mlp_c_fc_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    hidden_states_59 = x_60.view((2, 512, 3072));  x_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_28 = 0.5 * hidden_states_59
    pow_8 = torch.pow(hidden_states_59, 3.0)
    mul_29 = 0.044715 * pow_8;  pow_8 = None
    add_30 = hidden_states_59 + mul_29;  hidden_states_59 = mul_29 = None
    mul_30 = 0.7978845608028654 * add_30;  add_30 = None
    tanh_7 = torch.tanh(mul_30);  mul_30 = None
    add_31 = 1.0 + tanh_7;  tanh_7 = None
    hidden_states_60 = mul_28 * add_31;  mul_28 = add_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_7_mlp_c_proj_bias = self.L__mod___transformer_h_7_mlp_c_proj_bias
    view_96 = hidden_states_60.view(-1, 3072);  hidden_states_60 = None
    l__mod___transformer_h_7_mlp_c_proj_weight = self.L__mod___transformer_h_7_mlp_c_proj_weight
    x_62 = torch.addmm(l__mod___transformer_h_7_mlp_c_proj_bias, view_96, l__mod___transformer_h_7_mlp_c_proj_weight);  l__mod___transformer_h_7_mlp_c_proj_bias = view_96 = l__mod___transformer_h_7_mlp_c_proj_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    hidden_states_61 = x_62.view((2, 512, 768));  x_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_7 = self.L__mod___transformer_h_7_mlp_dropout(hidden_states_61);  hidden_states_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    residual_16 = residual_15 + feed_forward_hidden_states_7;  residual_15 = feed_forward_hidden_states_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_64 = self.L__mod___transformer_h_8_ln_1(residual_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_8_attn_c_attn_bias = self.L__mod___transformer_h_8_attn_c_attn_bias
    view_98 = hidden_states_64.view(-1, 768);  hidden_states_64 = None
    l__mod___transformer_h_8_attn_c_attn_weight = self.L__mod___transformer_h_8_attn_c_attn_weight
    x_64 = torch.addmm(l__mod___transformer_h_8_attn_c_attn_bias, view_98, l__mod___transformer_h_8_attn_c_attn_weight);  l__mod___transformer_h_8_attn_c_attn_bias = view_98 = l__mod___transformer_h_8_attn_c_attn_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    x_65 = x_64.view((2, 512, 2304));  x_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_8 = x_65.split(768, dim = 2);  x_65 = None
    query_16 = split_8[0]
    key_16 = split_8[1]
    value_16 = split_8[2];  split_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_32 = query_16.view((2, 512, 12, 64));  query_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_17 = tensor_32.permute(0, 2, 1, 3);  tensor_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_33 = key_16.view((2, 512, 12, 64));  key_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_17 = tensor_33.permute(0, 2, 1, 3);  tensor_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_34 = value_16.view((2, 512, 12, 64));  value_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_17 = tensor_34.permute(0, 2, 1, 3);  tensor_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_8 = key_17.transpose(-1, -2)
    attn_weights_56 = torch.matmul(query_17, transpose_8);  query_17 = transpose_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_16 = torch.full([], 8.0, dtype = torch.float32, device = device(type='cpu'))
    attn_weights_57 = attn_weights_56 / full_16;  attn_weights_56 = full_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_8_attn_bias = self.L__mod___transformer_h_8_attn_bias
    causal_mask_8 = l__mod___transformer_h_8_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 512, None), slice(None, 512, None))];  l__mod___transformer_h_8_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_17 = torch.full([], -3.4028234663852886e+38, dtype = torch.float32)
    mask_value_8 = full_17.to(device(type='cpu'));  full_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    to_17 = attn_weights_57.to(torch.float32);  attn_weights_57 = None
    attn_weights_58 = torch.where(causal_mask_8, to_17, mask_value_8);  causal_mask_8 = to_17 = mask_value_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_59 = torch.nn.functional.softmax(attn_weights_58, dim = -1);  attn_weights_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:211, code: attn_weights = attn_weights.type(value.dtype)
    attn_weights_60 = attn_weights_59.type(torch.float32);  attn_weights_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_62 = self.L__mod___transformer_h_8_attn_attn_dropout(attn_weights_60);  attn_weights_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_48 = torch.matmul(attn_weights_62, value_17);  attn_weights_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_35 = attn_output_48.permute(0, 2, 1, 3);  attn_output_48 = None
    tensor_35 = permute_35.contiguous();  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    attn_output_49 = tensor_35.view((2, 512, 768));  tensor_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_8_attn_c_proj_bias = self.L__mod___transformer_h_8_attn_c_proj_bias
    view_104 = attn_output_49.view(-1, 768);  attn_output_49 = None
    l__mod___transformer_h_8_attn_c_proj_weight = self.L__mod___transformer_h_8_attn_c_proj_weight
    x_66 = torch.addmm(l__mod___transformer_h_8_attn_c_proj_bias, view_104, l__mod___transformer_h_8_attn_c_proj_weight);  l__mod___transformer_h_8_attn_c_proj_bias = view_104 = l__mod___transformer_h_8_attn_c_proj_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    attn_output_50 = x_66.view((2, 512, 768));  x_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    attn_output_52 = self.L__mod___transformer_h_8_attn_resid_dropout(attn_output_50);  attn_output_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    residual_17 = attn_output_52 + residual_16;  attn_output_52 = residual_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    hidden_states_66 = self.L__mod___transformer_h_8_ln_2(residual_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_8_mlp_c_fc_bias = self.L__mod___transformer_h_8_mlp_c_fc_bias
    view_106 = hidden_states_66.view(-1, 768);  hidden_states_66 = None
    l__mod___transformer_h_8_mlp_c_fc_weight = self.L__mod___transformer_h_8_mlp_c_fc_weight
    x_68 = torch.addmm(l__mod___transformer_h_8_mlp_c_fc_bias, view_106, l__mod___transformer_h_8_mlp_c_fc_weight);  l__mod___transformer_h_8_mlp_c_fc_bias = view_106 = l__mod___transformer_h_8_mlp_c_fc_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    hidden_states_67 = x_68.view((2, 512, 3072));  x_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_32 = 0.5 * hidden_states_67
    pow_9 = torch.pow(hidden_states_67, 3.0)
    mul_33 = 0.044715 * pow_9;  pow_9 = None
    add_34 = hidden_states_67 + mul_33;  hidden_states_67 = mul_33 = None
    mul_34 = 0.7978845608028654 * add_34;  add_34 = None
    tanh_8 = torch.tanh(mul_34);  mul_34 = None
    add_35 = 1.0 + tanh_8;  tanh_8 = None
    hidden_states_68 = mul_32 * add_35;  mul_32 = add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_8_mlp_c_proj_bias = self.L__mod___transformer_h_8_mlp_c_proj_bias
    view_108 = hidden_states_68.view(-1, 3072);  hidden_states_68 = None
    l__mod___transformer_h_8_mlp_c_proj_weight = self.L__mod___transformer_h_8_mlp_c_proj_weight
    x_70 = torch.addmm(l__mod___transformer_h_8_mlp_c_proj_bias, view_108, l__mod___transformer_h_8_mlp_c_proj_weight);  l__mod___transformer_h_8_mlp_c_proj_bias = view_108 = l__mod___transformer_h_8_mlp_c_proj_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    hidden_states_69 = x_70.view((2, 512, 768));  x_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_8 = self.L__mod___transformer_h_8_mlp_dropout(hidden_states_69);  hidden_states_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    residual_18 = residual_17 + feed_forward_hidden_states_8;  residual_17 = feed_forward_hidden_states_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_72 = self.L__mod___transformer_h_9_ln_1(residual_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_9_attn_c_attn_bias = self.L__mod___transformer_h_9_attn_c_attn_bias
    view_110 = hidden_states_72.view(-1, 768);  hidden_states_72 = None
    l__mod___transformer_h_9_attn_c_attn_weight = self.L__mod___transformer_h_9_attn_c_attn_weight
    x_72 = torch.addmm(l__mod___transformer_h_9_attn_c_attn_bias, view_110, l__mod___transformer_h_9_attn_c_attn_weight);  l__mod___transformer_h_9_attn_c_attn_bias = view_110 = l__mod___transformer_h_9_attn_c_attn_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    x_73 = x_72.view((2, 512, 2304));  x_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_9 = x_73.split(768, dim = 2);  x_73 = None
    query_18 = split_9[0]
    key_18 = split_9[1]
    value_18 = split_9[2];  split_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_36 = query_18.view((2, 512, 12, 64));  query_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_19 = tensor_36.permute(0, 2, 1, 3);  tensor_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_37 = key_18.view((2, 512, 12, 64));  key_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_19 = tensor_37.permute(0, 2, 1, 3);  tensor_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_38 = value_18.view((2, 512, 12, 64));  value_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_19 = tensor_38.permute(0, 2, 1, 3);  tensor_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_9 = key_19.transpose(-1, -2)
    attn_weights_63 = torch.matmul(query_19, transpose_9);  query_19 = transpose_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_18 = torch.full([], 8.0, dtype = torch.float32, device = device(type='cpu'))
    attn_weights_64 = attn_weights_63 / full_18;  attn_weights_63 = full_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_9_attn_bias = self.L__mod___transformer_h_9_attn_bias
    causal_mask_9 = l__mod___transformer_h_9_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 512, None), slice(None, 512, None))];  l__mod___transformer_h_9_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_19 = torch.full([], -3.4028234663852886e+38, dtype = torch.float32)
    mask_value_9 = full_19.to(device(type='cpu'));  full_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    to_19 = attn_weights_64.to(torch.float32);  attn_weights_64 = None
    attn_weights_65 = torch.where(causal_mask_9, to_19, mask_value_9);  causal_mask_9 = to_19 = mask_value_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_66 = torch.nn.functional.softmax(attn_weights_65, dim = -1);  attn_weights_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:211, code: attn_weights = attn_weights.type(value.dtype)
    attn_weights_67 = attn_weights_66.type(torch.float32);  attn_weights_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_69 = self.L__mod___transformer_h_9_attn_attn_dropout(attn_weights_67);  attn_weights_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_54 = torch.matmul(attn_weights_69, value_19);  attn_weights_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_39 = attn_output_54.permute(0, 2, 1, 3);  attn_output_54 = None
    tensor_39 = permute_39.contiguous();  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    attn_output_55 = tensor_39.view((2, 512, 768));  tensor_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_9_attn_c_proj_bias = self.L__mod___transformer_h_9_attn_c_proj_bias
    view_116 = attn_output_55.view(-1, 768);  attn_output_55 = None
    l__mod___transformer_h_9_attn_c_proj_weight = self.L__mod___transformer_h_9_attn_c_proj_weight
    x_74 = torch.addmm(l__mod___transformer_h_9_attn_c_proj_bias, view_116, l__mod___transformer_h_9_attn_c_proj_weight);  l__mod___transformer_h_9_attn_c_proj_bias = view_116 = l__mod___transformer_h_9_attn_c_proj_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    attn_output_56 = x_74.view((2, 512, 768));  x_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    attn_output_58 = self.L__mod___transformer_h_9_attn_resid_dropout(attn_output_56);  attn_output_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    residual_19 = attn_output_58 + residual_18;  attn_output_58 = residual_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    hidden_states_74 = self.L__mod___transformer_h_9_ln_2(residual_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_9_mlp_c_fc_bias = self.L__mod___transformer_h_9_mlp_c_fc_bias
    view_118 = hidden_states_74.view(-1, 768);  hidden_states_74 = None
    l__mod___transformer_h_9_mlp_c_fc_weight = self.L__mod___transformer_h_9_mlp_c_fc_weight
    x_76 = torch.addmm(l__mod___transformer_h_9_mlp_c_fc_bias, view_118, l__mod___transformer_h_9_mlp_c_fc_weight);  l__mod___transformer_h_9_mlp_c_fc_bias = view_118 = l__mod___transformer_h_9_mlp_c_fc_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    hidden_states_75 = x_76.view((2, 512, 3072));  x_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_36 = 0.5 * hidden_states_75
    pow_10 = torch.pow(hidden_states_75, 3.0)
    mul_37 = 0.044715 * pow_10;  pow_10 = None
    add_38 = hidden_states_75 + mul_37;  hidden_states_75 = mul_37 = None
    mul_38 = 0.7978845608028654 * add_38;  add_38 = None
    tanh_9 = torch.tanh(mul_38);  mul_38 = None
    add_39 = 1.0 + tanh_9;  tanh_9 = None
    hidden_states_76 = mul_36 * add_39;  mul_36 = add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_9_mlp_c_proj_bias = self.L__mod___transformer_h_9_mlp_c_proj_bias
    view_120 = hidden_states_76.view(-1, 3072);  hidden_states_76 = None
    l__mod___transformer_h_9_mlp_c_proj_weight = self.L__mod___transformer_h_9_mlp_c_proj_weight
    x_78 = torch.addmm(l__mod___transformer_h_9_mlp_c_proj_bias, view_120, l__mod___transformer_h_9_mlp_c_proj_weight);  l__mod___transformer_h_9_mlp_c_proj_bias = view_120 = l__mod___transformer_h_9_mlp_c_proj_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    hidden_states_77 = x_78.view((2, 512, 768));  x_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_9 = self.L__mod___transformer_h_9_mlp_dropout(hidden_states_77);  hidden_states_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    residual_20 = residual_19 + feed_forward_hidden_states_9;  residual_19 = feed_forward_hidden_states_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_80 = self.L__mod___transformer_h_10_ln_1(residual_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_10_attn_c_attn_bias = self.L__mod___transformer_h_10_attn_c_attn_bias
    view_122 = hidden_states_80.view(-1, 768);  hidden_states_80 = None
    l__mod___transformer_h_10_attn_c_attn_weight = self.L__mod___transformer_h_10_attn_c_attn_weight
    x_80 = torch.addmm(l__mod___transformer_h_10_attn_c_attn_bias, view_122, l__mod___transformer_h_10_attn_c_attn_weight);  l__mod___transformer_h_10_attn_c_attn_bias = view_122 = l__mod___transformer_h_10_attn_c_attn_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    x_81 = x_80.view((2, 512, 2304));  x_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_10 = x_81.split(768, dim = 2);  x_81 = None
    query_20 = split_10[0]
    key_20 = split_10[1]
    value_20 = split_10[2];  split_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_40 = query_20.view((2, 512, 12, 64));  query_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_21 = tensor_40.permute(0, 2, 1, 3);  tensor_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_41 = key_20.view((2, 512, 12, 64));  key_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_21 = tensor_41.permute(0, 2, 1, 3);  tensor_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_42 = value_20.view((2, 512, 12, 64));  value_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_21 = tensor_42.permute(0, 2, 1, 3);  tensor_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_10 = key_21.transpose(-1, -2)
    attn_weights_70 = torch.matmul(query_21, transpose_10);  query_21 = transpose_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_20 = torch.full([], 8.0, dtype = torch.float32, device = device(type='cpu'))
    attn_weights_71 = attn_weights_70 / full_20;  attn_weights_70 = full_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_10_attn_bias = self.L__mod___transformer_h_10_attn_bias
    causal_mask_10 = l__mod___transformer_h_10_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 512, None), slice(None, 512, None))];  l__mod___transformer_h_10_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_21 = torch.full([], -3.4028234663852886e+38, dtype = torch.float32)
    mask_value_10 = full_21.to(device(type='cpu'));  full_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    to_21 = attn_weights_71.to(torch.float32);  attn_weights_71 = None
    attn_weights_72 = torch.where(causal_mask_10, to_21, mask_value_10);  causal_mask_10 = to_21 = mask_value_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_73 = torch.nn.functional.softmax(attn_weights_72, dim = -1);  attn_weights_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:211, code: attn_weights = attn_weights.type(value.dtype)
    attn_weights_74 = attn_weights_73.type(torch.float32);  attn_weights_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_76 = self.L__mod___transformer_h_10_attn_attn_dropout(attn_weights_74);  attn_weights_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_60 = torch.matmul(attn_weights_76, value_21);  attn_weights_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_43 = attn_output_60.permute(0, 2, 1, 3);  attn_output_60 = None
    tensor_43 = permute_43.contiguous();  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    attn_output_61 = tensor_43.view((2, 512, 768));  tensor_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_10_attn_c_proj_bias = self.L__mod___transformer_h_10_attn_c_proj_bias
    view_128 = attn_output_61.view(-1, 768);  attn_output_61 = None
    l__mod___transformer_h_10_attn_c_proj_weight = self.L__mod___transformer_h_10_attn_c_proj_weight
    x_82 = torch.addmm(l__mod___transformer_h_10_attn_c_proj_bias, view_128, l__mod___transformer_h_10_attn_c_proj_weight);  l__mod___transformer_h_10_attn_c_proj_bias = view_128 = l__mod___transformer_h_10_attn_c_proj_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    attn_output_62 = x_82.view((2, 512, 768));  x_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    attn_output_64 = self.L__mod___transformer_h_10_attn_resid_dropout(attn_output_62);  attn_output_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    residual_21 = attn_output_64 + residual_20;  attn_output_64 = residual_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    hidden_states_82 = self.L__mod___transformer_h_10_ln_2(residual_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_10_mlp_c_fc_bias = self.L__mod___transformer_h_10_mlp_c_fc_bias
    view_130 = hidden_states_82.view(-1, 768);  hidden_states_82 = None
    l__mod___transformer_h_10_mlp_c_fc_weight = self.L__mod___transformer_h_10_mlp_c_fc_weight
    x_84 = torch.addmm(l__mod___transformer_h_10_mlp_c_fc_bias, view_130, l__mod___transformer_h_10_mlp_c_fc_weight);  l__mod___transformer_h_10_mlp_c_fc_bias = view_130 = l__mod___transformer_h_10_mlp_c_fc_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    hidden_states_83 = x_84.view((2, 512, 3072));  x_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_40 = 0.5 * hidden_states_83
    pow_11 = torch.pow(hidden_states_83, 3.0)
    mul_41 = 0.044715 * pow_11;  pow_11 = None
    add_42 = hidden_states_83 + mul_41;  hidden_states_83 = mul_41 = None
    mul_42 = 0.7978845608028654 * add_42;  add_42 = None
    tanh_10 = torch.tanh(mul_42);  mul_42 = None
    add_43 = 1.0 + tanh_10;  tanh_10 = None
    hidden_states_84 = mul_40 * add_43;  mul_40 = add_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_10_mlp_c_proj_bias = self.L__mod___transformer_h_10_mlp_c_proj_bias
    view_132 = hidden_states_84.view(-1, 3072);  hidden_states_84 = None
    l__mod___transformer_h_10_mlp_c_proj_weight = self.L__mod___transformer_h_10_mlp_c_proj_weight
    x_86 = torch.addmm(l__mod___transformer_h_10_mlp_c_proj_bias, view_132, l__mod___transformer_h_10_mlp_c_proj_weight);  l__mod___transformer_h_10_mlp_c_proj_bias = view_132 = l__mod___transformer_h_10_mlp_c_proj_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    hidden_states_85 = x_86.view((2, 512, 768));  x_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_10 = self.L__mod___transformer_h_10_mlp_dropout(hidden_states_85);  hidden_states_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    residual_22 = residual_21 + feed_forward_hidden_states_10;  residual_21 = feed_forward_hidden_states_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    hidden_states_88 = self.L__mod___transformer_h_11_ln_1(residual_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_11_attn_c_attn_bias = self.L__mod___transformer_h_11_attn_c_attn_bias
    view_134 = hidden_states_88.view(-1, 768);  hidden_states_88 = None
    l__mod___transformer_h_11_attn_c_attn_weight = self.L__mod___transformer_h_11_attn_c_attn_weight
    x_88 = torch.addmm(l__mod___transformer_h_11_attn_c_attn_bias, view_134, l__mod___transformer_h_11_attn_c_attn_weight);  l__mod___transformer_h_11_attn_c_attn_bias = view_134 = l__mod___transformer_h_11_attn_c_attn_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    x_89 = x_88.view((2, 512, 2304));  x_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_11 = x_89.split(768, dim = 2);  x_89 = None
    query_22 = split_11[0]
    key_22 = split_11[1]
    value_22 = split_11[2];  split_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_44 = query_22.view((2, 512, 12, 64));  query_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    query_23 = tensor_44.permute(0, 2, 1, 3);  tensor_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_45 = key_22.view((2, 512, 12, 64));  key_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    key_23 = tensor_45.permute(0, 2, 1, 3);  tensor_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    tensor_46 = value_22.view((2, 512, 12, 64));  value_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    value_23 = tensor_46.permute(0, 2, 1, 3);  tensor_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    transpose_11 = key_23.transpose(-1, -2)
    attn_weights_77 = torch.matmul(query_23, transpose_11);  query_23 = transpose_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_22 = torch.full([], 8.0, dtype = torch.float32, device = device(type='cpu'))
    attn_weights_78 = attn_weights_77 / full_22;  attn_weights_77 = full_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    l__mod___transformer_h_11_attn_bias = self.L__mod___transformer_h_11_attn_bias
    causal_mask_11 = l__mod___transformer_h_11_attn_bias[(slice(None, None, None), slice(None, None, None), slice(0, 512, None), slice(None, 512, None))];  l__mod___transformer_h_11_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_23 = torch.full([], -3.4028234663852886e+38, dtype = torch.float32)
    mask_value_11 = full_23.to(device(type='cpu'));  full_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    to_23 = attn_weights_78.to(torch.float32);  attn_weights_78 = None
    attn_weights_79 = torch.where(causal_mask_11, to_23, mask_value_11);  causal_mask_11 = to_23 = mask_value_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights_80 = torch.nn.functional.softmax(attn_weights_79, dim = -1);  attn_weights_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:211, code: attn_weights = attn_weights.type(value.dtype)
    attn_weights_81 = attn_weights_80.type(torch.float32);  attn_weights_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    attn_weights_83 = self.L__mod___transformer_h_11_attn_attn_dropout(attn_weights_81);  attn_weights_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    attn_output_66 = torch.matmul(attn_weights_83, value_23);  attn_weights_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_47 = attn_output_66.permute(0, 2, 1, 3);  attn_output_66 = None
    tensor_47 = permute_47.contiguous();  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    attn_output_67 = tensor_47.view((2, 512, 768));  tensor_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_11_attn_c_proj_bias = self.L__mod___transformer_h_11_attn_c_proj_bias
    view_140 = attn_output_67.view(-1, 768);  attn_output_67 = None
    l__mod___transformer_h_11_attn_c_proj_weight = self.L__mod___transformer_h_11_attn_c_proj_weight
    x_90 = torch.addmm(l__mod___transformer_h_11_attn_c_proj_bias, view_140, l__mod___transformer_h_11_attn_c_proj_weight);  l__mod___transformer_h_11_attn_c_proj_bias = view_140 = l__mod___transformer_h_11_attn_c_proj_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    attn_output_68 = x_90.view((2, 512, 768));  x_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    attn_output_70 = self.L__mod___transformer_h_11_attn_resid_dropout(attn_output_68);  attn_output_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    residual_23 = attn_output_70 + residual_22;  attn_output_70 = residual_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    hidden_states_90 = self.L__mod___transformer_h_11_ln_2(residual_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_11_mlp_c_fc_bias = self.L__mod___transformer_h_11_mlp_c_fc_bias
    view_142 = hidden_states_90.view(-1, 768);  hidden_states_90 = None
    l__mod___transformer_h_11_mlp_c_fc_weight = self.L__mod___transformer_h_11_mlp_c_fc_weight
    x_92 = torch.addmm(l__mod___transformer_h_11_mlp_c_fc_bias, view_142, l__mod___transformer_h_11_mlp_c_fc_weight);  l__mod___transformer_h_11_mlp_c_fc_bias = view_142 = l__mod___transformer_h_11_mlp_c_fc_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    hidden_states_91 = x_92.view((2, 512, 3072));  x_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_44 = 0.5 * hidden_states_91
    pow_12 = torch.pow(hidden_states_91, 3.0)
    mul_45 = 0.044715 * pow_12;  pow_12 = None
    add_46 = hidden_states_91 + mul_45;  hidden_states_91 = mul_45 = None
    mul_46 = 0.7978845608028654 * add_46;  add_46 = None
    tanh_11 = torch.tanh(mul_46);  mul_46 = None
    add_47 = 1.0 + tanh_11;  tanh_11 = None
    hidden_states_92 = mul_44 * add_47;  mul_44 = add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    l__mod___transformer_h_11_mlp_c_proj_bias = self.L__mod___transformer_h_11_mlp_c_proj_bias
    view_144 = hidden_states_92.view(-1, 3072);  hidden_states_92 = None
    l__mod___transformer_h_11_mlp_c_proj_weight = self.L__mod___transformer_h_11_mlp_c_proj_weight
    x_94 = torch.addmm(l__mod___transformer_h_11_mlp_c_proj_bias, view_144, l__mod___transformer_h_11_mlp_c_proj_weight);  l__mod___transformer_h_11_mlp_c_proj_bias = view_144 = l__mod___transformer_h_11_mlp_c_proj_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    hidden_states_93 = x_94.view((2, 512, 768));  x_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    feed_forward_hidden_states_11 = self.L__mod___transformer_h_11_mlp_dropout(hidden_states_93);  hidden_states_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    hidden_states_95 = residual_23 + feed_forward_hidden_states_11;  residual_23 = feed_forward_hidden_states_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:926, code: hidden_states = self.ln_f(hidden_states)
    l__mod___transformer_ln_f = self.L__mod___transformer_ln_f(hidden_states_95);  hidden_states_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:928, code: hidden_states = hidden_states.view(output_shape)
    hidden_states_96 = l__mod___transformer_ln_f.view((-1, 512, 768));  l__mod___transformer_ln_f = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1098, code: lm_logits = self.lm_head(hidden_states)
    lm_logits = self.L__mod___lm_head(hidden_states_96);  hidden_states_96 = None
    return (lm_logits, key_1, value_1, key_3, value_3, key_5, value_5, key_7, value_7, key_9, value_9, key_11, value_11, key_13, value_13, key_15, value_15, key_17, value_17, key_19, value_19, key_21, value_21, key_23, value_23)
    