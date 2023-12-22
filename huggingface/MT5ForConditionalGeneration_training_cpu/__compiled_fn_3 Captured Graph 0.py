from __future__ import annotations



def forward(self, L_cloned_inputs_input_ids_ : torch.Tensor, L_cloned_inputs_labels_ : torch.Tensor, L_cloned_inputs_decoder_input_ids_ : torch.Tensor):
    l_cloned_inputs_input_ids_ = L_cloned_inputs_input_ids_
    l_cloned_inputs_labels_ = L_cloned_inputs_labels_
    l_cloned_inputs_decoder_input_ids_ = L_cloned_inputs_decoder_input_ids_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:984, code: input_ids = input_ids.view(-1, input_shape[-1])
    input_ids = l_cloned_inputs_input_ids_.view(-1, 128);  l_cloned_inputs_input_ids_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:994, code: inputs_embeds = self.embed_tokens(input_ids)
    inputs_embeds = self.L__mod___encoder_embed_tokens(input_ids);  input_ids = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:1006, code: attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
    attention_mask = torch.ones(1, 128, device = device(type='cpu'))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:916, code: extended_attention_mask = attention_mask[:, None, None, :]
    extended_attention_mask = attention_mask[(slice(None, None, None), None, None, slice(None, None, None))];  attention_mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:927, code: extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask_1 = extended_attention_mask.to(dtype = torch.float32);  extended_attention_mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:928, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    sub = 1.0 - extended_attention_mask_1;  extended_attention_mask_1 = None
    extended_attention_mask_3 = sub * -3.4028234663852886e+38;  sub = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:1049, code: hidden_states = self.dropout(inputs_embeds)
    hidden_states = self.L__mod___encoder_dropout(inputs_embeds);  inputs_embeds = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_1 = hidden_states.to(torch.float32)
    pow_1 = to_1.pow(2);  to_1 = None
    variance = pow_1.mean(-1, keepdim = True);  pow_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add = variance + 1e-06;  variance = None
    rsqrt = torch.rsqrt(add);  add = None
    hidden_states_1 = hidden_states * rsqrt;  rsqrt = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___encoder_block_0_layer_0_layer_norm_weight_1 = self.L__mod___encoder_block_0_layer_0_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    normed_hidden_states = l__mod___encoder_block_0_layer_0_layer_norm_weight_1 * hidden_states_1;  l__mod___encoder_block_0_layer_0_layer_norm_weight_1 = hidden_states_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    l__mod___encoder_block_0_layer_0_self_attention_q = self.L__mod___encoder_block_0_layer_0_SelfAttention_q(normed_hidden_states)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_1 = l__mod___encoder_block_0_layer_0_self_attention_q.view(1, -1, 6, 64);  l__mod___encoder_block_0_layer_0_self_attention_q = None
    query_states = view_1.transpose(1, 2);  view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    l__mod___encoder_block_0_layer_0_self_attention_k = self.L__mod___encoder_block_0_layer_0_SelfAttention_k(normed_hidden_states)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_2 = l__mod___encoder_block_0_layer_0_self_attention_k.view(1, -1, 6, 64);  l__mod___encoder_block_0_layer_0_self_attention_k = None
    key_states = view_2.transpose(1, 2);  view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    l__mod___encoder_block_0_layer_0_self_attention_v = self.L__mod___encoder_block_0_layer_0_SelfAttention_v(normed_hidden_states);  normed_hidden_states = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_3 = l__mod___encoder_block_0_layer_0_self_attention_v.view(1, -1, 6, 64);  l__mod___encoder_block_0_layer_0_self_attention_v = None
    value_states = view_3.transpose(1, 2);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    transpose_3 = key_states.transpose(3, 2);  key_states = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    scores = torch.matmul(query_states, transpose_3);  query_states = transpose_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:302, code: context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
    arange = torch.arange(128, dtype = torch.int64, device = device(type='cpu'))
    context_position = arange[(slice(None, None, None), None)];  arange = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:303, code: memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
    arange_1 = torch.arange(128, dtype = torch.int64, device = device(type='cpu'))
    memory_position = arange_1[(None, slice(None, None, None))];  arange_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:304, code: relative_position = memory_position - context_position  # shape (query_length, key_length)
    relative_position = memory_position - context_position;  memory_position = context_position = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:275, code: relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
    gt = relative_position > 0
    to_2 = gt.to(torch.int64);  gt = None
    mul_3 = to_2 * 16;  to_2 = None
    relative_buckets = 0 + mul_3;  mul_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:276, code: relative_position = torch.abs(relative_position)
    relative_position_1 = torch.abs(relative_position);  relative_position = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:283, code: is_small = relative_position < max_exact
    is_small = relative_position_1 < 8
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:287, code: torch.log(relative_position.float() / max_exact)
    float_1 = relative_position_1.float()
    truediv = float_1 / 8;  float_1 = None
    log = torch.log(truediv);  truediv = None
    truediv_1 = log / 2.772588722239781;  log = None
    mul_4 = truediv_1 * 8;  truediv_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:290, code: ).to(torch.long)
    to_3 = mul_4.to(torch.int64);  mul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:286, code: relative_position_if_large = max_exact + (
    relative_position_if_large = 8 + to_3;  to_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:292, code: relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
    full_like = torch.full_like(relative_position_if_large, 15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:291, code: relative_position_if_large = torch.min(
    relative_position_if_large_1 = torch.min(relative_position_if_large, full_like);  relative_position_if_large = full_like = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:295, code: relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
    where = torch.where(is_small, relative_position_1, relative_position_if_large_1);  is_small = relative_position_1 = relative_position_if_large_1 = None
    relative_buckets += where;  relative_position_bucket = relative_buckets;  relative_buckets = where = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:311, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
    values = self.L__mod___encoder_block_0_layer_0_SelfAttention_relative_attention_bias(relative_position_bucket);  relative_position_bucket = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:312, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
    permute = values.permute([2, 0, 1]);  values = None
    position_bias = permute.unsqueeze(0);  permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:413, code: position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)
    position_bias_9 = position_bias + extended_attention_mask_3;  position_bias = extended_attention_mask_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    scores += position_bias_9;  scores_1 = scores;  scores = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    float_2 = scores_1.float()
    softmax = torch.nn.functional.softmax(float_2, dim = -1);  float_2 = None
    attn_weights = softmax.type_as(scores_1);  softmax = scores_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    attn_weights_1 = torch.nn.functional.dropout(attn_weights, p = 0.1, training = True);  attn_weights = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    matmul_1 = torch.matmul(attn_weights_1, value_states);  attn_weights_1 = value_states = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    transpose_4 = matmul_1.transpose(1, 2);  matmul_1 = None
    contiguous = transpose_4.contiguous();  transpose_4 = None
    attn_output = contiguous.view(1, -1, 384);  contiguous = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    attn_output_1 = self.L__mod___encoder_block_0_layer_0_SelfAttention_o(attn_output);  attn_output = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    l__mod___encoder_block_0_layer_0_dropout = self.L__mod___encoder_block_0_layer_0_dropout(attn_output_1);  attn_output_1 = None
    hidden_states_5 = hidden_states + l__mod___encoder_block_0_layer_0_dropout;  hidden_states = l__mod___encoder_block_0_layer_0_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_4 = hidden_states_5.to(torch.float32)
    pow_2 = to_4.pow(2);  to_4 = None
    variance_1 = pow_2.mean(-1, keepdim = True);  pow_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_5 = variance_1 + 1e-06;  variance_1 = None
    rsqrt_1 = torch.rsqrt(add_5);  add_5 = None
    hidden_states_6 = hidden_states_5 * rsqrt_1;  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___encoder_block_0_layer_1_layer_norm_weight_1 = self.L__mod___encoder_block_0_layer_1_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    forwarded_states = l__mod___encoder_block_0_layer_1_layer_norm_weight_1 * hidden_states_6;  l__mod___encoder_block_0_layer_1_layer_norm_weight_1 = hidden_states_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    l__mod___encoder_block_0_layer__1__dense_relu_dense_wi_0 = self.L__mod___encoder_block_0_layer__1__DenseReluDense_wi_0(forwarded_states)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_7 = 0.5 * l__mod___encoder_block_0_layer__1__dense_relu_dense_wi_0
    pow_3 = torch.pow(l__mod___encoder_block_0_layer__1__dense_relu_dense_wi_0, 3.0)
    mul_8 = 0.044715 * pow_3;  pow_3 = None
    add_6 = l__mod___encoder_block_0_layer__1__dense_relu_dense_wi_0 + mul_8;  l__mod___encoder_block_0_layer__1__dense_relu_dense_wi_0 = mul_8 = None
    mul_9 = 0.7978845608028654 * add_6;  add_6 = None
    tanh = torch.tanh(mul_9);  mul_9 = None
    add_7 = 1.0 + tanh;  tanh = None
    hidden_gelu = mul_7 * add_7;  mul_7 = add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    hidden_linear = self.L__mod___encoder_block_0_layer__1__DenseReluDense_wi_1(forwarded_states);  forwarded_states = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    hidden_states_7 = hidden_gelu * hidden_linear;  hidden_gelu = hidden_linear = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    hidden_states_8 = self.L__mod___encoder_block_0_layer__1__DenseReluDense_dropout(hidden_states_7);  hidden_states_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    forwarded_states_1 = self.L__mod___encoder_block_0_layer__1__DenseReluDense_wo(hidden_states_8);  hidden_states_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    l__mod___encoder_block_0_layer__1__dropout = self.L__mod___encoder_block_0_layer__1__dropout(forwarded_states_1);  forwarded_states_1 = None
    hidden_states_12 = hidden_states_5 + l__mod___encoder_block_0_layer__1__dropout;  hidden_states_5 = l__mod___encoder_block_0_layer__1__dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_5 = hidden_states_12.to(torch.float32)
    pow_4 = to_5.pow(2);  to_5 = None
    variance_2 = pow_4.mean(-1, keepdim = True);  pow_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_9 = variance_2 + 1e-06;  variance_2 = None
    rsqrt_2 = torch.rsqrt(add_9);  add_9 = None
    hidden_states_13 = hidden_states_12 * rsqrt_2;  rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___encoder_block_1_layer_0_layer_norm_weight_1 = self.L__mod___encoder_block_1_layer_0_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    normed_hidden_states_1 = l__mod___encoder_block_1_layer_0_layer_norm_weight_1 * hidden_states_13;  l__mod___encoder_block_1_layer_0_layer_norm_weight_1 = hidden_states_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    l__mod___encoder_block_1_layer_0_self_attention_q = self.L__mod___encoder_block_1_layer_0_SelfAttention_q(normed_hidden_states_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_5 = l__mod___encoder_block_1_layer_0_self_attention_q.view(1, -1, 6, 64);  l__mod___encoder_block_1_layer_0_self_attention_q = None
    query_states_1 = view_5.transpose(1, 2);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    l__mod___encoder_block_1_layer_0_self_attention_k = self.L__mod___encoder_block_1_layer_0_SelfAttention_k(normed_hidden_states_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_6 = l__mod___encoder_block_1_layer_0_self_attention_k.view(1, -1, 6, 64);  l__mod___encoder_block_1_layer_0_self_attention_k = None
    key_states_1 = view_6.transpose(1, 2);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    l__mod___encoder_block_1_layer_0_self_attention_v = self.L__mod___encoder_block_1_layer_0_SelfAttention_v(normed_hidden_states_1);  normed_hidden_states_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_7 = l__mod___encoder_block_1_layer_0_self_attention_v.view(1, -1, 6, 64);  l__mod___encoder_block_1_layer_0_self_attention_v = None
    value_states_1 = view_7.transpose(1, 2);  view_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    transpose_8 = key_states_1.transpose(3, 2);  key_states_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    scores_2 = torch.matmul(query_states_1, transpose_8);  query_states_1 = transpose_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    scores_2 += position_bias_9;  scores_3 = scores_2;  scores_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    float_3 = scores_3.float()
    softmax_1 = torch.nn.functional.softmax(float_3, dim = -1);  float_3 = None
    attn_weights_2 = softmax_1.type_as(scores_3);  softmax_1 = scores_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    attn_weights_3 = torch.nn.functional.dropout(attn_weights_2, p = 0.1, training = True);  attn_weights_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    matmul_3 = torch.matmul(attn_weights_3, value_states_1);  attn_weights_3 = value_states_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    transpose_9 = matmul_3.transpose(1, 2);  matmul_3 = None
    contiguous_1 = transpose_9.contiguous();  transpose_9 = None
    attn_output_2 = contiguous_1.view(1, -1, 384);  contiguous_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    attn_output_3 = self.L__mod___encoder_block_1_layer_0_SelfAttention_o(attn_output_2);  attn_output_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    l__mod___encoder_block_1_layer_0_dropout = self.L__mod___encoder_block_1_layer_0_dropout(attn_output_3);  attn_output_3 = None
    hidden_states_17 = hidden_states_12 + l__mod___encoder_block_1_layer_0_dropout;  hidden_states_12 = l__mod___encoder_block_1_layer_0_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_6 = hidden_states_17.to(torch.float32)
    pow_5 = to_6.pow(2);  to_6 = None
    variance_3 = pow_5.mean(-1, keepdim = True);  pow_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_11 = variance_3 + 1e-06;  variance_3 = None
    rsqrt_3 = torch.rsqrt(add_11);  add_11 = None
    hidden_states_18 = hidden_states_17 * rsqrt_3;  rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___encoder_block_1_layer_1_layer_norm_weight_1 = self.L__mod___encoder_block_1_layer_1_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    forwarded_states_2 = l__mod___encoder_block_1_layer_1_layer_norm_weight_1 * hidden_states_18;  l__mod___encoder_block_1_layer_1_layer_norm_weight_1 = hidden_states_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    l__mod___encoder_block_1_layer__1__dense_relu_dense_wi_0 = self.L__mod___encoder_block_1_layer__1__DenseReluDense_wi_0(forwarded_states_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_16 = 0.5 * l__mod___encoder_block_1_layer__1__dense_relu_dense_wi_0
    pow_6 = torch.pow(l__mod___encoder_block_1_layer__1__dense_relu_dense_wi_0, 3.0)
    mul_17 = 0.044715 * pow_6;  pow_6 = None
    add_12 = l__mod___encoder_block_1_layer__1__dense_relu_dense_wi_0 + mul_17;  l__mod___encoder_block_1_layer__1__dense_relu_dense_wi_0 = mul_17 = None
    mul_18 = 0.7978845608028654 * add_12;  add_12 = None
    tanh_1 = torch.tanh(mul_18);  mul_18 = None
    add_13 = 1.0 + tanh_1;  tanh_1 = None
    hidden_gelu_1 = mul_16 * add_13;  mul_16 = add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    hidden_linear_1 = self.L__mod___encoder_block_1_layer__1__DenseReluDense_wi_1(forwarded_states_2);  forwarded_states_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    hidden_states_19 = hidden_gelu_1 * hidden_linear_1;  hidden_gelu_1 = hidden_linear_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    hidden_states_20 = self.L__mod___encoder_block_1_layer__1__DenseReluDense_dropout(hidden_states_19);  hidden_states_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    forwarded_states_3 = self.L__mod___encoder_block_1_layer__1__DenseReluDense_wo(hidden_states_20);  hidden_states_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    l__mod___encoder_block_1_layer__1__dropout = self.L__mod___encoder_block_1_layer__1__dropout(forwarded_states_3);  forwarded_states_3 = None
    hidden_states_24 = hidden_states_17 + l__mod___encoder_block_1_layer__1__dropout;  hidden_states_17 = l__mod___encoder_block_1_layer__1__dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_7 = hidden_states_24.to(torch.float32)
    pow_7 = to_7.pow(2);  to_7 = None
    variance_4 = pow_7.mean(-1, keepdim = True);  pow_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_15 = variance_4 + 1e-06;  variance_4 = None
    rsqrt_4 = torch.rsqrt(add_15);  add_15 = None
    hidden_states_25 = hidden_states_24 * rsqrt_4;  rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___encoder_block_2_layer_0_layer_norm_weight_1 = self.L__mod___encoder_block_2_layer_0_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    normed_hidden_states_2 = l__mod___encoder_block_2_layer_0_layer_norm_weight_1 * hidden_states_25;  l__mod___encoder_block_2_layer_0_layer_norm_weight_1 = hidden_states_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    l__mod___encoder_block_2_layer_0_self_attention_q = self.L__mod___encoder_block_2_layer_0_SelfAttention_q(normed_hidden_states_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_9 = l__mod___encoder_block_2_layer_0_self_attention_q.view(1, -1, 6, 64);  l__mod___encoder_block_2_layer_0_self_attention_q = None
    query_states_2 = view_9.transpose(1, 2);  view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    l__mod___encoder_block_2_layer_0_self_attention_k = self.L__mod___encoder_block_2_layer_0_SelfAttention_k(normed_hidden_states_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_10 = l__mod___encoder_block_2_layer_0_self_attention_k.view(1, -1, 6, 64);  l__mod___encoder_block_2_layer_0_self_attention_k = None
    key_states_2 = view_10.transpose(1, 2);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    l__mod___encoder_block_2_layer_0_self_attention_v = self.L__mod___encoder_block_2_layer_0_SelfAttention_v(normed_hidden_states_2);  normed_hidden_states_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_11 = l__mod___encoder_block_2_layer_0_self_attention_v.view(1, -1, 6, 64);  l__mod___encoder_block_2_layer_0_self_attention_v = None
    value_states_2 = view_11.transpose(1, 2);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    transpose_13 = key_states_2.transpose(3, 2);  key_states_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    scores_4 = torch.matmul(query_states_2, transpose_13);  query_states_2 = transpose_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    scores_4 += position_bias_9;  scores_5 = scores_4;  scores_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    float_4 = scores_5.float()
    softmax_2 = torch.nn.functional.softmax(float_4, dim = -1);  float_4 = None
    attn_weights_4 = softmax_2.type_as(scores_5);  softmax_2 = scores_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    attn_weights_5 = torch.nn.functional.dropout(attn_weights_4, p = 0.1, training = True);  attn_weights_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    matmul_5 = torch.matmul(attn_weights_5, value_states_2);  attn_weights_5 = value_states_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    transpose_14 = matmul_5.transpose(1, 2);  matmul_5 = None
    contiguous_2 = transpose_14.contiguous();  transpose_14 = None
    attn_output_4 = contiguous_2.view(1, -1, 384);  contiguous_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    attn_output_5 = self.L__mod___encoder_block_2_layer_0_SelfAttention_o(attn_output_4);  attn_output_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    l__mod___encoder_block_2_layer_0_dropout = self.L__mod___encoder_block_2_layer_0_dropout(attn_output_5);  attn_output_5 = None
    hidden_states_29 = hidden_states_24 + l__mod___encoder_block_2_layer_0_dropout;  hidden_states_24 = l__mod___encoder_block_2_layer_0_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_8 = hidden_states_29.to(torch.float32)
    pow_8 = to_8.pow(2);  to_8 = None
    variance_5 = pow_8.mean(-1, keepdim = True);  pow_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_17 = variance_5 + 1e-06;  variance_5 = None
    rsqrt_5 = torch.rsqrt(add_17);  add_17 = None
    hidden_states_30 = hidden_states_29 * rsqrt_5;  rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___encoder_block_2_layer_1_layer_norm_weight_1 = self.L__mod___encoder_block_2_layer_1_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    forwarded_states_4 = l__mod___encoder_block_2_layer_1_layer_norm_weight_1 * hidden_states_30;  l__mod___encoder_block_2_layer_1_layer_norm_weight_1 = hidden_states_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    l__mod___encoder_block_2_layer__1__dense_relu_dense_wi_0 = self.L__mod___encoder_block_2_layer__1__DenseReluDense_wi_0(forwarded_states_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_25 = 0.5 * l__mod___encoder_block_2_layer__1__dense_relu_dense_wi_0
    pow_9 = torch.pow(l__mod___encoder_block_2_layer__1__dense_relu_dense_wi_0, 3.0)
    mul_26 = 0.044715 * pow_9;  pow_9 = None
    add_18 = l__mod___encoder_block_2_layer__1__dense_relu_dense_wi_0 + mul_26;  l__mod___encoder_block_2_layer__1__dense_relu_dense_wi_0 = mul_26 = None
    mul_27 = 0.7978845608028654 * add_18;  add_18 = None
    tanh_2 = torch.tanh(mul_27);  mul_27 = None
    add_19 = 1.0 + tanh_2;  tanh_2 = None
    hidden_gelu_2 = mul_25 * add_19;  mul_25 = add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    hidden_linear_2 = self.L__mod___encoder_block_2_layer__1__DenseReluDense_wi_1(forwarded_states_4);  forwarded_states_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    hidden_states_31 = hidden_gelu_2 * hidden_linear_2;  hidden_gelu_2 = hidden_linear_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    hidden_states_32 = self.L__mod___encoder_block_2_layer__1__DenseReluDense_dropout(hidden_states_31);  hidden_states_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    forwarded_states_5 = self.L__mod___encoder_block_2_layer__1__DenseReluDense_wo(hidden_states_32);  hidden_states_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    l__mod___encoder_block_2_layer__1__dropout = self.L__mod___encoder_block_2_layer__1__dropout(forwarded_states_5);  forwarded_states_5 = None
    hidden_states_36 = hidden_states_29 + l__mod___encoder_block_2_layer__1__dropout;  hidden_states_29 = l__mod___encoder_block_2_layer__1__dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_9 = hidden_states_36.to(torch.float32)
    pow_10 = to_9.pow(2);  to_9 = None
    variance_6 = pow_10.mean(-1, keepdim = True);  pow_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_21 = variance_6 + 1e-06;  variance_6 = None
    rsqrt_6 = torch.rsqrt(add_21);  add_21 = None
    hidden_states_37 = hidden_states_36 * rsqrt_6;  rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___encoder_block_3_layer_0_layer_norm_weight_1 = self.L__mod___encoder_block_3_layer_0_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    normed_hidden_states_3 = l__mod___encoder_block_3_layer_0_layer_norm_weight_1 * hidden_states_37;  l__mod___encoder_block_3_layer_0_layer_norm_weight_1 = hidden_states_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    l__mod___encoder_block_3_layer_0_self_attention_q = self.L__mod___encoder_block_3_layer_0_SelfAttention_q(normed_hidden_states_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_13 = l__mod___encoder_block_3_layer_0_self_attention_q.view(1, -1, 6, 64);  l__mod___encoder_block_3_layer_0_self_attention_q = None
    query_states_3 = view_13.transpose(1, 2);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    l__mod___encoder_block_3_layer_0_self_attention_k = self.L__mod___encoder_block_3_layer_0_SelfAttention_k(normed_hidden_states_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_14 = l__mod___encoder_block_3_layer_0_self_attention_k.view(1, -1, 6, 64);  l__mod___encoder_block_3_layer_0_self_attention_k = None
    key_states_3 = view_14.transpose(1, 2);  view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    l__mod___encoder_block_3_layer_0_self_attention_v = self.L__mod___encoder_block_3_layer_0_SelfAttention_v(normed_hidden_states_3);  normed_hidden_states_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_15 = l__mod___encoder_block_3_layer_0_self_attention_v.view(1, -1, 6, 64);  l__mod___encoder_block_3_layer_0_self_attention_v = None
    value_states_3 = view_15.transpose(1, 2);  view_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    transpose_18 = key_states_3.transpose(3, 2);  key_states_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    scores_6 = torch.matmul(query_states_3, transpose_18);  query_states_3 = transpose_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    scores_6 += position_bias_9;  scores_7 = scores_6;  scores_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    float_5 = scores_7.float()
    softmax_3 = torch.nn.functional.softmax(float_5, dim = -1);  float_5 = None
    attn_weights_6 = softmax_3.type_as(scores_7);  softmax_3 = scores_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    attn_weights_7 = torch.nn.functional.dropout(attn_weights_6, p = 0.1, training = True);  attn_weights_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    matmul_7 = torch.matmul(attn_weights_7, value_states_3);  attn_weights_7 = value_states_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    transpose_19 = matmul_7.transpose(1, 2);  matmul_7 = None
    contiguous_3 = transpose_19.contiguous();  transpose_19 = None
    attn_output_6 = contiguous_3.view(1, -1, 384);  contiguous_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    attn_output_7 = self.L__mod___encoder_block_3_layer_0_SelfAttention_o(attn_output_6);  attn_output_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    l__mod___encoder_block_3_layer_0_dropout = self.L__mod___encoder_block_3_layer_0_dropout(attn_output_7);  attn_output_7 = None
    hidden_states_41 = hidden_states_36 + l__mod___encoder_block_3_layer_0_dropout;  hidden_states_36 = l__mod___encoder_block_3_layer_0_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_10 = hidden_states_41.to(torch.float32)
    pow_11 = to_10.pow(2);  to_10 = None
    variance_7 = pow_11.mean(-1, keepdim = True);  pow_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_23 = variance_7 + 1e-06;  variance_7 = None
    rsqrt_7 = torch.rsqrt(add_23);  add_23 = None
    hidden_states_42 = hidden_states_41 * rsqrt_7;  rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___encoder_block_3_layer_1_layer_norm_weight_1 = self.L__mod___encoder_block_3_layer_1_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    forwarded_states_6 = l__mod___encoder_block_3_layer_1_layer_norm_weight_1 * hidden_states_42;  l__mod___encoder_block_3_layer_1_layer_norm_weight_1 = hidden_states_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    l__mod___encoder_block_3_layer__1__dense_relu_dense_wi_0 = self.L__mod___encoder_block_3_layer__1__DenseReluDense_wi_0(forwarded_states_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_34 = 0.5 * l__mod___encoder_block_3_layer__1__dense_relu_dense_wi_0
    pow_12 = torch.pow(l__mod___encoder_block_3_layer__1__dense_relu_dense_wi_0, 3.0)
    mul_35 = 0.044715 * pow_12;  pow_12 = None
    add_24 = l__mod___encoder_block_3_layer__1__dense_relu_dense_wi_0 + mul_35;  l__mod___encoder_block_3_layer__1__dense_relu_dense_wi_0 = mul_35 = None
    mul_36 = 0.7978845608028654 * add_24;  add_24 = None
    tanh_3 = torch.tanh(mul_36);  mul_36 = None
    add_25 = 1.0 + tanh_3;  tanh_3 = None
    hidden_gelu_3 = mul_34 * add_25;  mul_34 = add_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    hidden_linear_3 = self.L__mod___encoder_block_3_layer__1__DenseReluDense_wi_1(forwarded_states_6);  forwarded_states_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    hidden_states_43 = hidden_gelu_3 * hidden_linear_3;  hidden_gelu_3 = hidden_linear_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    hidden_states_44 = self.L__mod___encoder_block_3_layer__1__DenseReluDense_dropout(hidden_states_43);  hidden_states_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    forwarded_states_7 = self.L__mod___encoder_block_3_layer__1__DenseReluDense_wo(hidden_states_44);  hidden_states_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    l__mod___encoder_block_3_layer__1__dropout = self.L__mod___encoder_block_3_layer__1__dropout(forwarded_states_7);  forwarded_states_7 = None
    hidden_states_48 = hidden_states_41 + l__mod___encoder_block_3_layer__1__dropout;  hidden_states_41 = l__mod___encoder_block_3_layer__1__dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_11 = hidden_states_48.to(torch.float32)
    pow_13 = to_11.pow(2);  to_11 = None
    variance_8 = pow_13.mean(-1, keepdim = True);  pow_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_27 = variance_8 + 1e-06;  variance_8 = None
    rsqrt_8 = torch.rsqrt(add_27);  add_27 = None
    hidden_states_49 = hidden_states_48 * rsqrt_8;  rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___encoder_block_4_layer_0_layer_norm_weight_1 = self.L__mod___encoder_block_4_layer_0_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    normed_hidden_states_4 = l__mod___encoder_block_4_layer_0_layer_norm_weight_1 * hidden_states_49;  l__mod___encoder_block_4_layer_0_layer_norm_weight_1 = hidden_states_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    l__mod___encoder_block_4_layer_0_self_attention_q = self.L__mod___encoder_block_4_layer_0_SelfAttention_q(normed_hidden_states_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_17 = l__mod___encoder_block_4_layer_0_self_attention_q.view(1, -1, 6, 64);  l__mod___encoder_block_4_layer_0_self_attention_q = None
    query_states_4 = view_17.transpose(1, 2);  view_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    l__mod___encoder_block_4_layer_0_self_attention_k = self.L__mod___encoder_block_4_layer_0_SelfAttention_k(normed_hidden_states_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_18 = l__mod___encoder_block_4_layer_0_self_attention_k.view(1, -1, 6, 64);  l__mod___encoder_block_4_layer_0_self_attention_k = None
    key_states_4 = view_18.transpose(1, 2);  view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    l__mod___encoder_block_4_layer_0_self_attention_v = self.L__mod___encoder_block_4_layer_0_SelfAttention_v(normed_hidden_states_4);  normed_hidden_states_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_19 = l__mod___encoder_block_4_layer_0_self_attention_v.view(1, -1, 6, 64);  l__mod___encoder_block_4_layer_0_self_attention_v = None
    value_states_4 = view_19.transpose(1, 2);  view_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    transpose_23 = key_states_4.transpose(3, 2);  key_states_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    scores_8 = torch.matmul(query_states_4, transpose_23);  query_states_4 = transpose_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    scores_8 += position_bias_9;  scores_9 = scores_8;  scores_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    float_6 = scores_9.float()
    softmax_4 = torch.nn.functional.softmax(float_6, dim = -1);  float_6 = None
    attn_weights_8 = softmax_4.type_as(scores_9);  softmax_4 = scores_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    attn_weights_9 = torch.nn.functional.dropout(attn_weights_8, p = 0.1, training = True);  attn_weights_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    matmul_9 = torch.matmul(attn_weights_9, value_states_4);  attn_weights_9 = value_states_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    transpose_24 = matmul_9.transpose(1, 2);  matmul_9 = None
    contiguous_4 = transpose_24.contiguous();  transpose_24 = None
    attn_output_8 = contiguous_4.view(1, -1, 384);  contiguous_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    attn_output_9 = self.L__mod___encoder_block_4_layer_0_SelfAttention_o(attn_output_8);  attn_output_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    l__mod___encoder_block_4_layer_0_dropout = self.L__mod___encoder_block_4_layer_0_dropout(attn_output_9);  attn_output_9 = None
    hidden_states_53 = hidden_states_48 + l__mod___encoder_block_4_layer_0_dropout;  hidden_states_48 = l__mod___encoder_block_4_layer_0_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_12 = hidden_states_53.to(torch.float32)
    pow_14 = to_12.pow(2);  to_12 = None
    variance_9 = pow_14.mean(-1, keepdim = True);  pow_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_29 = variance_9 + 1e-06;  variance_9 = None
    rsqrt_9 = torch.rsqrt(add_29);  add_29 = None
    hidden_states_54 = hidden_states_53 * rsqrt_9;  rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___encoder_block_4_layer_1_layer_norm_weight_1 = self.L__mod___encoder_block_4_layer_1_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    forwarded_states_8 = l__mod___encoder_block_4_layer_1_layer_norm_weight_1 * hidden_states_54;  l__mod___encoder_block_4_layer_1_layer_norm_weight_1 = hidden_states_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    l__mod___encoder_block_4_layer__1__dense_relu_dense_wi_0 = self.L__mod___encoder_block_4_layer__1__DenseReluDense_wi_0(forwarded_states_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_43 = 0.5 * l__mod___encoder_block_4_layer__1__dense_relu_dense_wi_0
    pow_15 = torch.pow(l__mod___encoder_block_4_layer__1__dense_relu_dense_wi_0, 3.0)
    mul_44 = 0.044715 * pow_15;  pow_15 = None
    add_30 = l__mod___encoder_block_4_layer__1__dense_relu_dense_wi_0 + mul_44;  l__mod___encoder_block_4_layer__1__dense_relu_dense_wi_0 = mul_44 = None
    mul_45 = 0.7978845608028654 * add_30;  add_30 = None
    tanh_4 = torch.tanh(mul_45);  mul_45 = None
    add_31 = 1.0 + tanh_4;  tanh_4 = None
    hidden_gelu_4 = mul_43 * add_31;  mul_43 = add_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    hidden_linear_4 = self.L__mod___encoder_block_4_layer__1__DenseReluDense_wi_1(forwarded_states_8);  forwarded_states_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    hidden_states_55 = hidden_gelu_4 * hidden_linear_4;  hidden_gelu_4 = hidden_linear_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    hidden_states_56 = self.L__mod___encoder_block_4_layer__1__DenseReluDense_dropout(hidden_states_55);  hidden_states_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    forwarded_states_9 = self.L__mod___encoder_block_4_layer__1__DenseReluDense_wo(hidden_states_56);  hidden_states_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    l__mod___encoder_block_4_layer__1__dropout = self.L__mod___encoder_block_4_layer__1__dropout(forwarded_states_9);  forwarded_states_9 = None
    hidden_states_60 = hidden_states_53 + l__mod___encoder_block_4_layer__1__dropout;  hidden_states_53 = l__mod___encoder_block_4_layer__1__dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_13 = hidden_states_60.to(torch.float32)
    pow_16 = to_13.pow(2);  to_13 = None
    variance_10 = pow_16.mean(-1, keepdim = True);  pow_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_33 = variance_10 + 1e-06;  variance_10 = None
    rsqrt_10 = torch.rsqrt(add_33);  add_33 = None
    hidden_states_61 = hidden_states_60 * rsqrt_10;  rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___encoder_block_5_layer_0_layer_norm_weight_1 = self.L__mod___encoder_block_5_layer_0_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    normed_hidden_states_5 = l__mod___encoder_block_5_layer_0_layer_norm_weight_1 * hidden_states_61;  l__mod___encoder_block_5_layer_0_layer_norm_weight_1 = hidden_states_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    l__mod___encoder_block_5_layer_0_self_attention_q = self.L__mod___encoder_block_5_layer_0_SelfAttention_q(normed_hidden_states_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_21 = l__mod___encoder_block_5_layer_0_self_attention_q.view(1, -1, 6, 64);  l__mod___encoder_block_5_layer_0_self_attention_q = None
    query_states_5 = view_21.transpose(1, 2);  view_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    l__mod___encoder_block_5_layer_0_self_attention_k = self.L__mod___encoder_block_5_layer_0_SelfAttention_k(normed_hidden_states_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_22 = l__mod___encoder_block_5_layer_0_self_attention_k.view(1, -1, 6, 64);  l__mod___encoder_block_5_layer_0_self_attention_k = None
    key_states_5 = view_22.transpose(1, 2);  view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    l__mod___encoder_block_5_layer_0_self_attention_v = self.L__mod___encoder_block_5_layer_0_SelfAttention_v(normed_hidden_states_5);  normed_hidden_states_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_23 = l__mod___encoder_block_5_layer_0_self_attention_v.view(1, -1, 6, 64);  l__mod___encoder_block_5_layer_0_self_attention_v = None
    value_states_5 = view_23.transpose(1, 2);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    transpose_28 = key_states_5.transpose(3, 2);  key_states_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    scores_10 = torch.matmul(query_states_5, transpose_28);  query_states_5 = transpose_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    scores_10 += position_bias_9;  scores_11 = scores_10;  scores_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    float_7 = scores_11.float()
    softmax_5 = torch.nn.functional.softmax(float_7, dim = -1);  float_7 = None
    attn_weights_10 = softmax_5.type_as(scores_11);  softmax_5 = scores_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    attn_weights_11 = torch.nn.functional.dropout(attn_weights_10, p = 0.1, training = True);  attn_weights_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    matmul_11 = torch.matmul(attn_weights_11, value_states_5);  attn_weights_11 = value_states_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    transpose_29 = matmul_11.transpose(1, 2);  matmul_11 = None
    contiguous_5 = transpose_29.contiguous();  transpose_29 = None
    attn_output_10 = contiguous_5.view(1, -1, 384);  contiguous_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    attn_output_11 = self.L__mod___encoder_block_5_layer_0_SelfAttention_o(attn_output_10);  attn_output_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    l__mod___encoder_block_5_layer_0_dropout = self.L__mod___encoder_block_5_layer_0_dropout(attn_output_11);  attn_output_11 = None
    hidden_states_65 = hidden_states_60 + l__mod___encoder_block_5_layer_0_dropout;  hidden_states_60 = l__mod___encoder_block_5_layer_0_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_14 = hidden_states_65.to(torch.float32)
    pow_17 = to_14.pow(2);  to_14 = None
    variance_11 = pow_17.mean(-1, keepdim = True);  pow_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_35 = variance_11 + 1e-06;  variance_11 = None
    rsqrt_11 = torch.rsqrt(add_35);  add_35 = None
    hidden_states_66 = hidden_states_65 * rsqrt_11;  rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___encoder_block_5_layer_1_layer_norm_weight_1 = self.L__mod___encoder_block_5_layer_1_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    forwarded_states_10 = l__mod___encoder_block_5_layer_1_layer_norm_weight_1 * hidden_states_66;  l__mod___encoder_block_5_layer_1_layer_norm_weight_1 = hidden_states_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    l__mod___encoder_block_5_layer__1__dense_relu_dense_wi_0 = self.L__mod___encoder_block_5_layer__1__DenseReluDense_wi_0(forwarded_states_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_52 = 0.5 * l__mod___encoder_block_5_layer__1__dense_relu_dense_wi_0
    pow_18 = torch.pow(l__mod___encoder_block_5_layer__1__dense_relu_dense_wi_0, 3.0)
    mul_53 = 0.044715 * pow_18;  pow_18 = None
    add_36 = l__mod___encoder_block_5_layer__1__dense_relu_dense_wi_0 + mul_53;  l__mod___encoder_block_5_layer__1__dense_relu_dense_wi_0 = mul_53 = None
    mul_54 = 0.7978845608028654 * add_36;  add_36 = None
    tanh_5 = torch.tanh(mul_54);  mul_54 = None
    add_37 = 1.0 + tanh_5;  tanh_5 = None
    hidden_gelu_5 = mul_52 * add_37;  mul_52 = add_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    hidden_linear_5 = self.L__mod___encoder_block_5_layer__1__DenseReluDense_wi_1(forwarded_states_10);  forwarded_states_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    hidden_states_67 = hidden_gelu_5 * hidden_linear_5;  hidden_gelu_5 = hidden_linear_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    hidden_states_68 = self.L__mod___encoder_block_5_layer__1__DenseReluDense_dropout(hidden_states_67);  hidden_states_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    forwarded_states_11 = self.L__mod___encoder_block_5_layer__1__DenseReluDense_wo(hidden_states_68);  hidden_states_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    l__mod___encoder_block_5_layer__1__dropout = self.L__mod___encoder_block_5_layer__1__dropout(forwarded_states_11);  forwarded_states_11 = None
    hidden_states_72 = hidden_states_65 + l__mod___encoder_block_5_layer__1__dropout;  hidden_states_65 = l__mod___encoder_block_5_layer__1__dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_15 = hidden_states_72.to(torch.float32)
    pow_19 = to_15.pow(2);  to_15 = None
    variance_12 = pow_19.mean(-1, keepdim = True);  pow_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_39 = variance_12 + 1e-06;  variance_12 = None
    rsqrt_12 = torch.rsqrt(add_39);  add_39 = None
    hidden_states_73 = hidden_states_72 * rsqrt_12;  rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___encoder_block_6_layer_0_layer_norm_weight_1 = self.L__mod___encoder_block_6_layer_0_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    normed_hidden_states_6 = l__mod___encoder_block_6_layer_0_layer_norm_weight_1 * hidden_states_73;  l__mod___encoder_block_6_layer_0_layer_norm_weight_1 = hidden_states_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    l__mod___encoder_block_6_layer_0_self_attention_q = self.L__mod___encoder_block_6_layer_0_SelfAttention_q(normed_hidden_states_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_25 = l__mod___encoder_block_6_layer_0_self_attention_q.view(1, -1, 6, 64);  l__mod___encoder_block_6_layer_0_self_attention_q = None
    query_states_6 = view_25.transpose(1, 2);  view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    l__mod___encoder_block_6_layer_0_self_attention_k = self.L__mod___encoder_block_6_layer_0_SelfAttention_k(normed_hidden_states_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_26 = l__mod___encoder_block_6_layer_0_self_attention_k.view(1, -1, 6, 64);  l__mod___encoder_block_6_layer_0_self_attention_k = None
    key_states_6 = view_26.transpose(1, 2);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    l__mod___encoder_block_6_layer_0_self_attention_v = self.L__mod___encoder_block_6_layer_0_SelfAttention_v(normed_hidden_states_6);  normed_hidden_states_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_27 = l__mod___encoder_block_6_layer_0_self_attention_v.view(1, -1, 6, 64);  l__mod___encoder_block_6_layer_0_self_attention_v = None
    value_states_6 = view_27.transpose(1, 2);  view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    transpose_33 = key_states_6.transpose(3, 2);  key_states_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    scores_12 = torch.matmul(query_states_6, transpose_33);  query_states_6 = transpose_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    scores_12 += position_bias_9;  scores_13 = scores_12;  scores_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    float_8 = scores_13.float()
    softmax_6 = torch.nn.functional.softmax(float_8, dim = -1);  float_8 = None
    attn_weights_12 = softmax_6.type_as(scores_13);  softmax_6 = scores_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    attn_weights_13 = torch.nn.functional.dropout(attn_weights_12, p = 0.1, training = True);  attn_weights_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    matmul_13 = torch.matmul(attn_weights_13, value_states_6);  attn_weights_13 = value_states_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    transpose_34 = matmul_13.transpose(1, 2);  matmul_13 = None
    contiguous_6 = transpose_34.contiguous();  transpose_34 = None
    attn_output_12 = contiguous_6.view(1, -1, 384);  contiguous_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    attn_output_13 = self.L__mod___encoder_block_6_layer_0_SelfAttention_o(attn_output_12);  attn_output_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    l__mod___encoder_block_6_layer_0_dropout = self.L__mod___encoder_block_6_layer_0_dropout(attn_output_13);  attn_output_13 = None
    hidden_states_77 = hidden_states_72 + l__mod___encoder_block_6_layer_0_dropout;  hidden_states_72 = l__mod___encoder_block_6_layer_0_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_16 = hidden_states_77.to(torch.float32)
    pow_20 = to_16.pow(2);  to_16 = None
    variance_13 = pow_20.mean(-1, keepdim = True);  pow_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_41 = variance_13 + 1e-06;  variance_13 = None
    rsqrt_13 = torch.rsqrt(add_41);  add_41 = None
    hidden_states_78 = hidden_states_77 * rsqrt_13;  rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___encoder_block_6_layer_1_layer_norm_weight_1 = self.L__mod___encoder_block_6_layer_1_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    forwarded_states_12 = l__mod___encoder_block_6_layer_1_layer_norm_weight_1 * hidden_states_78;  l__mod___encoder_block_6_layer_1_layer_norm_weight_1 = hidden_states_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    l__mod___encoder_block_6_layer__1__dense_relu_dense_wi_0 = self.L__mod___encoder_block_6_layer__1__DenseReluDense_wi_0(forwarded_states_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_61 = 0.5 * l__mod___encoder_block_6_layer__1__dense_relu_dense_wi_0
    pow_21 = torch.pow(l__mod___encoder_block_6_layer__1__dense_relu_dense_wi_0, 3.0)
    mul_62 = 0.044715 * pow_21;  pow_21 = None
    add_42 = l__mod___encoder_block_6_layer__1__dense_relu_dense_wi_0 + mul_62;  l__mod___encoder_block_6_layer__1__dense_relu_dense_wi_0 = mul_62 = None
    mul_63 = 0.7978845608028654 * add_42;  add_42 = None
    tanh_6 = torch.tanh(mul_63);  mul_63 = None
    add_43 = 1.0 + tanh_6;  tanh_6 = None
    hidden_gelu_6 = mul_61 * add_43;  mul_61 = add_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    hidden_linear_6 = self.L__mod___encoder_block_6_layer__1__DenseReluDense_wi_1(forwarded_states_12);  forwarded_states_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    hidden_states_79 = hidden_gelu_6 * hidden_linear_6;  hidden_gelu_6 = hidden_linear_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    hidden_states_80 = self.L__mod___encoder_block_6_layer__1__DenseReluDense_dropout(hidden_states_79);  hidden_states_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    forwarded_states_13 = self.L__mod___encoder_block_6_layer__1__DenseReluDense_wo(hidden_states_80);  hidden_states_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    l__mod___encoder_block_6_layer__1__dropout = self.L__mod___encoder_block_6_layer__1__dropout(forwarded_states_13);  forwarded_states_13 = None
    hidden_states_84 = hidden_states_77 + l__mod___encoder_block_6_layer__1__dropout;  hidden_states_77 = l__mod___encoder_block_6_layer__1__dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_17 = hidden_states_84.to(torch.float32)
    pow_22 = to_17.pow(2);  to_17 = None
    variance_14 = pow_22.mean(-1, keepdim = True);  pow_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_45 = variance_14 + 1e-06;  variance_14 = None
    rsqrt_14 = torch.rsqrt(add_45);  add_45 = None
    hidden_states_85 = hidden_states_84 * rsqrt_14;  rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___encoder_block_7_layer_0_layer_norm_weight_1 = self.L__mod___encoder_block_7_layer_0_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    normed_hidden_states_7 = l__mod___encoder_block_7_layer_0_layer_norm_weight_1 * hidden_states_85;  l__mod___encoder_block_7_layer_0_layer_norm_weight_1 = hidden_states_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    l__mod___encoder_block_7_layer_0_self_attention_q = self.L__mod___encoder_block_7_layer_0_SelfAttention_q(normed_hidden_states_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_29 = l__mod___encoder_block_7_layer_0_self_attention_q.view(1, -1, 6, 64);  l__mod___encoder_block_7_layer_0_self_attention_q = None
    query_states_7 = view_29.transpose(1, 2);  view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    l__mod___encoder_block_7_layer_0_self_attention_k = self.L__mod___encoder_block_7_layer_0_SelfAttention_k(normed_hidden_states_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_30 = l__mod___encoder_block_7_layer_0_self_attention_k.view(1, -1, 6, 64);  l__mod___encoder_block_7_layer_0_self_attention_k = None
    key_states_7 = view_30.transpose(1, 2);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    l__mod___encoder_block_7_layer_0_self_attention_v = self.L__mod___encoder_block_7_layer_0_SelfAttention_v(normed_hidden_states_7);  normed_hidden_states_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_31 = l__mod___encoder_block_7_layer_0_self_attention_v.view(1, -1, 6, 64);  l__mod___encoder_block_7_layer_0_self_attention_v = None
    value_states_7 = view_31.transpose(1, 2);  view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    transpose_38 = key_states_7.transpose(3, 2);  key_states_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    scores_14 = torch.matmul(query_states_7, transpose_38);  query_states_7 = transpose_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    scores_14 += position_bias_9;  scores_15 = scores_14;  scores_14 = position_bias_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    float_9 = scores_15.float()
    softmax_7 = torch.nn.functional.softmax(float_9, dim = -1);  float_9 = None
    attn_weights_14 = softmax_7.type_as(scores_15);  softmax_7 = scores_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    attn_weights_15 = torch.nn.functional.dropout(attn_weights_14, p = 0.1, training = True);  attn_weights_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    matmul_15 = torch.matmul(attn_weights_15, value_states_7);  attn_weights_15 = value_states_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    transpose_39 = matmul_15.transpose(1, 2);  matmul_15 = None
    contiguous_7 = transpose_39.contiguous();  transpose_39 = None
    attn_output_14 = contiguous_7.view(1, -1, 384);  contiguous_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    attn_output_15 = self.L__mod___encoder_block_7_layer_0_SelfAttention_o(attn_output_14);  attn_output_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    l__mod___encoder_block_7_layer_0_dropout = self.L__mod___encoder_block_7_layer_0_dropout(attn_output_15);  attn_output_15 = None
    hidden_states_89 = hidden_states_84 + l__mod___encoder_block_7_layer_0_dropout;  hidden_states_84 = l__mod___encoder_block_7_layer_0_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_18 = hidden_states_89.to(torch.float32)
    pow_23 = to_18.pow(2);  to_18 = None
    variance_15 = pow_23.mean(-1, keepdim = True);  pow_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_47 = variance_15 + 1e-06;  variance_15 = None
    rsqrt_15 = torch.rsqrt(add_47);  add_47 = None
    hidden_states_90 = hidden_states_89 * rsqrt_15;  rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___encoder_block_7_layer_1_layer_norm_weight_1 = self.L__mod___encoder_block_7_layer_1_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    forwarded_states_14 = l__mod___encoder_block_7_layer_1_layer_norm_weight_1 * hidden_states_90;  l__mod___encoder_block_7_layer_1_layer_norm_weight_1 = hidden_states_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    l__mod___encoder_block_7_layer__1__dense_relu_dense_wi_0 = self.L__mod___encoder_block_7_layer__1__DenseReluDense_wi_0(forwarded_states_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_70 = 0.5 * l__mod___encoder_block_7_layer__1__dense_relu_dense_wi_0
    pow_24 = torch.pow(l__mod___encoder_block_7_layer__1__dense_relu_dense_wi_0, 3.0)
    mul_71 = 0.044715 * pow_24;  pow_24 = None
    add_48 = l__mod___encoder_block_7_layer__1__dense_relu_dense_wi_0 + mul_71;  l__mod___encoder_block_7_layer__1__dense_relu_dense_wi_0 = mul_71 = None
    mul_72 = 0.7978845608028654 * add_48;  add_48 = None
    tanh_7 = torch.tanh(mul_72);  mul_72 = None
    add_49 = 1.0 + tanh_7;  tanh_7 = None
    hidden_gelu_7 = mul_70 * add_49;  mul_70 = add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    hidden_linear_7 = self.L__mod___encoder_block_7_layer__1__DenseReluDense_wi_1(forwarded_states_14);  forwarded_states_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    hidden_states_91 = hidden_gelu_7 * hidden_linear_7;  hidden_gelu_7 = hidden_linear_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    hidden_states_92 = self.L__mod___encoder_block_7_layer__1__DenseReluDense_dropout(hidden_states_91);  hidden_states_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    forwarded_states_15 = self.L__mod___encoder_block_7_layer__1__DenseReluDense_wo(hidden_states_92);  hidden_states_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    l__mod___encoder_block_7_layer__1__dropout = self.L__mod___encoder_block_7_layer__1__dropout(forwarded_states_15);  forwarded_states_15 = None
    hidden_states_96 = hidden_states_89 + l__mod___encoder_block_7_layer__1__dropout;  hidden_states_89 = l__mod___encoder_block_7_layer__1__dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_19 = hidden_states_96.to(torch.float32)
    pow_25 = to_19.pow(2);  to_19 = None
    variance_16 = pow_25.mean(-1, keepdim = True);  pow_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_51 = variance_16 + 1e-06;  variance_16 = None
    rsqrt_16 = torch.rsqrt(add_51);  add_51 = None
    hidden_states_97 = hidden_states_96 * rsqrt_16;  hidden_states_96 = rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___encoder_final_layer_norm_weight_1 = self.L__mod___encoder_final_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    hidden_states_98 = l__mod___encoder_final_layer_norm_weight_1 * hidden_states_97;  l__mod___encoder_final_layer_norm_weight_1 = hidden_states_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:1139, code: hidden_states = self.dropout(hidden_states)
    hidden_states_100 = self.L__mod___encoder_dropout(hidden_states_98);  hidden_states_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:984, code: input_ids = input_ids.view(-1, input_shape[-1])
    input_ids_1 = l_cloned_inputs_decoder_input_ids_.view(-1, 128);  l_cloned_inputs_decoder_input_ids_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:994, code: inputs_embeds = self.embed_tokens(input_ids)
    inputs_embeds_1 = self.L__mod___encoder_embed_tokens(input_ids_1);  input_ids_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:1006, code: attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
    attention_mask_1 = torch.ones(1, 128, device = device(type='cpu'))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:1009, code: encoder_attention_mask = torch.ones(
    encoder_attention_mask = torch.ones(1, 128, device = device(type='cpu'), dtype = torch.int64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:860, code: seq_ids = torch.arange(seq_length, device=device)
    seq_ids = torch.arange(128, device = device(type='cpu'))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:861, code: causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
    getitem_3 = seq_ids[(None, None, slice(None, None, None))]
    repeat = getitem_3.repeat(1, 128, 1);  getitem_3 = None
    getitem_4 = seq_ids[(None, slice(None, None, None), None)];  seq_ids = None
    causal_mask = repeat <= getitem_4;  repeat = getitem_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:864, code: causal_mask = causal_mask.to(attention_mask.dtype)
    causal_mask_1 = causal_mask.to(torch.float32);  causal_mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:876, code: extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
    getitem_5 = causal_mask_1[(slice(None, None, None), None, slice(None, None, None), slice(None, None, None))];  causal_mask_1 = None
    getitem_6 = attention_mask_1[(slice(None, None, None), None, None, slice(None, None, None))];  attention_mask_1 = None
    extended_attention_mask_5 = getitem_5 * getitem_6;  getitem_5 = getitem_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:927, code: extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask_6 = extended_attention_mask_5.to(dtype = torch.float32);  extended_attention_mask_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:928, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    sub_2 = 1.0 - extended_attention_mask_6;  extended_attention_mask_6 = None
    extended_attention_mask_8 = sub_2 * -3.4028234663852886e+38;  sub_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:840, code: encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
    encoder_extended_attention_mask = encoder_attention_mask[(slice(None, None, None), None, None, slice(None, None, None))];  encoder_attention_mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:846, code: encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
    encoder_extended_attention_mask_1 = encoder_extended_attention_mask.to(dtype = torch.float32);  encoder_extended_attention_mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:847, code: encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(self.dtype).min
    sub_3 = 1.0 - encoder_extended_attention_mask_1;  encoder_extended_attention_mask_1 = None
    encoder_extended_attention_mask_3 = sub_3 * -3.4028234663852886e+38;  sub_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:1049, code: hidden_states = self.dropout(inputs_embeds)
    hidden_states_101 = self.L__mod___decoder_dropout(inputs_embeds_1);  inputs_embeds_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_23 = hidden_states_101.to(torch.float32)
    pow_26 = to_23.pow(2);  to_23 = None
    variance_17 = pow_26.mean(-1, keepdim = True);  pow_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_52 = variance_17 + 1e-06;  variance_17 = None
    rsqrt_17 = torch.rsqrt(add_52);  add_52 = None
    hidden_states_102 = hidden_states_101 * rsqrt_17;  rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___decoder_block_0_layer_0_layer_norm_weight_3 = self.L__mod___decoder_block_0_layer_0_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    normed_hidden_states_8 = l__mod___decoder_block_0_layer_0_layer_norm_weight_3 * hidden_states_102;  l__mod___decoder_block_0_layer_0_layer_norm_weight_3 = hidden_states_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    l__mod___decoder_block_0_layer_0_self_attention_q = self.L__mod___decoder_block_0_layer_0_SelfAttention_q(normed_hidden_states_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_34 = l__mod___decoder_block_0_layer_0_self_attention_q.view(1, -1, 6, 64);  l__mod___decoder_block_0_layer_0_self_attention_q = None
    query_states_8 = view_34.transpose(1, 2);  view_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    l__mod___decoder_block_0_layer_0_self_attention_k = self.L__mod___decoder_block_0_layer_0_SelfAttention_k(normed_hidden_states_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_35 = l__mod___decoder_block_0_layer_0_self_attention_k.view(1, -1, 6, 64);  l__mod___decoder_block_0_layer_0_self_attention_k = None
    key_states_8 = view_35.transpose(1, 2);  view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    l__mod___decoder_block_0_layer_0_self_attention_v = self.L__mod___decoder_block_0_layer_0_SelfAttention_v(normed_hidden_states_8);  normed_hidden_states_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_36 = l__mod___decoder_block_0_layer_0_self_attention_v.view(1, -1, 6, 64);  l__mod___decoder_block_0_layer_0_self_attention_v = None
    value_states_8 = view_36.transpose(1, 2);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    transpose_43 = key_states_8.transpose(3, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    scores_16 = torch.matmul(query_states_8, transpose_43);  query_states_8 = transpose_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:302, code: context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
    arange_3 = torch.arange(128, dtype = torch.int64, device = device(type='cpu'))
    context_position_1 = arange_3[(slice(None, None, None), None)];  arange_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:303, code: memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
    arange_4 = torch.arange(128, dtype = torch.int64, device = device(type='cpu'))
    memory_position_1 = arange_4[(None, slice(None, None, None))];  arange_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:304, code: relative_position = memory_position - context_position  # shape (query_length, key_length)
    relative_position_2 = memory_position_1 - context_position_1;  memory_position_1 = context_position_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:278, code: relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
    zeros_like = torch.zeros_like(relative_position_2)
    min_2 = torch.min(relative_position_2, zeros_like);  relative_position_2 = zeros_like = None
    relative_position_3 = -min_2;  min_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:283, code: is_small = relative_position < max_exact
    is_small_1 = relative_position_3 < 16
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:287, code: torch.log(relative_position.float() / max_exact)
    float_10 = relative_position_3.float()
    truediv_2 = float_10 / 16;  float_10 = None
    log_1 = torch.log(truediv_2);  truediv_2 = None
    truediv_3 = log_1 / 2.0794415416798357;  log_1 = None
    mul_82 = truediv_3 * 16;  truediv_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:290, code: ).to(torch.long)
    to_24 = mul_82.to(torch.int64);  mul_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:286, code: relative_position_if_large = max_exact + (
    relative_position_if_large_2 = 16 + to_24;  to_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:292, code: relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
    full_like_1 = torch.full_like(relative_position_if_large_2, 31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:291, code: relative_position_if_large = torch.min(
    relative_position_if_large_3 = torch.min(relative_position_if_large_2, full_like_1);  relative_position_if_large_2 = full_like_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:295, code: relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
    where_1 = torch.where(is_small_1, relative_position_3, relative_position_if_large_3);  is_small_1 = relative_position_3 = relative_position_if_large_3 = None
    relative_position_bucket_1 = 0 + where_1;  where_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:311, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
    values_2 = self.L__mod___decoder_block_0_layer_0_SelfAttention_relative_attention_bias(relative_position_bucket_1);  relative_position_bucket_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:312, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
    permute_1 = values_2.permute([2, 0, 1]);  values_2 = None
    position_bias_10 = permute_1.unsqueeze(0);  permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:413, code: position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)
    position_bias_21 = position_bias_10 + extended_attention_mask_8;  position_bias_10 = extended_attention_mask_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    scores_16 += position_bias_21;  scores_17 = scores_16;  scores_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    float_11 = scores_17.float()
    softmax_8 = torch.nn.functional.softmax(float_11, dim = -1);  float_11 = None
    attn_weights_16 = softmax_8.type_as(scores_17);  softmax_8 = scores_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    attn_weights_17 = torch.nn.functional.dropout(attn_weights_16, p = 0.1, training = True);  attn_weights_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    matmul_17 = torch.matmul(attn_weights_17, value_states_8);  attn_weights_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    transpose_44 = matmul_17.transpose(1, 2);  matmul_17 = None
    contiguous_8 = transpose_44.contiguous();  transpose_44 = None
    attn_output_16 = contiguous_8.view(1, -1, 384);  contiguous_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    attn_output_17 = self.L__mod___decoder_block_0_layer_0_SelfAttention_o(attn_output_16);  attn_output_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    l__mod___decoder_block_0_layer_0_dropout = self.L__mod___decoder_block_0_layer_0_dropout(attn_output_17);  attn_output_17 = None
    hidden_states_106 = hidden_states_101 + l__mod___decoder_block_0_layer_0_dropout;  hidden_states_101 = l__mod___decoder_block_0_layer_0_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_25 = hidden_states_106.to(torch.float32)
    pow_27 = to_25.pow(2);  to_25 = None
    variance_18 = pow_27.mean(-1, keepdim = True);  pow_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_57 = variance_18 + 1e-06;  variance_18 = None
    rsqrt_18 = torch.rsqrt(add_57);  add_57 = None
    hidden_states_107 = hidden_states_106 * rsqrt_18;  rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___decoder_block_0_layer_1_layer_norm_weight_3 = self.L__mod___decoder_block_0_layer_1_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    normed_hidden_states_9 = l__mod___decoder_block_0_layer_1_layer_norm_weight_3 * hidden_states_107;  l__mod___decoder_block_0_layer_1_layer_norm_weight_3 = hidden_states_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    l__mod___decoder_block_0_layer_1_enc_dec_attention_q = self.L__mod___decoder_block_0_layer_1_EncDecAttention_q(normed_hidden_states_9);  normed_hidden_states_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_38 = l__mod___decoder_block_0_layer_1_enc_dec_attention_q.view(1, -1, 6, 64);  l__mod___decoder_block_0_layer_1_enc_dec_attention_q = None
    query_states_9 = view_38.transpose(1, 2);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    l__mod___decoder_block_0_layer_1_enc_dec_attention_k = self.L__mod___decoder_block_0_layer_1_EncDecAttention_k(hidden_states_100)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_39 = l__mod___decoder_block_0_layer_1_enc_dec_attention_k.view(1, -1, 6, 64);  l__mod___decoder_block_0_layer_1_enc_dec_attention_k = None
    key_states_9 = view_39.transpose(1, 2);  view_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    l__mod___decoder_block_0_layer_1_enc_dec_attention_v = self.L__mod___decoder_block_0_layer_1_EncDecAttention_v(hidden_states_100)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_40 = l__mod___decoder_block_0_layer_1_enc_dec_attention_v.view(1, -1, 6, 64);  l__mod___decoder_block_0_layer_1_enc_dec_attention_v = None
    value_states_9 = view_40.transpose(1, 2);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    transpose_48 = key_states_9.transpose(3, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    scores_18 = torch.matmul(query_states_9, transpose_48);  query_states_9 = transpose_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:399, code: position_bias = torch.zeros(
    position_bias_12 = torch.zeros((1, 6, 128, 128), device = device(type='cpu'), dtype = torch.float32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:413, code: position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)
    encoder_decoder_position_bias_7 = position_bias_12 + encoder_extended_attention_mask_3;  position_bias_12 = encoder_extended_attention_mask_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    scores_18 += encoder_decoder_position_bias_7;  scores_19 = scores_18;  scores_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    float_12 = scores_19.float()
    softmax_9 = torch.nn.functional.softmax(float_12, dim = -1);  float_12 = None
    attn_weights_18 = softmax_9.type_as(scores_19);  softmax_9 = scores_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    attn_weights_19 = torch.nn.functional.dropout(attn_weights_18, p = 0.1, training = True);  attn_weights_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    matmul_19 = torch.matmul(attn_weights_19, value_states_9);  attn_weights_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    transpose_49 = matmul_19.transpose(1, 2);  matmul_19 = None
    contiguous_9 = transpose_49.contiguous();  transpose_49 = None
    attn_output_18 = contiguous_9.view(1, -1, 384);  contiguous_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    attn_output_19 = self.L__mod___decoder_block_0_layer_1_EncDecAttention_o(attn_output_18);  attn_output_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:510, code: layer_output = hidden_states + self.dropout(attention_output[0])
    l__mod___decoder_block_0_layer_1_dropout = self.L__mod___decoder_block_0_layer_1_dropout(attn_output_19);  attn_output_19 = None
    hidden_states_110 = hidden_states_106 + l__mod___decoder_block_0_layer_1_dropout;  hidden_states_106 = l__mod___decoder_block_0_layer_1_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_26 = hidden_states_110.to(torch.float32)
    pow_28 = to_26.pow(2);  to_26 = None
    variance_19 = pow_28.mean(-1, keepdim = True);  pow_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_60 = variance_19 + 1e-06;  variance_19 = None
    rsqrt_19 = torch.rsqrt(add_60);  add_60 = None
    hidden_states_111 = hidden_states_110 * rsqrt_19;  rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___decoder_block_0_layer_2_layer_norm_weight_3 = self.L__mod___decoder_block_0_layer_2_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    forwarded_states_16 = l__mod___decoder_block_0_layer_2_layer_norm_weight_3 * hidden_states_111;  l__mod___decoder_block_0_layer_2_layer_norm_weight_3 = hidden_states_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    l__mod___decoder_block_0_layer__1__dense_relu_dense_wi_0 = self.L__mod___decoder_block_0_layer__1__DenseReluDense_wi_0(forwarded_states_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_87 = 0.5 * l__mod___decoder_block_0_layer__1__dense_relu_dense_wi_0
    pow_29 = torch.pow(l__mod___decoder_block_0_layer__1__dense_relu_dense_wi_0, 3.0)
    mul_88 = 0.044715 * pow_29;  pow_29 = None
    add_61 = l__mod___decoder_block_0_layer__1__dense_relu_dense_wi_0 + mul_88;  l__mod___decoder_block_0_layer__1__dense_relu_dense_wi_0 = mul_88 = None
    mul_89 = 0.7978845608028654 * add_61;  add_61 = None
    tanh_8 = torch.tanh(mul_89);  mul_89 = None
    add_62 = 1.0 + tanh_8;  tanh_8 = None
    hidden_gelu_8 = mul_87 * add_62;  mul_87 = add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    hidden_linear_8 = self.L__mod___decoder_block_0_layer__1__DenseReluDense_wi_1(forwarded_states_16);  forwarded_states_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    hidden_states_112 = hidden_gelu_8 * hidden_linear_8;  hidden_gelu_8 = hidden_linear_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    hidden_states_113 = self.L__mod___decoder_block_0_layer__1__DenseReluDense_dropout(hidden_states_112);  hidden_states_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    forwarded_states_17 = self.L__mod___decoder_block_0_layer__1__DenseReluDense_wo(hidden_states_113);  hidden_states_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    l__mod___decoder_block_0_layer__1__dropout = self.L__mod___decoder_block_0_layer__1__dropout(forwarded_states_17);  forwarded_states_17 = None
    hidden_states_117 = hidden_states_110 + l__mod___decoder_block_0_layer__1__dropout;  hidden_states_110 = l__mod___decoder_block_0_layer__1__dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_27 = hidden_states_117.to(torch.float32)
    pow_30 = to_27.pow(2);  to_27 = None
    variance_20 = pow_30.mean(-1, keepdim = True);  pow_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_64 = variance_20 + 1e-06;  variance_20 = None
    rsqrt_20 = torch.rsqrt(add_64);  add_64 = None
    hidden_states_118 = hidden_states_117 * rsqrt_20;  rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___decoder_block_1_layer_0_layer_norm_weight_3 = self.L__mod___decoder_block_1_layer_0_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    normed_hidden_states_10 = l__mod___decoder_block_1_layer_0_layer_norm_weight_3 * hidden_states_118;  l__mod___decoder_block_1_layer_0_layer_norm_weight_3 = hidden_states_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    l__mod___decoder_block_1_layer_0_self_attention_q = self.L__mod___decoder_block_1_layer_0_SelfAttention_q(normed_hidden_states_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_42 = l__mod___decoder_block_1_layer_0_self_attention_q.view(1, -1, 6, 64);  l__mod___decoder_block_1_layer_0_self_attention_q = None
    query_states_10 = view_42.transpose(1, 2);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    l__mod___decoder_block_1_layer_0_self_attention_k = self.L__mod___decoder_block_1_layer_0_SelfAttention_k(normed_hidden_states_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_43 = l__mod___decoder_block_1_layer_0_self_attention_k.view(1, -1, 6, 64);  l__mod___decoder_block_1_layer_0_self_attention_k = None
    key_states_10 = view_43.transpose(1, 2);  view_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    l__mod___decoder_block_1_layer_0_self_attention_v = self.L__mod___decoder_block_1_layer_0_SelfAttention_v(normed_hidden_states_10);  normed_hidden_states_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_44 = l__mod___decoder_block_1_layer_0_self_attention_v.view(1, -1, 6, 64);  l__mod___decoder_block_1_layer_0_self_attention_v = None
    value_states_10 = view_44.transpose(1, 2);  view_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    transpose_53 = key_states_10.transpose(3, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    scores_20 = torch.matmul(query_states_10, transpose_53);  query_states_10 = transpose_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    scores_20 += position_bias_21;  scores_21 = scores_20;  scores_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    float_13 = scores_21.float()
    softmax_10 = torch.nn.functional.softmax(float_13, dim = -1);  float_13 = None
    attn_weights_20 = softmax_10.type_as(scores_21);  softmax_10 = scores_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    attn_weights_21 = torch.nn.functional.dropout(attn_weights_20, p = 0.1, training = True);  attn_weights_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    matmul_21 = torch.matmul(attn_weights_21, value_states_10);  attn_weights_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    transpose_54 = matmul_21.transpose(1, 2);  matmul_21 = None
    contiguous_10 = transpose_54.contiguous();  transpose_54 = None
    attn_output_20 = contiguous_10.view(1, -1, 384);  contiguous_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    attn_output_21 = self.L__mod___decoder_block_1_layer_0_SelfAttention_o(attn_output_20);  attn_output_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    l__mod___decoder_block_1_layer_0_dropout = self.L__mod___decoder_block_1_layer_0_dropout(attn_output_21);  attn_output_21 = None
    hidden_states_122 = hidden_states_117 + l__mod___decoder_block_1_layer_0_dropout;  hidden_states_117 = l__mod___decoder_block_1_layer_0_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_28 = hidden_states_122.to(torch.float32)
    pow_31 = to_28.pow(2);  to_28 = None
    variance_21 = pow_31.mean(-1, keepdim = True);  pow_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_66 = variance_21 + 1e-06;  variance_21 = None
    rsqrt_21 = torch.rsqrt(add_66);  add_66 = None
    hidden_states_123 = hidden_states_122 * rsqrt_21;  rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___decoder_block_1_layer_1_layer_norm_weight_3 = self.L__mod___decoder_block_1_layer_1_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    normed_hidden_states_11 = l__mod___decoder_block_1_layer_1_layer_norm_weight_3 * hidden_states_123;  l__mod___decoder_block_1_layer_1_layer_norm_weight_3 = hidden_states_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    l__mod___decoder_block_1_layer_1_enc_dec_attention_q = self.L__mod___decoder_block_1_layer_1_EncDecAttention_q(normed_hidden_states_11);  normed_hidden_states_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_46 = l__mod___decoder_block_1_layer_1_enc_dec_attention_q.view(1, -1, 6, 64);  l__mod___decoder_block_1_layer_1_enc_dec_attention_q = None
    query_states_11 = view_46.transpose(1, 2);  view_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    l__mod___decoder_block_1_layer_1_enc_dec_attention_k = self.L__mod___decoder_block_1_layer_1_EncDecAttention_k(hidden_states_100)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_47 = l__mod___decoder_block_1_layer_1_enc_dec_attention_k.view(1, -1, 6, 64);  l__mod___decoder_block_1_layer_1_enc_dec_attention_k = None
    key_states_11 = view_47.transpose(1, 2);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    l__mod___decoder_block_1_layer_1_enc_dec_attention_v = self.L__mod___decoder_block_1_layer_1_EncDecAttention_v(hidden_states_100)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_48 = l__mod___decoder_block_1_layer_1_enc_dec_attention_v.view(1, -1, 6, 64);  l__mod___decoder_block_1_layer_1_enc_dec_attention_v = None
    value_states_11 = view_48.transpose(1, 2);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    transpose_58 = key_states_11.transpose(3, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    scores_22 = torch.matmul(query_states_11, transpose_58);  query_states_11 = transpose_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    scores_22 += encoder_decoder_position_bias_7;  scores_23 = scores_22;  scores_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    float_14 = scores_23.float()
    softmax_11 = torch.nn.functional.softmax(float_14, dim = -1);  float_14 = None
    attn_weights_22 = softmax_11.type_as(scores_23);  softmax_11 = scores_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    attn_weights_23 = torch.nn.functional.dropout(attn_weights_22, p = 0.1, training = True);  attn_weights_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    matmul_23 = torch.matmul(attn_weights_23, value_states_11);  attn_weights_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    transpose_59 = matmul_23.transpose(1, 2);  matmul_23 = None
    contiguous_11 = transpose_59.contiguous();  transpose_59 = None
    attn_output_22 = contiguous_11.view(1, -1, 384);  contiguous_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    attn_output_23 = self.L__mod___decoder_block_1_layer_1_EncDecAttention_o(attn_output_22);  attn_output_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:510, code: layer_output = hidden_states + self.dropout(attention_output[0])
    l__mod___decoder_block_1_layer_1_dropout = self.L__mod___decoder_block_1_layer_1_dropout(attn_output_23);  attn_output_23 = None
    hidden_states_126 = hidden_states_122 + l__mod___decoder_block_1_layer_1_dropout;  hidden_states_122 = l__mod___decoder_block_1_layer_1_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_29 = hidden_states_126.to(torch.float32)
    pow_32 = to_29.pow(2);  to_29 = None
    variance_22 = pow_32.mean(-1, keepdim = True);  pow_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_68 = variance_22 + 1e-06;  variance_22 = None
    rsqrt_22 = torch.rsqrt(add_68);  add_68 = None
    hidden_states_127 = hidden_states_126 * rsqrt_22;  rsqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___decoder_block_1_layer_2_layer_norm_weight_3 = self.L__mod___decoder_block_1_layer_2_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    forwarded_states_18 = l__mod___decoder_block_1_layer_2_layer_norm_weight_3 * hidden_states_127;  l__mod___decoder_block_1_layer_2_layer_norm_weight_3 = hidden_states_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    l__mod___decoder_block_1_layer__1__dense_relu_dense_wi_0 = self.L__mod___decoder_block_1_layer__1__DenseReluDense_wi_0(forwarded_states_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_98 = 0.5 * l__mod___decoder_block_1_layer__1__dense_relu_dense_wi_0
    pow_33 = torch.pow(l__mod___decoder_block_1_layer__1__dense_relu_dense_wi_0, 3.0)
    mul_99 = 0.044715 * pow_33;  pow_33 = None
    add_69 = l__mod___decoder_block_1_layer__1__dense_relu_dense_wi_0 + mul_99;  l__mod___decoder_block_1_layer__1__dense_relu_dense_wi_0 = mul_99 = None
    mul_100 = 0.7978845608028654 * add_69;  add_69 = None
    tanh_9 = torch.tanh(mul_100);  mul_100 = None
    add_70 = 1.0 + tanh_9;  tanh_9 = None
    hidden_gelu_9 = mul_98 * add_70;  mul_98 = add_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    hidden_linear_9 = self.L__mod___decoder_block_1_layer__1__DenseReluDense_wi_1(forwarded_states_18);  forwarded_states_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    hidden_states_128 = hidden_gelu_9 * hidden_linear_9;  hidden_gelu_9 = hidden_linear_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    hidden_states_129 = self.L__mod___decoder_block_1_layer__1__DenseReluDense_dropout(hidden_states_128);  hidden_states_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    forwarded_states_19 = self.L__mod___decoder_block_1_layer__1__DenseReluDense_wo(hidden_states_129);  hidden_states_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    l__mod___decoder_block_1_layer__1__dropout = self.L__mod___decoder_block_1_layer__1__dropout(forwarded_states_19);  forwarded_states_19 = None
    hidden_states_133 = hidden_states_126 + l__mod___decoder_block_1_layer__1__dropout;  hidden_states_126 = l__mod___decoder_block_1_layer__1__dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_30 = hidden_states_133.to(torch.float32)
    pow_34 = to_30.pow(2);  to_30 = None
    variance_23 = pow_34.mean(-1, keepdim = True);  pow_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_72 = variance_23 + 1e-06;  variance_23 = None
    rsqrt_23 = torch.rsqrt(add_72);  add_72 = None
    hidden_states_134 = hidden_states_133 * rsqrt_23;  rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___decoder_block_2_layer_0_layer_norm_weight_3 = self.L__mod___decoder_block_2_layer_0_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    normed_hidden_states_12 = l__mod___decoder_block_2_layer_0_layer_norm_weight_3 * hidden_states_134;  l__mod___decoder_block_2_layer_0_layer_norm_weight_3 = hidden_states_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    l__mod___decoder_block_2_layer_0_self_attention_q = self.L__mod___decoder_block_2_layer_0_SelfAttention_q(normed_hidden_states_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_50 = l__mod___decoder_block_2_layer_0_self_attention_q.view(1, -1, 6, 64);  l__mod___decoder_block_2_layer_0_self_attention_q = None
    query_states_12 = view_50.transpose(1, 2);  view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    l__mod___decoder_block_2_layer_0_self_attention_k = self.L__mod___decoder_block_2_layer_0_SelfAttention_k(normed_hidden_states_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_51 = l__mod___decoder_block_2_layer_0_self_attention_k.view(1, -1, 6, 64);  l__mod___decoder_block_2_layer_0_self_attention_k = None
    key_states_12 = view_51.transpose(1, 2);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    l__mod___decoder_block_2_layer_0_self_attention_v = self.L__mod___decoder_block_2_layer_0_SelfAttention_v(normed_hidden_states_12);  normed_hidden_states_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_52 = l__mod___decoder_block_2_layer_0_self_attention_v.view(1, -1, 6, 64);  l__mod___decoder_block_2_layer_0_self_attention_v = None
    value_states_12 = view_52.transpose(1, 2);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    transpose_63 = key_states_12.transpose(3, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    scores_24 = torch.matmul(query_states_12, transpose_63);  query_states_12 = transpose_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    scores_24 += position_bias_21;  scores_25 = scores_24;  scores_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    float_15 = scores_25.float()
    softmax_12 = torch.nn.functional.softmax(float_15, dim = -1);  float_15 = None
    attn_weights_24 = softmax_12.type_as(scores_25);  softmax_12 = scores_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    attn_weights_25 = torch.nn.functional.dropout(attn_weights_24, p = 0.1, training = True);  attn_weights_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    matmul_25 = torch.matmul(attn_weights_25, value_states_12);  attn_weights_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    transpose_64 = matmul_25.transpose(1, 2);  matmul_25 = None
    contiguous_12 = transpose_64.contiguous();  transpose_64 = None
    attn_output_24 = contiguous_12.view(1, -1, 384);  contiguous_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    attn_output_25 = self.L__mod___decoder_block_2_layer_0_SelfAttention_o(attn_output_24);  attn_output_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    l__mod___decoder_block_2_layer_0_dropout = self.L__mod___decoder_block_2_layer_0_dropout(attn_output_25);  attn_output_25 = None
    hidden_states_138 = hidden_states_133 + l__mod___decoder_block_2_layer_0_dropout;  hidden_states_133 = l__mod___decoder_block_2_layer_0_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_31 = hidden_states_138.to(torch.float32)
    pow_35 = to_31.pow(2);  to_31 = None
    variance_24 = pow_35.mean(-1, keepdim = True);  pow_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_74 = variance_24 + 1e-06;  variance_24 = None
    rsqrt_24 = torch.rsqrt(add_74);  add_74 = None
    hidden_states_139 = hidden_states_138 * rsqrt_24;  rsqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___decoder_block_2_layer_1_layer_norm_weight_3 = self.L__mod___decoder_block_2_layer_1_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    normed_hidden_states_13 = l__mod___decoder_block_2_layer_1_layer_norm_weight_3 * hidden_states_139;  l__mod___decoder_block_2_layer_1_layer_norm_weight_3 = hidden_states_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    l__mod___decoder_block_2_layer_1_enc_dec_attention_q = self.L__mod___decoder_block_2_layer_1_EncDecAttention_q(normed_hidden_states_13);  normed_hidden_states_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_54 = l__mod___decoder_block_2_layer_1_enc_dec_attention_q.view(1, -1, 6, 64);  l__mod___decoder_block_2_layer_1_enc_dec_attention_q = None
    query_states_13 = view_54.transpose(1, 2);  view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    l__mod___decoder_block_2_layer_1_enc_dec_attention_k = self.L__mod___decoder_block_2_layer_1_EncDecAttention_k(hidden_states_100)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_55 = l__mod___decoder_block_2_layer_1_enc_dec_attention_k.view(1, -1, 6, 64);  l__mod___decoder_block_2_layer_1_enc_dec_attention_k = None
    key_states_13 = view_55.transpose(1, 2);  view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    l__mod___decoder_block_2_layer_1_enc_dec_attention_v = self.L__mod___decoder_block_2_layer_1_EncDecAttention_v(hidden_states_100)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_56 = l__mod___decoder_block_2_layer_1_enc_dec_attention_v.view(1, -1, 6, 64);  l__mod___decoder_block_2_layer_1_enc_dec_attention_v = None
    value_states_13 = view_56.transpose(1, 2);  view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    transpose_68 = key_states_13.transpose(3, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    scores_26 = torch.matmul(query_states_13, transpose_68);  query_states_13 = transpose_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    scores_26 += encoder_decoder_position_bias_7;  scores_27 = scores_26;  scores_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    float_16 = scores_27.float()
    softmax_13 = torch.nn.functional.softmax(float_16, dim = -1);  float_16 = None
    attn_weights_26 = softmax_13.type_as(scores_27);  softmax_13 = scores_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    attn_weights_27 = torch.nn.functional.dropout(attn_weights_26, p = 0.1, training = True);  attn_weights_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    matmul_27 = torch.matmul(attn_weights_27, value_states_13);  attn_weights_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    transpose_69 = matmul_27.transpose(1, 2);  matmul_27 = None
    contiguous_13 = transpose_69.contiguous();  transpose_69 = None
    attn_output_26 = contiguous_13.view(1, -1, 384);  contiguous_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    attn_output_27 = self.L__mod___decoder_block_2_layer_1_EncDecAttention_o(attn_output_26);  attn_output_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:510, code: layer_output = hidden_states + self.dropout(attention_output[0])
    l__mod___decoder_block_2_layer_1_dropout = self.L__mod___decoder_block_2_layer_1_dropout(attn_output_27);  attn_output_27 = None
    hidden_states_142 = hidden_states_138 + l__mod___decoder_block_2_layer_1_dropout;  hidden_states_138 = l__mod___decoder_block_2_layer_1_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_32 = hidden_states_142.to(torch.float32)
    pow_36 = to_32.pow(2);  to_32 = None
    variance_25 = pow_36.mean(-1, keepdim = True);  pow_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_76 = variance_25 + 1e-06;  variance_25 = None
    rsqrt_25 = torch.rsqrt(add_76);  add_76 = None
    hidden_states_143 = hidden_states_142 * rsqrt_25;  rsqrt_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___decoder_block_2_layer_2_layer_norm_weight_3 = self.L__mod___decoder_block_2_layer_2_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    forwarded_states_20 = l__mod___decoder_block_2_layer_2_layer_norm_weight_3 * hidden_states_143;  l__mod___decoder_block_2_layer_2_layer_norm_weight_3 = hidden_states_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    l__mod___decoder_block_2_layer__1__dense_relu_dense_wi_0 = self.L__mod___decoder_block_2_layer__1__DenseReluDense_wi_0(forwarded_states_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_109 = 0.5 * l__mod___decoder_block_2_layer__1__dense_relu_dense_wi_0
    pow_37 = torch.pow(l__mod___decoder_block_2_layer__1__dense_relu_dense_wi_0, 3.0)
    mul_110 = 0.044715 * pow_37;  pow_37 = None
    add_77 = l__mod___decoder_block_2_layer__1__dense_relu_dense_wi_0 + mul_110;  l__mod___decoder_block_2_layer__1__dense_relu_dense_wi_0 = mul_110 = None
    mul_111 = 0.7978845608028654 * add_77;  add_77 = None
    tanh_10 = torch.tanh(mul_111);  mul_111 = None
    add_78 = 1.0 + tanh_10;  tanh_10 = None
    hidden_gelu_10 = mul_109 * add_78;  mul_109 = add_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    hidden_linear_10 = self.L__mod___decoder_block_2_layer__1__DenseReluDense_wi_1(forwarded_states_20);  forwarded_states_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    hidden_states_144 = hidden_gelu_10 * hidden_linear_10;  hidden_gelu_10 = hidden_linear_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    hidden_states_145 = self.L__mod___decoder_block_2_layer__1__DenseReluDense_dropout(hidden_states_144);  hidden_states_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    forwarded_states_21 = self.L__mod___decoder_block_2_layer__1__DenseReluDense_wo(hidden_states_145);  hidden_states_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    l__mod___decoder_block_2_layer__1__dropout = self.L__mod___decoder_block_2_layer__1__dropout(forwarded_states_21);  forwarded_states_21 = None
    hidden_states_149 = hidden_states_142 + l__mod___decoder_block_2_layer__1__dropout;  hidden_states_142 = l__mod___decoder_block_2_layer__1__dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_33 = hidden_states_149.to(torch.float32)
    pow_38 = to_33.pow(2);  to_33 = None
    variance_26 = pow_38.mean(-1, keepdim = True);  pow_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_80 = variance_26 + 1e-06;  variance_26 = None
    rsqrt_26 = torch.rsqrt(add_80);  add_80 = None
    hidden_states_150 = hidden_states_149 * rsqrt_26;  rsqrt_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___decoder_block_3_layer_0_layer_norm_weight_3 = self.L__mod___decoder_block_3_layer_0_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    normed_hidden_states_14 = l__mod___decoder_block_3_layer_0_layer_norm_weight_3 * hidden_states_150;  l__mod___decoder_block_3_layer_0_layer_norm_weight_3 = hidden_states_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    l__mod___decoder_block_3_layer_0_self_attention_q = self.L__mod___decoder_block_3_layer_0_SelfAttention_q(normed_hidden_states_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_58 = l__mod___decoder_block_3_layer_0_self_attention_q.view(1, -1, 6, 64);  l__mod___decoder_block_3_layer_0_self_attention_q = None
    query_states_14 = view_58.transpose(1, 2);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    l__mod___decoder_block_3_layer_0_self_attention_k = self.L__mod___decoder_block_3_layer_0_SelfAttention_k(normed_hidden_states_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_59 = l__mod___decoder_block_3_layer_0_self_attention_k.view(1, -1, 6, 64);  l__mod___decoder_block_3_layer_0_self_attention_k = None
    key_states_14 = view_59.transpose(1, 2);  view_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    l__mod___decoder_block_3_layer_0_self_attention_v = self.L__mod___decoder_block_3_layer_0_SelfAttention_v(normed_hidden_states_14);  normed_hidden_states_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_60 = l__mod___decoder_block_3_layer_0_self_attention_v.view(1, -1, 6, 64);  l__mod___decoder_block_3_layer_0_self_attention_v = None
    value_states_14 = view_60.transpose(1, 2);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    transpose_73 = key_states_14.transpose(3, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    scores_28 = torch.matmul(query_states_14, transpose_73);  query_states_14 = transpose_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    scores_28 += position_bias_21;  scores_29 = scores_28;  scores_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    float_17 = scores_29.float()
    softmax_14 = torch.nn.functional.softmax(float_17, dim = -1);  float_17 = None
    attn_weights_28 = softmax_14.type_as(scores_29);  softmax_14 = scores_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    attn_weights_29 = torch.nn.functional.dropout(attn_weights_28, p = 0.1, training = True);  attn_weights_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    matmul_29 = torch.matmul(attn_weights_29, value_states_14);  attn_weights_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    transpose_74 = matmul_29.transpose(1, 2);  matmul_29 = None
    contiguous_14 = transpose_74.contiguous();  transpose_74 = None
    attn_output_28 = contiguous_14.view(1, -1, 384);  contiguous_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    attn_output_29 = self.L__mod___decoder_block_3_layer_0_SelfAttention_o(attn_output_28);  attn_output_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    l__mod___decoder_block_3_layer_0_dropout = self.L__mod___decoder_block_3_layer_0_dropout(attn_output_29);  attn_output_29 = None
    hidden_states_154 = hidden_states_149 + l__mod___decoder_block_3_layer_0_dropout;  hidden_states_149 = l__mod___decoder_block_3_layer_0_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_34 = hidden_states_154.to(torch.float32)
    pow_39 = to_34.pow(2);  to_34 = None
    variance_27 = pow_39.mean(-1, keepdim = True);  pow_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_82 = variance_27 + 1e-06;  variance_27 = None
    rsqrt_27 = torch.rsqrt(add_82);  add_82 = None
    hidden_states_155 = hidden_states_154 * rsqrt_27;  rsqrt_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___decoder_block_3_layer_1_layer_norm_weight_3 = self.L__mod___decoder_block_3_layer_1_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    normed_hidden_states_15 = l__mod___decoder_block_3_layer_1_layer_norm_weight_3 * hidden_states_155;  l__mod___decoder_block_3_layer_1_layer_norm_weight_3 = hidden_states_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    l__mod___decoder_block_3_layer_1_enc_dec_attention_q = self.L__mod___decoder_block_3_layer_1_EncDecAttention_q(normed_hidden_states_15);  normed_hidden_states_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_62 = l__mod___decoder_block_3_layer_1_enc_dec_attention_q.view(1, -1, 6, 64);  l__mod___decoder_block_3_layer_1_enc_dec_attention_q = None
    query_states_15 = view_62.transpose(1, 2);  view_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    l__mod___decoder_block_3_layer_1_enc_dec_attention_k = self.L__mod___decoder_block_3_layer_1_EncDecAttention_k(hidden_states_100)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_63 = l__mod___decoder_block_3_layer_1_enc_dec_attention_k.view(1, -1, 6, 64);  l__mod___decoder_block_3_layer_1_enc_dec_attention_k = None
    key_states_15 = view_63.transpose(1, 2);  view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    l__mod___decoder_block_3_layer_1_enc_dec_attention_v = self.L__mod___decoder_block_3_layer_1_EncDecAttention_v(hidden_states_100)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_64 = l__mod___decoder_block_3_layer_1_enc_dec_attention_v.view(1, -1, 6, 64);  l__mod___decoder_block_3_layer_1_enc_dec_attention_v = None
    value_states_15 = view_64.transpose(1, 2);  view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    transpose_78 = key_states_15.transpose(3, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    scores_30 = torch.matmul(query_states_15, transpose_78);  query_states_15 = transpose_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    scores_30 += encoder_decoder_position_bias_7;  scores_31 = scores_30;  scores_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    float_18 = scores_31.float()
    softmax_15 = torch.nn.functional.softmax(float_18, dim = -1);  float_18 = None
    attn_weights_30 = softmax_15.type_as(scores_31);  softmax_15 = scores_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    attn_weights_31 = torch.nn.functional.dropout(attn_weights_30, p = 0.1, training = True);  attn_weights_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    matmul_31 = torch.matmul(attn_weights_31, value_states_15);  attn_weights_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    transpose_79 = matmul_31.transpose(1, 2);  matmul_31 = None
    contiguous_15 = transpose_79.contiguous();  transpose_79 = None
    attn_output_30 = contiguous_15.view(1, -1, 384);  contiguous_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    attn_output_31 = self.L__mod___decoder_block_3_layer_1_EncDecAttention_o(attn_output_30);  attn_output_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:510, code: layer_output = hidden_states + self.dropout(attention_output[0])
    l__mod___decoder_block_3_layer_1_dropout = self.L__mod___decoder_block_3_layer_1_dropout(attn_output_31);  attn_output_31 = None
    hidden_states_158 = hidden_states_154 + l__mod___decoder_block_3_layer_1_dropout;  hidden_states_154 = l__mod___decoder_block_3_layer_1_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_35 = hidden_states_158.to(torch.float32)
    pow_40 = to_35.pow(2);  to_35 = None
    variance_28 = pow_40.mean(-1, keepdim = True);  pow_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_84 = variance_28 + 1e-06;  variance_28 = None
    rsqrt_28 = torch.rsqrt(add_84);  add_84 = None
    hidden_states_159 = hidden_states_158 * rsqrt_28;  rsqrt_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___decoder_block_3_layer_2_layer_norm_weight_3 = self.L__mod___decoder_block_3_layer_2_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    forwarded_states_22 = l__mod___decoder_block_3_layer_2_layer_norm_weight_3 * hidden_states_159;  l__mod___decoder_block_3_layer_2_layer_norm_weight_3 = hidden_states_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    l__mod___decoder_block_3_layer__1__dense_relu_dense_wi_0 = self.L__mod___decoder_block_3_layer__1__DenseReluDense_wi_0(forwarded_states_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_120 = 0.5 * l__mod___decoder_block_3_layer__1__dense_relu_dense_wi_0
    pow_41 = torch.pow(l__mod___decoder_block_3_layer__1__dense_relu_dense_wi_0, 3.0)
    mul_121 = 0.044715 * pow_41;  pow_41 = None
    add_85 = l__mod___decoder_block_3_layer__1__dense_relu_dense_wi_0 + mul_121;  l__mod___decoder_block_3_layer__1__dense_relu_dense_wi_0 = mul_121 = None
    mul_122 = 0.7978845608028654 * add_85;  add_85 = None
    tanh_11 = torch.tanh(mul_122);  mul_122 = None
    add_86 = 1.0 + tanh_11;  tanh_11 = None
    hidden_gelu_11 = mul_120 * add_86;  mul_120 = add_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    hidden_linear_11 = self.L__mod___decoder_block_3_layer__1__DenseReluDense_wi_1(forwarded_states_22);  forwarded_states_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    hidden_states_160 = hidden_gelu_11 * hidden_linear_11;  hidden_gelu_11 = hidden_linear_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    hidden_states_161 = self.L__mod___decoder_block_3_layer__1__DenseReluDense_dropout(hidden_states_160);  hidden_states_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    forwarded_states_23 = self.L__mod___decoder_block_3_layer__1__DenseReluDense_wo(hidden_states_161);  hidden_states_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    l__mod___decoder_block_3_layer__1__dropout = self.L__mod___decoder_block_3_layer__1__dropout(forwarded_states_23);  forwarded_states_23 = None
    hidden_states_165 = hidden_states_158 + l__mod___decoder_block_3_layer__1__dropout;  hidden_states_158 = l__mod___decoder_block_3_layer__1__dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_36 = hidden_states_165.to(torch.float32)
    pow_42 = to_36.pow(2);  to_36 = None
    variance_29 = pow_42.mean(-1, keepdim = True);  pow_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_88 = variance_29 + 1e-06;  variance_29 = None
    rsqrt_29 = torch.rsqrt(add_88);  add_88 = None
    hidden_states_166 = hidden_states_165 * rsqrt_29;  rsqrt_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___decoder_block_4_layer_0_layer_norm_weight_3 = self.L__mod___decoder_block_4_layer_0_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    normed_hidden_states_16 = l__mod___decoder_block_4_layer_0_layer_norm_weight_3 * hidden_states_166;  l__mod___decoder_block_4_layer_0_layer_norm_weight_3 = hidden_states_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    l__mod___decoder_block_4_layer_0_self_attention_q = self.L__mod___decoder_block_4_layer_0_SelfAttention_q(normed_hidden_states_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_66 = l__mod___decoder_block_4_layer_0_self_attention_q.view(1, -1, 6, 64);  l__mod___decoder_block_4_layer_0_self_attention_q = None
    query_states_16 = view_66.transpose(1, 2);  view_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    l__mod___decoder_block_4_layer_0_self_attention_k = self.L__mod___decoder_block_4_layer_0_SelfAttention_k(normed_hidden_states_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_67 = l__mod___decoder_block_4_layer_0_self_attention_k.view(1, -1, 6, 64);  l__mod___decoder_block_4_layer_0_self_attention_k = None
    key_states_16 = view_67.transpose(1, 2);  view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    l__mod___decoder_block_4_layer_0_self_attention_v = self.L__mod___decoder_block_4_layer_0_SelfAttention_v(normed_hidden_states_16);  normed_hidden_states_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_68 = l__mod___decoder_block_4_layer_0_self_attention_v.view(1, -1, 6, 64);  l__mod___decoder_block_4_layer_0_self_attention_v = None
    value_states_16 = view_68.transpose(1, 2);  view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    transpose_83 = key_states_16.transpose(3, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    scores_32 = torch.matmul(query_states_16, transpose_83);  query_states_16 = transpose_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    scores_32 += position_bias_21;  scores_33 = scores_32;  scores_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    float_19 = scores_33.float()
    softmax_16 = torch.nn.functional.softmax(float_19, dim = -1);  float_19 = None
    attn_weights_32 = softmax_16.type_as(scores_33);  softmax_16 = scores_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    attn_weights_33 = torch.nn.functional.dropout(attn_weights_32, p = 0.1, training = True);  attn_weights_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    matmul_33 = torch.matmul(attn_weights_33, value_states_16);  attn_weights_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    transpose_84 = matmul_33.transpose(1, 2);  matmul_33 = None
    contiguous_16 = transpose_84.contiguous();  transpose_84 = None
    attn_output_32 = contiguous_16.view(1, -1, 384);  contiguous_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    attn_output_33 = self.L__mod___decoder_block_4_layer_0_SelfAttention_o(attn_output_32);  attn_output_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    l__mod___decoder_block_4_layer_0_dropout = self.L__mod___decoder_block_4_layer_0_dropout(attn_output_33);  attn_output_33 = None
    hidden_states_170 = hidden_states_165 + l__mod___decoder_block_4_layer_0_dropout;  hidden_states_165 = l__mod___decoder_block_4_layer_0_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_37 = hidden_states_170.to(torch.float32)
    pow_43 = to_37.pow(2);  to_37 = None
    variance_30 = pow_43.mean(-1, keepdim = True);  pow_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_90 = variance_30 + 1e-06;  variance_30 = None
    rsqrt_30 = torch.rsqrt(add_90);  add_90 = None
    hidden_states_171 = hidden_states_170 * rsqrt_30;  rsqrt_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___decoder_block_4_layer_1_layer_norm_weight_3 = self.L__mod___decoder_block_4_layer_1_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    normed_hidden_states_17 = l__mod___decoder_block_4_layer_1_layer_norm_weight_3 * hidden_states_171;  l__mod___decoder_block_4_layer_1_layer_norm_weight_3 = hidden_states_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    l__mod___decoder_block_4_layer_1_enc_dec_attention_q = self.L__mod___decoder_block_4_layer_1_EncDecAttention_q(normed_hidden_states_17);  normed_hidden_states_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_70 = l__mod___decoder_block_4_layer_1_enc_dec_attention_q.view(1, -1, 6, 64);  l__mod___decoder_block_4_layer_1_enc_dec_attention_q = None
    query_states_17 = view_70.transpose(1, 2);  view_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    l__mod___decoder_block_4_layer_1_enc_dec_attention_k = self.L__mod___decoder_block_4_layer_1_EncDecAttention_k(hidden_states_100)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_71 = l__mod___decoder_block_4_layer_1_enc_dec_attention_k.view(1, -1, 6, 64);  l__mod___decoder_block_4_layer_1_enc_dec_attention_k = None
    key_states_17 = view_71.transpose(1, 2);  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    l__mod___decoder_block_4_layer_1_enc_dec_attention_v = self.L__mod___decoder_block_4_layer_1_EncDecAttention_v(hidden_states_100)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_72 = l__mod___decoder_block_4_layer_1_enc_dec_attention_v.view(1, -1, 6, 64);  l__mod___decoder_block_4_layer_1_enc_dec_attention_v = None
    value_states_17 = view_72.transpose(1, 2);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    transpose_88 = key_states_17.transpose(3, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    scores_34 = torch.matmul(query_states_17, transpose_88);  query_states_17 = transpose_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    scores_34 += encoder_decoder_position_bias_7;  scores_35 = scores_34;  scores_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    float_20 = scores_35.float()
    softmax_17 = torch.nn.functional.softmax(float_20, dim = -1);  float_20 = None
    attn_weights_34 = softmax_17.type_as(scores_35);  softmax_17 = scores_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    attn_weights_35 = torch.nn.functional.dropout(attn_weights_34, p = 0.1, training = True);  attn_weights_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    matmul_35 = torch.matmul(attn_weights_35, value_states_17);  attn_weights_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    transpose_89 = matmul_35.transpose(1, 2);  matmul_35 = None
    contiguous_17 = transpose_89.contiguous();  transpose_89 = None
    attn_output_34 = contiguous_17.view(1, -1, 384);  contiguous_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    attn_output_35 = self.L__mod___decoder_block_4_layer_1_EncDecAttention_o(attn_output_34);  attn_output_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:510, code: layer_output = hidden_states + self.dropout(attention_output[0])
    l__mod___decoder_block_4_layer_1_dropout = self.L__mod___decoder_block_4_layer_1_dropout(attn_output_35);  attn_output_35 = None
    hidden_states_174 = hidden_states_170 + l__mod___decoder_block_4_layer_1_dropout;  hidden_states_170 = l__mod___decoder_block_4_layer_1_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_38 = hidden_states_174.to(torch.float32)
    pow_44 = to_38.pow(2);  to_38 = None
    variance_31 = pow_44.mean(-1, keepdim = True);  pow_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_92 = variance_31 + 1e-06;  variance_31 = None
    rsqrt_31 = torch.rsqrt(add_92);  add_92 = None
    hidden_states_175 = hidden_states_174 * rsqrt_31;  rsqrt_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___decoder_block_4_layer_2_layer_norm_weight_3 = self.L__mod___decoder_block_4_layer_2_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    forwarded_states_24 = l__mod___decoder_block_4_layer_2_layer_norm_weight_3 * hidden_states_175;  l__mod___decoder_block_4_layer_2_layer_norm_weight_3 = hidden_states_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    l__mod___decoder_block_4_layer__1__dense_relu_dense_wi_0 = self.L__mod___decoder_block_4_layer__1__DenseReluDense_wi_0(forwarded_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_131 = 0.5 * l__mod___decoder_block_4_layer__1__dense_relu_dense_wi_0
    pow_45 = torch.pow(l__mod___decoder_block_4_layer__1__dense_relu_dense_wi_0, 3.0)
    mul_132 = 0.044715 * pow_45;  pow_45 = None
    add_93 = l__mod___decoder_block_4_layer__1__dense_relu_dense_wi_0 + mul_132;  l__mod___decoder_block_4_layer__1__dense_relu_dense_wi_0 = mul_132 = None
    mul_133 = 0.7978845608028654 * add_93;  add_93 = None
    tanh_12 = torch.tanh(mul_133);  mul_133 = None
    add_94 = 1.0 + tanh_12;  tanh_12 = None
    hidden_gelu_12 = mul_131 * add_94;  mul_131 = add_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    hidden_linear_12 = self.L__mod___decoder_block_4_layer__1__DenseReluDense_wi_1(forwarded_states_24);  forwarded_states_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    hidden_states_176 = hidden_gelu_12 * hidden_linear_12;  hidden_gelu_12 = hidden_linear_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    hidden_states_177 = self.L__mod___decoder_block_4_layer__1__DenseReluDense_dropout(hidden_states_176);  hidden_states_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    forwarded_states_25 = self.L__mod___decoder_block_4_layer__1__DenseReluDense_wo(hidden_states_177);  hidden_states_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    l__mod___decoder_block_4_layer__1__dropout = self.L__mod___decoder_block_4_layer__1__dropout(forwarded_states_25);  forwarded_states_25 = None
    hidden_states_181 = hidden_states_174 + l__mod___decoder_block_4_layer__1__dropout;  hidden_states_174 = l__mod___decoder_block_4_layer__1__dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_39 = hidden_states_181.to(torch.float32)
    pow_46 = to_39.pow(2);  to_39 = None
    variance_32 = pow_46.mean(-1, keepdim = True);  pow_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_96 = variance_32 + 1e-06;  variance_32 = None
    rsqrt_32 = torch.rsqrt(add_96);  add_96 = None
    hidden_states_182 = hidden_states_181 * rsqrt_32;  rsqrt_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___decoder_block_5_layer_0_layer_norm_weight_3 = self.L__mod___decoder_block_5_layer_0_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    normed_hidden_states_18 = l__mod___decoder_block_5_layer_0_layer_norm_weight_3 * hidden_states_182;  l__mod___decoder_block_5_layer_0_layer_norm_weight_3 = hidden_states_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    l__mod___decoder_block_5_layer_0_self_attention_q = self.L__mod___decoder_block_5_layer_0_SelfAttention_q(normed_hidden_states_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_74 = l__mod___decoder_block_5_layer_0_self_attention_q.view(1, -1, 6, 64);  l__mod___decoder_block_5_layer_0_self_attention_q = None
    query_states_18 = view_74.transpose(1, 2);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    l__mod___decoder_block_5_layer_0_self_attention_k = self.L__mod___decoder_block_5_layer_0_SelfAttention_k(normed_hidden_states_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_75 = l__mod___decoder_block_5_layer_0_self_attention_k.view(1, -1, 6, 64);  l__mod___decoder_block_5_layer_0_self_attention_k = None
    key_states_18 = view_75.transpose(1, 2);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    l__mod___decoder_block_5_layer_0_self_attention_v = self.L__mod___decoder_block_5_layer_0_SelfAttention_v(normed_hidden_states_18);  normed_hidden_states_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_76 = l__mod___decoder_block_5_layer_0_self_attention_v.view(1, -1, 6, 64);  l__mod___decoder_block_5_layer_0_self_attention_v = None
    value_states_18 = view_76.transpose(1, 2);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    transpose_93 = key_states_18.transpose(3, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    scores_36 = torch.matmul(query_states_18, transpose_93);  query_states_18 = transpose_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    scores_36 += position_bias_21;  scores_37 = scores_36;  scores_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    float_21 = scores_37.float()
    softmax_18 = torch.nn.functional.softmax(float_21, dim = -1);  float_21 = None
    attn_weights_36 = softmax_18.type_as(scores_37);  softmax_18 = scores_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    attn_weights_37 = torch.nn.functional.dropout(attn_weights_36, p = 0.1, training = True);  attn_weights_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    matmul_37 = torch.matmul(attn_weights_37, value_states_18);  attn_weights_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    transpose_94 = matmul_37.transpose(1, 2);  matmul_37 = None
    contiguous_18 = transpose_94.contiguous();  transpose_94 = None
    attn_output_36 = contiguous_18.view(1, -1, 384);  contiguous_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    attn_output_37 = self.L__mod___decoder_block_5_layer_0_SelfAttention_o(attn_output_36);  attn_output_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    l__mod___decoder_block_5_layer_0_dropout = self.L__mod___decoder_block_5_layer_0_dropout(attn_output_37);  attn_output_37 = None
    hidden_states_186 = hidden_states_181 + l__mod___decoder_block_5_layer_0_dropout;  hidden_states_181 = l__mod___decoder_block_5_layer_0_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_40 = hidden_states_186.to(torch.float32)
    pow_47 = to_40.pow(2);  to_40 = None
    variance_33 = pow_47.mean(-1, keepdim = True);  pow_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_98 = variance_33 + 1e-06;  variance_33 = None
    rsqrt_33 = torch.rsqrt(add_98);  add_98 = None
    hidden_states_187 = hidden_states_186 * rsqrt_33;  rsqrt_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___decoder_block_5_layer_1_layer_norm_weight_3 = self.L__mod___decoder_block_5_layer_1_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    normed_hidden_states_19 = l__mod___decoder_block_5_layer_1_layer_norm_weight_3 * hidden_states_187;  l__mod___decoder_block_5_layer_1_layer_norm_weight_3 = hidden_states_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    l__mod___decoder_block_5_layer_1_enc_dec_attention_q = self.L__mod___decoder_block_5_layer_1_EncDecAttention_q(normed_hidden_states_19);  normed_hidden_states_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_78 = l__mod___decoder_block_5_layer_1_enc_dec_attention_q.view(1, -1, 6, 64);  l__mod___decoder_block_5_layer_1_enc_dec_attention_q = None
    query_states_19 = view_78.transpose(1, 2);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    l__mod___decoder_block_5_layer_1_enc_dec_attention_k = self.L__mod___decoder_block_5_layer_1_EncDecAttention_k(hidden_states_100)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_79 = l__mod___decoder_block_5_layer_1_enc_dec_attention_k.view(1, -1, 6, 64);  l__mod___decoder_block_5_layer_1_enc_dec_attention_k = None
    key_states_19 = view_79.transpose(1, 2);  view_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    l__mod___decoder_block_5_layer_1_enc_dec_attention_v = self.L__mod___decoder_block_5_layer_1_EncDecAttention_v(hidden_states_100)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_80 = l__mod___decoder_block_5_layer_1_enc_dec_attention_v.view(1, -1, 6, 64);  l__mod___decoder_block_5_layer_1_enc_dec_attention_v = None
    value_states_19 = view_80.transpose(1, 2);  view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    transpose_98 = key_states_19.transpose(3, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    scores_38 = torch.matmul(query_states_19, transpose_98);  query_states_19 = transpose_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    scores_38 += encoder_decoder_position_bias_7;  scores_39 = scores_38;  scores_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    float_22 = scores_39.float()
    softmax_19 = torch.nn.functional.softmax(float_22, dim = -1);  float_22 = None
    attn_weights_38 = softmax_19.type_as(scores_39);  softmax_19 = scores_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    attn_weights_39 = torch.nn.functional.dropout(attn_weights_38, p = 0.1, training = True);  attn_weights_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    matmul_39 = torch.matmul(attn_weights_39, value_states_19);  attn_weights_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    transpose_99 = matmul_39.transpose(1, 2);  matmul_39 = None
    contiguous_19 = transpose_99.contiguous();  transpose_99 = None
    attn_output_38 = contiguous_19.view(1, -1, 384);  contiguous_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    attn_output_39 = self.L__mod___decoder_block_5_layer_1_EncDecAttention_o(attn_output_38);  attn_output_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:510, code: layer_output = hidden_states + self.dropout(attention_output[0])
    l__mod___decoder_block_5_layer_1_dropout = self.L__mod___decoder_block_5_layer_1_dropout(attn_output_39);  attn_output_39 = None
    hidden_states_190 = hidden_states_186 + l__mod___decoder_block_5_layer_1_dropout;  hidden_states_186 = l__mod___decoder_block_5_layer_1_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_41 = hidden_states_190.to(torch.float32)
    pow_48 = to_41.pow(2);  to_41 = None
    variance_34 = pow_48.mean(-1, keepdim = True);  pow_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_100 = variance_34 + 1e-06;  variance_34 = None
    rsqrt_34 = torch.rsqrt(add_100);  add_100 = None
    hidden_states_191 = hidden_states_190 * rsqrt_34;  rsqrt_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___decoder_block_5_layer_2_layer_norm_weight_3 = self.L__mod___decoder_block_5_layer_2_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    forwarded_states_26 = l__mod___decoder_block_5_layer_2_layer_norm_weight_3 * hidden_states_191;  l__mod___decoder_block_5_layer_2_layer_norm_weight_3 = hidden_states_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    l__mod___decoder_block_5_layer__1__dense_relu_dense_wi_0 = self.L__mod___decoder_block_5_layer__1__DenseReluDense_wi_0(forwarded_states_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_142 = 0.5 * l__mod___decoder_block_5_layer__1__dense_relu_dense_wi_0
    pow_49 = torch.pow(l__mod___decoder_block_5_layer__1__dense_relu_dense_wi_0, 3.0)
    mul_143 = 0.044715 * pow_49;  pow_49 = None
    add_101 = l__mod___decoder_block_5_layer__1__dense_relu_dense_wi_0 + mul_143;  l__mod___decoder_block_5_layer__1__dense_relu_dense_wi_0 = mul_143 = None
    mul_144 = 0.7978845608028654 * add_101;  add_101 = None
    tanh_13 = torch.tanh(mul_144);  mul_144 = None
    add_102 = 1.0 + tanh_13;  tanh_13 = None
    hidden_gelu_13 = mul_142 * add_102;  mul_142 = add_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    hidden_linear_13 = self.L__mod___decoder_block_5_layer__1__DenseReluDense_wi_1(forwarded_states_26);  forwarded_states_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    hidden_states_192 = hidden_gelu_13 * hidden_linear_13;  hidden_gelu_13 = hidden_linear_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    hidden_states_193 = self.L__mod___decoder_block_5_layer__1__DenseReluDense_dropout(hidden_states_192);  hidden_states_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    forwarded_states_27 = self.L__mod___decoder_block_5_layer__1__DenseReluDense_wo(hidden_states_193);  hidden_states_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    l__mod___decoder_block_5_layer__1__dropout = self.L__mod___decoder_block_5_layer__1__dropout(forwarded_states_27);  forwarded_states_27 = None
    hidden_states_197 = hidden_states_190 + l__mod___decoder_block_5_layer__1__dropout;  hidden_states_190 = l__mod___decoder_block_5_layer__1__dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_42 = hidden_states_197.to(torch.float32)
    pow_50 = to_42.pow(2);  to_42 = None
    variance_35 = pow_50.mean(-1, keepdim = True);  pow_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_104 = variance_35 + 1e-06;  variance_35 = None
    rsqrt_35 = torch.rsqrt(add_104);  add_104 = None
    hidden_states_198 = hidden_states_197 * rsqrt_35;  rsqrt_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___decoder_block_6_layer_0_layer_norm_weight_3 = self.L__mod___decoder_block_6_layer_0_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    normed_hidden_states_20 = l__mod___decoder_block_6_layer_0_layer_norm_weight_3 * hidden_states_198;  l__mod___decoder_block_6_layer_0_layer_norm_weight_3 = hidden_states_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    l__mod___decoder_block_6_layer_0_self_attention_q = self.L__mod___decoder_block_6_layer_0_SelfAttention_q(normed_hidden_states_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_82 = l__mod___decoder_block_6_layer_0_self_attention_q.view(1, -1, 6, 64);  l__mod___decoder_block_6_layer_0_self_attention_q = None
    query_states_20 = view_82.transpose(1, 2);  view_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    l__mod___decoder_block_6_layer_0_self_attention_k = self.L__mod___decoder_block_6_layer_0_SelfAttention_k(normed_hidden_states_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_83 = l__mod___decoder_block_6_layer_0_self_attention_k.view(1, -1, 6, 64);  l__mod___decoder_block_6_layer_0_self_attention_k = None
    key_states_20 = view_83.transpose(1, 2);  view_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    l__mod___decoder_block_6_layer_0_self_attention_v = self.L__mod___decoder_block_6_layer_0_SelfAttention_v(normed_hidden_states_20);  normed_hidden_states_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_84 = l__mod___decoder_block_6_layer_0_self_attention_v.view(1, -1, 6, 64);  l__mod___decoder_block_6_layer_0_self_attention_v = None
    value_states_20 = view_84.transpose(1, 2);  view_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    transpose_103 = key_states_20.transpose(3, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    scores_40 = torch.matmul(query_states_20, transpose_103);  query_states_20 = transpose_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    scores_40 += position_bias_21;  scores_41 = scores_40;  scores_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    float_23 = scores_41.float()
    softmax_20 = torch.nn.functional.softmax(float_23, dim = -1);  float_23 = None
    attn_weights_40 = softmax_20.type_as(scores_41);  softmax_20 = scores_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    attn_weights_41 = torch.nn.functional.dropout(attn_weights_40, p = 0.1, training = True);  attn_weights_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    matmul_41 = torch.matmul(attn_weights_41, value_states_20);  attn_weights_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    transpose_104 = matmul_41.transpose(1, 2);  matmul_41 = None
    contiguous_20 = transpose_104.contiguous();  transpose_104 = None
    attn_output_40 = contiguous_20.view(1, -1, 384);  contiguous_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    attn_output_41 = self.L__mod___decoder_block_6_layer_0_SelfAttention_o(attn_output_40);  attn_output_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    l__mod___decoder_block_6_layer_0_dropout = self.L__mod___decoder_block_6_layer_0_dropout(attn_output_41);  attn_output_41 = None
    hidden_states_202 = hidden_states_197 + l__mod___decoder_block_6_layer_0_dropout;  hidden_states_197 = l__mod___decoder_block_6_layer_0_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_43 = hidden_states_202.to(torch.float32)
    pow_51 = to_43.pow(2);  to_43 = None
    variance_36 = pow_51.mean(-1, keepdim = True);  pow_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_106 = variance_36 + 1e-06;  variance_36 = None
    rsqrt_36 = torch.rsqrt(add_106);  add_106 = None
    hidden_states_203 = hidden_states_202 * rsqrt_36;  rsqrt_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___decoder_block_6_layer_1_layer_norm_weight_3 = self.L__mod___decoder_block_6_layer_1_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    normed_hidden_states_21 = l__mod___decoder_block_6_layer_1_layer_norm_weight_3 * hidden_states_203;  l__mod___decoder_block_6_layer_1_layer_norm_weight_3 = hidden_states_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    l__mod___decoder_block_6_layer_1_enc_dec_attention_q = self.L__mod___decoder_block_6_layer_1_EncDecAttention_q(normed_hidden_states_21);  normed_hidden_states_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_86 = l__mod___decoder_block_6_layer_1_enc_dec_attention_q.view(1, -1, 6, 64);  l__mod___decoder_block_6_layer_1_enc_dec_attention_q = None
    query_states_21 = view_86.transpose(1, 2);  view_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    l__mod___decoder_block_6_layer_1_enc_dec_attention_k = self.L__mod___decoder_block_6_layer_1_EncDecAttention_k(hidden_states_100)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_87 = l__mod___decoder_block_6_layer_1_enc_dec_attention_k.view(1, -1, 6, 64);  l__mod___decoder_block_6_layer_1_enc_dec_attention_k = None
    key_states_21 = view_87.transpose(1, 2);  view_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    l__mod___decoder_block_6_layer_1_enc_dec_attention_v = self.L__mod___decoder_block_6_layer_1_EncDecAttention_v(hidden_states_100)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_88 = l__mod___decoder_block_6_layer_1_enc_dec_attention_v.view(1, -1, 6, 64);  l__mod___decoder_block_6_layer_1_enc_dec_attention_v = None
    value_states_21 = view_88.transpose(1, 2);  view_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    transpose_108 = key_states_21.transpose(3, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    scores_42 = torch.matmul(query_states_21, transpose_108);  query_states_21 = transpose_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    scores_42 += encoder_decoder_position_bias_7;  scores_43 = scores_42;  scores_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    float_24 = scores_43.float()
    softmax_21 = torch.nn.functional.softmax(float_24, dim = -1);  float_24 = None
    attn_weights_42 = softmax_21.type_as(scores_43);  softmax_21 = scores_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    attn_weights_43 = torch.nn.functional.dropout(attn_weights_42, p = 0.1, training = True);  attn_weights_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    matmul_43 = torch.matmul(attn_weights_43, value_states_21);  attn_weights_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    transpose_109 = matmul_43.transpose(1, 2);  matmul_43 = None
    contiguous_21 = transpose_109.contiguous();  transpose_109 = None
    attn_output_42 = contiguous_21.view(1, -1, 384);  contiguous_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    attn_output_43 = self.L__mod___decoder_block_6_layer_1_EncDecAttention_o(attn_output_42);  attn_output_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:510, code: layer_output = hidden_states + self.dropout(attention_output[0])
    l__mod___decoder_block_6_layer_1_dropout = self.L__mod___decoder_block_6_layer_1_dropout(attn_output_43);  attn_output_43 = None
    hidden_states_206 = hidden_states_202 + l__mod___decoder_block_6_layer_1_dropout;  hidden_states_202 = l__mod___decoder_block_6_layer_1_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_44 = hidden_states_206.to(torch.float32)
    pow_52 = to_44.pow(2);  to_44 = None
    variance_37 = pow_52.mean(-1, keepdim = True);  pow_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_108 = variance_37 + 1e-06;  variance_37 = None
    rsqrt_37 = torch.rsqrt(add_108);  add_108 = None
    hidden_states_207 = hidden_states_206 * rsqrt_37;  rsqrt_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___decoder_block_6_layer_2_layer_norm_weight_3 = self.L__mod___decoder_block_6_layer_2_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    forwarded_states_28 = l__mod___decoder_block_6_layer_2_layer_norm_weight_3 * hidden_states_207;  l__mod___decoder_block_6_layer_2_layer_norm_weight_3 = hidden_states_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    l__mod___decoder_block_6_layer__1__dense_relu_dense_wi_0 = self.L__mod___decoder_block_6_layer__1__DenseReluDense_wi_0(forwarded_states_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_153 = 0.5 * l__mod___decoder_block_6_layer__1__dense_relu_dense_wi_0
    pow_53 = torch.pow(l__mod___decoder_block_6_layer__1__dense_relu_dense_wi_0, 3.0)
    mul_154 = 0.044715 * pow_53;  pow_53 = None
    add_109 = l__mod___decoder_block_6_layer__1__dense_relu_dense_wi_0 + mul_154;  l__mod___decoder_block_6_layer__1__dense_relu_dense_wi_0 = mul_154 = None
    mul_155 = 0.7978845608028654 * add_109;  add_109 = None
    tanh_14 = torch.tanh(mul_155);  mul_155 = None
    add_110 = 1.0 + tanh_14;  tanh_14 = None
    hidden_gelu_14 = mul_153 * add_110;  mul_153 = add_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    hidden_linear_14 = self.L__mod___decoder_block_6_layer__1__DenseReluDense_wi_1(forwarded_states_28);  forwarded_states_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    hidden_states_208 = hidden_gelu_14 * hidden_linear_14;  hidden_gelu_14 = hidden_linear_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    hidden_states_209 = self.L__mod___decoder_block_6_layer__1__DenseReluDense_dropout(hidden_states_208);  hidden_states_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    forwarded_states_29 = self.L__mod___decoder_block_6_layer__1__DenseReluDense_wo(hidden_states_209);  hidden_states_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    l__mod___decoder_block_6_layer__1__dropout = self.L__mod___decoder_block_6_layer__1__dropout(forwarded_states_29);  forwarded_states_29 = None
    hidden_states_213 = hidden_states_206 + l__mod___decoder_block_6_layer__1__dropout;  hidden_states_206 = l__mod___decoder_block_6_layer__1__dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_45 = hidden_states_213.to(torch.float32)
    pow_54 = to_45.pow(2);  to_45 = None
    variance_38 = pow_54.mean(-1, keepdim = True);  pow_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_112 = variance_38 + 1e-06;  variance_38 = None
    rsqrt_38 = torch.rsqrt(add_112);  add_112 = None
    hidden_states_214 = hidden_states_213 * rsqrt_38;  rsqrt_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___decoder_block_7_layer_0_layer_norm_weight_3 = self.L__mod___decoder_block_7_layer_0_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    normed_hidden_states_22 = l__mod___decoder_block_7_layer_0_layer_norm_weight_3 * hidden_states_214;  l__mod___decoder_block_7_layer_0_layer_norm_weight_3 = hidden_states_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    l__mod___decoder_block_7_layer_0_self_attention_q = self.L__mod___decoder_block_7_layer_0_SelfAttention_q(normed_hidden_states_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_90 = l__mod___decoder_block_7_layer_0_self_attention_q.view(1, -1, 6, 64);  l__mod___decoder_block_7_layer_0_self_attention_q = None
    query_states_22 = view_90.transpose(1, 2);  view_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    l__mod___decoder_block_7_layer_0_self_attention_k = self.L__mod___decoder_block_7_layer_0_SelfAttention_k(normed_hidden_states_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_91 = l__mod___decoder_block_7_layer_0_self_attention_k.view(1, -1, 6, 64);  l__mod___decoder_block_7_layer_0_self_attention_k = None
    key_states_22 = view_91.transpose(1, 2);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    l__mod___decoder_block_7_layer_0_self_attention_v = self.L__mod___decoder_block_7_layer_0_SelfAttention_v(normed_hidden_states_22);  normed_hidden_states_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_92 = l__mod___decoder_block_7_layer_0_self_attention_v.view(1, -1, 6, 64);  l__mod___decoder_block_7_layer_0_self_attention_v = None
    value_states_22 = view_92.transpose(1, 2);  view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    transpose_113 = key_states_22.transpose(3, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    scores_44 = torch.matmul(query_states_22, transpose_113);  query_states_22 = transpose_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    scores_44 += position_bias_21;  scores_45 = scores_44;  scores_44 = position_bias_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    float_25 = scores_45.float()
    softmax_22 = torch.nn.functional.softmax(float_25, dim = -1);  float_25 = None
    attn_weights_44 = softmax_22.type_as(scores_45);  softmax_22 = scores_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    attn_weights_45 = torch.nn.functional.dropout(attn_weights_44, p = 0.1, training = True);  attn_weights_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    matmul_45 = torch.matmul(attn_weights_45, value_states_22);  attn_weights_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    transpose_114 = matmul_45.transpose(1, 2);  matmul_45 = None
    contiguous_22 = transpose_114.contiguous();  transpose_114 = None
    attn_output_44 = contiguous_22.view(1, -1, 384);  contiguous_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    attn_output_45 = self.L__mod___decoder_block_7_layer_0_SelfAttention_o(attn_output_44);  attn_output_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    l__mod___decoder_block_7_layer_0_dropout = self.L__mod___decoder_block_7_layer_0_dropout(attn_output_45);  attn_output_45 = None
    hidden_states_218 = hidden_states_213 + l__mod___decoder_block_7_layer_0_dropout;  hidden_states_213 = l__mod___decoder_block_7_layer_0_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_46 = hidden_states_218.to(torch.float32)
    pow_55 = to_46.pow(2);  to_46 = None
    variance_39 = pow_55.mean(-1, keepdim = True);  pow_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_114 = variance_39 + 1e-06;  variance_39 = None
    rsqrt_39 = torch.rsqrt(add_114);  add_114 = None
    hidden_states_219 = hidden_states_218 * rsqrt_39;  rsqrt_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___decoder_block_7_layer_1_layer_norm_weight_3 = self.L__mod___decoder_block_7_layer_1_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    normed_hidden_states_23 = l__mod___decoder_block_7_layer_1_layer_norm_weight_3 * hidden_states_219;  l__mod___decoder_block_7_layer_1_layer_norm_weight_3 = hidden_states_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    l__mod___decoder_block_7_layer_1_enc_dec_attention_q = self.L__mod___decoder_block_7_layer_1_EncDecAttention_q(normed_hidden_states_23);  normed_hidden_states_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_94 = l__mod___decoder_block_7_layer_1_enc_dec_attention_q.view(1, -1, 6, 64);  l__mod___decoder_block_7_layer_1_enc_dec_attention_q = None
    query_states_23 = view_94.transpose(1, 2);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    l__mod___decoder_block_7_layer_1_enc_dec_attention_k = self.L__mod___decoder_block_7_layer_1_EncDecAttention_k(hidden_states_100)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_95 = l__mod___decoder_block_7_layer_1_enc_dec_attention_k.view(1, -1, 6, 64);  l__mod___decoder_block_7_layer_1_enc_dec_attention_k = None
    key_states_23 = view_95.transpose(1, 2);  view_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    l__mod___decoder_block_7_layer_1_enc_dec_attention_v = self.L__mod___decoder_block_7_layer_1_EncDecAttention_v(hidden_states_100)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_96 = l__mod___decoder_block_7_layer_1_enc_dec_attention_v.view(1, -1, 6, 64);  l__mod___decoder_block_7_layer_1_enc_dec_attention_v = None
    value_states_23 = view_96.transpose(1, 2);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    transpose_118 = key_states_23.transpose(3, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    scores_46 = torch.matmul(query_states_23, transpose_118);  query_states_23 = transpose_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    scores_46 += encoder_decoder_position_bias_7;  scores_47 = scores_46;  scores_46 = encoder_decoder_position_bias_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    float_26 = scores_47.float()
    softmax_23 = torch.nn.functional.softmax(float_26, dim = -1);  float_26 = None
    attn_weights_46 = softmax_23.type_as(scores_47);  softmax_23 = scores_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    attn_weights_47 = torch.nn.functional.dropout(attn_weights_46, p = 0.1, training = True);  attn_weights_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    matmul_47 = torch.matmul(attn_weights_47, value_states_23);  attn_weights_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    transpose_119 = matmul_47.transpose(1, 2);  matmul_47 = None
    contiguous_23 = transpose_119.contiguous();  transpose_119 = None
    attn_output_46 = contiguous_23.view(1, -1, 384);  contiguous_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    attn_output_47 = self.L__mod___decoder_block_7_layer_1_EncDecAttention_o(attn_output_46);  attn_output_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:510, code: layer_output = hidden_states + self.dropout(attention_output[0])
    l__mod___decoder_block_7_layer_1_dropout = self.L__mod___decoder_block_7_layer_1_dropout(attn_output_47);  attn_output_47 = None
    hidden_states_222 = hidden_states_218 + l__mod___decoder_block_7_layer_1_dropout;  hidden_states_218 = l__mod___decoder_block_7_layer_1_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_47 = hidden_states_222.to(torch.float32)
    pow_56 = to_47.pow(2);  to_47 = None
    variance_40 = pow_56.mean(-1, keepdim = True);  pow_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_116 = variance_40 + 1e-06;  variance_40 = None
    rsqrt_40 = torch.rsqrt(add_116);  add_116 = None
    hidden_states_223 = hidden_states_222 * rsqrt_40;  rsqrt_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___decoder_block_7_layer_2_layer_norm_weight_3 = self.L__mod___decoder_block_7_layer_2_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    forwarded_states_30 = l__mod___decoder_block_7_layer_2_layer_norm_weight_3 * hidden_states_223;  l__mod___decoder_block_7_layer_2_layer_norm_weight_3 = hidden_states_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    l__mod___decoder_block_7_layer__1__dense_relu_dense_wi_0 = self.L__mod___decoder_block_7_layer__1__DenseReluDense_wi_0(forwarded_states_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_164 = 0.5 * l__mod___decoder_block_7_layer__1__dense_relu_dense_wi_0
    pow_57 = torch.pow(l__mod___decoder_block_7_layer__1__dense_relu_dense_wi_0, 3.0)
    mul_165 = 0.044715 * pow_57;  pow_57 = None
    add_117 = l__mod___decoder_block_7_layer__1__dense_relu_dense_wi_0 + mul_165;  l__mod___decoder_block_7_layer__1__dense_relu_dense_wi_0 = mul_165 = None
    mul_166 = 0.7978845608028654 * add_117;  add_117 = None
    tanh_15 = torch.tanh(mul_166);  mul_166 = None
    add_118 = 1.0 + tanh_15;  tanh_15 = None
    hidden_gelu_15 = mul_164 * add_118;  mul_164 = add_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    hidden_linear_15 = self.L__mod___decoder_block_7_layer__1__DenseReluDense_wi_1(forwarded_states_30);  forwarded_states_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    hidden_states_224 = hidden_gelu_15 * hidden_linear_15;  hidden_gelu_15 = hidden_linear_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    hidden_states_225 = self.L__mod___decoder_block_7_layer__1__DenseReluDense_dropout(hidden_states_224);  hidden_states_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    forwarded_states_31 = self.L__mod___decoder_block_7_layer__1__DenseReluDense_wo(hidden_states_225);  hidden_states_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    l__mod___decoder_block_7_layer__1__dropout = self.L__mod___decoder_block_7_layer__1__dropout(forwarded_states_31);  forwarded_states_31 = None
    hidden_states_229 = hidden_states_222 + l__mod___decoder_block_7_layer__1__dropout;  hidden_states_222 = l__mod___decoder_block_7_layer__1__dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    to_48 = hidden_states_229.to(torch.float32)
    pow_58 = to_48.pow(2);  to_48 = None
    variance_41 = pow_58.mean(-1, keepdim = True);  pow_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_120 = variance_41 + 1e-06;  variance_41 = None
    rsqrt_41 = torch.rsqrt(add_120);  add_120 = None
    hidden_states_230 = hidden_states_229 * rsqrt_41;  hidden_states_229 = rsqrt_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:131, code: if self.weight.dtype in [torch.float16, torch.bfloat16]:
    l__mod___decoder_final_layer_norm_weight_3 = self.L__mod___decoder_final_layer_norm_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    hidden_states_231 = l__mod___decoder_final_layer_norm_weight_3 * hidden_states_230;  l__mod___decoder_final_layer_norm_weight_3 = hidden_states_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:1139, code: hidden_states = self.dropout(hidden_states)
    sequence_output = self.L__mod___decoder_dropout(hidden_states_231);  hidden_states_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:1799, code: lm_logits = self.lm_head(sequence_output)
    lm_logits = self.L__mod___lm_head(sequence_output);  sequence_output = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:1805, code: labels = labels.to(lm_logits.device)
    labels = l_cloned_inputs_labels_.to(device(type='cpu'));  l_cloned_inputs_labels_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:1806, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
    view_98 = lm_logits.view(-1, 250112)
    view_99 = labels.view(-1);  labels = None
    loss = torch.nn.functional.cross_entropy(view_98, view_99, None, None, -100, None, 'mean', 0.0);  view_98 = view_99 = None
    return (loss, lm_logits, key_states_8, value_states_8, key_states_9, value_states_9, key_states_10, value_states_10, key_states_11, value_states_11, key_states_12, value_states_12, key_states_13, value_states_13, key_states_14, value_states_14, key_states_15, value_states_15, key_states_16, value_states_16, key_states_17, value_states_17, key_states_18, value_states_18, key_states_19, value_states_19, key_states_20, value_states_20, key_states_21, value_states_21, key_states_22, value_states_22, key_states_23, value_states_23, hidden_states_100)
    