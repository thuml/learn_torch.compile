from __future__ import annotations



def forward(self, L_cloned_inputs_input_ids_ : torch.Tensor, L_cloned_inputs_start_positions_ : torch.Tensor, L_cloned_inputs_end_positions_ : torch.Tensor):
    l_cloned_inputs_input_ids_ = L_cloned_inputs_input_ids_
    l_cloned_inputs_start_positions_ = L_cloned_inputs_start_positions_
    l_cloned_inputs_end_positions_ = L_cloned_inputs_end_positions_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:972, code: attention_mask = torch.ones(input_shape, device=device)
    attention_mask = torch.ones((1, 512), device = device(type='cpu'))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:974, code: token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
    token_type_ids = torch.zeros((1, 512), dtype = torch.int64, device = device(type='cpu'))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:780, code: position_ids = self.position_ids[:, :seq_length]
    l__mod___deberta_embeddings_position_ids = self.L__mod___deberta_embeddings_position_ids
    position_ids = l__mod___deberta_embeddings_position_ids[(slice(None, None, None), slice(None, 512, None))];  l__mod___deberta_embeddings_position_ids = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:786, code: inputs_embeds = self.word_embeddings(input_ids)
    embeddings = self.L__mod___deberta_embeddings_word_embeddings(l_cloned_inputs_input_ids_);  l_cloned_inputs_input_ids_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:789, code: position_embeddings = self.position_embeddings(position_ids.long())
    long = position_ids.long();  position_ids = None
    position_embeddings = self.L__mod___deberta_embeddings_position_embeddings(long);  long = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:795, code: embeddings += position_embeddings
    embeddings += position_embeddings;  embeddings_1 = embeddings;  embeddings = position_embeddings = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:277, code: hidden_states = hidden_states.float()
    hidden_states = embeddings_1.float();  embeddings_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean = hidden_states.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub = hidden_states - mean
    pow_1 = sub.pow(2);  sub = None
    variance = pow_1.mean(-1, keepdim = True);  pow_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_1 = hidden_states - mean;  hidden_states = mean = None
    add = variance + 1e-07;  variance = None
    sqrt = torch.sqrt(add);  add = None
    hidden_states_1 = sub_1 / sqrt;  sub_1 = sqrt = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:281, code: hidden_states = hidden_states.to(input_type)
    hidden_states_2 = hidden_states_1.to(torch.float32);  hidden_states_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    l__mod___deberta_embeddings_layer_norm_weight = self.L__mod___deberta_embeddings_LayerNorm_weight
    mul = l__mod___deberta_embeddings_layer_norm_weight * hidden_states_2;  l__mod___deberta_embeddings_layer_norm_weight = hidden_states_2 = None
    l__mod___deberta_embeddings_layer_norm_bias = self.L__mod___deberta_embeddings_LayerNorm_bias
    embeddings_2 = mul + l__mod___deberta_embeddings_layer_norm_bias;  mul = l__mod___deberta_embeddings_layer_norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:809, code: mask = mask.unsqueeze(2)
    mask = attention_mask.unsqueeze(2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:810, code: mask = mask.to(embeddings.dtype)
    mask_1 = mask.to(torch.float32);  mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:812, code: embeddings = embeddings * mask
    embeddings_3 = embeddings_2 * mask_1;  embeddings_2 = mask_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    function_ctx = torch.autograd.function.FunctionCtx()
    query_states = torch__dynamo_variables_misc_trampoline_autograd_apply(embeddings_3, 0.1);  embeddings_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:421, code: extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    unsqueeze_1 = attention_mask.unsqueeze(1);  attention_mask = None
    extended_attention_mask = unsqueeze_1.unsqueeze(2);  unsqueeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:422, code: attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
    squeeze = extended_attention_mask.squeeze(-2)
    unsqueeze_3 = squeeze.unsqueeze(-1);  squeeze = None
    attention_mask_2 = extended_attention_mask * unsqueeze_3;  extended_attention_mask = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    qp = self.L__mod___deberta_encoder_layer_0_attention_self_in_proj(query_states)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    x = qp.view((1, 512, 12, -1));  qp = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute = x.permute(0, 2, 1, 3);  x = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    chunk = permute.chunk(3, dim = -1);  permute = None
    query_layer = chunk[0]
    key_layer = chunk[1]
    value_layer = chunk[2];  chunk = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    l__mod___deberta_encoder_layer_0_attention_self_q_bias = self.L__mod___deberta_encoder_layer_0_attention_self_q_bias
    getitem_4 = l__mod___deberta_encoder_layer_0_attention_self_q_bias[(None, None, slice(None, None, None))];  l__mod___deberta_encoder_layer_0_attention_self_q_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    x_1 = getitem_4.view((1, 1, 12, -1));  getitem_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_1 = x_1.permute(0, 2, 1, 3);  x_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    query_layer_1 = query_layer + permute_1;  query_layer = permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    l__mod___deberta_encoder_layer_0_attention_self_v_bias = self.L__mod___deberta_encoder_layer_0_attention_self_v_bias
    getitem_5 = l__mod___deberta_encoder_layer_0_attention_self_v_bias[(None, None, slice(None, None, None))];  l__mod___deberta_encoder_layer_0_attention_self_v_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    x_2 = getitem_5.view((1, 1, 12, -1));  getitem_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_2 = x_2.permute(0, 2, 1, 3);  x_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    value_layer_1 = value_layer + permute_2;  value_layer = permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    tensor = torch.tensor(64, dtype = torch.float32)
    mul_3 = tensor * 1;  tensor = None
    scale = torch.sqrt(mul_3);  mul_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    to_2 = scale.to(dtype = torch.float32);  scale = None
    query_layer_2 = query_layer_1 / to_2;  query_layer_1 = to_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose = key_layer.transpose(-1, -2);  key_layer = None
    attention_scores = torch.matmul(query_layer_2, transpose);  query_layer_2 = transpose = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    function_ctx_1 = torch.autograd.function.FunctionCtx()
    attention_probs = torch__dynamo_variables_misc_trampoline_autograd_apply_1(attention_scores, attention_mask_2, -1);  attention_scores = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    function_ctx_2 = torch.autograd.function.FunctionCtx()
    attention_probs_1 = torch__dynamo_variables_misc_trampoline_autograd_apply_2(attention_probs, 0.1);  attention_probs = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer = torch.matmul(attention_probs_1, value_layer_1);  attention_probs_1 = value_layer_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_3 = context_layer.permute(0, 2, 1, 3);  context_layer = None
    context_layer_1 = permute_3.contiguous();  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    self_output = context_layer_1.view((1, 512, -1));  context_layer_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    hidden_states_3 = self.L__mod___deberta_encoder_layer_0_attention_output_dense(self_output);  self_output = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    function_ctx_3 = torch.autograd.function.FunctionCtx()
    hidden_states_4 = torch__dynamo_variables_misc_trampoline_autograd_apply_3(hidden_states_3, 0.1);  hidden_states_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_4 = hidden_states_4 + query_states;  hidden_states_4 = query_states = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:277, code: hidden_states = hidden_states.float()
    hidden_states_5 = add_4.float();  add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_3 = hidden_states_5.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_2 = hidden_states_5 - mean_3
    pow_2 = sub_2.pow(2);  sub_2 = None
    variance_1 = pow_2.mean(-1, keepdim = True);  pow_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_3 = hidden_states_5 - mean_3;  hidden_states_5 = mean_3 = None
    add_5 = variance_1 + 1e-07;  variance_1 = None
    sqrt_2 = torch.sqrt(add_5);  add_5 = None
    hidden_states_6 = sub_3 / sqrt_2;  sub_3 = sqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:281, code: hidden_states = hidden_states.to(input_type)
    hidden_states_7 = hidden_states_6.to(torch.float32);  hidden_states_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    l__mod___deberta_encoder_layer_0_attention_output_layer_norm_weight = self.L__mod___deberta_encoder_layer_0_attention_output_LayerNorm_weight
    mul_4 = l__mod___deberta_encoder_layer_0_attention_output_layer_norm_weight * hidden_states_7;  l__mod___deberta_encoder_layer_0_attention_output_layer_norm_weight = hidden_states_7 = None
    l__mod___deberta_encoder_layer_0_attention_output_layer_norm_bias = self.L__mod___deberta_encoder_layer_0_attention_output_LayerNorm_bias
    attention_output = mul_4 + l__mod___deberta_encoder_layer_0_attention_output_layer_norm_bias;  mul_4 = l__mod___deberta_encoder_layer_0_attention_output_layer_norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    hidden_states_9 = self.L__mod___deberta_encoder_layer_0_intermediate_dense(attention_output)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output = torch._C._nn.gelu(hidden_states_9);  hidden_states_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    hidden_states_11 = self.L__mod___deberta_encoder_layer_0_output_dense(intermediate_output);  intermediate_output = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    function_ctx_4 = torch.autograd.function.FunctionCtx()
    hidden_states_12 = torch__dynamo_variables_misc_trampoline_autograd_apply_4(hidden_states_11, 0.1);  hidden_states_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_7 = hidden_states_12 + attention_output;  hidden_states_12 = attention_output = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:277, code: hidden_states = hidden_states.float()
    hidden_states_13 = add_7.float();  add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_6 = hidden_states_13.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_4 = hidden_states_13 - mean_6
    pow_3 = sub_4.pow(2);  sub_4 = None
    variance_2 = pow_3.mean(-1, keepdim = True);  pow_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_5 = hidden_states_13 - mean_6;  hidden_states_13 = mean_6 = None
    add_8 = variance_2 + 1e-07;  variance_2 = None
    sqrt_3 = torch.sqrt(add_8);  add_8 = None
    hidden_states_14 = sub_5 / sqrt_3;  sub_5 = sqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:281, code: hidden_states = hidden_states.to(input_type)
    hidden_states_15 = hidden_states_14.to(torch.float32);  hidden_states_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    l__mod___deberta_encoder_layer_0_output_layer_norm_weight = self.L__mod___deberta_encoder_layer_0_output_LayerNorm_weight
    mul_5 = l__mod___deberta_encoder_layer_0_output_layer_norm_weight * hidden_states_15;  l__mod___deberta_encoder_layer_0_output_layer_norm_weight = hidden_states_15 = None
    l__mod___deberta_encoder_layer_0_output_layer_norm_bias = self.L__mod___deberta_encoder_layer_0_output_LayerNorm_bias
    query_states_1 = mul_5 + l__mod___deberta_encoder_layer_0_output_layer_norm_bias;  mul_5 = l__mod___deberta_encoder_layer_0_output_layer_norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    qp_1 = self.L__mod___deberta_encoder_layer_1_attention_self_in_proj(query_states_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    x_3 = qp_1.view((1, 512, 12, -1));  qp_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_4 = x_3.permute(0, 2, 1, 3);  x_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    chunk_1 = permute_4.chunk(3, dim = -1);  permute_4 = None
    query_layer_3 = chunk_1[0]
    key_layer_1 = chunk_1[1]
    value_layer_2 = chunk_1[2];  chunk_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    l__mod___deberta_encoder_layer_1_attention_self_q_bias = self.L__mod___deberta_encoder_layer_1_attention_self_q_bias
    getitem_9 = l__mod___deberta_encoder_layer_1_attention_self_q_bias[(None, None, slice(None, None, None))];  l__mod___deberta_encoder_layer_1_attention_self_q_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    x_4 = getitem_9.view((1, 1, 12, -1));  getitem_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_5 = x_4.permute(0, 2, 1, 3);  x_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    query_layer_4 = query_layer_3 + permute_5;  query_layer_3 = permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    l__mod___deberta_encoder_layer_1_attention_self_v_bias = self.L__mod___deberta_encoder_layer_1_attention_self_v_bias
    getitem_10 = l__mod___deberta_encoder_layer_1_attention_self_v_bias[(None, None, slice(None, None, None))];  l__mod___deberta_encoder_layer_1_attention_self_v_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    x_5 = getitem_10.view((1, 1, 12, -1));  getitem_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_6 = x_5.permute(0, 2, 1, 3);  x_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    value_layer_3 = value_layer_2 + permute_6;  value_layer_2 = permute_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    tensor_1 = torch.tensor(64, dtype = torch.float32)
    mul_6 = tensor_1 * 1;  tensor_1 = None
    scale_1 = torch.sqrt(mul_6);  mul_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    to_5 = scale_1.to(dtype = torch.float32);  scale_1 = None
    query_layer_5 = query_layer_4 / to_5;  query_layer_4 = to_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_1 = key_layer_1.transpose(-1, -2);  key_layer_1 = None
    attention_scores_1 = torch.matmul(query_layer_5, transpose_1);  query_layer_5 = transpose_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    function_ctx_5 = torch.autograd.function.FunctionCtx()
    attention_probs_2 = torch__dynamo_variables_misc_trampoline_autograd_apply_5(attention_scores_1, attention_mask_2, -1);  attention_scores_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    function_ctx_6 = torch.autograd.function.FunctionCtx()
    attention_probs_3 = torch__dynamo_variables_misc_trampoline_autograd_apply_6(attention_probs_2, 0.1);  attention_probs_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_3 = torch.matmul(attention_probs_3, value_layer_3);  attention_probs_3 = value_layer_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_7 = context_layer_3.permute(0, 2, 1, 3);  context_layer_3 = None
    context_layer_4 = permute_7.contiguous();  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    self_output_1 = context_layer_4.view((1, 512, -1));  context_layer_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    hidden_states_18 = self.L__mod___deberta_encoder_layer_1_attention_output_dense(self_output_1);  self_output_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    function_ctx_7 = torch.autograd.function.FunctionCtx()
    hidden_states_19 = torch__dynamo_variables_misc_trampoline_autograd_apply_7(hidden_states_18, 0.1);  hidden_states_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_12 = hidden_states_19 + query_states_1;  hidden_states_19 = query_states_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:277, code: hidden_states = hidden_states.float()
    hidden_states_20 = add_12.float();  add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_9 = hidden_states_20.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_6 = hidden_states_20 - mean_9
    pow_4 = sub_6.pow(2);  sub_6 = None
    variance_3 = pow_4.mean(-1, keepdim = True);  pow_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_7 = hidden_states_20 - mean_9;  hidden_states_20 = mean_9 = None
    add_13 = variance_3 + 1e-07;  variance_3 = None
    sqrt_5 = torch.sqrt(add_13);  add_13 = None
    hidden_states_21 = sub_7 / sqrt_5;  sub_7 = sqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:281, code: hidden_states = hidden_states.to(input_type)
    hidden_states_22 = hidden_states_21.to(torch.float32);  hidden_states_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    l__mod___deberta_encoder_layer_1_attention_output_layer_norm_weight = self.L__mod___deberta_encoder_layer_1_attention_output_LayerNorm_weight
    mul_7 = l__mod___deberta_encoder_layer_1_attention_output_layer_norm_weight * hidden_states_22;  l__mod___deberta_encoder_layer_1_attention_output_layer_norm_weight = hidden_states_22 = None
    l__mod___deberta_encoder_layer_1_attention_output_layer_norm_bias = self.L__mod___deberta_encoder_layer_1_attention_output_LayerNorm_bias
    attention_output_2 = mul_7 + l__mod___deberta_encoder_layer_1_attention_output_layer_norm_bias;  mul_7 = l__mod___deberta_encoder_layer_1_attention_output_layer_norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    hidden_states_24 = self.L__mod___deberta_encoder_layer_1_intermediate_dense(attention_output_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_1 = torch._C._nn.gelu(hidden_states_24);  hidden_states_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    hidden_states_26 = self.L__mod___deberta_encoder_layer_1_output_dense(intermediate_output_1);  intermediate_output_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    function_ctx_8 = torch.autograd.function.FunctionCtx()
    hidden_states_27 = torch__dynamo_variables_misc_trampoline_autograd_apply_8(hidden_states_26, 0.1);  hidden_states_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_15 = hidden_states_27 + attention_output_2;  hidden_states_27 = attention_output_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:277, code: hidden_states = hidden_states.float()
    hidden_states_28 = add_15.float();  add_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_12 = hidden_states_28.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_8 = hidden_states_28 - mean_12
    pow_5 = sub_8.pow(2);  sub_8 = None
    variance_4 = pow_5.mean(-1, keepdim = True);  pow_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_9 = hidden_states_28 - mean_12;  hidden_states_28 = mean_12 = None
    add_16 = variance_4 + 1e-07;  variance_4 = None
    sqrt_6 = torch.sqrt(add_16);  add_16 = None
    hidden_states_29 = sub_9 / sqrt_6;  sub_9 = sqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:281, code: hidden_states = hidden_states.to(input_type)
    hidden_states_30 = hidden_states_29.to(torch.float32);  hidden_states_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    l__mod___deberta_encoder_layer_1_output_layer_norm_weight = self.L__mod___deberta_encoder_layer_1_output_LayerNorm_weight
    mul_8 = l__mod___deberta_encoder_layer_1_output_layer_norm_weight * hidden_states_30;  l__mod___deberta_encoder_layer_1_output_layer_norm_weight = hidden_states_30 = None
    l__mod___deberta_encoder_layer_1_output_layer_norm_bias = self.L__mod___deberta_encoder_layer_1_output_LayerNorm_bias
    query_states_2 = mul_8 + l__mod___deberta_encoder_layer_1_output_layer_norm_bias;  mul_8 = l__mod___deberta_encoder_layer_1_output_layer_norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    qp_2 = self.L__mod___deberta_encoder_layer_2_attention_self_in_proj(query_states_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    x_6 = qp_2.view((1, 512, 12, -1));  qp_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_8 = x_6.permute(0, 2, 1, 3);  x_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    chunk_2 = permute_8.chunk(3, dim = -1);  permute_8 = None
    query_layer_6 = chunk_2[0]
    key_layer_2 = chunk_2[1]
    value_layer_4 = chunk_2[2];  chunk_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    l__mod___deberta_encoder_layer_2_attention_self_q_bias = self.L__mod___deberta_encoder_layer_2_attention_self_q_bias
    getitem_14 = l__mod___deberta_encoder_layer_2_attention_self_q_bias[(None, None, slice(None, None, None))];  l__mod___deberta_encoder_layer_2_attention_self_q_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    x_7 = getitem_14.view((1, 1, 12, -1));  getitem_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_9 = x_7.permute(0, 2, 1, 3);  x_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    query_layer_7 = query_layer_6 + permute_9;  query_layer_6 = permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    l__mod___deberta_encoder_layer_2_attention_self_v_bias = self.L__mod___deberta_encoder_layer_2_attention_self_v_bias
    getitem_15 = l__mod___deberta_encoder_layer_2_attention_self_v_bias[(None, None, slice(None, None, None))];  l__mod___deberta_encoder_layer_2_attention_self_v_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    x_8 = getitem_15.view((1, 1, 12, -1));  getitem_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_10 = x_8.permute(0, 2, 1, 3);  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    value_layer_5 = value_layer_4 + permute_10;  value_layer_4 = permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    tensor_2 = torch.tensor(64, dtype = torch.float32)
    mul_9 = tensor_2 * 1;  tensor_2 = None
    scale_2 = torch.sqrt(mul_9);  mul_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    to_8 = scale_2.to(dtype = torch.float32);  scale_2 = None
    query_layer_8 = query_layer_7 / to_8;  query_layer_7 = to_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_2 = key_layer_2.transpose(-1, -2);  key_layer_2 = None
    attention_scores_2 = torch.matmul(query_layer_8, transpose_2);  query_layer_8 = transpose_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    function_ctx_9 = torch.autograd.function.FunctionCtx()
    attention_probs_4 = torch__dynamo_variables_misc_trampoline_autograd_apply_9(attention_scores_2, attention_mask_2, -1);  attention_scores_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    function_ctx_10 = torch.autograd.function.FunctionCtx()
    attention_probs_5 = torch__dynamo_variables_misc_trampoline_autograd_apply_10(attention_probs_4, 0.1);  attention_probs_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_6 = torch.matmul(attention_probs_5, value_layer_5);  attention_probs_5 = value_layer_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_11 = context_layer_6.permute(0, 2, 1, 3);  context_layer_6 = None
    context_layer_7 = permute_11.contiguous();  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    self_output_2 = context_layer_7.view((1, 512, -1));  context_layer_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    hidden_states_33 = self.L__mod___deberta_encoder_layer_2_attention_output_dense(self_output_2);  self_output_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    function_ctx_11 = torch.autograd.function.FunctionCtx()
    hidden_states_34 = torch__dynamo_variables_misc_trampoline_autograd_apply_11(hidden_states_33, 0.1);  hidden_states_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_20 = hidden_states_34 + query_states_2;  hidden_states_34 = query_states_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:277, code: hidden_states = hidden_states.float()
    hidden_states_35 = add_20.float();  add_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_15 = hidden_states_35.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_10 = hidden_states_35 - mean_15
    pow_6 = sub_10.pow(2);  sub_10 = None
    variance_5 = pow_6.mean(-1, keepdim = True);  pow_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_11 = hidden_states_35 - mean_15;  hidden_states_35 = mean_15 = None
    add_21 = variance_5 + 1e-07;  variance_5 = None
    sqrt_8 = torch.sqrt(add_21);  add_21 = None
    hidden_states_36 = sub_11 / sqrt_8;  sub_11 = sqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:281, code: hidden_states = hidden_states.to(input_type)
    hidden_states_37 = hidden_states_36.to(torch.float32);  hidden_states_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    l__mod___deberta_encoder_layer_2_attention_output_layer_norm_weight = self.L__mod___deberta_encoder_layer_2_attention_output_LayerNorm_weight
    mul_10 = l__mod___deberta_encoder_layer_2_attention_output_layer_norm_weight * hidden_states_37;  l__mod___deberta_encoder_layer_2_attention_output_layer_norm_weight = hidden_states_37 = None
    l__mod___deberta_encoder_layer_2_attention_output_layer_norm_bias = self.L__mod___deberta_encoder_layer_2_attention_output_LayerNorm_bias
    attention_output_4 = mul_10 + l__mod___deberta_encoder_layer_2_attention_output_layer_norm_bias;  mul_10 = l__mod___deberta_encoder_layer_2_attention_output_layer_norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    hidden_states_39 = self.L__mod___deberta_encoder_layer_2_intermediate_dense(attention_output_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_2 = torch._C._nn.gelu(hidden_states_39);  hidden_states_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    hidden_states_41 = self.L__mod___deberta_encoder_layer_2_output_dense(intermediate_output_2);  intermediate_output_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    function_ctx_12 = torch.autograd.function.FunctionCtx()
    hidden_states_42 = torch__dynamo_variables_misc_trampoline_autograd_apply_12(hidden_states_41, 0.1);  hidden_states_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_23 = hidden_states_42 + attention_output_4;  hidden_states_42 = attention_output_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:277, code: hidden_states = hidden_states.float()
    hidden_states_43 = add_23.float();  add_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_18 = hidden_states_43.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_12 = hidden_states_43 - mean_18
    pow_7 = sub_12.pow(2);  sub_12 = None
    variance_6 = pow_7.mean(-1, keepdim = True);  pow_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_13 = hidden_states_43 - mean_18;  hidden_states_43 = mean_18 = None
    add_24 = variance_6 + 1e-07;  variance_6 = None
    sqrt_9 = torch.sqrt(add_24);  add_24 = None
    hidden_states_44 = sub_13 / sqrt_9;  sub_13 = sqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:281, code: hidden_states = hidden_states.to(input_type)
    hidden_states_45 = hidden_states_44.to(torch.float32);  hidden_states_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    l__mod___deberta_encoder_layer_2_output_layer_norm_weight = self.L__mod___deberta_encoder_layer_2_output_LayerNorm_weight
    mul_11 = l__mod___deberta_encoder_layer_2_output_layer_norm_weight * hidden_states_45;  l__mod___deberta_encoder_layer_2_output_layer_norm_weight = hidden_states_45 = None
    l__mod___deberta_encoder_layer_2_output_layer_norm_bias = self.L__mod___deberta_encoder_layer_2_output_LayerNorm_bias
    query_states_3 = mul_11 + l__mod___deberta_encoder_layer_2_output_layer_norm_bias;  mul_11 = l__mod___deberta_encoder_layer_2_output_layer_norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    qp_3 = self.L__mod___deberta_encoder_layer_3_attention_self_in_proj(query_states_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    x_9 = qp_3.view((1, 512, 12, -1));  qp_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_12 = x_9.permute(0, 2, 1, 3);  x_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    chunk_3 = permute_12.chunk(3, dim = -1);  permute_12 = None
    query_layer_9 = chunk_3[0]
    key_layer_3 = chunk_3[1]
    value_layer_6 = chunk_3[2];  chunk_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    l__mod___deberta_encoder_layer_3_attention_self_q_bias = self.L__mod___deberta_encoder_layer_3_attention_self_q_bias
    getitem_19 = l__mod___deberta_encoder_layer_3_attention_self_q_bias[(None, None, slice(None, None, None))];  l__mod___deberta_encoder_layer_3_attention_self_q_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    x_10 = getitem_19.view((1, 1, 12, -1));  getitem_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_13 = x_10.permute(0, 2, 1, 3);  x_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    query_layer_10 = query_layer_9 + permute_13;  query_layer_9 = permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    l__mod___deberta_encoder_layer_3_attention_self_v_bias = self.L__mod___deberta_encoder_layer_3_attention_self_v_bias
    getitem_20 = l__mod___deberta_encoder_layer_3_attention_self_v_bias[(None, None, slice(None, None, None))];  l__mod___deberta_encoder_layer_3_attention_self_v_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    x_11 = getitem_20.view((1, 1, 12, -1));  getitem_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_14 = x_11.permute(0, 2, 1, 3);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    value_layer_7 = value_layer_6 + permute_14;  value_layer_6 = permute_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    tensor_3 = torch.tensor(64, dtype = torch.float32)
    mul_12 = tensor_3 * 1;  tensor_3 = None
    scale_3 = torch.sqrt(mul_12);  mul_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    to_11 = scale_3.to(dtype = torch.float32);  scale_3 = None
    query_layer_11 = query_layer_10 / to_11;  query_layer_10 = to_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_3 = key_layer_3.transpose(-1, -2);  key_layer_3 = None
    attention_scores_3 = torch.matmul(query_layer_11, transpose_3);  query_layer_11 = transpose_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    function_ctx_13 = torch.autograd.function.FunctionCtx()
    attention_probs_6 = torch__dynamo_variables_misc_trampoline_autograd_apply_13(attention_scores_3, attention_mask_2, -1);  attention_scores_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    function_ctx_14 = torch.autograd.function.FunctionCtx()
    attention_probs_7 = torch__dynamo_variables_misc_trampoline_autograd_apply_14(attention_probs_6, 0.1);  attention_probs_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_9 = torch.matmul(attention_probs_7, value_layer_7);  attention_probs_7 = value_layer_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_15 = context_layer_9.permute(0, 2, 1, 3);  context_layer_9 = None
    context_layer_10 = permute_15.contiguous();  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    self_output_3 = context_layer_10.view((1, 512, -1));  context_layer_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    hidden_states_48 = self.L__mod___deberta_encoder_layer_3_attention_output_dense(self_output_3);  self_output_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    function_ctx_15 = torch.autograd.function.FunctionCtx()
    hidden_states_49 = torch__dynamo_variables_misc_trampoline_autograd_apply_15(hidden_states_48, 0.1);  hidden_states_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_28 = hidden_states_49 + query_states_3;  hidden_states_49 = query_states_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:277, code: hidden_states = hidden_states.float()
    hidden_states_50 = add_28.float();  add_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_21 = hidden_states_50.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_14 = hidden_states_50 - mean_21
    pow_8 = sub_14.pow(2);  sub_14 = None
    variance_7 = pow_8.mean(-1, keepdim = True);  pow_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_15 = hidden_states_50 - mean_21;  hidden_states_50 = mean_21 = None
    add_29 = variance_7 + 1e-07;  variance_7 = None
    sqrt_11 = torch.sqrt(add_29);  add_29 = None
    hidden_states_51 = sub_15 / sqrt_11;  sub_15 = sqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:281, code: hidden_states = hidden_states.to(input_type)
    hidden_states_52 = hidden_states_51.to(torch.float32);  hidden_states_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    l__mod___deberta_encoder_layer_3_attention_output_layer_norm_weight = self.L__mod___deberta_encoder_layer_3_attention_output_LayerNorm_weight
    mul_13 = l__mod___deberta_encoder_layer_3_attention_output_layer_norm_weight * hidden_states_52;  l__mod___deberta_encoder_layer_3_attention_output_layer_norm_weight = hidden_states_52 = None
    l__mod___deberta_encoder_layer_3_attention_output_layer_norm_bias = self.L__mod___deberta_encoder_layer_3_attention_output_LayerNorm_bias
    attention_output_6 = mul_13 + l__mod___deberta_encoder_layer_3_attention_output_layer_norm_bias;  mul_13 = l__mod___deberta_encoder_layer_3_attention_output_layer_norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    hidden_states_54 = self.L__mod___deberta_encoder_layer_3_intermediate_dense(attention_output_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_3 = torch._C._nn.gelu(hidden_states_54);  hidden_states_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    hidden_states_56 = self.L__mod___deberta_encoder_layer_3_output_dense(intermediate_output_3);  intermediate_output_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    function_ctx_16 = torch.autograd.function.FunctionCtx()
    hidden_states_57 = torch__dynamo_variables_misc_trampoline_autograd_apply_16(hidden_states_56, 0.1);  hidden_states_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_31 = hidden_states_57 + attention_output_6;  hidden_states_57 = attention_output_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:277, code: hidden_states = hidden_states.float()
    hidden_states_58 = add_31.float();  add_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_24 = hidden_states_58.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_16 = hidden_states_58 - mean_24
    pow_9 = sub_16.pow(2);  sub_16 = None
    variance_8 = pow_9.mean(-1, keepdim = True);  pow_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_17 = hidden_states_58 - mean_24;  hidden_states_58 = mean_24 = None
    add_32 = variance_8 + 1e-07;  variance_8 = None
    sqrt_12 = torch.sqrt(add_32);  add_32 = None
    hidden_states_59 = sub_17 / sqrt_12;  sub_17 = sqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:281, code: hidden_states = hidden_states.to(input_type)
    hidden_states_60 = hidden_states_59.to(torch.float32);  hidden_states_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    l__mod___deberta_encoder_layer_3_output_layer_norm_weight = self.L__mod___deberta_encoder_layer_3_output_LayerNorm_weight
    mul_14 = l__mod___deberta_encoder_layer_3_output_layer_norm_weight * hidden_states_60;  l__mod___deberta_encoder_layer_3_output_layer_norm_weight = hidden_states_60 = None
    l__mod___deberta_encoder_layer_3_output_layer_norm_bias = self.L__mod___deberta_encoder_layer_3_output_LayerNorm_bias
    query_states_4 = mul_14 + l__mod___deberta_encoder_layer_3_output_layer_norm_bias;  mul_14 = l__mod___deberta_encoder_layer_3_output_layer_norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    qp_4 = self.L__mod___deberta_encoder_layer_4_attention_self_in_proj(query_states_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    x_12 = qp_4.view((1, 512, 12, -1));  qp_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_16 = x_12.permute(0, 2, 1, 3);  x_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    chunk_4 = permute_16.chunk(3, dim = -1);  permute_16 = None
    query_layer_12 = chunk_4[0]
    key_layer_4 = chunk_4[1]
    value_layer_8 = chunk_4[2];  chunk_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    l__mod___deberta_encoder_layer_4_attention_self_q_bias = self.L__mod___deberta_encoder_layer_4_attention_self_q_bias
    getitem_24 = l__mod___deberta_encoder_layer_4_attention_self_q_bias[(None, None, slice(None, None, None))];  l__mod___deberta_encoder_layer_4_attention_self_q_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    x_13 = getitem_24.view((1, 1, 12, -1));  getitem_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_17 = x_13.permute(0, 2, 1, 3);  x_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    query_layer_13 = query_layer_12 + permute_17;  query_layer_12 = permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    l__mod___deberta_encoder_layer_4_attention_self_v_bias = self.L__mod___deberta_encoder_layer_4_attention_self_v_bias
    getitem_25 = l__mod___deberta_encoder_layer_4_attention_self_v_bias[(None, None, slice(None, None, None))];  l__mod___deberta_encoder_layer_4_attention_self_v_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    x_14 = getitem_25.view((1, 1, 12, -1));  getitem_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_18 = x_14.permute(0, 2, 1, 3);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    value_layer_9 = value_layer_8 + permute_18;  value_layer_8 = permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    tensor_4 = torch.tensor(64, dtype = torch.float32)
    mul_15 = tensor_4 * 1;  tensor_4 = None
    scale_4 = torch.sqrt(mul_15);  mul_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    to_14 = scale_4.to(dtype = torch.float32);  scale_4 = None
    query_layer_14 = query_layer_13 / to_14;  query_layer_13 = to_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_4 = key_layer_4.transpose(-1, -2);  key_layer_4 = None
    attention_scores_4 = torch.matmul(query_layer_14, transpose_4);  query_layer_14 = transpose_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    function_ctx_17 = torch.autograd.function.FunctionCtx()
    attention_probs_8 = torch__dynamo_variables_misc_trampoline_autograd_apply_17(attention_scores_4, attention_mask_2, -1);  attention_scores_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    function_ctx_18 = torch.autograd.function.FunctionCtx()
    attention_probs_9 = torch__dynamo_variables_misc_trampoline_autograd_apply_18(attention_probs_8, 0.1);  attention_probs_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_12 = torch.matmul(attention_probs_9, value_layer_9);  attention_probs_9 = value_layer_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_19 = context_layer_12.permute(0, 2, 1, 3);  context_layer_12 = None
    context_layer_13 = permute_19.contiguous();  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    self_output_4 = context_layer_13.view((1, 512, -1));  context_layer_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    hidden_states_63 = self.L__mod___deberta_encoder_layer_4_attention_output_dense(self_output_4);  self_output_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    function_ctx_19 = torch.autograd.function.FunctionCtx()
    hidden_states_64 = torch__dynamo_variables_misc_trampoline_autograd_apply_19(hidden_states_63, 0.1);  hidden_states_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_36 = hidden_states_64 + query_states_4;  hidden_states_64 = query_states_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:277, code: hidden_states = hidden_states.float()
    hidden_states_65 = add_36.float();  add_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_27 = hidden_states_65.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_18 = hidden_states_65 - mean_27
    pow_10 = sub_18.pow(2);  sub_18 = None
    variance_9 = pow_10.mean(-1, keepdim = True);  pow_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_19 = hidden_states_65 - mean_27;  hidden_states_65 = mean_27 = None
    add_37 = variance_9 + 1e-07;  variance_9 = None
    sqrt_14 = torch.sqrt(add_37);  add_37 = None
    hidden_states_66 = sub_19 / sqrt_14;  sub_19 = sqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:281, code: hidden_states = hidden_states.to(input_type)
    hidden_states_67 = hidden_states_66.to(torch.float32);  hidden_states_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    l__mod___deberta_encoder_layer_4_attention_output_layer_norm_weight = self.L__mod___deberta_encoder_layer_4_attention_output_LayerNorm_weight
    mul_16 = l__mod___deberta_encoder_layer_4_attention_output_layer_norm_weight * hidden_states_67;  l__mod___deberta_encoder_layer_4_attention_output_layer_norm_weight = hidden_states_67 = None
    l__mod___deberta_encoder_layer_4_attention_output_layer_norm_bias = self.L__mod___deberta_encoder_layer_4_attention_output_LayerNorm_bias
    attention_output_8 = mul_16 + l__mod___deberta_encoder_layer_4_attention_output_layer_norm_bias;  mul_16 = l__mod___deberta_encoder_layer_4_attention_output_layer_norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    hidden_states_69 = self.L__mod___deberta_encoder_layer_4_intermediate_dense(attention_output_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_4 = torch._C._nn.gelu(hidden_states_69);  hidden_states_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    hidden_states_71 = self.L__mod___deberta_encoder_layer_4_output_dense(intermediate_output_4);  intermediate_output_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    function_ctx_20 = torch.autograd.function.FunctionCtx()
    hidden_states_72 = torch__dynamo_variables_misc_trampoline_autograd_apply_20(hidden_states_71, 0.1);  hidden_states_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_39 = hidden_states_72 + attention_output_8;  hidden_states_72 = attention_output_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:277, code: hidden_states = hidden_states.float()
    hidden_states_73 = add_39.float();  add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_30 = hidden_states_73.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_20 = hidden_states_73 - mean_30
    pow_11 = sub_20.pow(2);  sub_20 = None
    variance_10 = pow_11.mean(-1, keepdim = True);  pow_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_21 = hidden_states_73 - mean_30;  hidden_states_73 = mean_30 = None
    add_40 = variance_10 + 1e-07;  variance_10 = None
    sqrt_15 = torch.sqrt(add_40);  add_40 = None
    hidden_states_74 = sub_21 / sqrt_15;  sub_21 = sqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:281, code: hidden_states = hidden_states.to(input_type)
    hidden_states_75 = hidden_states_74.to(torch.float32);  hidden_states_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    l__mod___deberta_encoder_layer_4_output_layer_norm_weight = self.L__mod___deberta_encoder_layer_4_output_LayerNorm_weight
    mul_17 = l__mod___deberta_encoder_layer_4_output_layer_norm_weight * hidden_states_75;  l__mod___deberta_encoder_layer_4_output_layer_norm_weight = hidden_states_75 = None
    l__mod___deberta_encoder_layer_4_output_layer_norm_bias = self.L__mod___deberta_encoder_layer_4_output_LayerNorm_bias
    query_states_5 = mul_17 + l__mod___deberta_encoder_layer_4_output_layer_norm_bias;  mul_17 = l__mod___deberta_encoder_layer_4_output_layer_norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    qp_5 = self.L__mod___deberta_encoder_layer_5_attention_self_in_proj(query_states_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    x_15 = qp_5.view((1, 512, 12, -1));  qp_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_20 = x_15.permute(0, 2, 1, 3);  x_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    chunk_5 = permute_20.chunk(3, dim = -1);  permute_20 = None
    query_layer_15 = chunk_5[0]
    key_layer_5 = chunk_5[1]
    value_layer_10 = chunk_5[2];  chunk_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    l__mod___deberta_encoder_layer_5_attention_self_q_bias = self.L__mod___deberta_encoder_layer_5_attention_self_q_bias
    getitem_29 = l__mod___deberta_encoder_layer_5_attention_self_q_bias[(None, None, slice(None, None, None))];  l__mod___deberta_encoder_layer_5_attention_self_q_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    x_16 = getitem_29.view((1, 1, 12, -1));  getitem_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_21 = x_16.permute(0, 2, 1, 3);  x_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    query_layer_16 = query_layer_15 + permute_21;  query_layer_15 = permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    l__mod___deberta_encoder_layer_5_attention_self_v_bias = self.L__mod___deberta_encoder_layer_5_attention_self_v_bias
    getitem_30 = l__mod___deberta_encoder_layer_5_attention_self_v_bias[(None, None, slice(None, None, None))];  l__mod___deberta_encoder_layer_5_attention_self_v_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    x_17 = getitem_30.view((1, 1, 12, -1));  getitem_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_22 = x_17.permute(0, 2, 1, 3);  x_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    value_layer_11 = value_layer_10 + permute_22;  value_layer_10 = permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    tensor_5 = torch.tensor(64, dtype = torch.float32)
    mul_18 = tensor_5 * 1;  tensor_5 = None
    scale_5 = torch.sqrt(mul_18);  mul_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    to_17 = scale_5.to(dtype = torch.float32);  scale_5 = None
    query_layer_17 = query_layer_16 / to_17;  query_layer_16 = to_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_5 = key_layer_5.transpose(-1, -2);  key_layer_5 = None
    attention_scores_5 = torch.matmul(query_layer_17, transpose_5);  query_layer_17 = transpose_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    function_ctx_21 = torch.autograd.function.FunctionCtx()
    attention_probs_10 = torch__dynamo_variables_misc_trampoline_autograd_apply_21(attention_scores_5, attention_mask_2, -1);  attention_scores_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    function_ctx_22 = torch.autograd.function.FunctionCtx()
    attention_probs_11 = torch__dynamo_variables_misc_trampoline_autograd_apply_22(attention_probs_10, 0.1);  attention_probs_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_15 = torch.matmul(attention_probs_11, value_layer_11);  attention_probs_11 = value_layer_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_23 = context_layer_15.permute(0, 2, 1, 3);  context_layer_15 = None
    context_layer_16 = permute_23.contiguous();  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    self_output_5 = context_layer_16.view((1, 512, -1));  context_layer_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    hidden_states_78 = self.L__mod___deberta_encoder_layer_5_attention_output_dense(self_output_5);  self_output_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    function_ctx_23 = torch.autograd.function.FunctionCtx()
    hidden_states_79 = torch__dynamo_variables_misc_trampoline_autograd_apply_23(hidden_states_78, 0.1);  hidden_states_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_44 = hidden_states_79 + query_states_5;  hidden_states_79 = query_states_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:277, code: hidden_states = hidden_states.float()
    hidden_states_80 = add_44.float();  add_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_33 = hidden_states_80.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_22 = hidden_states_80 - mean_33
    pow_12 = sub_22.pow(2);  sub_22 = None
    variance_11 = pow_12.mean(-1, keepdim = True);  pow_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_23 = hidden_states_80 - mean_33;  hidden_states_80 = mean_33 = None
    add_45 = variance_11 + 1e-07;  variance_11 = None
    sqrt_17 = torch.sqrt(add_45);  add_45 = None
    hidden_states_81 = sub_23 / sqrt_17;  sub_23 = sqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:281, code: hidden_states = hidden_states.to(input_type)
    hidden_states_82 = hidden_states_81.to(torch.float32);  hidden_states_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    l__mod___deberta_encoder_layer_5_attention_output_layer_norm_weight = self.L__mod___deberta_encoder_layer_5_attention_output_LayerNorm_weight
    mul_19 = l__mod___deberta_encoder_layer_5_attention_output_layer_norm_weight * hidden_states_82;  l__mod___deberta_encoder_layer_5_attention_output_layer_norm_weight = hidden_states_82 = None
    l__mod___deberta_encoder_layer_5_attention_output_layer_norm_bias = self.L__mod___deberta_encoder_layer_5_attention_output_LayerNorm_bias
    attention_output_10 = mul_19 + l__mod___deberta_encoder_layer_5_attention_output_layer_norm_bias;  mul_19 = l__mod___deberta_encoder_layer_5_attention_output_layer_norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    hidden_states_84 = self.L__mod___deberta_encoder_layer_5_intermediate_dense(attention_output_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_5 = torch._C._nn.gelu(hidden_states_84);  hidden_states_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    hidden_states_86 = self.L__mod___deberta_encoder_layer_5_output_dense(intermediate_output_5);  intermediate_output_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    function_ctx_24 = torch.autograd.function.FunctionCtx()
    hidden_states_87 = torch__dynamo_variables_misc_trampoline_autograd_apply_24(hidden_states_86, 0.1);  hidden_states_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_47 = hidden_states_87 + attention_output_10;  hidden_states_87 = attention_output_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:277, code: hidden_states = hidden_states.float()
    hidden_states_88 = add_47.float();  add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_36 = hidden_states_88.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_24 = hidden_states_88 - mean_36
    pow_13 = sub_24.pow(2);  sub_24 = None
    variance_12 = pow_13.mean(-1, keepdim = True);  pow_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_25 = hidden_states_88 - mean_36;  hidden_states_88 = mean_36 = None
    add_48 = variance_12 + 1e-07;  variance_12 = None
    sqrt_18 = torch.sqrt(add_48);  add_48 = None
    hidden_states_89 = sub_25 / sqrt_18;  sub_25 = sqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:281, code: hidden_states = hidden_states.to(input_type)
    hidden_states_90 = hidden_states_89.to(torch.float32);  hidden_states_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    l__mod___deberta_encoder_layer_5_output_layer_norm_weight = self.L__mod___deberta_encoder_layer_5_output_LayerNorm_weight
    mul_20 = l__mod___deberta_encoder_layer_5_output_layer_norm_weight * hidden_states_90;  l__mod___deberta_encoder_layer_5_output_layer_norm_weight = hidden_states_90 = None
    l__mod___deberta_encoder_layer_5_output_layer_norm_bias = self.L__mod___deberta_encoder_layer_5_output_LayerNorm_bias
    query_states_6 = mul_20 + l__mod___deberta_encoder_layer_5_output_layer_norm_bias;  mul_20 = l__mod___deberta_encoder_layer_5_output_layer_norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    qp_6 = self.L__mod___deberta_encoder_layer_6_attention_self_in_proj(query_states_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    x_18 = qp_6.view((1, 512, 12, -1));  qp_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_24 = x_18.permute(0, 2, 1, 3);  x_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    chunk_6 = permute_24.chunk(3, dim = -1);  permute_24 = None
    query_layer_18 = chunk_6[0]
    key_layer_6 = chunk_6[1]
    value_layer_12 = chunk_6[2];  chunk_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    l__mod___deberta_encoder_layer_6_attention_self_q_bias = self.L__mod___deberta_encoder_layer_6_attention_self_q_bias
    getitem_34 = l__mod___deberta_encoder_layer_6_attention_self_q_bias[(None, None, slice(None, None, None))];  l__mod___deberta_encoder_layer_6_attention_self_q_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    x_19 = getitem_34.view((1, 1, 12, -1));  getitem_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_25 = x_19.permute(0, 2, 1, 3);  x_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    query_layer_19 = query_layer_18 + permute_25;  query_layer_18 = permute_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    l__mod___deberta_encoder_layer_6_attention_self_v_bias = self.L__mod___deberta_encoder_layer_6_attention_self_v_bias
    getitem_35 = l__mod___deberta_encoder_layer_6_attention_self_v_bias[(None, None, slice(None, None, None))];  l__mod___deberta_encoder_layer_6_attention_self_v_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    x_20 = getitem_35.view((1, 1, 12, -1));  getitem_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_26 = x_20.permute(0, 2, 1, 3);  x_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    value_layer_13 = value_layer_12 + permute_26;  value_layer_12 = permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    tensor_6 = torch.tensor(64, dtype = torch.float32)
    mul_21 = tensor_6 * 1;  tensor_6 = None
    scale_6 = torch.sqrt(mul_21);  mul_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    to_20 = scale_6.to(dtype = torch.float32);  scale_6 = None
    query_layer_20 = query_layer_19 / to_20;  query_layer_19 = to_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_6 = key_layer_6.transpose(-1, -2);  key_layer_6 = None
    attention_scores_6 = torch.matmul(query_layer_20, transpose_6);  query_layer_20 = transpose_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    function_ctx_25 = torch.autograd.function.FunctionCtx()
    attention_probs_12 = torch__dynamo_variables_misc_trampoline_autograd_apply_25(attention_scores_6, attention_mask_2, -1);  attention_scores_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    function_ctx_26 = torch.autograd.function.FunctionCtx()
    attention_probs_13 = torch__dynamo_variables_misc_trampoline_autograd_apply_26(attention_probs_12, 0.1);  attention_probs_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_18 = torch.matmul(attention_probs_13, value_layer_13);  attention_probs_13 = value_layer_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_27 = context_layer_18.permute(0, 2, 1, 3);  context_layer_18 = None
    context_layer_19 = permute_27.contiguous();  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    self_output_6 = context_layer_19.view((1, 512, -1));  context_layer_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    hidden_states_93 = self.L__mod___deberta_encoder_layer_6_attention_output_dense(self_output_6);  self_output_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    function_ctx_27 = torch.autograd.function.FunctionCtx()
    hidden_states_94 = torch__dynamo_variables_misc_trampoline_autograd_apply_27(hidden_states_93, 0.1);  hidden_states_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_52 = hidden_states_94 + query_states_6;  hidden_states_94 = query_states_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:277, code: hidden_states = hidden_states.float()
    hidden_states_95 = add_52.float();  add_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_39 = hidden_states_95.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_26 = hidden_states_95 - mean_39
    pow_14 = sub_26.pow(2);  sub_26 = None
    variance_13 = pow_14.mean(-1, keepdim = True);  pow_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_27 = hidden_states_95 - mean_39;  hidden_states_95 = mean_39 = None
    add_53 = variance_13 + 1e-07;  variance_13 = None
    sqrt_20 = torch.sqrt(add_53);  add_53 = None
    hidden_states_96 = sub_27 / sqrt_20;  sub_27 = sqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:281, code: hidden_states = hidden_states.to(input_type)
    hidden_states_97 = hidden_states_96.to(torch.float32);  hidden_states_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    l__mod___deberta_encoder_layer_6_attention_output_layer_norm_weight = self.L__mod___deberta_encoder_layer_6_attention_output_LayerNorm_weight
    mul_22 = l__mod___deberta_encoder_layer_6_attention_output_layer_norm_weight * hidden_states_97;  l__mod___deberta_encoder_layer_6_attention_output_layer_norm_weight = hidden_states_97 = None
    l__mod___deberta_encoder_layer_6_attention_output_layer_norm_bias = self.L__mod___deberta_encoder_layer_6_attention_output_LayerNorm_bias
    attention_output_12 = mul_22 + l__mod___deberta_encoder_layer_6_attention_output_layer_norm_bias;  mul_22 = l__mod___deberta_encoder_layer_6_attention_output_layer_norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    hidden_states_99 = self.L__mod___deberta_encoder_layer_6_intermediate_dense(attention_output_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_6 = torch._C._nn.gelu(hidden_states_99);  hidden_states_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    hidden_states_101 = self.L__mod___deberta_encoder_layer_6_output_dense(intermediate_output_6);  intermediate_output_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    function_ctx_28 = torch.autograd.function.FunctionCtx()
    hidden_states_102 = torch__dynamo_variables_misc_trampoline_autograd_apply_28(hidden_states_101, 0.1);  hidden_states_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_55 = hidden_states_102 + attention_output_12;  hidden_states_102 = attention_output_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:277, code: hidden_states = hidden_states.float()
    hidden_states_103 = add_55.float();  add_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_42 = hidden_states_103.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_28 = hidden_states_103 - mean_42
    pow_15 = sub_28.pow(2);  sub_28 = None
    variance_14 = pow_15.mean(-1, keepdim = True);  pow_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_29 = hidden_states_103 - mean_42;  hidden_states_103 = mean_42 = None
    add_56 = variance_14 + 1e-07;  variance_14 = None
    sqrt_21 = torch.sqrt(add_56);  add_56 = None
    hidden_states_104 = sub_29 / sqrt_21;  sub_29 = sqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:281, code: hidden_states = hidden_states.to(input_type)
    hidden_states_105 = hidden_states_104.to(torch.float32);  hidden_states_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    l__mod___deberta_encoder_layer_6_output_layer_norm_weight = self.L__mod___deberta_encoder_layer_6_output_LayerNorm_weight
    mul_23 = l__mod___deberta_encoder_layer_6_output_layer_norm_weight * hidden_states_105;  l__mod___deberta_encoder_layer_6_output_layer_norm_weight = hidden_states_105 = None
    l__mod___deberta_encoder_layer_6_output_layer_norm_bias = self.L__mod___deberta_encoder_layer_6_output_LayerNorm_bias
    query_states_7 = mul_23 + l__mod___deberta_encoder_layer_6_output_layer_norm_bias;  mul_23 = l__mod___deberta_encoder_layer_6_output_layer_norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    qp_7 = self.L__mod___deberta_encoder_layer_7_attention_self_in_proj(query_states_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    x_21 = qp_7.view((1, 512, 12, -1));  qp_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_28 = x_21.permute(0, 2, 1, 3);  x_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    chunk_7 = permute_28.chunk(3, dim = -1);  permute_28 = None
    query_layer_21 = chunk_7[0]
    key_layer_7 = chunk_7[1]
    value_layer_14 = chunk_7[2];  chunk_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    l__mod___deberta_encoder_layer_7_attention_self_q_bias = self.L__mod___deberta_encoder_layer_7_attention_self_q_bias
    getitem_39 = l__mod___deberta_encoder_layer_7_attention_self_q_bias[(None, None, slice(None, None, None))];  l__mod___deberta_encoder_layer_7_attention_self_q_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    x_22 = getitem_39.view((1, 1, 12, -1));  getitem_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_29 = x_22.permute(0, 2, 1, 3);  x_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    query_layer_22 = query_layer_21 + permute_29;  query_layer_21 = permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    l__mod___deberta_encoder_layer_7_attention_self_v_bias = self.L__mod___deberta_encoder_layer_7_attention_self_v_bias
    getitem_40 = l__mod___deberta_encoder_layer_7_attention_self_v_bias[(None, None, slice(None, None, None))];  l__mod___deberta_encoder_layer_7_attention_self_v_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    x_23 = getitem_40.view((1, 1, 12, -1));  getitem_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_30 = x_23.permute(0, 2, 1, 3);  x_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    value_layer_15 = value_layer_14 + permute_30;  value_layer_14 = permute_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    tensor_7 = torch.tensor(64, dtype = torch.float32)
    mul_24 = tensor_7 * 1;  tensor_7 = None
    scale_7 = torch.sqrt(mul_24);  mul_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    to_23 = scale_7.to(dtype = torch.float32);  scale_7 = None
    query_layer_23 = query_layer_22 / to_23;  query_layer_22 = to_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_7 = key_layer_7.transpose(-1, -2);  key_layer_7 = None
    attention_scores_7 = torch.matmul(query_layer_23, transpose_7);  query_layer_23 = transpose_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    function_ctx_29 = torch.autograd.function.FunctionCtx()
    attention_probs_14 = torch__dynamo_variables_misc_trampoline_autograd_apply_29(attention_scores_7, attention_mask_2, -1);  attention_scores_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    function_ctx_30 = torch.autograd.function.FunctionCtx()
    attention_probs_15 = torch__dynamo_variables_misc_trampoline_autograd_apply_30(attention_probs_14, 0.1);  attention_probs_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_21 = torch.matmul(attention_probs_15, value_layer_15);  attention_probs_15 = value_layer_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_31 = context_layer_21.permute(0, 2, 1, 3);  context_layer_21 = None
    context_layer_22 = permute_31.contiguous();  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    self_output_7 = context_layer_22.view((1, 512, -1));  context_layer_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    hidden_states_108 = self.L__mod___deberta_encoder_layer_7_attention_output_dense(self_output_7);  self_output_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    function_ctx_31 = torch.autograd.function.FunctionCtx()
    hidden_states_109 = torch__dynamo_variables_misc_trampoline_autograd_apply_31(hidden_states_108, 0.1);  hidden_states_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_60 = hidden_states_109 + query_states_7;  hidden_states_109 = query_states_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:277, code: hidden_states = hidden_states.float()
    hidden_states_110 = add_60.float();  add_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_45 = hidden_states_110.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_30 = hidden_states_110 - mean_45
    pow_16 = sub_30.pow(2);  sub_30 = None
    variance_15 = pow_16.mean(-1, keepdim = True);  pow_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_31 = hidden_states_110 - mean_45;  hidden_states_110 = mean_45 = None
    add_61 = variance_15 + 1e-07;  variance_15 = None
    sqrt_23 = torch.sqrt(add_61);  add_61 = None
    hidden_states_111 = sub_31 / sqrt_23;  sub_31 = sqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:281, code: hidden_states = hidden_states.to(input_type)
    hidden_states_112 = hidden_states_111.to(torch.float32);  hidden_states_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    l__mod___deberta_encoder_layer_7_attention_output_layer_norm_weight = self.L__mod___deberta_encoder_layer_7_attention_output_LayerNorm_weight
    mul_25 = l__mod___deberta_encoder_layer_7_attention_output_layer_norm_weight * hidden_states_112;  l__mod___deberta_encoder_layer_7_attention_output_layer_norm_weight = hidden_states_112 = None
    l__mod___deberta_encoder_layer_7_attention_output_layer_norm_bias = self.L__mod___deberta_encoder_layer_7_attention_output_LayerNorm_bias
    attention_output_14 = mul_25 + l__mod___deberta_encoder_layer_7_attention_output_layer_norm_bias;  mul_25 = l__mod___deberta_encoder_layer_7_attention_output_layer_norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    hidden_states_114 = self.L__mod___deberta_encoder_layer_7_intermediate_dense(attention_output_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_7 = torch._C._nn.gelu(hidden_states_114);  hidden_states_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    hidden_states_116 = self.L__mod___deberta_encoder_layer_7_output_dense(intermediate_output_7);  intermediate_output_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    function_ctx_32 = torch.autograd.function.FunctionCtx()
    hidden_states_117 = torch__dynamo_variables_misc_trampoline_autograd_apply_32(hidden_states_116, 0.1);  hidden_states_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_63 = hidden_states_117 + attention_output_14;  hidden_states_117 = attention_output_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:277, code: hidden_states = hidden_states.float()
    hidden_states_118 = add_63.float();  add_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_48 = hidden_states_118.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_32 = hidden_states_118 - mean_48
    pow_17 = sub_32.pow(2);  sub_32 = None
    variance_16 = pow_17.mean(-1, keepdim = True);  pow_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_33 = hidden_states_118 - mean_48;  hidden_states_118 = mean_48 = None
    add_64 = variance_16 + 1e-07;  variance_16 = None
    sqrt_24 = torch.sqrt(add_64);  add_64 = None
    hidden_states_119 = sub_33 / sqrt_24;  sub_33 = sqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:281, code: hidden_states = hidden_states.to(input_type)
    hidden_states_120 = hidden_states_119.to(torch.float32);  hidden_states_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    l__mod___deberta_encoder_layer_7_output_layer_norm_weight = self.L__mod___deberta_encoder_layer_7_output_LayerNorm_weight
    mul_26 = l__mod___deberta_encoder_layer_7_output_layer_norm_weight * hidden_states_120;  l__mod___deberta_encoder_layer_7_output_layer_norm_weight = hidden_states_120 = None
    l__mod___deberta_encoder_layer_7_output_layer_norm_bias = self.L__mod___deberta_encoder_layer_7_output_LayerNorm_bias
    query_states_8 = mul_26 + l__mod___deberta_encoder_layer_7_output_layer_norm_bias;  mul_26 = l__mod___deberta_encoder_layer_7_output_layer_norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    qp_8 = self.L__mod___deberta_encoder_layer_8_attention_self_in_proj(query_states_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    x_24 = qp_8.view((1, 512, 12, -1));  qp_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_32 = x_24.permute(0, 2, 1, 3);  x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    chunk_8 = permute_32.chunk(3, dim = -1);  permute_32 = None
    query_layer_24 = chunk_8[0]
    key_layer_8 = chunk_8[1]
    value_layer_16 = chunk_8[2];  chunk_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    l__mod___deberta_encoder_layer_8_attention_self_q_bias = self.L__mod___deberta_encoder_layer_8_attention_self_q_bias
    getitem_44 = l__mod___deberta_encoder_layer_8_attention_self_q_bias[(None, None, slice(None, None, None))];  l__mod___deberta_encoder_layer_8_attention_self_q_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    x_25 = getitem_44.view((1, 1, 12, -1));  getitem_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_33 = x_25.permute(0, 2, 1, 3);  x_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    query_layer_25 = query_layer_24 + permute_33;  query_layer_24 = permute_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    l__mod___deberta_encoder_layer_8_attention_self_v_bias = self.L__mod___deberta_encoder_layer_8_attention_self_v_bias
    getitem_45 = l__mod___deberta_encoder_layer_8_attention_self_v_bias[(None, None, slice(None, None, None))];  l__mod___deberta_encoder_layer_8_attention_self_v_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    x_26 = getitem_45.view((1, 1, 12, -1));  getitem_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_34 = x_26.permute(0, 2, 1, 3);  x_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    value_layer_17 = value_layer_16 + permute_34;  value_layer_16 = permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    tensor_8 = torch.tensor(64, dtype = torch.float32)
    mul_27 = tensor_8 * 1;  tensor_8 = None
    scale_8 = torch.sqrt(mul_27);  mul_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    to_26 = scale_8.to(dtype = torch.float32);  scale_8 = None
    query_layer_26 = query_layer_25 / to_26;  query_layer_25 = to_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_8 = key_layer_8.transpose(-1, -2);  key_layer_8 = None
    attention_scores_8 = torch.matmul(query_layer_26, transpose_8);  query_layer_26 = transpose_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    function_ctx_33 = torch.autograd.function.FunctionCtx()
    attention_probs_16 = torch__dynamo_variables_misc_trampoline_autograd_apply_33(attention_scores_8, attention_mask_2, -1);  attention_scores_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    function_ctx_34 = torch.autograd.function.FunctionCtx()
    attention_probs_17 = torch__dynamo_variables_misc_trampoline_autograd_apply_34(attention_probs_16, 0.1);  attention_probs_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_24 = torch.matmul(attention_probs_17, value_layer_17);  attention_probs_17 = value_layer_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_35 = context_layer_24.permute(0, 2, 1, 3);  context_layer_24 = None
    context_layer_25 = permute_35.contiguous();  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    self_output_8 = context_layer_25.view((1, 512, -1));  context_layer_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    hidden_states_123 = self.L__mod___deberta_encoder_layer_8_attention_output_dense(self_output_8);  self_output_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    function_ctx_35 = torch.autograd.function.FunctionCtx()
    hidden_states_124 = torch__dynamo_variables_misc_trampoline_autograd_apply_35(hidden_states_123, 0.1);  hidden_states_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_68 = hidden_states_124 + query_states_8;  hidden_states_124 = query_states_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:277, code: hidden_states = hidden_states.float()
    hidden_states_125 = add_68.float();  add_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_51 = hidden_states_125.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_34 = hidden_states_125 - mean_51
    pow_18 = sub_34.pow(2);  sub_34 = None
    variance_17 = pow_18.mean(-1, keepdim = True);  pow_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_35 = hidden_states_125 - mean_51;  hidden_states_125 = mean_51 = None
    add_69 = variance_17 + 1e-07;  variance_17 = None
    sqrt_26 = torch.sqrt(add_69);  add_69 = None
    hidden_states_126 = sub_35 / sqrt_26;  sub_35 = sqrt_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:281, code: hidden_states = hidden_states.to(input_type)
    hidden_states_127 = hidden_states_126.to(torch.float32);  hidden_states_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    l__mod___deberta_encoder_layer_8_attention_output_layer_norm_weight = self.L__mod___deberta_encoder_layer_8_attention_output_LayerNorm_weight
    mul_28 = l__mod___deberta_encoder_layer_8_attention_output_layer_norm_weight * hidden_states_127;  l__mod___deberta_encoder_layer_8_attention_output_layer_norm_weight = hidden_states_127 = None
    l__mod___deberta_encoder_layer_8_attention_output_layer_norm_bias = self.L__mod___deberta_encoder_layer_8_attention_output_LayerNorm_bias
    attention_output_16 = mul_28 + l__mod___deberta_encoder_layer_8_attention_output_layer_norm_bias;  mul_28 = l__mod___deberta_encoder_layer_8_attention_output_layer_norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    hidden_states_129 = self.L__mod___deberta_encoder_layer_8_intermediate_dense(attention_output_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_8 = torch._C._nn.gelu(hidden_states_129);  hidden_states_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    hidden_states_131 = self.L__mod___deberta_encoder_layer_8_output_dense(intermediate_output_8);  intermediate_output_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    function_ctx_36 = torch.autograd.function.FunctionCtx()
    hidden_states_132 = torch__dynamo_variables_misc_trampoline_autograd_apply_36(hidden_states_131, 0.1);  hidden_states_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_71 = hidden_states_132 + attention_output_16;  hidden_states_132 = attention_output_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:277, code: hidden_states = hidden_states.float()
    hidden_states_133 = add_71.float();  add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_54 = hidden_states_133.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_36 = hidden_states_133 - mean_54
    pow_19 = sub_36.pow(2);  sub_36 = None
    variance_18 = pow_19.mean(-1, keepdim = True);  pow_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_37 = hidden_states_133 - mean_54;  hidden_states_133 = mean_54 = None
    add_72 = variance_18 + 1e-07;  variance_18 = None
    sqrt_27 = torch.sqrt(add_72);  add_72 = None
    hidden_states_134 = sub_37 / sqrt_27;  sub_37 = sqrt_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:281, code: hidden_states = hidden_states.to(input_type)
    hidden_states_135 = hidden_states_134.to(torch.float32);  hidden_states_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    l__mod___deberta_encoder_layer_8_output_layer_norm_weight = self.L__mod___deberta_encoder_layer_8_output_LayerNorm_weight
    mul_29 = l__mod___deberta_encoder_layer_8_output_layer_norm_weight * hidden_states_135;  l__mod___deberta_encoder_layer_8_output_layer_norm_weight = hidden_states_135 = None
    l__mod___deberta_encoder_layer_8_output_layer_norm_bias = self.L__mod___deberta_encoder_layer_8_output_LayerNorm_bias
    query_states_9 = mul_29 + l__mod___deberta_encoder_layer_8_output_layer_norm_bias;  mul_29 = l__mod___deberta_encoder_layer_8_output_layer_norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    qp_9 = self.L__mod___deberta_encoder_layer_9_attention_self_in_proj(query_states_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    x_27 = qp_9.view((1, 512, 12, -1));  qp_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_36 = x_27.permute(0, 2, 1, 3);  x_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    chunk_9 = permute_36.chunk(3, dim = -1);  permute_36 = None
    query_layer_27 = chunk_9[0]
    key_layer_9 = chunk_9[1]
    value_layer_18 = chunk_9[2];  chunk_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    l__mod___deberta_encoder_layer_9_attention_self_q_bias = self.L__mod___deberta_encoder_layer_9_attention_self_q_bias
    getitem_49 = l__mod___deberta_encoder_layer_9_attention_self_q_bias[(None, None, slice(None, None, None))];  l__mod___deberta_encoder_layer_9_attention_self_q_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    x_28 = getitem_49.view((1, 1, 12, -1));  getitem_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_37 = x_28.permute(0, 2, 1, 3);  x_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    query_layer_28 = query_layer_27 + permute_37;  query_layer_27 = permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    l__mod___deberta_encoder_layer_9_attention_self_v_bias = self.L__mod___deberta_encoder_layer_9_attention_self_v_bias
    getitem_50 = l__mod___deberta_encoder_layer_9_attention_self_v_bias[(None, None, slice(None, None, None))];  l__mod___deberta_encoder_layer_9_attention_self_v_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    x_29 = getitem_50.view((1, 1, 12, -1));  getitem_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_38 = x_29.permute(0, 2, 1, 3);  x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    value_layer_19 = value_layer_18 + permute_38;  value_layer_18 = permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    tensor_9 = torch.tensor(64, dtype = torch.float32)
    mul_30 = tensor_9 * 1;  tensor_9 = None
    scale_9 = torch.sqrt(mul_30);  mul_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    to_29 = scale_9.to(dtype = torch.float32);  scale_9 = None
    query_layer_29 = query_layer_28 / to_29;  query_layer_28 = to_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_9 = key_layer_9.transpose(-1, -2);  key_layer_9 = None
    attention_scores_9 = torch.matmul(query_layer_29, transpose_9);  query_layer_29 = transpose_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    function_ctx_37 = torch.autograd.function.FunctionCtx()
    attention_probs_18 = torch__dynamo_variables_misc_trampoline_autograd_apply_37(attention_scores_9, attention_mask_2, -1);  attention_scores_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    function_ctx_38 = torch.autograd.function.FunctionCtx()
    attention_probs_19 = torch__dynamo_variables_misc_trampoline_autograd_apply_38(attention_probs_18, 0.1);  attention_probs_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_27 = torch.matmul(attention_probs_19, value_layer_19);  attention_probs_19 = value_layer_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_39 = context_layer_27.permute(0, 2, 1, 3);  context_layer_27 = None
    context_layer_28 = permute_39.contiguous();  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    self_output_9 = context_layer_28.view((1, 512, -1));  context_layer_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    hidden_states_138 = self.L__mod___deberta_encoder_layer_9_attention_output_dense(self_output_9);  self_output_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    function_ctx_39 = torch.autograd.function.FunctionCtx()
    hidden_states_139 = torch__dynamo_variables_misc_trampoline_autograd_apply_39(hidden_states_138, 0.1);  hidden_states_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_76 = hidden_states_139 + query_states_9;  hidden_states_139 = query_states_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:277, code: hidden_states = hidden_states.float()
    hidden_states_140 = add_76.float();  add_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_57 = hidden_states_140.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_38 = hidden_states_140 - mean_57
    pow_20 = sub_38.pow(2);  sub_38 = None
    variance_19 = pow_20.mean(-1, keepdim = True);  pow_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_39 = hidden_states_140 - mean_57;  hidden_states_140 = mean_57 = None
    add_77 = variance_19 + 1e-07;  variance_19 = None
    sqrt_29 = torch.sqrt(add_77);  add_77 = None
    hidden_states_141 = sub_39 / sqrt_29;  sub_39 = sqrt_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:281, code: hidden_states = hidden_states.to(input_type)
    hidden_states_142 = hidden_states_141.to(torch.float32);  hidden_states_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    l__mod___deberta_encoder_layer_9_attention_output_layer_norm_weight = self.L__mod___deberta_encoder_layer_9_attention_output_LayerNorm_weight
    mul_31 = l__mod___deberta_encoder_layer_9_attention_output_layer_norm_weight * hidden_states_142;  l__mod___deberta_encoder_layer_9_attention_output_layer_norm_weight = hidden_states_142 = None
    l__mod___deberta_encoder_layer_9_attention_output_layer_norm_bias = self.L__mod___deberta_encoder_layer_9_attention_output_LayerNorm_bias
    attention_output_18 = mul_31 + l__mod___deberta_encoder_layer_9_attention_output_layer_norm_bias;  mul_31 = l__mod___deberta_encoder_layer_9_attention_output_layer_norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    hidden_states_144 = self.L__mod___deberta_encoder_layer_9_intermediate_dense(attention_output_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_9 = torch._C._nn.gelu(hidden_states_144);  hidden_states_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    hidden_states_146 = self.L__mod___deberta_encoder_layer_9_output_dense(intermediate_output_9);  intermediate_output_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    function_ctx_40 = torch.autograd.function.FunctionCtx()
    hidden_states_147 = torch__dynamo_variables_misc_trampoline_autograd_apply_40(hidden_states_146, 0.1);  hidden_states_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_79 = hidden_states_147 + attention_output_18;  hidden_states_147 = attention_output_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:277, code: hidden_states = hidden_states.float()
    hidden_states_148 = add_79.float();  add_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_60 = hidden_states_148.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_40 = hidden_states_148 - mean_60
    pow_21 = sub_40.pow(2);  sub_40 = None
    variance_20 = pow_21.mean(-1, keepdim = True);  pow_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_41 = hidden_states_148 - mean_60;  hidden_states_148 = mean_60 = None
    add_80 = variance_20 + 1e-07;  variance_20 = None
    sqrt_30 = torch.sqrt(add_80);  add_80 = None
    hidden_states_149 = sub_41 / sqrt_30;  sub_41 = sqrt_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:281, code: hidden_states = hidden_states.to(input_type)
    hidden_states_150 = hidden_states_149.to(torch.float32);  hidden_states_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    l__mod___deberta_encoder_layer_9_output_layer_norm_weight = self.L__mod___deberta_encoder_layer_9_output_LayerNorm_weight
    mul_32 = l__mod___deberta_encoder_layer_9_output_layer_norm_weight * hidden_states_150;  l__mod___deberta_encoder_layer_9_output_layer_norm_weight = hidden_states_150 = None
    l__mod___deberta_encoder_layer_9_output_layer_norm_bias = self.L__mod___deberta_encoder_layer_9_output_LayerNorm_bias
    query_states_10 = mul_32 + l__mod___deberta_encoder_layer_9_output_layer_norm_bias;  mul_32 = l__mod___deberta_encoder_layer_9_output_layer_norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    qp_10 = self.L__mod___deberta_encoder_layer_10_attention_self_in_proj(query_states_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    x_30 = qp_10.view((1, 512, 12, -1));  qp_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_40 = x_30.permute(0, 2, 1, 3);  x_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    chunk_10 = permute_40.chunk(3, dim = -1);  permute_40 = None
    query_layer_30 = chunk_10[0]
    key_layer_10 = chunk_10[1]
    value_layer_20 = chunk_10[2];  chunk_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    l__mod___deberta_encoder_layer_10_attention_self_q_bias = self.L__mod___deberta_encoder_layer_10_attention_self_q_bias
    getitem_54 = l__mod___deberta_encoder_layer_10_attention_self_q_bias[(None, None, slice(None, None, None))];  l__mod___deberta_encoder_layer_10_attention_self_q_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    x_31 = getitem_54.view((1, 1, 12, -1));  getitem_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_41 = x_31.permute(0, 2, 1, 3);  x_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    query_layer_31 = query_layer_30 + permute_41;  query_layer_30 = permute_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    l__mod___deberta_encoder_layer_10_attention_self_v_bias = self.L__mod___deberta_encoder_layer_10_attention_self_v_bias
    getitem_55 = l__mod___deberta_encoder_layer_10_attention_self_v_bias[(None, None, slice(None, None, None))];  l__mod___deberta_encoder_layer_10_attention_self_v_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    x_32 = getitem_55.view((1, 1, 12, -1));  getitem_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_42 = x_32.permute(0, 2, 1, 3);  x_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    value_layer_21 = value_layer_20 + permute_42;  value_layer_20 = permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    tensor_10 = torch.tensor(64, dtype = torch.float32)
    mul_33 = tensor_10 * 1;  tensor_10 = None
    scale_10 = torch.sqrt(mul_33);  mul_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    to_32 = scale_10.to(dtype = torch.float32);  scale_10 = None
    query_layer_32 = query_layer_31 / to_32;  query_layer_31 = to_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_10 = key_layer_10.transpose(-1, -2);  key_layer_10 = None
    attention_scores_10 = torch.matmul(query_layer_32, transpose_10);  query_layer_32 = transpose_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    function_ctx_41 = torch.autograd.function.FunctionCtx()
    attention_probs_20 = torch__dynamo_variables_misc_trampoline_autograd_apply_41(attention_scores_10, attention_mask_2, -1);  attention_scores_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    function_ctx_42 = torch.autograd.function.FunctionCtx()
    attention_probs_21 = torch__dynamo_variables_misc_trampoline_autograd_apply_42(attention_probs_20, 0.1);  attention_probs_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_30 = torch.matmul(attention_probs_21, value_layer_21);  attention_probs_21 = value_layer_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_43 = context_layer_30.permute(0, 2, 1, 3);  context_layer_30 = None
    context_layer_31 = permute_43.contiguous();  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    self_output_10 = context_layer_31.view((1, 512, -1));  context_layer_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    hidden_states_153 = self.L__mod___deberta_encoder_layer_10_attention_output_dense(self_output_10);  self_output_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    function_ctx_43 = torch.autograd.function.FunctionCtx()
    hidden_states_154 = torch__dynamo_variables_misc_trampoline_autograd_apply_43(hidden_states_153, 0.1);  hidden_states_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_84 = hidden_states_154 + query_states_10;  hidden_states_154 = query_states_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:277, code: hidden_states = hidden_states.float()
    hidden_states_155 = add_84.float();  add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_63 = hidden_states_155.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_42 = hidden_states_155 - mean_63
    pow_22 = sub_42.pow(2);  sub_42 = None
    variance_21 = pow_22.mean(-1, keepdim = True);  pow_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_43 = hidden_states_155 - mean_63;  hidden_states_155 = mean_63 = None
    add_85 = variance_21 + 1e-07;  variance_21 = None
    sqrt_32 = torch.sqrt(add_85);  add_85 = None
    hidden_states_156 = sub_43 / sqrt_32;  sub_43 = sqrt_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:281, code: hidden_states = hidden_states.to(input_type)
    hidden_states_157 = hidden_states_156.to(torch.float32);  hidden_states_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    l__mod___deberta_encoder_layer_10_attention_output_layer_norm_weight = self.L__mod___deberta_encoder_layer_10_attention_output_LayerNorm_weight
    mul_34 = l__mod___deberta_encoder_layer_10_attention_output_layer_norm_weight * hidden_states_157;  l__mod___deberta_encoder_layer_10_attention_output_layer_norm_weight = hidden_states_157 = None
    l__mod___deberta_encoder_layer_10_attention_output_layer_norm_bias = self.L__mod___deberta_encoder_layer_10_attention_output_LayerNorm_bias
    attention_output_20 = mul_34 + l__mod___deberta_encoder_layer_10_attention_output_layer_norm_bias;  mul_34 = l__mod___deberta_encoder_layer_10_attention_output_layer_norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    hidden_states_159 = self.L__mod___deberta_encoder_layer_10_intermediate_dense(attention_output_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_10 = torch._C._nn.gelu(hidden_states_159);  hidden_states_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    hidden_states_161 = self.L__mod___deberta_encoder_layer_10_output_dense(intermediate_output_10);  intermediate_output_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    function_ctx_44 = torch.autograd.function.FunctionCtx()
    hidden_states_162 = torch__dynamo_variables_misc_trampoline_autograd_apply_44(hidden_states_161, 0.1);  hidden_states_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_87 = hidden_states_162 + attention_output_20;  hidden_states_162 = attention_output_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:277, code: hidden_states = hidden_states.float()
    hidden_states_163 = add_87.float();  add_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_66 = hidden_states_163.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_44 = hidden_states_163 - mean_66
    pow_23 = sub_44.pow(2);  sub_44 = None
    variance_22 = pow_23.mean(-1, keepdim = True);  pow_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_45 = hidden_states_163 - mean_66;  hidden_states_163 = mean_66 = None
    add_88 = variance_22 + 1e-07;  variance_22 = None
    sqrt_33 = torch.sqrt(add_88);  add_88 = None
    hidden_states_164 = sub_45 / sqrt_33;  sub_45 = sqrt_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:281, code: hidden_states = hidden_states.to(input_type)
    hidden_states_165 = hidden_states_164.to(torch.float32);  hidden_states_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    l__mod___deberta_encoder_layer_10_output_layer_norm_weight = self.L__mod___deberta_encoder_layer_10_output_LayerNorm_weight
    mul_35 = l__mod___deberta_encoder_layer_10_output_layer_norm_weight * hidden_states_165;  l__mod___deberta_encoder_layer_10_output_layer_norm_weight = hidden_states_165 = None
    l__mod___deberta_encoder_layer_10_output_layer_norm_bias = self.L__mod___deberta_encoder_layer_10_output_LayerNorm_bias
    query_states_11 = mul_35 + l__mod___deberta_encoder_layer_10_output_layer_norm_bias;  mul_35 = l__mod___deberta_encoder_layer_10_output_layer_norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    qp_11 = self.L__mod___deberta_encoder_layer_11_attention_self_in_proj(query_states_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    x_33 = qp_11.view((1, 512, 12, -1));  qp_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_44 = x_33.permute(0, 2, 1, 3);  x_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    chunk_11 = permute_44.chunk(3, dim = -1);  permute_44 = None
    query_layer_33 = chunk_11[0]
    key_layer_11 = chunk_11[1]
    value_layer_22 = chunk_11[2];  chunk_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    l__mod___deberta_encoder_layer_11_attention_self_q_bias = self.L__mod___deberta_encoder_layer_11_attention_self_q_bias
    getitem_59 = l__mod___deberta_encoder_layer_11_attention_self_q_bias[(None, None, slice(None, None, None))];  l__mod___deberta_encoder_layer_11_attention_self_q_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    x_34 = getitem_59.view((1, 1, 12, -1));  getitem_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_45 = x_34.permute(0, 2, 1, 3);  x_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    query_layer_34 = query_layer_33 + permute_45;  query_layer_33 = permute_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    l__mod___deberta_encoder_layer_11_attention_self_v_bias = self.L__mod___deberta_encoder_layer_11_attention_self_v_bias
    getitem_60 = l__mod___deberta_encoder_layer_11_attention_self_v_bias[(None, None, slice(None, None, None))];  l__mod___deberta_encoder_layer_11_attention_self_v_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    x_35 = getitem_60.view((1, 1, 12, -1));  getitem_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_46 = x_35.permute(0, 2, 1, 3);  x_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    value_layer_23 = value_layer_22 + permute_46;  value_layer_22 = permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    tensor_11 = torch.tensor(64, dtype = torch.float32)
    mul_36 = tensor_11 * 1;  tensor_11 = None
    scale_11 = torch.sqrt(mul_36);  mul_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    to_35 = scale_11.to(dtype = torch.float32);  scale_11 = None
    query_layer_35 = query_layer_34 / to_35;  query_layer_34 = to_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_11 = key_layer_11.transpose(-1, -2);  key_layer_11 = None
    attention_scores_11 = torch.matmul(query_layer_35, transpose_11);  query_layer_35 = transpose_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    function_ctx_45 = torch.autograd.function.FunctionCtx()
    attention_probs_22 = torch__dynamo_variables_misc_trampoline_autograd_apply_45(attention_scores_11, attention_mask_2, -1);  attention_scores_11 = attention_mask_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    function_ctx_46 = torch.autograd.function.FunctionCtx()
    attention_probs_23 = torch__dynamo_variables_misc_trampoline_autograd_apply_46(attention_probs_22, 0.1);  attention_probs_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_33 = torch.matmul(attention_probs_23, value_layer_23);  attention_probs_23 = value_layer_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_47 = context_layer_33.permute(0, 2, 1, 3);  context_layer_33 = None
    context_layer_34 = permute_47.contiguous();  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    self_output_11 = context_layer_34.view((1, 512, -1));  context_layer_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    hidden_states_168 = self.L__mod___deberta_encoder_layer_11_attention_output_dense(self_output_11);  self_output_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    function_ctx_47 = torch.autograd.function.FunctionCtx()
    hidden_states_169 = torch__dynamo_variables_misc_trampoline_autograd_apply_47(hidden_states_168, 0.1);  hidden_states_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_92 = hidden_states_169 + query_states_11;  hidden_states_169 = query_states_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:277, code: hidden_states = hidden_states.float()
    hidden_states_170 = add_92.float();  add_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_69 = hidden_states_170.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_46 = hidden_states_170 - mean_69
    pow_24 = sub_46.pow(2);  sub_46 = None
    variance_23 = pow_24.mean(-1, keepdim = True);  pow_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_47 = hidden_states_170 - mean_69;  hidden_states_170 = mean_69 = None
    add_93 = variance_23 + 1e-07;  variance_23 = None
    sqrt_35 = torch.sqrt(add_93);  add_93 = None
    hidden_states_171 = sub_47 / sqrt_35;  sub_47 = sqrt_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:281, code: hidden_states = hidden_states.to(input_type)
    hidden_states_172 = hidden_states_171.to(torch.float32);  hidden_states_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    l__mod___deberta_encoder_layer_11_attention_output_layer_norm_weight = self.L__mod___deberta_encoder_layer_11_attention_output_LayerNorm_weight
    mul_37 = l__mod___deberta_encoder_layer_11_attention_output_layer_norm_weight * hidden_states_172;  l__mod___deberta_encoder_layer_11_attention_output_layer_norm_weight = hidden_states_172 = None
    l__mod___deberta_encoder_layer_11_attention_output_layer_norm_bias = self.L__mod___deberta_encoder_layer_11_attention_output_LayerNorm_bias
    attention_output_22 = mul_37 + l__mod___deberta_encoder_layer_11_attention_output_layer_norm_bias;  mul_37 = l__mod___deberta_encoder_layer_11_attention_output_layer_norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    hidden_states_174 = self.L__mod___deberta_encoder_layer_11_intermediate_dense(attention_output_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_11 = torch._C._nn.gelu(hidden_states_174);  hidden_states_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    hidden_states_176 = self.L__mod___deberta_encoder_layer_11_output_dense(intermediate_output_11);  intermediate_output_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    function_ctx_48 = torch.autograd.function.FunctionCtx()
    hidden_states_177 = torch__dynamo_variables_misc_trampoline_autograd_apply_48(hidden_states_176, 0.1);  hidden_states_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_95 = hidden_states_177 + attention_output_22;  hidden_states_177 = attention_output_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:277, code: hidden_states = hidden_states.float()
    hidden_states_178 = add_95.float();  add_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_72 = hidden_states_178.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_48 = hidden_states_178 - mean_72
    pow_25 = sub_48.pow(2);  sub_48 = None
    variance_24 = pow_25.mean(-1, keepdim = True);  pow_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_49 = hidden_states_178 - mean_72;  hidden_states_178 = mean_72 = None
    add_96 = variance_24 + 1e-07;  variance_24 = None
    sqrt_36 = torch.sqrt(add_96);  add_96 = None
    hidden_states_179 = sub_49 / sqrt_36;  sub_49 = sqrt_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:281, code: hidden_states = hidden_states.to(input_type)
    hidden_states_180 = hidden_states_179.to(torch.float32);  hidden_states_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    l__mod___deberta_encoder_layer_11_output_layer_norm_weight = self.L__mod___deberta_encoder_layer_11_output_LayerNorm_weight
    mul_38 = l__mod___deberta_encoder_layer_11_output_layer_norm_weight * hidden_states_180;  l__mod___deberta_encoder_layer_11_output_layer_norm_weight = hidden_states_180 = None
    l__mod___deberta_encoder_layer_11_output_layer_norm_bias = self.L__mod___deberta_encoder_layer_11_output_LayerNorm_bias
    sequence_output = mul_38 + l__mod___deberta_encoder_layer_11_output_layer_norm_bias;  mul_38 = l__mod___deberta_encoder_layer_11_output_layer_norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1411, code: logits = self.qa_outputs(sequence_output)
    logits = self.L__mod___qa_outputs(sequence_output);  sequence_output = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1412, code: start_logits, end_logits = logits.split(1, dim=-1)
    split = logits.split(1, dim = -1);  logits = None
    start_logits = split[0]
    end_logits = split[1];  split = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1413, code: start_logits = start_logits.squeeze(-1).contiguous()
    squeeze_1 = start_logits.squeeze(-1);  start_logits = None
    start_logits_1 = squeeze_1.contiguous();  squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1414, code: end_logits = end_logits.squeeze(-1).contiguous()
    squeeze_2 = end_logits.squeeze(-1);  end_logits = None
    end_logits_1 = squeeze_2.contiguous();  squeeze_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1425, code: start_positions = start_positions.clamp(0, ignored_index)
    start_positions = l_cloned_inputs_start_positions_.clamp(0, 512);  l_cloned_inputs_start_positions_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1426, code: end_positions = end_positions.clamp(0, ignored_index)
    end_positions = l_cloned_inputs_end_positions_.clamp(0, 512);  l_cloned_inputs_end_positions_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1429, code: start_loss = loss_fct(start_logits, start_positions)
    start_loss = torch.nn.functional.cross_entropy(start_logits_1, start_positions, None, None, 512, None, 'mean', 0.0);  start_positions = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1430, code: end_loss = loss_fct(end_logits, end_positions)
    end_loss = torch.nn.functional.cross_entropy(end_logits_1, end_positions, None, None, 512, None, 'mean', 0.0);  end_positions = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1431, code: total_loss = (start_loss + end_loss) / 2
    add_98 = start_loss + end_loss;  start_loss = end_loss = None
    loss = add_98 / 2;  add_98 = None
    return (loss, start_logits_1, end_logits_1)
    