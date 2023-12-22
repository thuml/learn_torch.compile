from __future__ import annotations



def forward(self, L_cloned_inputs_input_ids_ : torch.Tensor, L_cloned_inputs_start_positions_ : torch.Tensor, L_cloned_inputs_end_positions_ : torch.Tensor):
    l_cloned_inputs_input_ids_ = L_cloned_inputs_input_ids_
    l_cloned_inputs_start_positions_ = L_cloned_inputs_start_positions_
    l_cloned_inputs_end_positions_ = L_cloned_inputs_end_positions_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1072, code: attention_mask = torch.ones(input_shape, device=device)
    input_mask = torch.ones((1, 512), device = device(type='cpu'))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1074, code: token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
    token_type_ids = torch.zeros((1, 512), dtype = torch.int64, device = device(type='cpu'))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:878, code: position_ids = self.position_ids[:, :seq_length]
    l__mod___deberta_embeddings_position_ids = self.L__mod___deberta_embeddings_position_ids
    position_ids = l__mod___deberta_embeddings_position_ids[(slice(None, None, None), slice(None, 512, None))];  l__mod___deberta_embeddings_position_ids = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:884, code: inputs_embeds = self.word_embeddings(input_ids)
    embeddings = self.L__mod___deberta_embeddings_word_embeddings(l_cloned_inputs_input_ids_);  l_cloned_inputs_input_ids_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:887, code: position_embeddings = self.position_embeddings(position_ids.long())
    long = position_ids.long();  position_ids = None
    position_embeddings = self.L__mod___deberta_embeddings_position_embeddings(long);  long = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:893, code: embeddings += position_embeddings
    embeddings += position_embeddings;  embeddings_1 = embeddings;  embeddings = position_embeddings = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:901, code: embeddings = self.LayerNorm(embeddings)
    embeddings_2 = self.L__mod___deberta_embeddings_LayerNorm(embeddings_1);  embeddings_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:907, code: mask = mask.unsqueeze(2)
    mask = input_mask.unsqueeze(2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:908, code: mask = mask.to(embeddings.dtype)
    mask_1 = mask.to(torch.float32);  mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:910, code: embeddings = embeddings * mask
    embeddings_3 = embeddings_2 * mask_1;  embeddings_2 = mask_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx = torch.autograd.function.FunctionCtx()
    query_states = torch__dynamo_variables_misc_trampoline_autograd_apply(embeddings_3, 0.1);  embeddings_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:454, code: extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    unsqueeze_1 = input_mask.unsqueeze(1);  input_mask = None
    extended_attention_mask = unsqueeze_1.unsqueeze(2);  unsqueeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:455, code: attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
    squeeze = extended_attention_mask.squeeze(-2)
    unsqueeze_3 = squeeze.unsqueeze(-1);  squeeze = None
    attention_mask_2 = extended_attention_mask * unsqueeze_3;  extended_attention_mask = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_0_attention_self_query_proj = self.L__mod___deberta_encoder_layer_0_attention_self_query_proj(query_states)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x = l__mod___deberta_encoder_layer_0_attention_self_query_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_0_attention_self_query_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute = x.permute(0, 2, 1, 3);  x = None
    contiguous = permute.contiguous();  permute = None
    query_layer = contiguous.view(-1, 512, 64);  contiguous = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_0_attention_self_key_proj = self.L__mod___deberta_encoder_layer_0_attention_self_key_proj(query_states)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_1 = l__mod___deberta_encoder_layer_0_attention_self_key_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_0_attention_self_key_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_1 = x_1.permute(0, 2, 1, 3);  x_1 = None
    contiguous_1 = permute_1.contiguous();  permute_1 = None
    key_layer = contiguous_1.view(-1, 512, 64);  contiguous_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_0_attention_self_value_proj = self.L__mod___deberta_encoder_layer_0_attention_self_value_proj(query_states)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_2 = l__mod___deberta_encoder_layer_0_attention_self_value_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_0_attention_self_value_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_2 = x_2.permute(0, 2, 1, 3);  x_2 = None
    contiguous_2 = permute_2.contiguous();  permute_2 = None
    value_layer = contiguous_2.view(-1, 512, 64);  contiguous_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    tensor = torch.tensor(64, dtype = torch.float32)
    mul_2 = tensor * 1;  tensor = None
    scale = torch.sqrt(mul_2);  mul_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    transpose = key_layer.transpose(-1, -2);  key_layer = None
    to_1 = scale.to(dtype = torch.float32);  scale = None
    truediv = transpose / to_1;  transpose = to_1 = None
    attention_scores = torch.bmm(query_layer, truediv);  query_layer = truediv = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    attention_scores_1 = attention_scores.view(-1, 24, 512, 512);  attention_scores = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    function_ctx_1 = torch.autograd.function.FunctionCtx()
    attention_probs = torch__dynamo_variables_misc_trampoline_autograd_apply_1(attention_scores_1, attention_mask_2, -1);  attention_scores_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_2 = torch.autograd.function.FunctionCtx()
    attention_probs_1 = torch__dynamo_variables_misc_trampoline_autograd_apply_2(attention_probs, 0.1);  attention_probs = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_7 = attention_probs_1.view(-1, 512, 512);  attention_probs_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    context_layer = torch.bmm(view_7, value_layer);  view_7 = value_layer = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_8 = context_layer.view(-1, 24, 512, 64);  context_layer = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_3 = view_8.permute(0, 2, 1, 3);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    context_layer_1 = permute_3.contiguous();  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    self_output = context_layer_1.view((1, 512, -1));  context_layer_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    hidden_states = self.L__mod___deberta_encoder_layer_0_attention_output_dense(self_output);  self_output = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_3 = torch.autograd.function.FunctionCtx()
    hidden_states_1 = torch__dynamo_variables_misc_trampoline_autograd_apply_3(hidden_states, 0.1);  hidden_states = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add = hidden_states_1 + query_states;  hidden_states_1 = query_states = None
    attention_output = self.L__mod___deberta_encoder_layer_0_attention_output_LayerNorm(add);  add = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    hidden_states_3 = self.L__mod___deberta_encoder_layer_0_intermediate_dense(attention_output)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output = torch._C._nn.gelu(hidden_states_3);  hidden_states_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    hidden_states_5 = self.L__mod___deberta_encoder_layer_0_output_dense(intermediate_output);  intermediate_output = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_4 = torch.autograd.function.FunctionCtx()
    hidden_states_6 = torch__dynamo_variables_misc_trampoline_autograd_apply_4(hidden_states_5, 0.1);  hidden_states_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_1 = hidden_states_6 + attention_output;  hidden_states_6 = attention_output = None
    query_states_2 = self.L__mod___deberta_encoder_layer_0_output_LayerNorm(add_1);  add_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_1_attention_self_query_proj = self.L__mod___deberta_encoder_layer_1_attention_self_query_proj(query_states_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_3 = l__mod___deberta_encoder_layer_1_attention_self_query_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_1_attention_self_query_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_4 = x_3.permute(0, 2, 1, 3);  x_3 = None
    contiguous_4 = permute_4.contiguous();  permute_4 = None
    query_layer_1 = contiguous_4.view(-1, 512, 64);  contiguous_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_1_attention_self_key_proj = self.L__mod___deberta_encoder_layer_1_attention_self_key_proj(query_states_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_4 = l__mod___deberta_encoder_layer_1_attention_self_key_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_1_attention_self_key_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_5 = x_4.permute(0, 2, 1, 3);  x_4 = None
    contiguous_5 = permute_5.contiguous();  permute_5 = None
    key_layer_1 = contiguous_5.view(-1, 512, 64);  contiguous_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_1_attention_self_value_proj = self.L__mod___deberta_encoder_layer_1_attention_self_value_proj(query_states_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_5 = l__mod___deberta_encoder_layer_1_attention_self_value_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_1_attention_self_value_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_6 = x_5.permute(0, 2, 1, 3);  x_5 = None
    contiguous_6 = permute_6.contiguous();  permute_6 = None
    value_layer_1 = contiguous_6.view(-1, 512, 64);  contiguous_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    tensor_1 = torch.tensor(64, dtype = torch.float32)
    mul_3 = tensor_1 * 1;  tensor_1 = None
    scale_1 = torch.sqrt(mul_3);  mul_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    transpose_1 = key_layer_1.transpose(-1, -2);  key_layer_1 = None
    to_2 = scale_1.to(dtype = torch.float32);  scale_1 = None
    truediv_1 = transpose_1 / to_2;  transpose_1 = to_2 = None
    attention_scores_3 = torch.bmm(query_layer_1, truediv_1);  query_layer_1 = truediv_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    attention_scores_4 = attention_scores_3.view(-1, 24, 512, 512);  attention_scores_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    function_ctx_5 = torch.autograd.function.FunctionCtx()
    attention_probs_2 = torch__dynamo_variables_misc_trampoline_autograd_apply_5(attention_scores_4, attention_mask_2, -1);  attention_scores_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_6 = torch.autograd.function.FunctionCtx()
    attention_probs_3 = torch__dynamo_variables_misc_trampoline_autograd_apply_6(attention_probs_2, 0.1);  attention_probs_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_17 = attention_probs_3.view(-1, 512, 512);  attention_probs_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    context_layer_3 = torch.bmm(view_17, value_layer_1);  view_17 = value_layer_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_18 = context_layer_3.view(-1, 24, 512, 64);  context_layer_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_7 = view_18.permute(0, 2, 1, 3);  view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    context_layer_4 = permute_7.contiguous();  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    self_output_1 = context_layer_4.view((1, 512, -1));  context_layer_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    hidden_states_8 = self.L__mod___deberta_encoder_layer_1_attention_output_dense(self_output_1);  self_output_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_7 = torch.autograd.function.FunctionCtx()
    hidden_states_9 = torch__dynamo_variables_misc_trampoline_autograd_apply_7(hidden_states_8, 0.1);  hidden_states_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_2 = hidden_states_9 + query_states_2;  hidden_states_9 = query_states_2 = None
    attention_output_2 = self.L__mod___deberta_encoder_layer_1_attention_output_LayerNorm(add_2);  add_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    hidden_states_11 = self.L__mod___deberta_encoder_layer_1_intermediate_dense(attention_output_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_1 = torch._C._nn.gelu(hidden_states_11);  hidden_states_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    hidden_states_13 = self.L__mod___deberta_encoder_layer_1_output_dense(intermediate_output_1);  intermediate_output_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_8 = torch.autograd.function.FunctionCtx()
    hidden_states_14 = torch__dynamo_variables_misc_trampoline_autograd_apply_8(hidden_states_13, 0.1);  hidden_states_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_3 = hidden_states_14 + attention_output_2;  hidden_states_14 = attention_output_2 = None
    query_states_4 = self.L__mod___deberta_encoder_layer_1_output_LayerNorm(add_3);  add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_2_attention_self_query_proj = self.L__mod___deberta_encoder_layer_2_attention_self_query_proj(query_states_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_6 = l__mod___deberta_encoder_layer_2_attention_self_query_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_2_attention_self_query_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_8 = x_6.permute(0, 2, 1, 3);  x_6 = None
    contiguous_8 = permute_8.contiguous();  permute_8 = None
    query_layer_2 = contiguous_8.view(-1, 512, 64);  contiguous_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_2_attention_self_key_proj = self.L__mod___deberta_encoder_layer_2_attention_self_key_proj(query_states_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_7 = l__mod___deberta_encoder_layer_2_attention_self_key_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_2_attention_self_key_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_9 = x_7.permute(0, 2, 1, 3);  x_7 = None
    contiguous_9 = permute_9.contiguous();  permute_9 = None
    key_layer_2 = contiguous_9.view(-1, 512, 64);  contiguous_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_2_attention_self_value_proj = self.L__mod___deberta_encoder_layer_2_attention_self_value_proj(query_states_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_8 = l__mod___deberta_encoder_layer_2_attention_self_value_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_2_attention_self_value_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_10 = x_8.permute(0, 2, 1, 3);  x_8 = None
    contiguous_10 = permute_10.contiguous();  permute_10 = None
    value_layer_2 = contiguous_10.view(-1, 512, 64);  contiguous_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    tensor_2 = torch.tensor(64, dtype = torch.float32)
    mul_4 = tensor_2 * 1;  tensor_2 = None
    scale_2 = torch.sqrt(mul_4);  mul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    transpose_2 = key_layer_2.transpose(-1, -2);  key_layer_2 = None
    to_3 = scale_2.to(dtype = torch.float32);  scale_2 = None
    truediv_2 = transpose_2 / to_3;  transpose_2 = to_3 = None
    attention_scores_6 = torch.bmm(query_layer_2, truediv_2);  query_layer_2 = truediv_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    attention_scores_7 = attention_scores_6.view(-1, 24, 512, 512);  attention_scores_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    function_ctx_9 = torch.autograd.function.FunctionCtx()
    attention_probs_4 = torch__dynamo_variables_misc_trampoline_autograd_apply_9(attention_scores_7, attention_mask_2, -1);  attention_scores_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_10 = torch.autograd.function.FunctionCtx()
    attention_probs_5 = torch__dynamo_variables_misc_trampoline_autograd_apply_10(attention_probs_4, 0.1);  attention_probs_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_27 = attention_probs_5.view(-1, 512, 512);  attention_probs_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    context_layer_6 = torch.bmm(view_27, value_layer_2);  view_27 = value_layer_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_28 = context_layer_6.view(-1, 24, 512, 64);  context_layer_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_11 = view_28.permute(0, 2, 1, 3);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    context_layer_7 = permute_11.contiguous();  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    self_output_2 = context_layer_7.view((1, 512, -1));  context_layer_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    hidden_states_16 = self.L__mod___deberta_encoder_layer_2_attention_output_dense(self_output_2);  self_output_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_11 = torch.autograd.function.FunctionCtx()
    hidden_states_17 = torch__dynamo_variables_misc_trampoline_autograd_apply_11(hidden_states_16, 0.1);  hidden_states_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_4 = hidden_states_17 + query_states_4;  hidden_states_17 = query_states_4 = None
    attention_output_4 = self.L__mod___deberta_encoder_layer_2_attention_output_LayerNorm(add_4);  add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    hidden_states_19 = self.L__mod___deberta_encoder_layer_2_intermediate_dense(attention_output_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_2 = torch._C._nn.gelu(hidden_states_19);  hidden_states_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    hidden_states_21 = self.L__mod___deberta_encoder_layer_2_output_dense(intermediate_output_2);  intermediate_output_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_12 = torch.autograd.function.FunctionCtx()
    hidden_states_22 = torch__dynamo_variables_misc_trampoline_autograd_apply_12(hidden_states_21, 0.1);  hidden_states_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_5 = hidden_states_22 + attention_output_4;  hidden_states_22 = attention_output_4 = None
    query_states_6 = self.L__mod___deberta_encoder_layer_2_output_LayerNorm(add_5);  add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_3_attention_self_query_proj = self.L__mod___deberta_encoder_layer_3_attention_self_query_proj(query_states_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_9 = l__mod___deberta_encoder_layer_3_attention_self_query_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_3_attention_self_query_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_12 = x_9.permute(0, 2, 1, 3);  x_9 = None
    contiguous_12 = permute_12.contiguous();  permute_12 = None
    query_layer_3 = contiguous_12.view(-1, 512, 64);  contiguous_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_3_attention_self_key_proj = self.L__mod___deberta_encoder_layer_3_attention_self_key_proj(query_states_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_10 = l__mod___deberta_encoder_layer_3_attention_self_key_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_3_attention_self_key_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_13 = x_10.permute(0, 2, 1, 3);  x_10 = None
    contiguous_13 = permute_13.contiguous();  permute_13 = None
    key_layer_3 = contiguous_13.view(-1, 512, 64);  contiguous_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_3_attention_self_value_proj = self.L__mod___deberta_encoder_layer_3_attention_self_value_proj(query_states_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_11 = l__mod___deberta_encoder_layer_3_attention_self_value_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_3_attention_self_value_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_14 = x_11.permute(0, 2, 1, 3);  x_11 = None
    contiguous_14 = permute_14.contiguous();  permute_14 = None
    value_layer_3 = contiguous_14.view(-1, 512, 64);  contiguous_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    tensor_3 = torch.tensor(64, dtype = torch.float32)
    mul_5 = tensor_3 * 1;  tensor_3 = None
    scale_3 = torch.sqrt(mul_5);  mul_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    transpose_3 = key_layer_3.transpose(-1, -2);  key_layer_3 = None
    to_4 = scale_3.to(dtype = torch.float32);  scale_3 = None
    truediv_3 = transpose_3 / to_4;  transpose_3 = to_4 = None
    attention_scores_9 = torch.bmm(query_layer_3, truediv_3);  query_layer_3 = truediv_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    attention_scores_10 = attention_scores_9.view(-1, 24, 512, 512);  attention_scores_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    function_ctx_13 = torch.autograd.function.FunctionCtx()
    attention_probs_6 = torch__dynamo_variables_misc_trampoline_autograd_apply_13(attention_scores_10, attention_mask_2, -1);  attention_scores_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_14 = torch.autograd.function.FunctionCtx()
    attention_probs_7 = torch__dynamo_variables_misc_trampoline_autograd_apply_14(attention_probs_6, 0.1);  attention_probs_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_37 = attention_probs_7.view(-1, 512, 512);  attention_probs_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    context_layer_9 = torch.bmm(view_37, value_layer_3);  view_37 = value_layer_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_38 = context_layer_9.view(-1, 24, 512, 64);  context_layer_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_15 = view_38.permute(0, 2, 1, 3);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    context_layer_10 = permute_15.contiguous();  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    self_output_3 = context_layer_10.view((1, 512, -1));  context_layer_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    hidden_states_24 = self.L__mod___deberta_encoder_layer_3_attention_output_dense(self_output_3);  self_output_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_15 = torch.autograd.function.FunctionCtx()
    hidden_states_25 = torch__dynamo_variables_misc_trampoline_autograd_apply_15(hidden_states_24, 0.1);  hidden_states_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_6 = hidden_states_25 + query_states_6;  hidden_states_25 = query_states_6 = None
    attention_output_6 = self.L__mod___deberta_encoder_layer_3_attention_output_LayerNorm(add_6);  add_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    hidden_states_27 = self.L__mod___deberta_encoder_layer_3_intermediate_dense(attention_output_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_3 = torch._C._nn.gelu(hidden_states_27);  hidden_states_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    hidden_states_29 = self.L__mod___deberta_encoder_layer_3_output_dense(intermediate_output_3);  intermediate_output_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_16 = torch.autograd.function.FunctionCtx()
    hidden_states_30 = torch__dynamo_variables_misc_trampoline_autograd_apply_16(hidden_states_29, 0.1);  hidden_states_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_7 = hidden_states_30 + attention_output_6;  hidden_states_30 = attention_output_6 = None
    query_states_8 = self.L__mod___deberta_encoder_layer_3_output_LayerNorm(add_7);  add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_4_attention_self_query_proj = self.L__mod___deberta_encoder_layer_4_attention_self_query_proj(query_states_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_12 = l__mod___deberta_encoder_layer_4_attention_self_query_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_4_attention_self_query_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_16 = x_12.permute(0, 2, 1, 3);  x_12 = None
    contiguous_16 = permute_16.contiguous();  permute_16 = None
    query_layer_4 = contiguous_16.view(-1, 512, 64);  contiguous_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_4_attention_self_key_proj = self.L__mod___deberta_encoder_layer_4_attention_self_key_proj(query_states_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_13 = l__mod___deberta_encoder_layer_4_attention_self_key_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_4_attention_self_key_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_17 = x_13.permute(0, 2, 1, 3);  x_13 = None
    contiguous_17 = permute_17.contiguous();  permute_17 = None
    key_layer_4 = contiguous_17.view(-1, 512, 64);  contiguous_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_4_attention_self_value_proj = self.L__mod___deberta_encoder_layer_4_attention_self_value_proj(query_states_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_14 = l__mod___deberta_encoder_layer_4_attention_self_value_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_4_attention_self_value_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_18 = x_14.permute(0, 2, 1, 3);  x_14 = None
    contiguous_18 = permute_18.contiguous();  permute_18 = None
    value_layer_4 = contiguous_18.view(-1, 512, 64);  contiguous_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    tensor_4 = torch.tensor(64, dtype = torch.float32)
    mul_6 = tensor_4 * 1;  tensor_4 = None
    scale_4 = torch.sqrt(mul_6);  mul_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    transpose_4 = key_layer_4.transpose(-1, -2);  key_layer_4 = None
    to_5 = scale_4.to(dtype = torch.float32);  scale_4 = None
    truediv_4 = transpose_4 / to_5;  transpose_4 = to_5 = None
    attention_scores_12 = torch.bmm(query_layer_4, truediv_4);  query_layer_4 = truediv_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    attention_scores_13 = attention_scores_12.view(-1, 24, 512, 512);  attention_scores_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    function_ctx_17 = torch.autograd.function.FunctionCtx()
    attention_probs_8 = torch__dynamo_variables_misc_trampoline_autograd_apply_17(attention_scores_13, attention_mask_2, -1);  attention_scores_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_18 = torch.autograd.function.FunctionCtx()
    attention_probs_9 = torch__dynamo_variables_misc_trampoline_autograd_apply_18(attention_probs_8, 0.1);  attention_probs_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_47 = attention_probs_9.view(-1, 512, 512);  attention_probs_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    context_layer_12 = torch.bmm(view_47, value_layer_4);  view_47 = value_layer_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_48 = context_layer_12.view(-1, 24, 512, 64);  context_layer_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_19 = view_48.permute(0, 2, 1, 3);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    context_layer_13 = permute_19.contiguous();  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    self_output_4 = context_layer_13.view((1, 512, -1));  context_layer_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    hidden_states_32 = self.L__mod___deberta_encoder_layer_4_attention_output_dense(self_output_4);  self_output_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_19 = torch.autograd.function.FunctionCtx()
    hidden_states_33 = torch__dynamo_variables_misc_trampoline_autograd_apply_19(hidden_states_32, 0.1);  hidden_states_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_8 = hidden_states_33 + query_states_8;  hidden_states_33 = query_states_8 = None
    attention_output_8 = self.L__mod___deberta_encoder_layer_4_attention_output_LayerNorm(add_8);  add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    hidden_states_35 = self.L__mod___deberta_encoder_layer_4_intermediate_dense(attention_output_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_4 = torch._C._nn.gelu(hidden_states_35);  hidden_states_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    hidden_states_37 = self.L__mod___deberta_encoder_layer_4_output_dense(intermediate_output_4);  intermediate_output_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_20 = torch.autograd.function.FunctionCtx()
    hidden_states_38 = torch__dynamo_variables_misc_trampoline_autograd_apply_20(hidden_states_37, 0.1);  hidden_states_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_9 = hidden_states_38 + attention_output_8;  hidden_states_38 = attention_output_8 = None
    query_states_10 = self.L__mod___deberta_encoder_layer_4_output_LayerNorm(add_9);  add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_5_attention_self_query_proj = self.L__mod___deberta_encoder_layer_5_attention_self_query_proj(query_states_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_15 = l__mod___deberta_encoder_layer_5_attention_self_query_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_5_attention_self_query_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_20 = x_15.permute(0, 2, 1, 3);  x_15 = None
    contiguous_20 = permute_20.contiguous();  permute_20 = None
    query_layer_5 = contiguous_20.view(-1, 512, 64);  contiguous_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_5_attention_self_key_proj = self.L__mod___deberta_encoder_layer_5_attention_self_key_proj(query_states_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_16 = l__mod___deberta_encoder_layer_5_attention_self_key_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_5_attention_self_key_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_21 = x_16.permute(0, 2, 1, 3);  x_16 = None
    contiguous_21 = permute_21.contiguous();  permute_21 = None
    key_layer_5 = contiguous_21.view(-1, 512, 64);  contiguous_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_5_attention_self_value_proj = self.L__mod___deberta_encoder_layer_5_attention_self_value_proj(query_states_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_17 = l__mod___deberta_encoder_layer_5_attention_self_value_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_5_attention_self_value_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_22 = x_17.permute(0, 2, 1, 3);  x_17 = None
    contiguous_22 = permute_22.contiguous();  permute_22 = None
    value_layer_5 = contiguous_22.view(-1, 512, 64);  contiguous_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    tensor_5 = torch.tensor(64, dtype = torch.float32)
    mul_7 = tensor_5 * 1;  tensor_5 = None
    scale_5 = torch.sqrt(mul_7);  mul_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    transpose_5 = key_layer_5.transpose(-1, -2);  key_layer_5 = None
    to_6 = scale_5.to(dtype = torch.float32);  scale_5 = None
    truediv_5 = transpose_5 / to_6;  transpose_5 = to_6 = None
    attention_scores_15 = torch.bmm(query_layer_5, truediv_5);  query_layer_5 = truediv_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    attention_scores_16 = attention_scores_15.view(-1, 24, 512, 512);  attention_scores_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    function_ctx_21 = torch.autograd.function.FunctionCtx()
    attention_probs_10 = torch__dynamo_variables_misc_trampoline_autograd_apply_21(attention_scores_16, attention_mask_2, -1);  attention_scores_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_22 = torch.autograd.function.FunctionCtx()
    attention_probs_11 = torch__dynamo_variables_misc_trampoline_autograd_apply_22(attention_probs_10, 0.1);  attention_probs_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_57 = attention_probs_11.view(-1, 512, 512);  attention_probs_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    context_layer_15 = torch.bmm(view_57, value_layer_5);  view_57 = value_layer_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_58 = context_layer_15.view(-1, 24, 512, 64);  context_layer_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_23 = view_58.permute(0, 2, 1, 3);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    context_layer_16 = permute_23.contiguous();  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    self_output_5 = context_layer_16.view((1, 512, -1));  context_layer_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    hidden_states_40 = self.L__mod___deberta_encoder_layer_5_attention_output_dense(self_output_5);  self_output_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_23 = torch.autograd.function.FunctionCtx()
    hidden_states_41 = torch__dynamo_variables_misc_trampoline_autograd_apply_23(hidden_states_40, 0.1);  hidden_states_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_10 = hidden_states_41 + query_states_10;  hidden_states_41 = query_states_10 = None
    attention_output_10 = self.L__mod___deberta_encoder_layer_5_attention_output_LayerNorm(add_10);  add_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    hidden_states_43 = self.L__mod___deberta_encoder_layer_5_intermediate_dense(attention_output_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_5 = torch._C._nn.gelu(hidden_states_43);  hidden_states_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    hidden_states_45 = self.L__mod___deberta_encoder_layer_5_output_dense(intermediate_output_5);  intermediate_output_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_24 = torch.autograd.function.FunctionCtx()
    hidden_states_46 = torch__dynamo_variables_misc_trampoline_autograd_apply_24(hidden_states_45, 0.1);  hidden_states_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_11 = hidden_states_46 + attention_output_10;  hidden_states_46 = attention_output_10 = None
    query_states_12 = self.L__mod___deberta_encoder_layer_5_output_LayerNorm(add_11);  add_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_6_attention_self_query_proj = self.L__mod___deberta_encoder_layer_6_attention_self_query_proj(query_states_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_18 = l__mod___deberta_encoder_layer_6_attention_self_query_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_6_attention_self_query_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_24 = x_18.permute(0, 2, 1, 3);  x_18 = None
    contiguous_24 = permute_24.contiguous();  permute_24 = None
    query_layer_6 = contiguous_24.view(-1, 512, 64);  contiguous_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_6_attention_self_key_proj = self.L__mod___deberta_encoder_layer_6_attention_self_key_proj(query_states_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_19 = l__mod___deberta_encoder_layer_6_attention_self_key_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_6_attention_self_key_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_25 = x_19.permute(0, 2, 1, 3);  x_19 = None
    contiguous_25 = permute_25.contiguous();  permute_25 = None
    key_layer_6 = contiguous_25.view(-1, 512, 64);  contiguous_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_6_attention_self_value_proj = self.L__mod___deberta_encoder_layer_6_attention_self_value_proj(query_states_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_20 = l__mod___deberta_encoder_layer_6_attention_self_value_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_6_attention_self_value_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_26 = x_20.permute(0, 2, 1, 3);  x_20 = None
    contiguous_26 = permute_26.contiguous();  permute_26 = None
    value_layer_6 = contiguous_26.view(-1, 512, 64);  contiguous_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    tensor_6 = torch.tensor(64, dtype = torch.float32)
    mul_8 = tensor_6 * 1;  tensor_6 = None
    scale_6 = torch.sqrt(mul_8);  mul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    transpose_6 = key_layer_6.transpose(-1, -2);  key_layer_6 = None
    to_7 = scale_6.to(dtype = torch.float32);  scale_6 = None
    truediv_6 = transpose_6 / to_7;  transpose_6 = to_7 = None
    attention_scores_18 = torch.bmm(query_layer_6, truediv_6);  query_layer_6 = truediv_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    attention_scores_19 = attention_scores_18.view(-1, 24, 512, 512);  attention_scores_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    function_ctx_25 = torch.autograd.function.FunctionCtx()
    attention_probs_12 = torch__dynamo_variables_misc_trampoline_autograd_apply_25(attention_scores_19, attention_mask_2, -1);  attention_scores_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_26 = torch.autograd.function.FunctionCtx()
    attention_probs_13 = torch__dynamo_variables_misc_trampoline_autograd_apply_26(attention_probs_12, 0.1);  attention_probs_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_67 = attention_probs_13.view(-1, 512, 512);  attention_probs_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    context_layer_18 = torch.bmm(view_67, value_layer_6);  view_67 = value_layer_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_68 = context_layer_18.view(-1, 24, 512, 64);  context_layer_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_27 = view_68.permute(0, 2, 1, 3);  view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    context_layer_19 = permute_27.contiguous();  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    self_output_6 = context_layer_19.view((1, 512, -1));  context_layer_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    hidden_states_48 = self.L__mod___deberta_encoder_layer_6_attention_output_dense(self_output_6);  self_output_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_27 = torch.autograd.function.FunctionCtx()
    hidden_states_49 = torch__dynamo_variables_misc_trampoline_autograd_apply_27(hidden_states_48, 0.1);  hidden_states_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_12 = hidden_states_49 + query_states_12;  hidden_states_49 = query_states_12 = None
    attention_output_12 = self.L__mod___deberta_encoder_layer_6_attention_output_LayerNorm(add_12);  add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    hidden_states_51 = self.L__mod___deberta_encoder_layer_6_intermediate_dense(attention_output_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_6 = torch._C._nn.gelu(hidden_states_51);  hidden_states_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    hidden_states_53 = self.L__mod___deberta_encoder_layer_6_output_dense(intermediate_output_6);  intermediate_output_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_28 = torch.autograd.function.FunctionCtx()
    hidden_states_54 = torch__dynamo_variables_misc_trampoline_autograd_apply_28(hidden_states_53, 0.1);  hidden_states_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_13 = hidden_states_54 + attention_output_12;  hidden_states_54 = attention_output_12 = None
    query_states_14 = self.L__mod___deberta_encoder_layer_6_output_LayerNorm(add_13);  add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_7_attention_self_query_proj = self.L__mod___deberta_encoder_layer_7_attention_self_query_proj(query_states_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_21 = l__mod___deberta_encoder_layer_7_attention_self_query_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_7_attention_self_query_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_28 = x_21.permute(0, 2, 1, 3);  x_21 = None
    contiguous_28 = permute_28.contiguous();  permute_28 = None
    query_layer_7 = contiguous_28.view(-1, 512, 64);  contiguous_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_7_attention_self_key_proj = self.L__mod___deberta_encoder_layer_7_attention_self_key_proj(query_states_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_22 = l__mod___deberta_encoder_layer_7_attention_self_key_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_7_attention_self_key_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_29 = x_22.permute(0, 2, 1, 3);  x_22 = None
    contiguous_29 = permute_29.contiguous();  permute_29 = None
    key_layer_7 = contiguous_29.view(-1, 512, 64);  contiguous_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_7_attention_self_value_proj = self.L__mod___deberta_encoder_layer_7_attention_self_value_proj(query_states_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_23 = l__mod___deberta_encoder_layer_7_attention_self_value_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_7_attention_self_value_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_30 = x_23.permute(0, 2, 1, 3);  x_23 = None
    contiguous_30 = permute_30.contiguous();  permute_30 = None
    value_layer_7 = contiguous_30.view(-1, 512, 64);  contiguous_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    tensor_7 = torch.tensor(64, dtype = torch.float32)
    mul_9 = tensor_7 * 1;  tensor_7 = None
    scale_7 = torch.sqrt(mul_9);  mul_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    transpose_7 = key_layer_7.transpose(-1, -2);  key_layer_7 = None
    to_8 = scale_7.to(dtype = torch.float32);  scale_7 = None
    truediv_7 = transpose_7 / to_8;  transpose_7 = to_8 = None
    attention_scores_21 = torch.bmm(query_layer_7, truediv_7);  query_layer_7 = truediv_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    attention_scores_22 = attention_scores_21.view(-1, 24, 512, 512);  attention_scores_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    function_ctx_29 = torch.autograd.function.FunctionCtx()
    attention_probs_14 = torch__dynamo_variables_misc_trampoline_autograd_apply_29(attention_scores_22, attention_mask_2, -1);  attention_scores_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_30 = torch.autograd.function.FunctionCtx()
    attention_probs_15 = torch__dynamo_variables_misc_trampoline_autograd_apply_30(attention_probs_14, 0.1);  attention_probs_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_77 = attention_probs_15.view(-1, 512, 512);  attention_probs_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    context_layer_21 = torch.bmm(view_77, value_layer_7);  view_77 = value_layer_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_78 = context_layer_21.view(-1, 24, 512, 64);  context_layer_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_31 = view_78.permute(0, 2, 1, 3);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    context_layer_22 = permute_31.contiguous();  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    self_output_7 = context_layer_22.view((1, 512, -1));  context_layer_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    hidden_states_56 = self.L__mod___deberta_encoder_layer_7_attention_output_dense(self_output_7);  self_output_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_31 = torch.autograd.function.FunctionCtx()
    hidden_states_57 = torch__dynamo_variables_misc_trampoline_autograd_apply_31(hidden_states_56, 0.1);  hidden_states_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_14 = hidden_states_57 + query_states_14;  hidden_states_57 = query_states_14 = None
    attention_output_14 = self.L__mod___deberta_encoder_layer_7_attention_output_LayerNorm(add_14);  add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    hidden_states_59 = self.L__mod___deberta_encoder_layer_7_intermediate_dense(attention_output_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_7 = torch._C._nn.gelu(hidden_states_59);  hidden_states_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    hidden_states_61 = self.L__mod___deberta_encoder_layer_7_output_dense(intermediate_output_7);  intermediate_output_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_32 = torch.autograd.function.FunctionCtx()
    hidden_states_62 = torch__dynamo_variables_misc_trampoline_autograd_apply_32(hidden_states_61, 0.1);  hidden_states_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_15 = hidden_states_62 + attention_output_14;  hidden_states_62 = attention_output_14 = None
    query_states_16 = self.L__mod___deberta_encoder_layer_7_output_LayerNorm(add_15);  add_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_8_attention_self_query_proj = self.L__mod___deberta_encoder_layer_8_attention_self_query_proj(query_states_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_24 = l__mod___deberta_encoder_layer_8_attention_self_query_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_8_attention_self_query_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_32 = x_24.permute(0, 2, 1, 3);  x_24 = None
    contiguous_32 = permute_32.contiguous();  permute_32 = None
    query_layer_8 = contiguous_32.view(-1, 512, 64);  contiguous_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_8_attention_self_key_proj = self.L__mod___deberta_encoder_layer_8_attention_self_key_proj(query_states_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_25 = l__mod___deberta_encoder_layer_8_attention_self_key_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_8_attention_self_key_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_33 = x_25.permute(0, 2, 1, 3);  x_25 = None
    contiguous_33 = permute_33.contiguous();  permute_33 = None
    key_layer_8 = contiguous_33.view(-1, 512, 64);  contiguous_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_8_attention_self_value_proj = self.L__mod___deberta_encoder_layer_8_attention_self_value_proj(query_states_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_26 = l__mod___deberta_encoder_layer_8_attention_self_value_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_8_attention_self_value_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_34 = x_26.permute(0, 2, 1, 3);  x_26 = None
    contiguous_34 = permute_34.contiguous();  permute_34 = None
    value_layer_8 = contiguous_34.view(-1, 512, 64);  contiguous_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    tensor_8 = torch.tensor(64, dtype = torch.float32)
    mul_10 = tensor_8 * 1;  tensor_8 = None
    scale_8 = torch.sqrt(mul_10);  mul_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    transpose_8 = key_layer_8.transpose(-1, -2);  key_layer_8 = None
    to_9 = scale_8.to(dtype = torch.float32);  scale_8 = None
    truediv_8 = transpose_8 / to_9;  transpose_8 = to_9 = None
    attention_scores_24 = torch.bmm(query_layer_8, truediv_8);  query_layer_8 = truediv_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    attention_scores_25 = attention_scores_24.view(-1, 24, 512, 512);  attention_scores_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    function_ctx_33 = torch.autograd.function.FunctionCtx()
    attention_probs_16 = torch__dynamo_variables_misc_trampoline_autograd_apply_33(attention_scores_25, attention_mask_2, -1);  attention_scores_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_34 = torch.autograd.function.FunctionCtx()
    attention_probs_17 = torch__dynamo_variables_misc_trampoline_autograd_apply_34(attention_probs_16, 0.1);  attention_probs_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_87 = attention_probs_17.view(-1, 512, 512);  attention_probs_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    context_layer_24 = torch.bmm(view_87, value_layer_8);  view_87 = value_layer_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_88 = context_layer_24.view(-1, 24, 512, 64);  context_layer_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_35 = view_88.permute(0, 2, 1, 3);  view_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    context_layer_25 = permute_35.contiguous();  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    self_output_8 = context_layer_25.view((1, 512, -1));  context_layer_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    hidden_states_64 = self.L__mod___deberta_encoder_layer_8_attention_output_dense(self_output_8);  self_output_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_35 = torch.autograd.function.FunctionCtx()
    hidden_states_65 = torch__dynamo_variables_misc_trampoline_autograd_apply_35(hidden_states_64, 0.1);  hidden_states_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_16 = hidden_states_65 + query_states_16;  hidden_states_65 = query_states_16 = None
    attention_output_16 = self.L__mod___deberta_encoder_layer_8_attention_output_LayerNorm(add_16);  add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    hidden_states_67 = self.L__mod___deberta_encoder_layer_8_intermediate_dense(attention_output_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_8 = torch._C._nn.gelu(hidden_states_67);  hidden_states_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    hidden_states_69 = self.L__mod___deberta_encoder_layer_8_output_dense(intermediate_output_8);  intermediate_output_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_36 = torch.autograd.function.FunctionCtx()
    hidden_states_70 = torch__dynamo_variables_misc_trampoline_autograd_apply_36(hidden_states_69, 0.1);  hidden_states_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_17 = hidden_states_70 + attention_output_16;  hidden_states_70 = attention_output_16 = None
    query_states_18 = self.L__mod___deberta_encoder_layer_8_output_LayerNorm(add_17);  add_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_9_attention_self_query_proj = self.L__mod___deberta_encoder_layer_9_attention_self_query_proj(query_states_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_27 = l__mod___deberta_encoder_layer_9_attention_self_query_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_9_attention_self_query_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_36 = x_27.permute(0, 2, 1, 3);  x_27 = None
    contiguous_36 = permute_36.contiguous();  permute_36 = None
    query_layer_9 = contiguous_36.view(-1, 512, 64);  contiguous_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_9_attention_self_key_proj = self.L__mod___deberta_encoder_layer_9_attention_self_key_proj(query_states_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_28 = l__mod___deberta_encoder_layer_9_attention_self_key_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_9_attention_self_key_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_37 = x_28.permute(0, 2, 1, 3);  x_28 = None
    contiguous_37 = permute_37.contiguous();  permute_37 = None
    key_layer_9 = contiguous_37.view(-1, 512, 64);  contiguous_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_9_attention_self_value_proj = self.L__mod___deberta_encoder_layer_9_attention_self_value_proj(query_states_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_29 = l__mod___deberta_encoder_layer_9_attention_self_value_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_9_attention_self_value_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_38 = x_29.permute(0, 2, 1, 3);  x_29 = None
    contiguous_38 = permute_38.contiguous();  permute_38 = None
    value_layer_9 = contiguous_38.view(-1, 512, 64);  contiguous_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    tensor_9 = torch.tensor(64, dtype = torch.float32)
    mul_11 = tensor_9 * 1;  tensor_9 = None
    scale_9 = torch.sqrt(mul_11);  mul_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    transpose_9 = key_layer_9.transpose(-1, -2);  key_layer_9 = None
    to_10 = scale_9.to(dtype = torch.float32);  scale_9 = None
    truediv_9 = transpose_9 / to_10;  transpose_9 = to_10 = None
    attention_scores_27 = torch.bmm(query_layer_9, truediv_9);  query_layer_9 = truediv_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    attention_scores_28 = attention_scores_27.view(-1, 24, 512, 512);  attention_scores_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    function_ctx_37 = torch.autograd.function.FunctionCtx()
    attention_probs_18 = torch__dynamo_variables_misc_trampoline_autograd_apply_37(attention_scores_28, attention_mask_2, -1);  attention_scores_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_38 = torch.autograd.function.FunctionCtx()
    attention_probs_19 = torch__dynamo_variables_misc_trampoline_autograd_apply_38(attention_probs_18, 0.1);  attention_probs_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_97 = attention_probs_19.view(-1, 512, 512);  attention_probs_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    context_layer_27 = torch.bmm(view_97, value_layer_9);  view_97 = value_layer_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_98 = context_layer_27.view(-1, 24, 512, 64);  context_layer_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_39 = view_98.permute(0, 2, 1, 3);  view_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    context_layer_28 = permute_39.contiguous();  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    self_output_9 = context_layer_28.view((1, 512, -1));  context_layer_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    hidden_states_72 = self.L__mod___deberta_encoder_layer_9_attention_output_dense(self_output_9);  self_output_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_39 = torch.autograd.function.FunctionCtx()
    hidden_states_73 = torch__dynamo_variables_misc_trampoline_autograd_apply_39(hidden_states_72, 0.1);  hidden_states_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_18 = hidden_states_73 + query_states_18;  hidden_states_73 = query_states_18 = None
    attention_output_18 = self.L__mod___deberta_encoder_layer_9_attention_output_LayerNorm(add_18);  add_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    hidden_states_75 = self.L__mod___deberta_encoder_layer_9_intermediate_dense(attention_output_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_9 = torch._C._nn.gelu(hidden_states_75);  hidden_states_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    hidden_states_77 = self.L__mod___deberta_encoder_layer_9_output_dense(intermediate_output_9);  intermediate_output_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_40 = torch.autograd.function.FunctionCtx()
    hidden_states_78 = torch__dynamo_variables_misc_trampoline_autograd_apply_40(hidden_states_77, 0.1);  hidden_states_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_19 = hidden_states_78 + attention_output_18;  hidden_states_78 = attention_output_18 = None
    query_states_20 = self.L__mod___deberta_encoder_layer_9_output_LayerNorm(add_19);  add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_10_attention_self_query_proj = self.L__mod___deberta_encoder_layer_10_attention_self_query_proj(query_states_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_30 = l__mod___deberta_encoder_layer_10_attention_self_query_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_10_attention_self_query_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_40 = x_30.permute(0, 2, 1, 3);  x_30 = None
    contiguous_40 = permute_40.contiguous();  permute_40 = None
    query_layer_10 = contiguous_40.view(-1, 512, 64);  contiguous_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_10_attention_self_key_proj = self.L__mod___deberta_encoder_layer_10_attention_self_key_proj(query_states_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_31 = l__mod___deberta_encoder_layer_10_attention_self_key_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_10_attention_self_key_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_41 = x_31.permute(0, 2, 1, 3);  x_31 = None
    contiguous_41 = permute_41.contiguous();  permute_41 = None
    key_layer_10 = contiguous_41.view(-1, 512, 64);  contiguous_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_10_attention_self_value_proj = self.L__mod___deberta_encoder_layer_10_attention_self_value_proj(query_states_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_32 = l__mod___deberta_encoder_layer_10_attention_self_value_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_10_attention_self_value_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_42 = x_32.permute(0, 2, 1, 3);  x_32 = None
    contiguous_42 = permute_42.contiguous();  permute_42 = None
    value_layer_10 = contiguous_42.view(-1, 512, 64);  contiguous_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    tensor_10 = torch.tensor(64, dtype = torch.float32)
    mul_12 = tensor_10 * 1;  tensor_10 = None
    scale_10 = torch.sqrt(mul_12);  mul_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    transpose_10 = key_layer_10.transpose(-1, -2);  key_layer_10 = None
    to_11 = scale_10.to(dtype = torch.float32);  scale_10 = None
    truediv_10 = transpose_10 / to_11;  transpose_10 = to_11 = None
    attention_scores_30 = torch.bmm(query_layer_10, truediv_10);  query_layer_10 = truediv_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    attention_scores_31 = attention_scores_30.view(-1, 24, 512, 512);  attention_scores_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    function_ctx_41 = torch.autograd.function.FunctionCtx()
    attention_probs_20 = torch__dynamo_variables_misc_trampoline_autograd_apply_41(attention_scores_31, attention_mask_2, -1);  attention_scores_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_42 = torch.autograd.function.FunctionCtx()
    attention_probs_21 = torch__dynamo_variables_misc_trampoline_autograd_apply_42(attention_probs_20, 0.1);  attention_probs_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_107 = attention_probs_21.view(-1, 512, 512);  attention_probs_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    context_layer_30 = torch.bmm(view_107, value_layer_10);  view_107 = value_layer_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_108 = context_layer_30.view(-1, 24, 512, 64);  context_layer_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_43 = view_108.permute(0, 2, 1, 3);  view_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    context_layer_31 = permute_43.contiguous();  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    self_output_10 = context_layer_31.view((1, 512, -1));  context_layer_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    hidden_states_80 = self.L__mod___deberta_encoder_layer_10_attention_output_dense(self_output_10);  self_output_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_43 = torch.autograd.function.FunctionCtx()
    hidden_states_81 = torch__dynamo_variables_misc_trampoline_autograd_apply_43(hidden_states_80, 0.1);  hidden_states_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_20 = hidden_states_81 + query_states_20;  hidden_states_81 = query_states_20 = None
    attention_output_20 = self.L__mod___deberta_encoder_layer_10_attention_output_LayerNorm(add_20);  add_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    hidden_states_83 = self.L__mod___deberta_encoder_layer_10_intermediate_dense(attention_output_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_10 = torch._C._nn.gelu(hidden_states_83);  hidden_states_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    hidden_states_85 = self.L__mod___deberta_encoder_layer_10_output_dense(intermediate_output_10);  intermediate_output_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_44 = torch.autograd.function.FunctionCtx()
    hidden_states_86 = torch__dynamo_variables_misc_trampoline_autograd_apply_44(hidden_states_85, 0.1);  hidden_states_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_21 = hidden_states_86 + attention_output_20;  hidden_states_86 = attention_output_20 = None
    query_states_22 = self.L__mod___deberta_encoder_layer_10_output_LayerNorm(add_21);  add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_11_attention_self_query_proj = self.L__mod___deberta_encoder_layer_11_attention_self_query_proj(query_states_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_33 = l__mod___deberta_encoder_layer_11_attention_self_query_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_11_attention_self_query_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_44 = x_33.permute(0, 2, 1, 3);  x_33 = None
    contiguous_44 = permute_44.contiguous();  permute_44 = None
    query_layer_11 = contiguous_44.view(-1, 512, 64);  contiguous_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_11_attention_self_key_proj = self.L__mod___deberta_encoder_layer_11_attention_self_key_proj(query_states_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_34 = l__mod___deberta_encoder_layer_11_attention_self_key_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_11_attention_self_key_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_45 = x_34.permute(0, 2, 1, 3);  x_34 = None
    contiguous_45 = permute_45.contiguous();  permute_45 = None
    key_layer_11 = contiguous_45.view(-1, 512, 64);  contiguous_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_11_attention_self_value_proj = self.L__mod___deberta_encoder_layer_11_attention_self_value_proj(query_states_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_35 = l__mod___deberta_encoder_layer_11_attention_self_value_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_11_attention_self_value_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_46 = x_35.permute(0, 2, 1, 3);  x_35 = None
    contiguous_46 = permute_46.contiguous();  permute_46 = None
    value_layer_11 = contiguous_46.view(-1, 512, 64);  contiguous_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    tensor_11 = torch.tensor(64, dtype = torch.float32)
    mul_13 = tensor_11 * 1;  tensor_11 = None
    scale_11 = torch.sqrt(mul_13);  mul_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    transpose_11 = key_layer_11.transpose(-1, -2);  key_layer_11 = None
    to_12 = scale_11.to(dtype = torch.float32);  scale_11 = None
    truediv_11 = transpose_11 / to_12;  transpose_11 = to_12 = None
    attention_scores_33 = torch.bmm(query_layer_11, truediv_11);  query_layer_11 = truediv_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    attention_scores_34 = attention_scores_33.view(-1, 24, 512, 512);  attention_scores_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    function_ctx_45 = torch.autograd.function.FunctionCtx()
    attention_probs_22 = torch__dynamo_variables_misc_trampoline_autograd_apply_45(attention_scores_34, attention_mask_2, -1);  attention_scores_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_46 = torch.autograd.function.FunctionCtx()
    attention_probs_23 = torch__dynamo_variables_misc_trampoline_autograd_apply_46(attention_probs_22, 0.1);  attention_probs_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_117 = attention_probs_23.view(-1, 512, 512);  attention_probs_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    context_layer_33 = torch.bmm(view_117, value_layer_11);  view_117 = value_layer_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_118 = context_layer_33.view(-1, 24, 512, 64);  context_layer_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_47 = view_118.permute(0, 2, 1, 3);  view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    context_layer_34 = permute_47.contiguous();  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    self_output_11 = context_layer_34.view((1, 512, -1));  context_layer_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    hidden_states_88 = self.L__mod___deberta_encoder_layer_11_attention_output_dense(self_output_11);  self_output_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_47 = torch.autograd.function.FunctionCtx()
    hidden_states_89 = torch__dynamo_variables_misc_trampoline_autograd_apply_47(hidden_states_88, 0.1);  hidden_states_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_22 = hidden_states_89 + query_states_22;  hidden_states_89 = query_states_22 = None
    attention_output_22 = self.L__mod___deberta_encoder_layer_11_attention_output_LayerNorm(add_22);  add_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    hidden_states_91 = self.L__mod___deberta_encoder_layer_11_intermediate_dense(attention_output_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_11 = torch._C._nn.gelu(hidden_states_91);  hidden_states_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    hidden_states_93 = self.L__mod___deberta_encoder_layer_11_output_dense(intermediate_output_11);  intermediate_output_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_48 = torch.autograd.function.FunctionCtx()
    hidden_states_94 = torch__dynamo_variables_misc_trampoline_autograd_apply_48(hidden_states_93, 0.1);  hidden_states_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_23 = hidden_states_94 + attention_output_22;  hidden_states_94 = attention_output_22 = None
    query_states_24 = self.L__mod___deberta_encoder_layer_11_output_LayerNorm(add_23);  add_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_12_attention_self_query_proj = self.L__mod___deberta_encoder_layer_12_attention_self_query_proj(query_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_36 = l__mod___deberta_encoder_layer_12_attention_self_query_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_12_attention_self_query_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_48 = x_36.permute(0, 2, 1, 3);  x_36 = None
    contiguous_48 = permute_48.contiguous();  permute_48 = None
    query_layer_12 = contiguous_48.view(-1, 512, 64);  contiguous_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_12_attention_self_key_proj = self.L__mod___deberta_encoder_layer_12_attention_self_key_proj(query_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_37 = l__mod___deberta_encoder_layer_12_attention_self_key_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_12_attention_self_key_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_49 = x_37.permute(0, 2, 1, 3);  x_37 = None
    contiguous_49 = permute_49.contiguous();  permute_49 = None
    key_layer_12 = contiguous_49.view(-1, 512, 64);  contiguous_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_12_attention_self_value_proj = self.L__mod___deberta_encoder_layer_12_attention_self_value_proj(query_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_38 = l__mod___deberta_encoder_layer_12_attention_self_value_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_12_attention_self_value_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_50 = x_38.permute(0, 2, 1, 3);  x_38 = None
    contiguous_50 = permute_50.contiguous();  permute_50 = None
    value_layer_12 = contiguous_50.view(-1, 512, 64);  contiguous_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    tensor_12 = torch.tensor(64, dtype = torch.float32)
    mul_14 = tensor_12 * 1;  tensor_12 = None
    scale_12 = torch.sqrt(mul_14);  mul_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    transpose_12 = key_layer_12.transpose(-1, -2);  key_layer_12 = None
    to_13 = scale_12.to(dtype = torch.float32);  scale_12 = None
    truediv_12 = transpose_12 / to_13;  transpose_12 = to_13 = None
    attention_scores_36 = torch.bmm(query_layer_12, truediv_12);  query_layer_12 = truediv_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    attention_scores_37 = attention_scores_36.view(-1, 24, 512, 512);  attention_scores_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    function_ctx_49 = torch.autograd.function.FunctionCtx()
    attention_probs_24 = torch__dynamo_variables_misc_trampoline_autograd_apply_49(attention_scores_37, attention_mask_2, -1);  attention_scores_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_50 = torch.autograd.function.FunctionCtx()
    attention_probs_25 = torch__dynamo_variables_misc_trampoline_autograd_apply_50(attention_probs_24, 0.1);  attention_probs_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_127 = attention_probs_25.view(-1, 512, 512);  attention_probs_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    context_layer_36 = torch.bmm(view_127, value_layer_12);  view_127 = value_layer_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_128 = context_layer_36.view(-1, 24, 512, 64);  context_layer_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_51 = view_128.permute(0, 2, 1, 3);  view_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    context_layer_37 = permute_51.contiguous();  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    self_output_12 = context_layer_37.view((1, 512, -1));  context_layer_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    hidden_states_96 = self.L__mod___deberta_encoder_layer_12_attention_output_dense(self_output_12);  self_output_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_51 = torch.autograd.function.FunctionCtx()
    hidden_states_97 = torch__dynamo_variables_misc_trampoline_autograd_apply_51(hidden_states_96, 0.1);  hidden_states_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_24 = hidden_states_97 + query_states_24;  hidden_states_97 = query_states_24 = None
    attention_output_24 = self.L__mod___deberta_encoder_layer_12_attention_output_LayerNorm(add_24);  add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    hidden_states_99 = self.L__mod___deberta_encoder_layer_12_intermediate_dense(attention_output_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_12 = torch._C._nn.gelu(hidden_states_99);  hidden_states_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    hidden_states_101 = self.L__mod___deberta_encoder_layer_12_output_dense(intermediate_output_12);  intermediate_output_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_52 = torch.autograd.function.FunctionCtx()
    hidden_states_102 = torch__dynamo_variables_misc_trampoline_autograd_apply_52(hidden_states_101, 0.1);  hidden_states_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_25 = hidden_states_102 + attention_output_24;  hidden_states_102 = attention_output_24 = None
    query_states_26 = self.L__mod___deberta_encoder_layer_12_output_LayerNorm(add_25);  add_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_13_attention_self_query_proj = self.L__mod___deberta_encoder_layer_13_attention_self_query_proj(query_states_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_39 = l__mod___deberta_encoder_layer_13_attention_self_query_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_13_attention_self_query_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_52 = x_39.permute(0, 2, 1, 3);  x_39 = None
    contiguous_52 = permute_52.contiguous();  permute_52 = None
    query_layer_13 = contiguous_52.view(-1, 512, 64);  contiguous_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_13_attention_self_key_proj = self.L__mod___deberta_encoder_layer_13_attention_self_key_proj(query_states_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_40 = l__mod___deberta_encoder_layer_13_attention_self_key_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_13_attention_self_key_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_53 = x_40.permute(0, 2, 1, 3);  x_40 = None
    contiguous_53 = permute_53.contiguous();  permute_53 = None
    key_layer_13 = contiguous_53.view(-1, 512, 64);  contiguous_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_13_attention_self_value_proj = self.L__mod___deberta_encoder_layer_13_attention_self_value_proj(query_states_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_41 = l__mod___deberta_encoder_layer_13_attention_self_value_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_13_attention_self_value_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_54 = x_41.permute(0, 2, 1, 3);  x_41 = None
    contiguous_54 = permute_54.contiguous();  permute_54 = None
    value_layer_13 = contiguous_54.view(-1, 512, 64);  contiguous_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    tensor_13 = torch.tensor(64, dtype = torch.float32)
    mul_15 = tensor_13 * 1;  tensor_13 = None
    scale_13 = torch.sqrt(mul_15);  mul_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    transpose_13 = key_layer_13.transpose(-1, -2);  key_layer_13 = None
    to_14 = scale_13.to(dtype = torch.float32);  scale_13 = None
    truediv_13 = transpose_13 / to_14;  transpose_13 = to_14 = None
    attention_scores_39 = torch.bmm(query_layer_13, truediv_13);  query_layer_13 = truediv_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    attention_scores_40 = attention_scores_39.view(-1, 24, 512, 512);  attention_scores_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    function_ctx_53 = torch.autograd.function.FunctionCtx()
    attention_probs_26 = torch__dynamo_variables_misc_trampoline_autograd_apply_53(attention_scores_40, attention_mask_2, -1);  attention_scores_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_54 = torch.autograd.function.FunctionCtx()
    attention_probs_27 = torch__dynamo_variables_misc_trampoline_autograd_apply_54(attention_probs_26, 0.1);  attention_probs_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_137 = attention_probs_27.view(-1, 512, 512);  attention_probs_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    context_layer_39 = torch.bmm(view_137, value_layer_13);  view_137 = value_layer_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_138 = context_layer_39.view(-1, 24, 512, 64);  context_layer_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_55 = view_138.permute(0, 2, 1, 3);  view_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    context_layer_40 = permute_55.contiguous();  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    self_output_13 = context_layer_40.view((1, 512, -1));  context_layer_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    hidden_states_104 = self.L__mod___deberta_encoder_layer_13_attention_output_dense(self_output_13);  self_output_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_55 = torch.autograd.function.FunctionCtx()
    hidden_states_105 = torch__dynamo_variables_misc_trampoline_autograd_apply_55(hidden_states_104, 0.1);  hidden_states_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_26 = hidden_states_105 + query_states_26;  hidden_states_105 = query_states_26 = None
    attention_output_26 = self.L__mod___deberta_encoder_layer_13_attention_output_LayerNorm(add_26);  add_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    hidden_states_107 = self.L__mod___deberta_encoder_layer_13_intermediate_dense(attention_output_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_13 = torch._C._nn.gelu(hidden_states_107);  hidden_states_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    hidden_states_109 = self.L__mod___deberta_encoder_layer_13_output_dense(intermediate_output_13);  intermediate_output_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_56 = torch.autograd.function.FunctionCtx()
    hidden_states_110 = torch__dynamo_variables_misc_trampoline_autograd_apply_56(hidden_states_109, 0.1);  hidden_states_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_27 = hidden_states_110 + attention_output_26;  hidden_states_110 = attention_output_26 = None
    query_states_28 = self.L__mod___deberta_encoder_layer_13_output_LayerNorm(add_27);  add_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_14_attention_self_query_proj = self.L__mod___deberta_encoder_layer_14_attention_self_query_proj(query_states_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_42 = l__mod___deberta_encoder_layer_14_attention_self_query_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_14_attention_self_query_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_56 = x_42.permute(0, 2, 1, 3);  x_42 = None
    contiguous_56 = permute_56.contiguous();  permute_56 = None
    query_layer_14 = contiguous_56.view(-1, 512, 64);  contiguous_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_14_attention_self_key_proj = self.L__mod___deberta_encoder_layer_14_attention_self_key_proj(query_states_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_43 = l__mod___deberta_encoder_layer_14_attention_self_key_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_14_attention_self_key_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_57 = x_43.permute(0, 2, 1, 3);  x_43 = None
    contiguous_57 = permute_57.contiguous();  permute_57 = None
    key_layer_14 = contiguous_57.view(-1, 512, 64);  contiguous_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_14_attention_self_value_proj = self.L__mod___deberta_encoder_layer_14_attention_self_value_proj(query_states_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_44 = l__mod___deberta_encoder_layer_14_attention_self_value_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_14_attention_self_value_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_58 = x_44.permute(0, 2, 1, 3);  x_44 = None
    contiguous_58 = permute_58.contiguous();  permute_58 = None
    value_layer_14 = contiguous_58.view(-1, 512, 64);  contiguous_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    tensor_14 = torch.tensor(64, dtype = torch.float32)
    mul_16 = tensor_14 * 1;  tensor_14 = None
    scale_14 = torch.sqrt(mul_16);  mul_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    transpose_14 = key_layer_14.transpose(-1, -2);  key_layer_14 = None
    to_15 = scale_14.to(dtype = torch.float32);  scale_14 = None
    truediv_14 = transpose_14 / to_15;  transpose_14 = to_15 = None
    attention_scores_42 = torch.bmm(query_layer_14, truediv_14);  query_layer_14 = truediv_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    attention_scores_43 = attention_scores_42.view(-1, 24, 512, 512);  attention_scores_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    function_ctx_57 = torch.autograd.function.FunctionCtx()
    attention_probs_28 = torch__dynamo_variables_misc_trampoline_autograd_apply_57(attention_scores_43, attention_mask_2, -1);  attention_scores_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_58 = torch.autograd.function.FunctionCtx()
    attention_probs_29 = torch__dynamo_variables_misc_trampoline_autograd_apply_58(attention_probs_28, 0.1);  attention_probs_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_147 = attention_probs_29.view(-1, 512, 512);  attention_probs_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    context_layer_42 = torch.bmm(view_147, value_layer_14);  view_147 = value_layer_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_148 = context_layer_42.view(-1, 24, 512, 64);  context_layer_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_59 = view_148.permute(0, 2, 1, 3);  view_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    context_layer_43 = permute_59.contiguous();  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    self_output_14 = context_layer_43.view((1, 512, -1));  context_layer_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    hidden_states_112 = self.L__mod___deberta_encoder_layer_14_attention_output_dense(self_output_14);  self_output_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_59 = torch.autograd.function.FunctionCtx()
    hidden_states_113 = torch__dynamo_variables_misc_trampoline_autograd_apply_59(hidden_states_112, 0.1);  hidden_states_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_28 = hidden_states_113 + query_states_28;  hidden_states_113 = query_states_28 = None
    attention_output_28 = self.L__mod___deberta_encoder_layer_14_attention_output_LayerNorm(add_28);  add_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    hidden_states_115 = self.L__mod___deberta_encoder_layer_14_intermediate_dense(attention_output_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_14 = torch._C._nn.gelu(hidden_states_115);  hidden_states_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    hidden_states_117 = self.L__mod___deberta_encoder_layer_14_output_dense(intermediate_output_14);  intermediate_output_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_60 = torch.autograd.function.FunctionCtx()
    hidden_states_118 = torch__dynamo_variables_misc_trampoline_autograd_apply_60(hidden_states_117, 0.1);  hidden_states_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_29 = hidden_states_118 + attention_output_28;  hidden_states_118 = attention_output_28 = None
    query_states_30 = self.L__mod___deberta_encoder_layer_14_output_LayerNorm(add_29);  add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_15_attention_self_query_proj = self.L__mod___deberta_encoder_layer_15_attention_self_query_proj(query_states_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_45 = l__mod___deberta_encoder_layer_15_attention_self_query_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_15_attention_self_query_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_60 = x_45.permute(0, 2, 1, 3);  x_45 = None
    contiguous_60 = permute_60.contiguous();  permute_60 = None
    query_layer_15 = contiguous_60.view(-1, 512, 64);  contiguous_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_15_attention_self_key_proj = self.L__mod___deberta_encoder_layer_15_attention_self_key_proj(query_states_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_46 = l__mod___deberta_encoder_layer_15_attention_self_key_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_15_attention_self_key_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_61 = x_46.permute(0, 2, 1, 3);  x_46 = None
    contiguous_61 = permute_61.contiguous();  permute_61 = None
    key_layer_15 = contiguous_61.view(-1, 512, 64);  contiguous_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_15_attention_self_value_proj = self.L__mod___deberta_encoder_layer_15_attention_self_value_proj(query_states_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_47 = l__mod___deberta_encoder_layer_15_attention_self_value_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_15_attention_self_value_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_62 = x_47.permute(0, 2, 1, 3);  x_47 = None
    contiguous_62 = permute_62.contiguous();  permute_62 = None
    value_layer_15 = contiguous_62.view(-1, 512, 64);  contiguous_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    tensor_15 = torch.tensor(64, dtype = torch.float32)
    mul_17 = tensor_15 * 1;  tensor_15 = None
    scale_15 = torch.sqrt(mul_17);  mul_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    transpose_15 = key_layer_15.transpose(-1, -2);  key_layer_15 = None
    to_16 = scale_15.to(dtype = torch.float32);  scale_15 = None
    truediv_15 = transpose_15 / to_16;  transpose_15 = to_16 = None
    attention_scores_45 = torch.bmm(query_layer_15, truediv_15);  query_layer_15 = truediv_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    attention_scores_46 = attention_scores_45.view(-1, 24, 512, 512);  attention_scores_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    function_ctx_61 = torch.autograd.function.FunctionCtx()
    attention_probs_30 = torch__dynamo_variables_misc_trampoline_autograd_apply_61(attention_scores_46, attention_mask_2, -1);  attention_scores_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_62 = torch.autograd.function.FunctionCtx()
    attention_probs_31 = torch__dynamo_variables_misc_trampoline_autograd_apply_62(attention_probs_30, 0.1);  attention_probs_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_157 = attention_probs_31.view(-1, 512, 512);  attention_probs_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    context_layer_45 = torch.bmm(view_157, value_layer_15);  view_157 = value_layer_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_158 = context_layer_45.view(-1, 24, 512, 64);  context_layer_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_63 = view_158.permute(0, 2, 1, 3);  view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    context_layer_46 = permute_63.contiguous();  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    self_output_15 = context_layer_46.view((1, 512, -1));  context_layer_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    hidden_states_120 = self.L__mod___deberta_encoder_layer_15_attention_output_dense(self_output_15);  self_output_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_63 = torch.autograd.function.FunctionCtx()
    hidden_states_121 = torch__dynamo_variables_misc_trampoline_autograd_apply_63(hidden_states_120, 0.1);  hidden_states_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_30 = hidden_states_121 + query_states_30;  hidden_states_121 = query_states_30 = None
    attention_output_30 = self.L__mod___deberta_encoder_layer_15_attention_output_LayerNorm(add_30);  add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    hidden_states_123 = self.L__mod___deberta_encoder_layer_15_intermediate_dense(attention_output_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_15 = torch._C._nn.gelu(hidden_states_123);  hidden_states_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    hidden_states_125 = self.L__mod___deberta_encoder_layer_15_output_dense(intermediate_output_15);  intermediate_output_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_64 = torch.autograd.function.FunctionCtx()
    hidden_states_126 = torch__dynamo_variables_misc_trampoline_autograd_apply_64(hidden_states_125, 0.1);  hidden_states_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_31 = hidden_states_126 + attention_output_30;  hidden_states_126 = attention_output_30 = None
    query_states_32 = self.L__mod___deberta_encoder_layer_15_output_LayerNorm(add_31);  add_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_16_attention_self_query_proj = self.L__mod___deberta_encoder_layer_16_attention_self_query_proj(query_states_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_48 = l__mod___deberta_encoder_layer_16_attention_self_query_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_16_attention_self_query_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_64 = x_48.permute(0, 2, 1, 3);  x_48 = None
    contiguous_64 = permute_64.contiguous();  permute_64 = None
    query_layer_16 = contiguous_64.view(-1, 512, 64);  contiguous_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_16_attention_self_key_proj = self.L__mod___deberta_encoder_layer_16_attention_self_key_proj(query_states_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_49 = l__mod___deberta_encoder_layer_16_attention_self_key_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_16_attention_self_key_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_65 = x_49.permute(0, 2, 1, 3);  x_49 = None
    contiguous_65 = permute_65.contiguous();  permute_65 = None
    key_layer_16 = contiguous_65.view(-1, 512, 64);  contiguous_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_16_attention_self_value_proj = self.L__mod___deberta_encoder_layer_16_attention_self_value_proj(query_states_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_50 = l__mod___deberta_encoder_layer_16_attention_self_value_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_16_attention_self_value_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_66 = x_50.permute(0, 2, 1, 3);  x_50 = None
    contiguous_66 = permute_66.contiguous();  permute_66 = None
    value_layer_16 = contiguous_66.view(-1, 512, 64);  contiguous_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    tensor_16 = torch.tensor(64, dtype = torch.float32)
    mul_18 = tensor_16 * 1;  tensor_16 = None
    scale_16 = torch.sqrt(mul_18);  mul_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    transpose_16 = key_layer_16.transpose(-1, -2);  key_layer_16 = None
    to_17 = scale_16.to(dtype = torch.float32);  scale_16 = None
    truediv_16 = transpose_16 / to_17;  transpose_16 = to_17 = None
    attention_scores_48 = torch.bmm(query_layer_16, truediv_16);  query_layer_16 = truediv_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    attention_scores_49 = attention_scores_48.view(-1, 24, 512, 512);  attention_scores_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    function_ctx_65 = torch.autograd.function.FunctionCtx()
    attention_probs_32 = torch__dynamo_variables_misc_trampoline_autograd_apply_65(attention_scores_49, attention_mask_2, -1);  attention_scores_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_66 = torch.autograd.function.FunctionCtx()
    attention_probs_33 = torch__dynamo_variables_misc_trampoline_autograd_apply_66(attention_probs_32, 0.1);  attention_probs_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_167 = attention_probs_33.view(-1, 512, 512);  attention_probs_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    context_layer_48 = torch.bmm(view_167, value_layer_16);  view_167 = value_layer_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_168 = context_layer_48.view(-1, 24, 512, 64);  context_layer_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_67 = view_168.permute(0, 2, 1, 3);  view_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    context_layer_49 = permute_67.contiguous();  permute_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    self_output_16 = context_layer_49.view((1, 512, -1));  context_layer_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    hidden_states_128 = self.L__mod___deberta_encoder_layer_16_attention_output_dense(self_output_16);  self_output_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_67 = torch.autograd.function.FunctionCtx()
    hidden_states_129 = torch__dynamo_variables_misc_trampoline_autograd_apply_67(hidden_states_128, 0.1);  hidden_states_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_32 = hidden_states_129 + query_states_32;  hidden_states_129 = query_states_32 = None
    attention_output_32 = self.L__mod___deberta_encoder_layer_16_attention_output_LayerNorm(add_32);  add_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    hidden_states_131 = self.L__mod___deberta_encoder_layer_16_intermediate_dense(attention_output_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_16 = torch._C._nn.gelu(hidden_states_131);  hidden_states_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    hidden_states_133 = self.L__mod___deberta_encoder_layer_16_output_dense(intermediate_output_16);  intermediate_output_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_68 = torch.autograd.function.FunctionCtx()
    hidden_states_134 = torch__dynamo_variables_misc_trampoline_autograd_apply_68(hidden_states_133, 0.1);  hidden_states_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_33 = hidden_states_134 + attention_output_32;  hidden_states_134 = attention_output_32 = None
    query_states_34 = self.L__mod___deberta_encoder_layer_16_output_LayerNorm(add_33);  add_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_17_attention_self_query_proj = self.L__mod___deberta_encoder_layer_17_attention_self_query_proj(query_states_34)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_51 = l__mod___deberta_encoder_layer_17_attention_self_query_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_17_attention_self_query_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_68 = x_51.permute(0, 2, 1, 3);  x_51 = None
    contiguous_68 = permute_68.contiguous();  permute_68 = None
    query_layer_17 = contiguous_68.view(-1, 512, 64);  contiguous_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_17_attention_self_key_proj = self.L__mod___deberta_encoder_layer_17_attention_self_key_proj(query_states_34)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_52 = l__mod___deberta_encoder_layer_17_attention_self_key_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_17_attention_self_key_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_69 = x_52.permute(0, 2, 1, 3);  x_52 = None
    contiguous_69 = permute_69.contiguous();  permute_69 = None
    key_layer_17 = contiguous_69.view(-1, 512, 64);  contiguous_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_17_attention_self_value_proj = self.L__mod___deberta_encoder_layer_17_attention_self_value_proj(query_states_34)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_53 = l__mod___deberta_encoder_layer_17_attention_self_value_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_17_attention_self_value_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_70 = x_53.permute(0, 2, 1, 3);  x_53 = None
    contiguous_70 = permute_70.contiguous();  permute_70 = None
    value_layer_17 = contiguous_70.view(-1, 512, 64);  contiguous_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    tensor_17 = torch.tensor(64, dtype = torch.float32)
    mul_19 = tensor_17 * 1;  tensor_17 = None
    scale_17 = torch.sqrt(mul_19);  mul_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    transpose_17 = key_layer_17.transpose(-1, -2);  key_layer_17 = None
    to_18 = scale_17.to(dtype = torch.float32);  scale_17 = None
    truediv_17 = transpose_17 / to_18;  transpose_17 = to_18 = None
    attention_scores_51 = torch.bmm(query_layer_17, truediv_17);  query_layer_17 = truediv_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    attention_scores_52 = attention_scores_51.view(-1, 24, 512, 512);  attention_scores_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    function_ctx_69 = torch.autograd.function.FunctionCtx()
    attention_probs_34 = torch__dynamo_variables_misc_trampoline_autograd_apply_69(attention_scores_52, attention_mask_2, -1);  attention_scores_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_70 = torch.autograd.function.FunctionCtx()
    attention_probs_35 = torch__dynamo_variables_misc_trampoline_autograd_apply_70(attention_probs_34, 0.1);  attention_probs_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_177 = attention_probs_35.view(-1, 512, 512);  attention_probs_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    context_layer_51 = torch.bmm(view_177, value_layer_17);  view_177 = value_layer_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_178 = context_layer_51.view(-1, 24, 512, 64);  context_layer_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_71 = view_178.permute(0, 2, 1, 3);  view_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    context_layer_52 = permute_71.contiguous();  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    self_output_17 = context_layer_52.view((1, 512, -1));  context_layer_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    hidden_states_136 = self.L__mod___deberta_encoder_layer_17_attention_output_dense(self_output_17);  self_output_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_71 = torch.autograd.function.FunctionCtx()
    hidden_states_137 = torch__dynamo_variables_misc_trampoline_autograd_apply_71(hidden_states_136, 0.1);  hidden_states_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_34 = hidden_states_137 + query_states_34;  hidden_states_137 = query_states_34 = None
    attention_output_34 = self.L__mod___deberta_encoder_layer_17_attention_output_LayerNorm(add_34);  add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    hidden_states_139 = self.L__mod___deberta_encoder_layer_17_intermediate_dense(attention_output_34)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_17 = torch._C._nn.gelu(hidden_states_139);  hidden_states_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    hidden_states_141 = self.L__mod___deberta_encoder_layer_17_output_dense(intermediate_output_17);  intermediate_output_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_72 = torch.autograd.function.FunctionCtx()
    hidden_states_142 = torch__dynamo_variables_misc_trampoline_autograd_apply_72(hidden_states_141, 0.1);  hidden_states_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_35 = hidden_states_142 + attention_output_34;  hidden_states_142 = attention_output_34 = None
    query_states_36 = self.L__mod___deberta_encoder_layer_17_output_LayerNorm(add_35);  add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_18_attention_self_query_proj = self.L__mod___deberta_encoder_layer_18_attention_self_query_proj(query_states_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_54 = l__mod___deberta_encoder_layer_18_attention_self_query_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_18_attention_self_query_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_72 = x_54.permute(0, 2, 1, 3);  x_54 = None
    contiguous_72 = permute_72.contiguous();  permute_72 = None
    query_layer_18 = contiguous_72.view(-1, 512, 64);  contiguous_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_18_attention_self_key_proj = self.L__mod___deberta_encoder_layer_18_attention_self_key_proj(query_states_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_55 = l__mod___deberta_encoder_layer_18_attention_self_key_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_18_attention_self_key_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_73 = x_55.permute(0, 2, 1, 3);  x_55 = None
    contiguous_73 = permute_73.contiguous();  permute_73 = None
    key_layer_18 = contiguous_73.view(-1, 512, 64);  contiguous_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_18_attention_self_value_proj = self.L__mod___deberta_encoder_layer_18_attention_self_value_proj(query_states_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_56 = l__mod___deberta_encoder_layer_18_attention_self_value_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_18_attention_self_value_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_74 = x_56.permute(0, 2, 1, 3);  x_56 = None
    contiguous_74 = permute_74.contiguous();  permute_74 = None
    value_layer_18 = contiguous_74.view(-1, 512, 64);  contiguous_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    tensor_18 = torch.tensor(64, dtype = torch.float32)
    mul_20 = tensor_18 * 1;  tensor_18 = None
    scale_18 = torch.sqrt(mul_20);  mul_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    transpose_18 = key_layer_18.transpose(-1, -2);  key_layer_18 = None
    to_19 = scale_18.to(dtype = torch.float32);  scale_18 = None
    truediv_18 = transpose_18 / to_19;  transpose_18 = to_19 = None
    attention_scores_54 = torch.bmm(query_layer_18, truediv_18);  query_layer_18 = truediv_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    attention_scores_55 = attention_scores_54.view(-1, 24, 512, 512);  attention_scores_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    function_ctx_73 = torch.autograd.function.FunctionCtx()
    attention_probs_36 = torch__dynamo_variables_misc_trampoline_autograd_apply_73(attention_scores_55, attention_mask_2, -1);  attention_scores_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_74 = torch.autograd.function.FunctionCtx()
    attention_probs_37 = torch__dynamo_variables_misc_trampoline_autograd_apply_74(attention_probs_36, 0.1);  attention_probs_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_187 = attention_probs_37.view(-1, 512, 512);  attention_probs_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    context_layer_54 = torch.bmm(view_187, value_layer_18);  view_187 = value_layer_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_188 = context_layer_54.view(-1, 24, 512, 64);  context_layer_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_75 = view_188.permute(0, 2, 1, 3);  view_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    context_layer_55 = permute_75.contiguous();  permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    self_output_18 = context_layer_55.view((1, 512, -1));  context_layer_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    hidden_states_144 = self.L__mod___deberta_encoder_layer_18_attention_output_dense(self_output_18);  self_output_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_75 = torch.autograd.function.FunctionCtx()
    hidden_states_145 = torch__dynamo_variables_misc_trampoline_autograd_apply_75(hidden_states_144, 0.1);  hidden_states_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_36 = hidden_states_145 + query_states_36;  hidden_states_145 = query_states_36 = None
    attention_output_36 = self.L__mod___deberta_encoder_layer_18_attention_output_LayerNorm(add_36);  add_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    hidden_states_147 = self.L__mod___deberta_encoder_layer_18_intermediate_dense(attention_output_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_18 = torch._C._nn.gelu(hidden_states_147);  hidden_states_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    hidden_states_149 = self.L__mod___deberta_encoder_layer_18_output_dense(intermediate_output_18);  intermediate_output_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_76 = torch.autograd.function.FunctionCtx()
    hidden_states_150 = torch__dynamo_variables_misc_trampoline_autograd_apply_76(hidden_states_149, 0.1);  hidden_states_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_37 = hidden_states_150 + attention_output_36;  hidden_states_150 = attention_output_36 = None
    query_states_38 = self.L__mod___deberta_encoder_layer_18_output_LayerNorm(add_37);  add_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_19_attention_self_query_proj = self.L__mod___deberta_encoder_layer_19_attention_self_query_proj(query_states_38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_57 = l__mod___deberta_encoder_layer_19_attention_self_query_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_19_attention_self_query_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_76 = x_57.permute(0, 2, 1, 3);  x_57 = None
    contiguous_76 = permute_76.contiguous();  permute_76 = None
    query_layer_19 = contiguous_76.view(-1, 512, 64);  contiguous_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_19_attention_self_key_proj = self.L__mod___deberta_encoder_layer_19_attention_self_key_proj(query_states_38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_58 = l__mod___deberta_encoder_layer_19_attention_self_key_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_19_attention_self_key_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_77 = x_58.permute(0, 2, 1, 3);  x_58 = None
    contiguous_77 = permute_77.contiguous();  permute_77 = None
    key_layer_19 = contiguous_77.view(-1, 512, 64);  contiguous_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_19_attention_self_value_proj = self.L__mod___deberta_encoder_layer_19_attention_self_value_proj(query_states_38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_59 = l__mod___deberta_encoder_layer_19_attention_self_value_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_19_attention_self_value_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_78 = x_59.permute(0, 2, 1, 3);  x_59 = None
    contiguous_78 = permute_78.contiguous();  permute_78 = None
    value_layer_19 = contiguous_78.view(-1, 512, 64);  contiguous_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    tensor_19 = torch.tensor(64, dtype = torch.float32)
    mul_21 = tensor_19 * 1;  tensor_19 = None
    scale_19 = torch.sqrt(mul_21);  mul_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    transpose_19 = key_layer_19.transpose(-1, -2);  key_layer_19 = None
    to_20 = scale_19.to(dtype = torch.float32);  scale_19 = None
    truediv_19 = transpose_19 / to_20;  transpose_19 = to_20 = None
    attention_scores_57 = torch.bmm(query_layer_19, truediv_19);  query_layer_19 = truediv_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    attention_scores_58 = attention_scores_57.view(-1, 24, 512, 512);  attention_scores_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    function_ctx_77 = torch.autograd.function.FunctionCtx()
    attention_probs_38 = torch__dynamo_variables_misc_trampoline_autograd_apply_77(attention_scores_58, attention_mask_2, -1);  attention_scores_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_78 = torch.autograd.function.FunctionCtx()
    attention_probs_39 = torch__dynamo_variables_misc_trampoline_autograd_apply_78(attention_probs_38, 0.1);  attention_probs_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_197 = attention_probs_39.view(-1, 512, 512);  attention_probs_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    context_layer_57 = torch.bmm(view_197, value_layer_19);  view_197 = value_layer_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_198 = context_layer_57.view(-1, 24, 512, 64);  context_layer_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_79 = view_198.permute(0, 2, 1, 3);  view_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    context_layer_58 = permute_79.contiguous();  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    self_output_19 = context_layer_58.view((1, 512, -1));  context_layer_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    hidden_states_152 = self.L__mod___deberta_encoder_layer_19_attention_output_dense(self_output_19);  self_output_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_79 = torch.autograd.function.FunctionCtx()
    hidden_states_153 = torch__dynamo_variables_misc_trampoline_autograd_apply_79(hidden_states_152, 0.1);  hidden_states_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_38 = hidden_states_153 + query_states_38;  hidden_states_153 = query_states_38 = None
    attention_output_38 = self.L__mod___deberta_encoder_layer_19_attention_output_LayerNorm(add_38);  add_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    hidden_states_155 = self.L__mod___deberta_encoder_layer_19_intermediate_dense(attention_output_38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_19 = torch._C._nn.gelu(hidden_states_155);  hidden_states_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    hidden_states_157 = self.L__mod___deberta_encoder_layer_19_output_dense(intermediate_output_19);  intermediate_output_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_80 = torch.autograd.function.FunctionCtx()
    hidden_states_158 = torch__dynamo_variables_misc_trampoline_autograd_apply_80(hidden_states_157, 0.1);  hidden_states_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_39 = hidden_states_158 + attention_output_38;  hidden_states_158 = attention_output_38 = None
    query_states_40 = self.L__mod___deberta_encoder_layer_19_output_LayerNorm(add_39);  add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_20_attention_self_query_proj = self.L__mod___deberta_encoder_layer_20_attention_self_query_proj(query_states_40)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_60 = l__mod___deberta_encoder_layer_20_attention_self_query_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_20_attention_self_query_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_80 = x_60.permute(0, 2, 1, 3);  x_60 = None
    contiguous_80 = permute_80.contiguous();  permute_80 = None
    query_layer_20 = contiguous_80.view(-1, 512, 64);  contiguous_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_20_attention_self_key_proj = self.L__mod___deberta_encoder_layer_20_attention_self_key_proj(query_states_40)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_61 = l__mod___deberta_encoder_layer_20_attention_self_key_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_20_attention_self_key_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_81 = x_61.permute(0, 2, 1, 3);  x_61 = None
    contiguous_81 = permute_81.contiguous();  permute_81 = None
    key_layer_20 = contiguous_81.view(-1, 512, 64);  contiguous_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_20_attention_self_value_proj = self.L__mod___deberta_encoder_layer_20_attention_self_value_proj(query_states_40)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_62 = l__mod___deberta_encoder_layer_20_attention_self_value_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_20_attention_self_value_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_82 = x_62.permute(0, 2, 1, 3);  x_62 = None
    contiguous_82 = permute_82.contiguous();  permute_82 = None
    value_layer_20 = contiguous_82.view(-1, 512, 64);  contiguous_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    tensor_20 = torch.tensor(64, dtype = torch.float32)
    mul_22 = tensor_20 * 1;  tensor_20 = None
    scale_20 = torch.sqrt(mul_22);  mul_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    transpose_20 = key_layer_20.transpose(-1, -2);  key_layer_20 = None
    to_21 = scale_20.to(dtype = torch.float32);  scale_20 = None
    truediv_20 = transpose_20 / to_21;  transpose_20 = to_21 = None
    attention_scores_60 = torch.bmm(query_layer_20, truediv_20);  query_layer_20 = truediv_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    attention_scores_61 = attention_scores_60.view(-1, 24, 512, 512);  attention_scores_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    function_ctx_81 = torch.autograd.function.FunctionCtx()
    attention_probs_40 = torch__dynamo_variables_misc_trampoline_autograd_apply_81(attention_scores_61, attention_mask_2, -1);  attention_scores_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_82 = torch.autograd.function.FunctionCtx()
    attention_probs_41 = torch__dynamo_variables_misc_trampoline_autograd_apply_82(attention_probs_40, 0.1);  attention_probs_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_207 = attention_probs_41.view(-1, 512, 512);  attention_probs_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    context_layer_60 = torch.bmm(view_207, value_layer_20);  view_207 = value_layer_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_208 = context_layer_60.view(-1, 24, 512, 64);  context_layer_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_83 = view_208.permute(0, 2, 1, 3);  view_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    context_layer_61 = permute_83.contiguous();  permute_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    self_output_20 = context_layer_61.view((1, 512, -1));  context_layer_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    hidden_states_160 = self.L__mod___deberta_encoder_layer_20_attention_output_dense(self_output_20);  self_output_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_83 = torch.autograd.function.FunctionCtx()
    hidden_states_161 = torch__dynamo_variables_misc_trampoline_autograd_apply_83(hidden_states_160, 0.1);  hidden_states_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_40 = hidden_states_161 + query_states_40;  hidden_states_161 = query_states_40 = None
    attention_output_40 = self.L__mod___deberta_encoder_layer_20_attention_output_LayerNorm(add_40);  add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    hidden_states_163 = self.L__mod___deberta_encoder_layer_20_intermediate_dense(attention_output_40)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_20 = torch._C._nn.gelu(hidden_states_163);  hidden_states_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    hidden_states_165 = self.L__mod___deberta_encoder_layer_20_output_dense(intermediate_output_20);  intermediate_output_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_84 = torch.autograd.function.FunctionCtx()
    hidden_states_166 = torch__dynamo_variables_misc_trampoline_autograd_apply_84(hidden_states_165, 0.1);  hidden_states_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_41 = hidden_states_166 + attention_output_40;  hidden_states_166 = attention_output_40 = None
    query_states_42 = self.L__mod___deberta_encoder_layer_20_output_LayerNorm(add_41);  add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_21_attention_self_query_proj = self.L__mod___deberta_encoder_layer_21_attention_self_query_proj(query_states_42)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_63 = l__mod___deberta_encoder_layer_21_attention_self_query_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_21_attention_self_query_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_84 = x_63.permute(0, 2, 1, 3);  x_63 = None
    contiguous_84 = permute_84.contiguous();  permute_84 = None
    query_layer_21 = contiguous_84.view(-1, 512, 64);  contiguous_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_21_attention_self_key_proj = self.L__mod___deberta_encoder_layer_21_attention_self_key_proj(query_states_42)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_64 = l__mod___deberta_encoder_layer_21_attention_self_key_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_21_attention_self_key_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_85 = x_64.permute(0, 2, 1, 3);  x_64 = None
    contiguous_85 = permute_85.contiguous();  permute_85 = None
    key_layer_21 = contiguous_85.view(-1, 512, 64);  contiguous_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_21_attention_self_value_proj = self.L__mod___deberta_encoder_layer_21_attention_self_value_proj(query_states_42)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_65 = l__mod___deberta_encoder_layer_21_attention_self_value_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_21_attention_self_value_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_86 = x_65.permute(0, 2, 1, 3);  x_65 = None
    contiguous_86 = permute_86.contiguous();  permute_86 = None
    value_layer_21 = contiguous_86.view(-1, 512, 64);  contiguous_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    tensor_21 = torch.tensor(64, dtype = torch.float32)
    mul_23 = tensor_21 * 1;  tensor_21 = None
    scale_21 = torch.sqrt(mul_23);  mul_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    transpose_21 = key_layer_21.transpose(-1, -2);  key_layer_21 = None
    to_22 = scale_21.to(dtype = torch.float32);  scale_21 = None
    truediv_21 = transpose_21 / to_22;  transpose_21 = to_22 = None
    attention_scores_63 = torch.bmm(query_layer_21, truediv_21);  query_layer_21 = truediv_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    attention_scores_64 = attention_scores_63.view(-1, 24, 512, 512);  attention_scores_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    function_ctx_85 = torch.autograd.function.FunctionCtx()
    attention_probs_42 = torch__dynamo_variables_misc_trampoline_autograd_apply_85(attention_scores_64, attention_mask_2, -1);  attention_scores_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_86 = torch.autograd.function.FunctionCtx()
    attention_probs_43 = torch__dynamo_variables_misc_trampoline_autograd_apply_86(attention_probs_42, 0.1);  attention_probs_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_217 = attention_probs_43.view(-1, 512, 512);  attention_probs_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    context_layer_63 = torch.bmm(view_217, value_layer_21);  view_217 = value_layer_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_218 = context_layer_63.view(-1, 24, 512, 64);  context_layer_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_87 = view_218.permute(0, 2, 1, 3);  view_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    context_layer_64 = permute_87.contiguous();  permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    self_output_21 = context_layer_64.view((1, 512, -1));  context_layer_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    hidden_states_168 = self.L__mod___deberta_encoder_layer_21_attention_output_dense(self_output_21);  self_output_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_87 = torch.autograd.function.FunctionCtx()
    hidden_states_169 = torch__dynamo_variables_misc_trampoline_autograd_apply_87(hidden_states_168, 0.1);  hidden_states_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_42 = hidden_states_169 + query_states_42;  hidden_states_169 = query_states_42 = None
    attention_output_42 = self.L__mod___deberta_encoder_layer_21_attention_output_LayerNorm(add_42);  add_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    hidden_states_171 = self.L__mod___deberta_encoder_layer_21_intermediate_dense(attention_output_42)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_21 = torch._C._nn.gelu(hidden_states_171);  hidden_states_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    hidden_states_173 = self.L__mod___deberta_encoder_layer_21_output_dense(intermediate_output_21);  intermediate_output_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_88 = torch.autograd.function.FunctionCtx()
    hidden_states_174 = torch__dynamo_variables_misc_trampoline_autograd_apply_88(hidden_states_173, 0.1);  hidden_states_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_43 = hidden_states_174 + attention_output_42;  hidden_states_174 = attention_output_42 = None
    query_states_44 = self.L__mod___deberta_encoder_layer_21_output_LayerNorm(add_43);  add_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_22_attention_self_query_proj = self.L__mod___deberta_encoder_layer_22_attention_self_query_proj(query_states_44)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_66 = l__mod___deberta_encoder_layer_22_attention_self_query_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_22_attention_self_query_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_88 = x_66.permute(0, 2, 1, 3);  x_66 = None
    contiguous_88 = permute_88.contiguous();  permute_88 = None
    query_layer_22 = contiguous_88.view(-1, 512, 64);  contiguous_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_22_attention_self_key_proj = self.L__mod___deberta_encoder_layer_22_attention_self_key_proj(query_states_44)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_67 = l__mod___deberta_encoder_layer_22_attention_self_key_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_22_attention_self_key_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_89 = x_67.permute(0, 2, 1, 3);  x_67 = None
    contiguous_89 = permute_89.contiguous();  permute_89 = None
    key_layer_22 = contiguous_89.view(-1, 512, 64);  contiguous_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_22_attention_self_value_proj = self.L__mod___deberta_encoder_layer_22_attention_self_value_proj(query_states_44)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_68 = l__mod___deberta_encoder_layer_22_attention_self_value_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_22_attention_self_value_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_90 = x_68.permute(0, 2, 1, 3);  x_68 = None
    contiguous_90 = permute_90.contiguous();  permute_90 = None
    value_layer_22 = contiguous_90.view(-1, 512, 64);  contiguous_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    tensor_22 = torch.tensor(64, dtype = torch.float32)
    mul_24 = tensor_22 * 1;  tensor_22 = None
    scale_22 = torch.sqrt(mul_24);  mul_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    transpose_22 = key_layer_22.transpose(-1, -2);  key_layer_22 = None
    to_23 = scale_22.to(dtype = torch.float32);  scale_22 = None
    truediv_22 = transpose_22 / to_23;  transpose_22 = to_23 = None
    attention_scores_66 = torch.bmm(query_layer_22, truediv_22);  query_layer_22 = truediv_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    attention_scores_67 = attention_scores_66.view(-1, 24, 512, 512);  attention_scores_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    function_ctx_89 = torch.autograd.function.FunctionCtx()
    attention_probs_44 = torch__dynamo_variables_misc_trampoline_autograd_apply_89(attention_scores_67, attention_mask_2, -1);  attention_scores_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_90 = torch.autograd.function.FunctionCtx()
    attention_probs_45 = torch__dynamo_variables_misc_trampoline_autograd_apply_90(attention_probs_44, 0.1);  attention_probs_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_227 = attention_probs_45.view(-1, 512, 512);  attention_probs_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    context_layer_66 = torch.bmm(view_227, value_layer_22);  view_227 = value_layer_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_228 = context_layer_66.view(-1, 24, 512, 64);  context_layer_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_91 = view_228.permute(0, 2, 1, 3);  view_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    context_layer_67 = permute_91.contiguous();  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    self_output_22 = context_layer_67.view((1, 512, -1));  context_layer_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    hidden_states_176 = self.L__mod___deberta_encoder_layer_22_attention_output_dense(self_output_22);  self_output_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_91 = torch.autograd.function.FunctionCtx()
    hidden_states_177 = torch__dynamo_variables_misc_trampoline_autograd_apply_91(hidden_states_176, 0.1);  hidden_states_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_44 = hidden_states_177 + query_states_44;  hidden_states_177 = query_states_44 = None
    attention_output_44 = self.L__mod___deberta_encoder_layer_22_attention_output_LayerNorm(add_44);  add_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    hidden_states_179 = self.L__mod___deberta_encoder_layer_22_intermediate_dense(attention_output_44)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_22 = torch._C._nn.gelu(hidden_states_179);  hidden_states_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    hidden_states_181 = self.L__mod___deberta_encoder_layer_22_output_dense(intermediate_output_22);  intermediate_output_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_92 = torch.autograd.function.FunctionCtx()
    hidden_states_182 = torch__dynamo_variables_misc_trampoline_autograd_apply_92(hidden_states_181, 0.1);  hidden_states_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_45 = hidden_states_182 + attention_output_44;  hidden_states_182 = attention_output_44 = None
    query_states_46 = self.L__mod___deberta_encoder_layer_22_output_LayerNorm(add_45);  add_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_23_attention_self_query_proj = self.L__mod___deberta_encoder_layer_23_attention_self_query_proj(query_states_46)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_69 = l__mod___deberta_encoder_layer_23_attention_self_query_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_23_attention_self_query_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_92 = x_69.permute(0, 2, 1, 3);  x_69 = None
    contiguous_92 = permute_92.contiguous();  permute_92 = None
    query_layer_23 = contiguous_92.view(-1, 512, 64);  contiguous_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_23_attention_self_key_proj = self.L__mod___deberta_encoder_layer_23_attention_self_key_proj(query_states_46)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_70 = l__mod___deberta_encoder_layer_23_attention_self_key_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_23_attention_self_key_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_93 = x_70.permute(0, 2, 1, 3);  x_70 = None
    contiguous_93 = permute_93.contiguous();  permute_93 = None
    key_layer_23 = contiguous_93.view(-1, 512, 64);  contiguous_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    l__mod___deberta_encoder_layer_23_attention_self_value_proj = self.L__mod___deberta_encoder_layer_23_attention_self_value_proj(query_states_46)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    x_71 = l__mod___deberta_encoder_layer_23_attention_self_value_proj.view((1, 512, 24, -1));  l__mod___deberta_encoder_layer_23_attention_self_value_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_94 = x_71.permute(0, 2, 1, 3);  x_71 = None
    contiguous_94 = permute_94.contiguous();  permute_94 = None
    value_layer_23 = contiguous_94.view(-1, 512, 64);  contiguous_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    tensor_23 = torch.tensor(64, dtype = torch.float32)
    mul_25 = tensor_23 * 1;  tensor_23 = None
    scale_23 = torch.sqrt(mul_25);  mul_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    transpose_23 = key_layer_23.transpose(-1, -2);  key_layer_23 = None
    to_24 = scale_23.to(dtype = torch.float32);  scale_23 = None
    truediv_23 = transpose_23 / to_24;  transpose_23 = to_24 = None
    attention_scores_69 = torch.bmm(query_layer_23, truediv_23);  query_layer_23 = truediv_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    attention_scores_70 = attention_scores_69.view(-1, 24, 512, 512);  attention_scores_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    function_ctx_93 = torch.autograd.function.FunctionCtx()
    attention_probs_46 = torch__dynamo_variables_misc_trampoline_autograd_apply_93(attention_scores_70, attention_mask_2, -1);  attention_scores_70 = attention_mask_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_94 = torch.autograd.function.FunctionCtx()
    attention_probs_47 = torch__dynamo_variables_misc_trampoline_autograd_apply_94(attention_probs_46, 0.1);  attention_probs_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_237 = attention_probs_47.view(-1, 512, 512);  attention_probs_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    context_layer_69 = torch.bmm(view_237, value_layer_23);  view_237 = value_layer_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_238 = context_layer_69.view(-1, 24, 512, 64);  context_layer_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_95 = view_238.permute(0, 2, 1, 3);  view_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    context_layer_70 = permute_95.contiguous();  permute_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    self_output_23 = context_layer_70.view((1, 512, -1));  context_layer_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    hidden_states_184 = self.L__mod___deberta_encoder_layer_23_attention_output_dense(self_output_23);  self_output_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_95 = torch.autograd.function.FunctionCtx()
    hidden_states_185 = torch__dynamo_variables_misc_trampoline_autograd_apply_95(hidden_states_184, 0.1);  hidden_states_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_46 = hidden_states_185 + query_states_46;  hidden_states_185 = query_states_46 = None
    attention_output_46 = self.L__mod___deberta_encoder_layer_23_attention_output_LayerNorm(add_46);  add_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    hidden_states_187 = self.L__mod___deberta_encoder_layer_23_intermediate_dense(attention_output_46)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_23 = torch._C._nn.gelu(hidden_states_187);  hidden_states_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    hidden_states_189 = self.L__mod___deberta_encoder_layer_23_output_dense(intermediate_output_23);  intermediate_output_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    function_ctx_96 = torch.autograd.function.FunctionCtx()
    hidden_states_190 = torch__dynamo_variables_misc_trampoline_autograd_apply_96(hidden_states_189, 0.1);  hidden_states_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_47 = hidden_states_190 + attention_output_46;  hidden_states_190 = attention_output_46 = None
    sequence_output = self.L__mod___deberta_encoder_layer_23_output_LayerNorm(add_47);  add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1513, code: logits = self.qa_outputs(sequence_output)
    logits = self.L__mod___qa_outputs(sequence_output);  sequence_output = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1514, code: start_logits, end_logits = logits.split(1, dim=-1)
    split = logits.split(1, dim = -1);  logits = None
    start_logits = split[0]
    end_logits = split[1];  split = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1515, code: start_logits = start_logits.squeeze(-1).contiguous()
    squeeze_1 = start_logits.squeeze(-1);  start_logits = None
    start_logits_1 = squeeze_1.contiguous();  squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1516, code: end_logits = end_logits.squeeze(-1).contiguous()
    squeeze_2 = end_logits.squeeze(-1);  end_logits = None
    end_logits_1 = squeeze_2.contiguous();  squeeze_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1527, code: start_positions = start_positions.clamp(0, ignored_index)
    start_positions = l_cloned_inputs_start_positions_.clamp(0, 512);  l_cloned_inputs_start_positions_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1528, code: end_positions = end_positions.clamp(0, ignored_index)
    end_positions = l_cloned_inputs_end_positions_.clamp(0, 512);  l_cloned_inputs_end_positions_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1531, code: start_loss = loss_fct(start_logits, start_positions)
    start_loss = torch.nn.functional.cross_entropy(start_logits_1, start_positions, None, None, 512, None, 'mean', 0.0);  start_positions = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1532, code: end_loss = loss_fct(end_logits, end_positions)
    end_loss = torch.nn.functional.cross_entropy(end_logits_1, end_positions, None, None, 512, None, 'mean', 0.0);  end_positions = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1533, code: total_loss = (start_loss + end_loss) / 2
    add_48 = start_loss + end_loss;  start_loss = end_loss = None
    loss = add_48 / 2;  add_48 = None
    return (loss, start_logits_1, end_logits_1)
    