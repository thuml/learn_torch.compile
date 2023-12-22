from __future__ import annotations



def forward(self, L_cloned_inputs_input_ids_ : torch.Tensor, L_cloned_inputs_start_positions_ : torch.Tensor, L_cloned_inputs_end_positions_ : torch.Tensor):
    l_cloned_inputs_input_ids_ = L_cloned_inputs_input_ids_
    l_cloned_inputs_start_positions_ = L_cloned_inputs_start_positions_
    l_cloned_inputs_end_positions_ = L_cloned_inputs_end_positions_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:880, code: attention_mask = torch.ones(input_shape, device=device)
    attention_mask = torch.ones((1, 128), device = device(type='cpu'))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:882, code: token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
    token_type_ids = torch.zeros((1, 128), dtype = torch.int64, device = device(type='cpu'))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:916, code: extended_attention_mask = attention_mask[:, None, None, :]
    extended_attention_mask = attention_mask[(slice(None, None, None), None, None, slice(None, None, None))];  attention_mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:927, code: extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask_1 = extended_attention_mask.to(dtype = torch.float32);  extended_attention_mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:928, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    sub = 1.0 - extended_attention_mask_1;  extended_attention_mask_1 = None
    extended_attention_mask_3 = sub * -3.4028234663852886e+38;  sub = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:213, code: position_ids = self.position_ids[:, :seq_length]
    l__mod___mobilebert_embeddings_position_ids = self.L__mod___mobilebert_embeddings_position_ids
    position_ids = l__mod___mobilebert_embeddings_position_ids[(slice(None, None, None), slice(None, 128, None))];  l__mod___mobilebert_embeddings_position_ids = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:218, code: inputs_embeds = self.word_embeddings(input_ids)
    inputs_embeds = self.L__mod___mobilebert_embeddings_word_embeddings(l_cloned_inputs_input_ids_);  l_cloned_inputs_input_ids_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:230, code: nn.functional.pad(inputs_embeds[:, 1:], [0, 0, 0, 1, 0, 0], value=0.0),
    getitem_2 = inputs_embeds[(slice(None, None, None), slice(1, None, None))]
    pad = torch.nn.functional.pad(getitem_2, [0, 0, 0, 1, 0, 0], value = 0.0);  getitem_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:232, code: nn.functional.pad(inputs_embeds[:, :-1], [0, 0, 1, 0, 0, 0], value=0.0),
    getitem_3 = inputs_embeds[(slice(None, None, None), slice(None, -1, None))]
    pad_1 = torch.nn.functional.pad(getitem_3, [0, 0, 1, 0, 0, 0], value = 0.0);  getitem_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:228, code: inputs_embeds = torch.cat(
    inputs_embeds_1 = torch.cat([pad, inputs_embeds, pad_1], dim = 2);  pad = inputs_embeds = pad_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:237, code: inputs_embeds = self.embedding_transformation(inputs_embeds)
    inputs_embeds_2 = self.L__mod___mobilebert_embeddings_embedding_transformation(inputs_embeds_1);  inputs_embeds_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:241, code: position_embeddings = self.position_embeddings(position_ids)
    position_embeddings = self.L__mod___mobilebert_embeddings_position_embeddings(position_ids);  position_ids = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:242, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    token_type_embeddings = self.L__mod___mobilebert_embeddings_token_type_embeddings(token_type_ids);  token_type_ids = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:243, code: embeddings = inputs_embeds + position_embeddings + token_type_embeddings
    add = inputs_embeds_2 + position_embeddings;  inputs_embeds_2 = position_embeddings = None
    embeddings = add + token_type_embeddings;  add = token_type_embeddings = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_embeddings_layer_norm_weight_1 = self.L__mod___mobilebert_embeddings_LayerNorm_weight
    mul_1 = embeddings * l__mod___mobilebert_embeddings_layer_norm_weight_1;  embeddings = l__mod___mobilebert_embeddings_layer_norm_weight_1 = None
    l__mod___mobilebert_embeddings_layer_norm_bias_1 = self.L__mod___mobilebert_embeddings_LayerNorm_bias
    embeddings_1 = mul_1 + l__mod___mobilebert_embeddings_layer_norm_bias_1;  mul_1 = l__mod___mobilebert_embeddings_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:245, code: embeddings = self.dropout(embeddings)
    value_tensor = self.L__mod___mobilebert_embeddings_dropout(embeddings_1);  embeddings_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input = self.L__mod___mobilebert_encoder_layer_0_bottleneck_input_dense(value_tensor)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_0_bottleneck_input_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_0_bottleneck_input_LayerNorm_weight
    mul_2 = layer_input * l__mod___mobilebert_encoder_layer_0_bottleneck_input_layer_norm_weight_1;  layer_input = l__mod___mobilebert_encoder_layer_0_bottleneck_input_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_0_bottleneck_input_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_0_bottleneck_input_LayerNorm_bias
    layer_input_4 = mul_2 + l__mod___mobilebert_encoder_layer_0_bottleneck_input_layer_norm_bias_1;  mul_2 = l__mod___mobilebert_encoder_layer_0_bottleneck_input_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_2 = self.L__mod___mobilebert_encoder_layer_0_bottleneck_attention_dense(value_tensor)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_0_bottleneck_attention_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_0_bottleneck_attention_LayerNorm_weight
    mul_3 = layer_input_2 * l__mod___mobilebert_encoder_layer_0_bottleneck_attention_layer_norm_weight_1;  layer_input_2 = l__mod___mobilebert_encoder_layer_0_bottleneck_attention_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_0_bottleneck_attention_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_0_bottleneck_attention_LayerNorm_bias
    key_tensor = mul_3 + l__mod___mobilebert_encoder_layer_0_bottleneck_attention_layer_norm_bias_1;  mul_3 = l__mod___mobilebert_encoder_layer_0_bottleneck_attention_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    mixed_query_layer = self.L__mod___mobilebert_encoder_layer_0_attention_self_query(key_tensor)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    mixed_key_layer = self.L__mod___mobilebert_encoder_layer_0_attention_self_key(key_tensor);  key_tensor = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    mixed_value_layer = self.L__mod___mobilebert_encoder_layer_0_attention_self_value(value_tensor)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x = mixed_query_layer.view((1, 128, 4, 32));  mixed_query_layer = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    query_layer = x.permute(0, 2, 1, 3);  x = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_1 = mixed_key_layer.view((1, 128, 4, 32));  mixed_key_layer = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    key_layer = x_1.permute(0, 2, 1, 3);  x_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_2 = mixed_value_layer.view((1, 128, 4, 32));  mixed_value_layer = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    value_layer = x_2.permute(0, 2, 1, 3);  x_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:286, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose = key_layer.transpose(-1, -2);  key_layer = None
    attention_scores = torch.matmul(query_layer, transpose);  query_layer = transpose = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:287, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_1 = attention_scores / 5.656854249492381;  attention_scores = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:290, code: attention_scores = attention_scores + attention_mask
    attention_scores_2 = attention_scores_1 + extended_attention_mask_3;  attention_scores_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:292, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs = torch.nn.functional.softmax(attention_scores_2, dim = -1);  attention_scores_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295, code: attention_probs = self.dropout(attention_probs)
    attention_probs_1 = self.L__mod___mobilebert_encoder_layer_0_attention_self_dropout(attention_probs);  attention_probs = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:299, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer = torch.matmul(attention_probs_1, value_layer);  attention_probs_1 = value_layer = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_3 = context_layer.permute(0, 2, 1, 3);  context_layer = None
    context_layer_1 = permute_3.contiguous();  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    context_layer_2 = context_layer_1.view((1, 128, 128));  context_layer_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    layer_outputs = self.L__mod___mobilebert_encoder_layer_0_attention_output_dense(context_layer_2);  context_layer_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_6 = layer_outputs + layer_input_4;  layer_outputs = layer_input_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_0_attention_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_0_attention_output_LayerNorm_weight
    mul_4 = add_6 * l__mod___mobilebert_encoder_layer_0_attention_output_layer_norm_weight_1;  add_6 = l__mod___mobilebert_encoder_layer_0_attention_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_0_attention_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_0_attention_output_LayerNorm_bias
    attention_output = mul_4 + l__mod___mobilebert_encoder_layer_0_attention_output_layer_norm_bias_1;  mul_4 = l__mod___mobilebert_encoder_layer_0_attention_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states = self.L__mod___mobilebert_encoder_layer_0_ffn_0_intermediate_dense(attention_output)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output = self.L__mod___mobilebert_encoder_layer_0_ffn_0_intermediate_intermediate_act_fn(hidden_states);  hidden_states = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_2 = self.L__mod___mobilebert_encoder_layer_0_ffn_0_output_dense(intermediate_output);  intermediate_output = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_8 = layer_outputs_2 + attention_output;  layer_outputs_2 = attention_output = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_0_ffn_0_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_0_ffn_0_output_LayerNorm_weight
    mul_5 = add_8 * l__mod___mobilebert_encoder_layer_0_ffn_0_output_layer_norm_weight_1;  add_8 = l__mod___mobilebert_encoder_layer_0_ffn_0_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_0_ffn_0_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_0_ffn_0_output_LayerNorm_bias
    attention_output_1 = mul_5 + l__mod___mobilebert_encoder_layer_0_ffn_0_output_layer_norm_bias_1;  mul_5 = l__mod___mobilebert_encoder_layer_0_ffn_0_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_2 = self.L__mod___mobilebert_encoder_layer_0_ffn_1_intermediate_dense(attention_output_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_1 = self.L__mod___mobilebert_encoder_layer_0_ffn_1_intermediate_intermediate_act_fn(hidden_states_2);  hidden_states_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_5 = self.L__mod___mobilebert_encoder_layer_0_ffn_1_output_dense(intermediate_output_1);  intermediate_output_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_10 = layer_outputs_5 + attention_output_1;  layer_outputs_5 = attention_output_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_0_ffn_1_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_0_ffn_1_output_LayerNorm_weight
    mul_6 = add_10 * l__mod___mobilebert_encoder_layer_0_ffn_1_output_layer_norm_weight_1;  add_10 = l__mod___mobilebert_encoder_layer_0_ffn_1_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_0_ffn_1_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_0_ffn_1_output_LayerNorm_bias
    attention_output_2 = mul_6 + l__mod___mobilebert_encoder_layer_0_ffn_1_output_layer_norm_bias_1;  mul_6 = l__mod___mobilebert_encoder_layer_0_ffn_1_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_4 = self.L__mod___mobilebert_encoder_layer_0_ffn_2_intermediate_dense(attention_output_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_2 = self.L__mod___mobilebert_encoder_layer_0_ffn_2_intermediate_intermediate_act_fn(hidden_states_4);  hidden_states_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_8 = self.L__mod___mobilebert_encoder_layer_0_ffn_2_output_dense(intermediate_output_2);  intermediate_output_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_12 = layer_outputs_8 + attention_output_2;  layer_outputs_8 = attention_output_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_0_ffn_2_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_0_ffn_2_output_LayerNorm_weight
    mul_7 = add_12 * l__mod___mobilebert_encoder_layer_0_ffn_2_output_layer_norm_weight_1;  add_12 = l__mod___mobilebert_encoder_layer_0_ffn_2_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_0_ffn_2_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_0_ffn_2_output_LayerNorm_bias
    attention_output_3 = mul_7 + l__mod___mobilebert_encoder_layer_0_ffn_2_output_layer_norm_bias_1;  mul_7 = l__mod___mobilebert_encoder_layer_0_ffn_2_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_6 = self.L__mod___mobilebert_encoder_layer_0_intermediate_dense(attention_output_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_3 = self.L__mod___mobilebert_encoder_layer_0_intermediate_intermediate_act_fn(hidden_states_6);  hidden_states_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    layer_output = self.L__mod___mobilebert_encoder_layer_0_output_dense(intermediate_output_3);  intermediate_output_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_14 = layer_output + attention_output_3;  layer_output = attention_output_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_0_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_0_output_LayerNorm_weight
    mul_8 = add_14 * l__mod___mobilebert_encoder_layer_0_output_layer_norm_weight_1;  add_14 = l__mod___mobilebert_encoder_layer_0_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_0_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_0_output_LayerNorm_bias
    layer_output_1 = mul_8 + l__mod___mobilebert_encoder_layer_0_output_layer_norm_bias_1;  mul_8 = l__mod___mobilebert_encoder_layer_0_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_11 = self.L__mod___mobilebert_encoder_layer_0_output_bottleneck_dense(layer_output_1);  layer_output_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    layer_outputs_12 = self.L__mod___mobilebert_encoder_layer_0_output_bottleneck_dropout(layer_outputs_11);  layer_outputs_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_16 = layer_outputs_12 + value_tensor;  layer_outputs_12 = value_tensor = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_0_output_bottleneck_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_0_output_bottleneck_LayerNorm_weight
    mul_9 = add_16 * l__mod___mobilebert_encoder_layer_0_output_bottleneck_layer_norm_weight_1;  add_16 = l__mod___mobilebert_encoder_layer_0_output_bottleneck_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_0_output_bottleneck_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_0_output_bottleneck_LayerNorm_bias
    value_tensor_1 = mul_9 + l__mod___mobilebert_encoder_layer_0_output_bottleneck_layer_norm_bias_1;  mul_9 = l__mod___mobilebert_encoder_layer_0_output_bottleneck_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:549, code: torch.tensor(1000),
    tensor = torch.tensor(1000)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_5 = self.L__mod___mobilebert_encoder_layer_1_bottleneck_input_dense(value_tensor_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_1_bottleneck_input_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_1_bottleneck_input_LayerNorm_weight
    mul_10 = layer_input_5 * l__mod___mobilebert_encoder_layer_1_bottleneck_input_layer_norm_weight_1;  layer_input_5 = l__mod___mobilebert_encoder_layer_1_bottleneck_input_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_1_bottleneck_input_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_1_bottleneck_input_LayerNorm_bias
    layer_input_9 = mul_10 + l__mod___mobilebert_encoder_layer_1_bottleneck_input_layer_norm_bias_1;  mul_10 = l__mod___mobilebert_encoder_layer_1_bottleneck_input_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_7 = self.L__mod___mobilebert_encoder_layer_1_bottleneck_attention_dense(value_tensor_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_1_bottleneck_attention_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_1_bottleneck_attention_LayerNorm_weight
    mul_11 = layer_input_7 * l__mod___mobilebert_encoder_layer_1_bottleneck_attention_layer_norm_weight_1;  layer_input_7 = l__mod___mobilebert_encoder_layer_1_bottleneck_attention_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_1_bottleneck_attention_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_1_bottleneck_attention_LayerNorm_bias
    key_tensor_1 = mul_11 + l__mod___mobilebert_encoder_layer_1_bottleneck_attention_layer_norm_bias_1;  mul_11 = l__mod___mobilebert_encoder_layer_1_bottleneck_attention_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    mixed_query_layer_1 = self.L__mod___mobilebert_encoder_layer_1_attention_self_query(key_tensor_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    mixed_key_layer_1 = self.L__mod___mobilebert_encoder_layer_1_attention_self_key(key_tensor_1);  key_tensor_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    mixed_value_layer_1 = self.L__mod___mobilebert_encoder_layer_1_attention_self_value(value_tensor_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_3 = mixed_query_layer_1.view((1, 128, 4, 32));  mixed_query_layer_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    query_layer_1 = x_3.permute(0, 2, 1, 3);  x_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_4 = mixed_key_layer_1.view((1, 128, 4, 32));  mixed_key_layer_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    key_layer_1 = x_4.permute(0, 2, 1, 3);  x_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_5 = mixed_value_layer_1.view((1, 128, 4, 32));  mixed_value_layer_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    value_layer_1 = x_5.permute(0, 2, 1, 3);  x_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:286, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_1 = key_layer_1.transpose(-1, -2);  key_layer_1 = None
    attention_scores_3 = torch.matmul(query_layer_1, transpose_1);  query_layer_1 = transpose_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:287, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_4 = attention_scores_3 / 5.656854249492381;  attention_scores_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:290, code: attention_scores = attention_scores + attention_mask
    attention_scores_5 = attention_scores_4 + extended_attention_mask_3;  attention_scores_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:292, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_2 = torch.nn.functional.softmax(attention_scores_5, dim = -1);  attention_scores_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295, code: attention_probs = self.dropout(attention_probs)
    attention_probs_3 = self.L__mod___mobilebert_encoder_layer_1_attention_self_dropout(attention_probs_2);  attention_probs_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:299, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_3 = torch.matmul(attention_probs_3, value_layer_1);  attention_probs_3 = value_layer_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_7 = context_layer_3.permute(0, 2, 1, 3);  context_layer_3 = None
    context_layer_4 = permute_7.contiguous();  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    context_layer_5 = context_layer_4.view((1, 128, 128));  context_layer_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_14 = self.L__mod___mobilebert_encoder_layer_1_attention_output_dense(context_layer_5);  context_layer_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_21 = layer_outputs_14 + layer_input_9;  layer_outputs_14 = layer_input_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_1_attention_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_1_attention_output_LayerNorm_weight
    mul_12 = add_21 * l__mod___mobilebert_encoder_layer_1_attention_output_layer_norm_weight_1;  add_21 = l__mod___mobilebert_encoder_layer_1_attention_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_1_attention_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_1_attention_output_LayerNorm_bias
    attention_output_5 = mul_12 + l__mod___mobilebert_encoder_layer_1_attention_output_layer_norm_bias_1;  mul_12 = l__mod___mobilebert_encoder_layer_1_attention_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_9 = self.L__mod___mobilebert_encoder_layer_1_ffn_0_intermediate_dense(attention_output_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_4 = self.L__mod___mobilebert_encoder_layer_1_ffn_0_intermediate_intermediate_act_fn(hidden_states_9);  hidden_states_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_16 = self.L__mod___mobilebert_encoder_layer_1_ffn_0_output_dense(intermediate_output_4);  intermediate_output_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_23 = layer_outputs_16 + attention_output_5;  layer_outputs_16 = attention_output_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_1_ffn_0_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_1_ffn_0_output_LayerNorm_weight
    mul_13 = add_23 * l__mod___mobilebert_encoder_layer_1_ffn_0_output_layer_norm_weight_1;  add_23 = l__mod___mobilebert_encoder_layer_1_ffn_0_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_1_ffn_0_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_1_ffn_0_output_LayerNorm_bias
    attention_output_6 = mul_13 + l__mod___mobilebert_encoder_layer_1_ffn_0_output_layer_norm_bias_1;  mul_13 = l__mod___mobilebert_encoder_layer_1_ffn_0_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_11 = self.L__mod___mobilebert_encoder_layer_1_ffn_1_intermediate_dense(attention_output_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_5 = self.L__mod___mobilebert_encoder_layer_1_ffn_1_intermediate_intermediate_act_fn(hidden_states_11);  hidden_states_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_19 = self.L__mod___mobilebert_encoder_layer_1_ffn_1_output_dense(intermediate_output_5);  intermediate_output_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_25 = layer_outputs_19 + attention_output_6;  layer_outputs_19 = attention_output_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_1_ffn_1_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_1_ffn_1_output_LayerNorm_weight
    mul_14 = add_25 * l__mod___mobilebert_encoder_layer_1_ffn_1_output_layer_norm_weight_1;  add_25 = l__mod___mobilebert_encoder_layer_1_ffn_1_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_1_ffn_1_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_1_ffn_1_output_LayerNorm_bias
    attention_output_7 = mul_14 + l__mod___mobilebert_encoder_layer_1_ffn_1_output_layer_norm_bias_1;  mul_14 = l__mod___mobilebert_encoder_layer_1_ffn_1_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_13 = self.L__mod___mobilebert_encoder_layer_1_ffn_2_intermediate_dense(attention_output_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_6 = self.L__mod___mobilebert_encoder_layer_1_ffn_2_intermediate_intermediate_act_fn(hidden_states_13);  hidden_states_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_22 = self.L__mod___mobilebert_encoder_layer_1_ffn_2_output_dense(intermediate_output_6);  intermediate_output_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_27 = layer_outputs_22 + attention_output_7;  layer_outputs_22 = attention_output_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_1_ffn_2_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_1_ffn_2_output_LayerNorm_weight
    mul_15 = add_27 * l__mod___mobilebert_encoder_layer_1_ffn_2_output_layer_norm_weight_1;  add_27 = l__mod___mobilebert_encoder_layer_1_ffn_2_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_1_ffn_2_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_1_ffn_2_output_LayerNorm_bias
    attention_output_8 = mul_15 + l__mod___mobilebert_encoder_layer_1_ffn_2_output_layer_norm_bias_1;  mul_15 = l__mod___mobilebert_encoder_layer_1_ffn_2_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_15 = self.L__mod___mobilebert_encoder_layer_1_intermediate_dense(attention_output_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_7 = self.L__mod___mobilebert_encoder_layer_1_intermediate_intermediate_act_fn(hidden_states_15);  hidden_states_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    layer_output_4 = self.L__mod___mobilebert_encoder_layer_1_output_dense(intermediate_output_7);  intermediate_output_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_29 = layer_output_4 + attention_output_8;  layer_output_4 = attention_output_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_1_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_1_output_LayerNorm_weight
    mul_16 = add_29 * l__mod___mobilebert_encoder_layer_1_output_layer_norm_weight_1;  add_29 = l__mod___mobilebert_encoder_layer_1_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_1_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_1_output_LayerNorm_bias
    layer_output_5 = mul_16 + l__mod___mobilebert_encoder_layer_1_output_layer_norm_bias_1;  mul_16 = l__mod___mobilebert_encoder_layer_1_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_25 = self.L__mod___mobilebert_encoder_layer_1_output_bottleneck_dense(layer_output_5);  layer_output_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    layer_outputs_26 = self.L__mod___mobilebert_encoder_layer_1_output_bottleneck_dropout(layer_outputs_25);  layer_outputs_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_31 = layer_outputs_26 + value_tensor_1;  layer_outputs_26 = value_tensor_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_1_output_bottleneck_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_1_output_bottleneck_LayerNorm_weight
    mul_17 = add_31 * l__mod___mobilebert_encoder_layer_1_output_bottleneck_layer_norm_weight_1;  add_31 = l__mod___mobilebert_encoder_layer_1_output_bottleneck_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_1_output_bottleneck_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_1_output_bottleneck_LayerNorm_bias
    value_tensor_2 = mul_17 + l__mod___mobilebert_encoder_layer_1_output_bottleneck_layer_norm_bias_1;  mul_17 = l__mod___mobilebert_encoder_layer_1_output_bottleneck_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:549, code: torch.tensor(1000),
    tensor_1 = torch.tensor(1000)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_10 = self.L__mod___mobilebert_encoder_layer_2_bottleneck_input_dense(value_tensor_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_2_bottleneck_input_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_2_bottleneck_input_LayerNorm_weight
    mul_18 = layer_input_10 * l__mod___mobilebert_encoder_layer_2_bottleneck_input_layer_norm_weight_1;  layer_input_10 = l__mod___mobilebert_encoder_layer_2_bottleneck_input_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_2_bottleneck_input_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_2_bottleneck_input_LayerNorm_bias
    layer_input_14 = mul_18 + l__mod___mobilebert_encoder_layer_2_bottleneck_input_layer_norm_bias_1;  mul_18 = l__mod___mobilebert_encoder_layer_2_bottleneck_input_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_12 = self.L__mod___mobilebert_encoder_layer_2_bottleneck_attention_dense(value_tensor_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_2_bottleneck_attention_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_2_bottleneck_attention_LayerNorm_weight
    mul_19 = layer_input_12 * l__mod___mobilebert_encoder_layer_2_bottleneck_attention_layer_norm_weight_1;  layer_input_12 = l__mod___mobilebert_encoder_layer_2_bottleneck_attention_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_2_bottleneck_attention_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_2_bottleneck_attention_LayerNorm_bias
    key_tensor_2 = mul_19 + l__mod___mobilebert_encoder_layer_2_bottleneck_attention_layer_norm_bias_1;  mul_19 = l__mod___mobilebert_encoder_layer_2_bottleneck_attention_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    mixed_query_layer_2 = self.L__mod___mobilebert_encoder_layer_2_attention_self_query(key_tensor_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    mixed_key_layer_2 = self.L__mod___mobilebert_encoder_layer_2_attention_self_key(key_tensor_2);  key_tensor_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    mixed_value_layer_2 = self.L__mod___mobilebert_encoder_layer_2_attention_self_value(value_tensor_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_6 = mixed_query_layer_2.view((1, 128, 4, 32));  mixed_query_layer_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    query_layer_2 = x_6.permute(0, 2, 1, 3);  x_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_7 = mixed_key_layer_2.view((1, 128, 4, 32));  mixed_key_layer_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    key_layer_2 = x_7.permute(0, 2, 1, 3);  x_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_8 = mixed_value_layer_2.view((1, 128, 4, 32));  mixed_value_layer_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    value_layer_2 = x_8.permute(0, 2, 1, 3);  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:286, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_2 = key_layer_2.transpose(-1, -2);  key_layer_2 = None
    attention_scores_6 = torch.matmul(query_layer_2, transpose_2);  query_layer_2 = transpose_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:287, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_7 = attention_scores_6 / 5.656854249492381;  attention_scores_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:290, code: attention_scores = attention_scores + attention_mask
    attention_scores_8 = attention_scores_7 + extended_attention_mask_3;  attention_scores_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:292, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_4 = torch.nn.functional.softmax(attention_scores_8, dim = -1);  attention_scores_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295, code: attention_probs = self.dropout(attention_probs)
    attention_probs_5 = self.L__mod___mobilebert_encoder_layer_2_attention_self_dropout(attention_probs_4);  attention_probs_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:299, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_6 = torch.matmul(attention_probs_5, value_layer_2);  attention_probs_5 = value_layer_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_11 = context_layer_6.permute(0, 2, 1, 3);  context_layer_6 = None
    context_layer_7 = permute_11.contiguous();  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    context_layer_8 = context_layer_7.view((1, 128, 128));  context_layer_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_28 = self.L__mod___mobilebert_encoder_layer_2_attention_output_dense(context_layer_8);  context_layer_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_36 = layer_outputs_28 + layer_input_14;  layer_outputs_28 = layer_input_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_2_attention_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_2_attention_output_LayerNorm_weight
    mul_20 = add_36 * l__mod___mobilebert_encoder_layer_2_attention_output_layer_norm_weight_1;  add_36 = l__mod___mobilebert_encoder_layer_2_attention_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_2_attention_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_2_attention_output_LayerNorm_bias
    attention_output_10 = mul_20 + l__mod___mobilebert_encoder_layer_2_attention_output_layer_norm_bias_1;  mul_20 = l__mod___mobilebert_encoder_layer_2_attention_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_18 = self.L__mod___mobilebert_encoder_layer_2_ffn_0_intermediate_dense(attention_output_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_8 = self.L__mod___mobilebert_encoder_layer_2_ffn_0_intermediate_intermediate_act_fn(hidden_states_18);  hidden_states_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_30 = self.L__mod___mobilebert_encoder_layer_2_ffn_0_output_dense(intermediate_output_8);  intermediate_output_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_38 = layer_outputs_30 + attention_output_10;  layer_outputs_30 = attention_output_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_2_ffn_0_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_2_ffn_0_output_LayerNorm_weight
    mul_21 = add_38 * l__mod___mobilebert_encoder_layer_2_ffn_0_output_layer_norm_weight_1;  add_38 = l__mod___mobilebert_encoder_layer_2_ffn_0_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_2_ffn_0_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_2_ffn_0_output_LayerNorm_bias
    attention_output_11 = mul_21 + l__mod___mobilebert_encoder_layer_2_ffn_0_output_layer_norm_bias_1;  mul_21 = l__mod___mobilebert_encoder_layer_2_ffn_0_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_20 = self.L__mod___mobilebert_encoder_layer_2_ffn_1_intermediate_dense(attention_output_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_9 = self.L__mod___mobilebert_encoder_layer_2_ffn_1_intermediate_intermediate_act_fn(hidden_states_20);  hidden_states_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_33 = self.L__mod___mobilebert_encoder_layer_2_ffn_1_output_dense(intermediate_output_9);  intermediate_output_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_40 = layer_outputs_33 + attention_output_11;  layer_outputs_33 = attention_output_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_2_ffn_1_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_2_ffn_1_output_LayerNorm_weight
    mul_22 = add_40 * l__mod___mobilebert_encoder_layer_2_ffn_1_output_layer_norm_weight_1;  add_40 = l__mod___mobilebert_encoder_layer_2_ffn_1_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_2_ffn_1_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_2_ffn_1_output_LayerNorm_bias
    attention_output_12 = mul_22 + l__mod___mobilebert_encoder_layer_2_ffn_1_output_layer_norm_bias_1;  mul_22 = l__mod___mobilebert_encoder_layer_2_ffn_1_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_22 = self.L__mod___mobilebert_encoder_layer_2_ffn_2_intermediate_dense(attention_output_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_10 = self.L__mod___mobilebert_encoder_layer_2_ffn_2_intermediate_intermediate_act_fn(hidden_states_22);  hidden_states_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_36 = self.L__mod___mobilebert_encoder_layer_2_ffn_2_output_dense(intermediate_output_10);  intermediate_output_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_42 = layer_outputs_36 + attention_output_12;  layer_outputs_36 = attention_output_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_2_ffn_2_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_2_ffn_2_output_LayerNorm_weight
    mul_23 = add_42 * l__mod___mobilebert_encoder_layer_2_ffn_2_output_layer_norm_weight_1;  add_42 = l__mod___mobilebert_encoder_layer_2_ffn_2_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_2_ffn_2_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_2_ffn_2_output_LayerNorm_bias
    attention_output_13 = mul_23 + l__mod___mobilebert_encoder_layer_2_ffn_2_output_layer_norm_bias_1;  mul_23 = l__mod___mobilebert_encoder_layer_2_ffn_2_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_24 = self.L__mod___mobilebert_encoder_layer_2_intermediate_dense(attention_output_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_11 = self.L__mod___mobilebert_encoder_layer_2_intermediate_intermediate_act_fn(hidden_states_24);  hidden_states_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    layer_output_8 = self.L__mod___mobilebert_encoder_layer_2_output_dense(intermediate_output_11);  intermediate_output_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_44 = layer_output_8 + attention_output_13;  layer_output_8 = attention_output_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_2_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_2_output_LayerNorm_weight
    mul_24 = add_44 * l__mod___mobilebert_encoder_layer_2_output_layer_norm_weight_1;  add_44 = l__mod___mobilebert_encoder_layer_2_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_2_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_2_output_LayerNorm_bias
    layer_output_9 = mul_24 + l__mod___mobilebert_encoder_layer_2_output_layer_norm_bias_1;  mul_24 = l__mod___mobilebert_encoder_layer_2_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_39 = self.L__mod___mobilebert_encoder_layer_2_output_bottleneck_dense(layer_output_9);  layer_output_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    layer_outputs_40 = self.L__mod___mobilebert_encoder_layer_2_output_bottleneck_dropout(layer_outputs_39);  layer_outputs_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_46 = layer_outputs_40 + value_tensor_2;  layer_outputs_40 = value_tensor_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_2_output_bottleneck_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_2_output_bottleneck_LayerNorm_weight
    mul_25 = add_46 * l__mod___mobilebert_encoder_layer_2_output_bottleneck_layer_norm_weight_1;  add_46 = l__mod___mobilebert_encoder_layer_2_output_bottleneck_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_2_output_bottleneck_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_2_output_bottleneck_LayerNorm_bias
    value_tensor_3 = mul_25 + l__mod___mobilebert_encoder_layer_2_output_bottleneck_layer_norm_bias_1;  mul_25 = l__mod___mobilebert_encoder_layer_2_output_bottleneck_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:549, code: torch.tensor(1000),
    tensor_2 = torch.tensor(1000)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_15 = self.L__mod___mobilebert_encoder_layer_3_bottleneck_input_dense(value_tensor_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_3_bottleneck_input_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_3_bottleneck_input_LayerNorm_weight
    mul_26 = layer_input_15 * l__mod___mobilebert_encoder_layer_3_bottleneck_input_layer_norm_weight_1;  layer_input_15 = l__mod___mobilebert_encoder_layer_3_bottleneck_input_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_3_bottleneck_input_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_3_bottleneck_input_LayerNorm_bias
    layer_input_19 = mul_26 + l__mod___mobilebert_encoder_layer_3_bottleneck_input_layer_norm_bias_1;  mul_26 = l__mod___mobilebert_encoder_layer_3_bottleneck_input_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_17 = self.L__mod___mobilebert_encoder_layer_3_bottleneck_attention_dense(value_tensor_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_3_bottleneck_attention_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_3_bottleneck_attention_LayerNorm_weight
    mul_27 = layer_input_17 * l__mod___mobilebert_encoder_layer_3_bottleneck_attention_layer_norm_weight_1;  layer_input_17 = l__mod___mobilebert_encoder_layer_3_bottleneck_attention_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_3_bottleneck_attention_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_3_bottleneck_attention_LayerNorm_bias
    key_tensor_3 = mul_27 + l__mod___mobilebert_encoder_layer_3_bottleneck_attention_layer_norm_bias_1;  mul_27 = l__mod___mobilebert_encoder_layer_3_bottleneck_attention_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    mixed_query_layer_3 = self.L__mod___mobilebert_encoder_layer_3_attention_self_query(key_tensor_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    mixed_key_layer_3 = self.L__mod___mobilebert_encoder_layer_3_attention_self_key(key_tensor_3);  key_tensor_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    mixed_value_layer_3 = self.L__mod___mobilebert_encoder_layer_3_attention_self_value(value_tensor_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_9 = mixed_query_layer_3.view((1, 128, 4, 32));  mixed_query_layer_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    query_layer_3 = x_9.permute(0, 2, 1, 3);  x_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_10 = mixed_key_layer_3.view((1, 128, 4, 32));  mixed_key_layer_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    key_layer_3 = x_10.permute(0, 2, 1, 3);  x_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_11 = mixed_value_layer_3.view((1, 128, 4, 32));  mixed_value_layer_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    value_layer_3 = x_11.permute(0, 2, 1, 3);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:286, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_3 = key_layer_3.transpose(-1, -2);  key_layer_3 = None
    attention_scores_9 = torch.matmul(query_layer_3, transpose_3);  query_layer_3 = transpose_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:287, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_10 = attention_scores_9 / 5.656854249492381;  attention_scores_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:290, code: attention_scores = attention_scores + attention_mask
    attention_scores_11 = attention_scores_10 + extended_attention_mask_3;  attention_scores_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:292, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_6 = torch.nn.functional.softmax(attention_scores_11, dim = -1);  attention_scores_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295, code: attention_probs = self.dropout(attention_probs)
    attention_probs_7 = self.L__mod___mobilebert_encoder_layer_3_attention_self_dropout(attention_probs_6);  attention_probs_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:299, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_9 = torch.matmul(attention_probs_7, value_layer_3);  attention_probs_7 = value_layer_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_15 = context_layer_9.permute(0, 2, 1, 3);  context_layer_9 = None
    context_layer_10 = permute_15.contiguous();  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    context_layer_11 = context_layer_10.view((1, 128, 128));  context_layer_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_42 = self.L__mod___mobilebert_encoder_layer_3_attention_output_dense(context_layer_11);  context_layer_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_51 = layer_outputs_42 + layer_input_19;  layer_outputs_42 = layer_input_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_3_attention_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_3_attention_output_LayerNorm_weight
    mul_28 = add_51 * l__mod___mobilebert_encoder_layer_3_attention_output_layer_norm_weight_1;  add_51 = l__mod___mobilebert_encoder_layer_3_attention_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_3_attention_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_3_attention_output_LayerNorm_bias
    attention_output_15 = mul_28 + l__mod___mobilebert_encoder_layer_3_attention_output_layer_norm_bias_1;  mul_28 = l__mod___mobilebert_encoder_layer_3_attention_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_27 = self.L__mod___mobilebert_encoder_layer_3_ffn_0_intermediate_dense(attention_output_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_12 = self.L__mod___mobilebert_encoder_layer_3_ffn_0_intermediate_intermediate_act_fn(hidden_states_27);  hidden_states_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_44 = self.L__mod___mobilebert_encoder_layer_3_ffn_0_output_dense(intermediate_output_12);  intermediate_output_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_53 = layer_outputs_44 + attention_output_15;  layer_outputs_44 = attention_output_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_3_ffn_0_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_3_ffn_0_output_LayerNorm_weight
    mul_29 = add_53 * l__mod___mobilebert_encoder_layer_3_ffn_0_output_layer_norm_weight_1;  add_53 = l__mod___mobilebert_encoder_layer_3_ffn_0_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_3_ffn_0_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_3_ffn_0_output_LayerNorm_bias
    attention_output_16 = mul_29 + l__mod___mobilebert_encoder_layer_3_ffn_0_output_layer_norm_bias_1;  mul_29 = l__mod___mobilebert_encoder_layer_3_ffn_0_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_29 = self.L__mod___mobilebert_encoder_layer_3_ffn_1_intermediate_dense(attention_output_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_13 = self.L__mod___mobilebert_encoder_layer_3_ffn_1_intermediate_intermediate_act_fn(hidden_states_29);  hidden_states_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_47 = self.L__mod___mobilebert_encoder_layer_3_ffn_1_output_dense(intermediate_output_13);  intermediate_output_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_55 = layer_outputs_47 + attention_output_16;  layer_outputs_47 = attention_output_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_3_ffn_1_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_3_ffn_1_output_LayerNorm_weight
    mul_30 = add_55 * l__mod___mobilebert_encoder_layer_3_ffn_1_output_layer_norm_weight_1;  add_55 = l__mod___mobilebert_encoder_layer_3_ffn_1_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_3_ffn_1_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_3_ffn_1_output_LayerNorm_bias
    attention_output_17 = mul_30 + l__mod___mobilebert_encoder_layer_3_ffn_1_output_layer_norm_bias_1;  mul_30 = l__mod___mobilebert_encoder_layer_3_ffn_1_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_31 = self.L__mod___mobilebert_encoder_layer_3_ffn_2_intermediate_dense(attention_output_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_14 = self.L__mod___mobilebert_encoder_layer_3_ffn_2_intermediate_intermediate_act_fn(hidden_states_31);  hidden_states_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_50 = self.L__mod___mobilebert_encoder_layer_3_ffn_2_output_dense(intermediate_output_14);  intermediate_output_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_57 = layer_outputs_50 + attention_output_17;  layer_outputs_50 = attention_output_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_3_ffn_2_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_3_ffn_2_output_LayerNorm_weight
    mul_31 = add_57 * l__mod___mobilebert_encoder_layer_3_ffn_2_output_layer_norm_weight_1;  add_57 = l__mod___mobilebert_encoder_layer_3_ffn_2_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_3_ffn_2_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_3_ffn_2_output_LayerNorm_bias
    attention_output_18 = mul_31 + l__mod___mobilebert_encoder_layer_3_ffn_2_output_layer_norm_bias_1;  mul_31 = l__mod___mobilebert_encoder_layer_3_ffn_2_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_33 = self.L__mod___mobilebert_encoder_layer_3_intermediate_dense(attention_output_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_15 = self.L__mod___mobilebert_encoder_layer_3_intermediate_intermediate_act_fn(hidden_states_33);  hidden_states_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    layer_output_12 = self.L__mod___mobilebert_encoder_layer_3_output_dense(intermediate_output_15);  intermediate_output_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_59 = layer_output_12 + attention_output_18;  layer_output_12 = attention_output_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_3_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_3_output_LayerNorm_weight
    mul_32 = add_59 * l__mod___mobilebert_encoder_layer_3_output_layer_norm_weight_1;  add_59 = l__mod___mobilebert_encoder_layer_3_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_3_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_3_output_LayerNorm_bias
    layer_output_13 = mul_32 + l__mod___mobilebert_encoder_layer_3_output_layer_norm_bias_1;  mul_32 = l__mod___mobilebert_encoder_layer_3_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_53 = self.L__mod___mobilebert_encoder_layer_3_output_bottleneck_dense(layer_output_13);  layer_output_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    layer_outputs_54 = self.L__mod___mobilebert_encoder_layer_3_output_bottleneck_dropout(layer_outputs_53);  layer_outputs_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_61 = layer_outputs_54 + value_tensor_3;  layer_outputs_54 = value_tensor_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_3_output_bottleneck_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_3_output_bottleneck_LayerNorm_weight
    mul_33 = add_61 * l__mod___mobilebert_encoder_layer_3_output_bottleneck_layer_norm_weight_1;  add_61 = l__mod___mobilebert_encoder_layer_3_output_bottleneck_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_3_output_bottleneck_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_3_output_bottleneck_LayerNorm_bias
    value_tensor_4 = mul_33 + l__mod___mobilebert_encoder_layer_3_output_bottleneck_layer_norm_bias_1;  mul_33 = l__mod___mobilebert_encoder_layer_3_output_bottleneck_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:549, code: torch.tensor(1000),
    tensor_3 = torch.tensor(1000)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_20 = self.L__mod___mobilebert_encoder_layer_4_bottleneck_input_dense(value_tensor_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_4_bottleneck_input_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_4_bottleneck_input_LayerNorm_weight
    mul_34 = layer_input_20 * l__mod___mobilebert_encoder_layer_4_bottleneck_input_layer_norm_weight_1;  layer_input_20 = l__mod___mobilebert_encoder_layer_4_bottleneck_input_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_4_bottleneck_input_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_4_bottleneck_input_LayerNorm_bias
    layer_input_24 = mul_34 + l__mod___mobilebert_encoder_layer_4_bottleneck_input_layer_norm_bias_1;  mul_34 = l__mod___mobilebert_encoder_layer_4_bottleneck_input_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_22 = self.L__mod___mobilebert_encoder_layer_4_bottleneck_attention_dense(value_tensor_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_4_bottleneck_attention_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_4_bottleneck_attention_LayerNorm_weight
    mul_35 = layer_input_22 * l__mod___mobilebert_encoder_layer_4_bottleneck_attention_layer_norm_weight_1;  layer_input_22 = l__mod___mobilebert_encoder_layer_4_bottleneck_attention_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_4_bottleneck_attention_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_4_bottleneck_attention_LayerNorm_bias
    key_tensor_4 = mul_35 + l__mod___mobilebert_encoder_layer_4_bottleneck_attention_layer_norm_bias_1;  mul_35 = l__mod___mobilebert_encoder_layer_4_bottleneck_attention_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    mixed_query_layer_4 = self.L__mod___mobilebert_encoder_layer_4_attention_self_query(key_tensor_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    mixed_key_layer_4 = self.L__mod___mobilebert_encoder_layer_4_attention_self_key(key_tensor_4);  key_tensor_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    mixed_value_layer_4 = self.L__mod___mobilebert_encoder_layer_4_attention_self_value(value_tensor_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_12 = mixed_query_layer_4.view((1, 128, 4, 32));  mixed_query_layer_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    query_layer_4 = x_12.permute(0, 2, 1, 3);  x_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_13 = mixed_key_layer_4.view((1, 128, 4, 32));  mixed_key_layer_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    key_layer_4 = x_13.permute(0, 2, 1, 3);  x_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_14 = mixed_value_layer_4.view((1, 128, 4, 32));  mixed_value_layer_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    value_layer_4 = x_14.permute(0, 2, 1, 3);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:286, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_4 = key_layer_4.transpose(-1, -2);  key_layer_4 = None
    attention_scores_12 = torch.matmul(query_layer_4, transpose_4);  query_layer_4 = transpose_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:287, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_13 = attention_scores_12 / 5.656854249492381;  attention_scores_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:290, code: attention_scores = attention_scores + attention_mask
    attention_scores_14 = attention_scores_13 + extended_attention_mask_3;  attention_scores_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:292, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_8 = torch.nn.functional.softmax(attention_scores_14, dim = -1);  attention_scores_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295, code: attention_probs = self.dropout(attention_probs)
    attention_probs_9 = self.L__mod___mobilebert_encoder_layer_4_attention_self_dropout(attention_probs_8);  attention_probs_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:299, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_12 = torch.matmul(attention_probs_9, value_layer_4);  attention_probs_9 = value_layer_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_19 = context_layer_12.permute(0, 2, 1, 3);  context_layer_12 = None
    context_layer_13 = permute_19.contiguous();  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    context_layer_14 = context_layer_13.view((1, 128, 128));  context_layer_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_56 = self.L__mod___mobilebert_encoder_layer_4_attention_output_dense(context_layer_14);  context_layer_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_66 = layer_outputs_56 + layer_input_24;  layer_outputs_56 = layer_input_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_4_attention_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_4_attention_output_LayerNorm_weight
    mul_36 = add_66 * l__mod___mobilebert_encoder_layer_4_attention_output_layer_norm_weight_1;  add_66 = l__mod___mobilebert_encoder_layer_4_attention_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_4_attention_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_4_attention_output_LayerNorm_bias
    attention_output_20 = mul_36 + l__mod___mobilebert_encoder_layer_4_attention_output_layer_norm_bias_1;  mul_36 = l__mod___mobilebert_encoder_layer_4_attention_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_36 = self.L__mod___mobilebert_encoder_layer_4_ffn_0_intermediate_dense(attention_output_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_16 = self.L__mod___mobilebert_encoder_layer_4_ffn_0_intermediate_intermediate_act_fn(hidden_states_36);  hidden_states_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_58 = self.L__mod___mobilebert_encoder_layer_4_ffn_0_output_dense(intermediate_output_16);  intermediate_output_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_68 = layer_outputs_58 + attention_output_20;  layer_outputs_58 = attention_output_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_4_ffn_0_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_4_ffn_0_output_LayerNorm_weight
    mul_37 = add_68 * l__mod___mobilebert_encoder_layer_4_ffn_0_output_layer_norm_weight_1;  add_68 = l__mod___mobilebert_encoder_layer_4_ffn_0_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_4_ffn_0_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_4_ffn_0_output_LayerNorm_bias
    attention_output_21 = mul_37 + l__mod___mobilebert_encoder_layer_4_ffn_0_output_layer_norm_bias_1;  mul_37 = l__mod___mobilebert_encoder_layer_4_ffn_0_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_38 = self.L__mod___mobilebert_encoder_layer_4_ffn_1_intermediate_dense(attention_output_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_17 = self.L__mod___mobilebert_encoder_layer_4_ffn_1_intermediate_intermediate_act_fn(hidden_states_38);  hidden_states_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_61 = self.L__mod___mobilebert_encoder_layer_4_ffn_1_output_dense(intermediate_output_17);  intermediate_output_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_70 = layer_outputs_61 + attention_output_21;  layer_outputs_61 = attention_output_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_4_ffn_1_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_4_ffn_1_output_LayerNorm_weight
    mul_38 = add_70 * l__mod___mobilebert_encoder_layer_4_ffn_1_output_layer_norm_weight_1;  add_70 = l__mod___mobilebert_encoder_layer_4_ffn_1_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_4_ffn_1_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_4_ffn_1_output_LayerNorm_bias
    attention_output_22 = mul_38 + l__mod___mobilebert_encoder_layer_4_ffn_1_output_layer_norm_bias_1;  mul_38 = l__mod___mobilebert_encoder_layer_4_ffn_1_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_40 = self.L__mod___mobilebert_encoder_layer_4_ffn_2_intermediate_dense(attention_output_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_18 = self.L__mod___mobilebert_encoder_layer_4_ffn_2_intermediate_intermediate_act_fn(hidden_states_40);  hidden_states_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_64 = self.L__mod___mobilebert_encoder_layer_4_ffn_2_output_dense(intermediate_output_18);  intermediate_output_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_72 = layer_outputs_64 + attention_output_22;  layer_outputs_64 = attention_output_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_4_ffn_2_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_4_ffn_2_output_LayerNorm_weight
    mul_39 = add_72 * l__mod___mobilebert_encoder_layer_4_ffn_2_output_layer_norm_weight_1;  add_72 = l__mod___mobilebert_encoder_layer_4_ffn_2_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_4_ffn_2_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_4_ffn_2_output_LayerNorm_bias
    attention_output_23 = mul_39 + l__mod___mobilebert_encoder_layer_4_ffn_2_output_layer_norm_bias_1;  mul_39 = l__mod___mobilebert_encoder_layer_4_ffn_2_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_42 = self.L__mod___mobilebert_encoder_layer_4_intermediate_dense(attention_output_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_19 = self.L__mod___mobilebert_encoder_layer_4_intermediate_intermediate_act_fn(hidden_states_42);  hidden_states_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    layer_output_16 = self.L__mod___mobilebert_encoder_layer_4_output_dense(intermediate_output_19);  intermediate_output_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_74 = layer_output_16 + attention_output_23;  layer_output_16 = attention_output_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_4_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_4_output_LayerNorm_weight
    mul_40 = add_74 * l__mod___mobilebert_encoder_layer_4_output_layer_norm_weight_1;  add_74 = l__mod___mobilebert_encoder_layer_4_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_4_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_4_output_LayerNorm_bias
    layer_output_17 = mul_40 + l__mod___mobilebert_encoder_layer_4_output_layer_norm_bias_1;  mul_40 = l__mod___mobilebert_encoder_layer_4_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_67 = self.L__mod___mobilebert_encoder_layer_4_output_bottleneck_dense(layer_output_17);  layer_output_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    layer_outputs_68 = self.L__mod___mobilebert_encoder_layer_4_output_bottleneck_dropout(layer_outputs_67);  layer_outputs_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_76 = layer_outputs_68 + value_tensor_4;  layer_outputs_68 = value_tensor_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_4_output_bottleneck_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_4_output_bottleneck_LayerNorm_weight
    mul_41 = add_76 * l__mod___mobilebert_encoder_layer_4_output_bottleneck_layer_norm_weight_1;  add_76 = l__mod___mobilebert_encoder_layer_4_output_bottleneck_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_4_output_bottleneck_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_4_output_bottleneck_LayerNorm_bias
    value_tensor_5 = mul_41 + l__mod___mobilebert_encoder_layer_4_output_bottleneck_layer_norm_bias_1;  mul_41 = l__mod___mobilebert_encoder_layer_4_output_bottleneck_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:549, code: torch.tensor(1000),
    tensor_4 = torch.tensor(1000)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_25 = self.L__mod___mobilebert_encoder_layer_5_bottleneck_input_dense(value_tensor_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_5_bottleneck_input_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_5_bottleneck_input_LayerNorm_weight
    mul_42 = layer_input_25 * l__mod___mobilebert_encoder_layer_5_bottleneck_input_layer_norm_weight_1;  layer_input_25 = l__mod___mobilebert_encoder_layer_5_bottleneck_input_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_5_bottleneck_input_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_5_bottleneck_input_LayerNorm_bias
    layer_input_29 = mul_42 + l__mod___mobilebert_encoder_layer_5_bottleneck_input_layer_norm_bias_1;  mul_42 = l__mod___mobilebert_encoder_layer_5_bottleneck_input_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_27 = self.L__mod___mobilebert_encoder_layer_5_bottleneck_attention_dense(value_tensor_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_5_bottleneck_attention_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_5_bottleneck_attention_LayerNorm_weight
    mul_43 = layer_input_27 * l__mod___mobilebert_encoder_layer_5_bottleneck_attention_layer_norm_weight_1;  layer_input_27 = l__mod___mobilebert_encoder_layer_5_bottleneck_attention_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_5_bottleneck_attention_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_5_bottleneck_attention_LayerNorm_bias
    key_tensor_5 = mul_43 + l__mod___mobilebert_encoder_layer_5_bottleneck_attention_layer_norm_bias_1;  mul_43 = l__mod___mobilebert_encoder_layer_5_bottleneck_attention_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    mixed_query_layer_5 = self.L__mod___mobilebert_encoder_layer_5_attention_self_query(key_tensor_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    mixed_key_layer_5 = self.L__mod___mobilebert_encoder_layer_5_attention_self_key(key_tensor_5);  key_tensor_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    mixed_value_layer_5 = self.L__mod___mobilebert_encoder_layer_5_attention_self_value(value_tensor_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_15 = mixed_query_layer_5.view((1, 128, 4, 32));  mixed_query_layer_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    query_layer_5 = x_15.permute(0, 2, 1, 3);  x_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_16 = mixed_key_layer_5.view((1, 128, 4, 32));  mixed_key_layer_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    key_layer_5 = x_16.permute(0, 2, 1, 3);  x_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_17 = mixed_value_layer_5.view((1, 128, 4, 32));  mixed_value_layer_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    value_layer_5 = x_17.permute(0, 2, 1, 3);  x_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:286, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_5 = key_layer_5.transpose(-1, -2);  key_layer_5 = None
    attention_scores_15 = torch.matmul(query_layer_5, transpose_5);  query_layer_5 = transpose_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:287, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_16 = attention_scores_15 / 5.656854249492381;  attention_scores_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:290, code: attention_scores = attention_scores + attention_mask
    attention_scores_17 = attention_scores_16 + extended_attention_mask_3;  attention_scores_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:292, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_10 = torch.nn.functional.softmax(attention_scores_17, dim = -1);  attention_scores_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295, code: attention_probs = self.dropout(attention_probs)
    attention_probs_11 = self.L__mod___mobilebert_encoder_layer_5_attention_self_dropout(attention_probs_10);  attention_probs_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:299, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_15 = torch.matmul(attention_probs_11, value_layer_5);  attention_probs_11 = value_layer_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_23 = context_layer_15.permute(0, 2, 1, 3);  context_layer_15 = None
    context_layer_16 = permute_23.contiguous();  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    context_layer_17 = context_layer_16.view((1, 128, 128));  context_layer_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_70 = self.L__mod___mobilebert_encoder_layer_5_attention_output_dense(context_layer_17);  context_layer_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_81 = layer_outputs_70 + layer_input_29;  layer_outputs_70 = layer_input_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_5_attention_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_5_attention_output_LayerNorm_weight
    mul_44 = add_81 * l__mod___mobilebert_encoder_layer_5_attention_output_layer_norm_weight_1;  add_81 = l__mod___mobilebert_encoder_layer_5_attention_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_5_attention_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_5_attention_output_LayerNorm_bias
    attention_output_25 = mul_44 + l__mod___mobilebert_encoder_layer_5_attention_output_layer_norm_bias_1;  mul_44 = l__mod___mobilebert_encoder_layer_5_attention_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_45 = self.L__mod___mobilebert_encoder_layer_5_ffn_0_intermediate_dense(attention_output_25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_20 = self.L__mod___mobilebert_encoder_layer_5_ffn_0_intermediate_intermediate_act_fn(hidden_states_45);  hidden_states_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_72 = self.L__mod___mobilebert_encoder_layer_5_ffn_0_output_dense(intermediate_output_20);  intermediate_output_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_83 = layer_outputs_72 + attention_output_25;  layer_outputs_72 = attention_output_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_5_ffn_0_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_5_ffn_0_output_LayerNorm_weight
    mul_45 = add_83 * l__mod___mobilebert_encoder_layer_5_ffn_0_output_layer_norm_weight_1;  add_83 = l__mod___mobilebert_encoder_layer_5_ffn_0_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_5_ffn_0_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_5_ffn_0_output_LayerNorm_bias
    attention_output_26 = mul_45 + l__mod___mobilebert_encoder_layer_5_ffn_0_output_layer_norm_bias_1;  mul_45 = l__mod___mobilebert_encoder_layer_5_ffn_0_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_47 = self.L__mod___mobilebert_encoder_layer_5_ffn_1_intermediate_dense(attention_output_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_21 = self.L__mod___mobilebert_encoder_layer_5_ffn_1_intermediate_intermediate_act_fn(hidden_states_47);  hidden_states_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_75 = self.L__mod___mobilebert_encoder_layer_5_ffn_1_output_dense(intermediate_output_21);  intermediate_output_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_85 = layer_outputs_75 + attention_output_26;  layer_outputs_75 = attention_output_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_5_ffn_1_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_5_ffn_1_output_LayerNorm_weight
    mul_46 = add_85 * l__mod___mobilebert_encoder_layer_5_ffn_1_output_layer_norm_weight_1;  add_85 = l__mod___mobilebert_encoder_layer_5_ffn_1_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_5_ffn_1_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_5_ffn_1_output_LayerNorm_bias
    attention_output_27 = mul_46 + l__mod___mobilebert_encoder_layer_5_ffn_1_output_layer_norm_bias_1;  mul_46 = l__mod___mobilebert_encoder_layer_5_ffn_1_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_49 = self.L__mod___mobilebert_encoder_layer_5_ffn_2_intermediate_dense(attention_output_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_22 = self.L__mod___mobilebert_encoder_layer_5_ffn_2_intermediate_intermediate_act_fn(hidden_states_49);  hidden_states_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_78 = self.L__mod___mobilebert_encoder_layer_5_ffn_2_output_dense(intermediate_output_22);  intermediate_output_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_87 = layer_outputs_78 + attention_output_27;  layer_outputs_78 = attention_output_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_5_ffn_2_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_5_ffn_2_output_LayerNorm_weight
    mul_47 = add_87 * l__mod___mobilebert_encoder_layer_5_ffn_2_output_layer_norm_weight_1;  add_87 = l__mod___mobilebert_encoder_layer_5_ffn_2_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_5_ffn_2_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_5_ffn_2_output_LayerNorm_bias
    attention_output_28 = mul_47 + l__mod___mobilebert_encoder_layer_5_ffn_2_output_layer_norm_bias_1;  mul_47 = l__mod___mobilebert_encoder_layer_5_ffn_2_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_51 = self.L__mod___mobilebert_encoder_layer_5_intermediate_dense(attention_output_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_23 = self.L__mod___mobilebert_encoder_layer_5_intermediate_intermediate_act_fn(hidden_states_51);  hidden_states_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    layer_output_20 = self.L__mod___mobilebert_encoder_layer_5_output_dense(intermediate_output_23);  intermediate_output_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_89 = layer_output_20 + attention_output_28;  layer_output_20 = attention_output_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_5_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_5_output_LayerNorm_weight
    mul_48 = add_89 * l__mod___mobilebert_encoder_layer_5_output_layer_norm_weight_1;  add_89 = l__mod___mobilebert_encoder_layer_5_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_5_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_5_output_LayerNorm_bias
    layer_output_21 = mul_48 + l__mod___mobilebert_encoder_layer_5_output_layer_norm_bias_1;  mul_48 = l__mod___mobilebert_encoder_layer_5_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_81 = self.L__mod___mobilebert_encoder_layer_5_output_bottleneck_dense(layer_output_21);  layer_output_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    layer_outputs_82 = self.L__mod___mobilebert_encoder_layer_5_output_bottleneck_dropout(layer_outputs_81);  layer_outputs_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_91 = layer_outputs_82 + value_tensor_5;  layer_outputs_82 = value_tensor_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_5_output_bottleneck_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_5_output_bottleneck_LayerNorm_weight
    mul_49 = add_91 * l__mod___mobilebert_encoder_layer_5_output_bottleneck_layer_norm_weight_1;  add_91 = l__mod___mobilebert_encoder_layer_5_output_bottleneck_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_5_output_bottleneck_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_5_output_bottleneck_LayerNorm_bias
    value_tensor_6 = mul_49 + l__mod___mobilebert_encoder_layer_5_output_bottleneck_layer_norm_bias_1;  mul_49 = l__mod___mobilebert_encoder_layer_5_output_bottleneck_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:549, code: torch.tensor(1000),
    tensor_5 = torch.tensor(1000)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_30 = self.L__mod___mobilebert_encoder_layer_6_bottleneck_input_dense(value_tensor_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_6_bottleneck_input_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_6_bottleneck_input_LayerNorm_weight
    mul_50 = layer_input_30 * l__mod___mobilebert_encoder_layer_6_bottleneck_input_layer_norm_weight_1;  layer_input_30 = l__mod___mobilebert_encoder_layer_6_bottleneck_input_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_6_bottleneck_input_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_6_bottleneck_input_LayerNorm_bias
    layer_input_34 = mul_50 + l__mod___mobilebert_encoder_layer_6_bottleneck_input_layer_norm_bias_1;  mul_50 = l__mod___mobilebert_encoder_layer_6_bottleneck_input_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_32 = self.L__mod___mobilebert_encoder_layer_6_bottleneck_attention_dense(value_tensor_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_6_bottleneck_attention_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_6_bottleneck_attention_LayerNorm_weight
    mul_51 = layer_input_32 * l__mod___mobilebert_encoder_layer_6_bottleneck_attention_layer_norm_weight_1;  layer_input_32 = l__mod___mobilebert_encoder_layer_6_bottleneck_attention_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_6_bottleneck_attention_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_6_bottleneck_attention_LayerNorm_bias
    key_tensor_6 = mul_51 + l__mod___mobilebert_encoder_layer_6_bottleneck_attention_layer_norm_bias_1;  mul_51 = l__mod___mobilebert_encoder_layer_6_bottleneck_attention_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    mixed_query_layer_6 = self.L__mod___mobilebert_encoder_layer_6_attention_self_query(key_tensor_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    mixed_key_layer_6 = self.L__mod___mobilebert_encoder_layer_6_attention_self_key(key_tensor_6);  key_tensor_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    mixed_value_layer_6 = self.L__mod___mobilebert_encoder_layer_6_attention_self_value(value_tensor_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_18 = mixed_query_layer_6.view((1, 128, 4, 32));  mixed_query_layer_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    query_layer_6 = x_18.permute(0, 2, 1, 3);  x_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_19 = mixed_key_layer_6.view((1, 128, 4, 32));  mixed_key_layer_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    key_layer_6 = x_19.permute(0, 2, 1, 3);  x_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_20 = mixed_value_layer_6.view((1, 128, 4, 32));  mixed_value_layer_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    value_layer_6 = x_20.permute(0, 2, 1, 3);  x_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:286, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_6 = key_layer_6.transpose(-1, -2);  key_layer_6 = None
    attention_scores_18 = torch.matmul(query_layer_6, transpose_6);  query_layer_6 = transpose_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:287, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_19 = attention_scores_18 / 5.656854249492381;  attention_scores_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:290, code: attention_scores = attention_scores + attention_mask
    attention_scores_20 = attention_scores_19 + extended_attention_mask_3;  attention_scores_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:292, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_12 = torch.nn.functional.softmax(attention_scores_20, dim = -1);  attention_scores_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295, code: attention_probs = self.dropout(attention_probs)
    attention_probs_13 = self.L__mod___mobilebert_encoder_layer_6_attention_self_dropout(attention_probs_12);  attention_probs_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:299, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_18 = torch.matmul(attention_probs_13, value_layer_6);  attention_probs_13 = value_layer_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_27 = context_layer_18.permute(0, 2, 1, 3);  context_layer_18 = None
    context_layer_19 = permute_27.contiguous();  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    context_layer_20 = context_layer_19.view((1, 128, 128));  context_layer_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_84 = self.L__mod___mobilebert_encoder_layer_6_attention_output_dense(context_layer_20);  context_layer_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_96 = layer_outputs_84 + layer_input_34;  layer_outputs_84 = layer_input_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_6_attention_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_6_attention_output_LayerNorm_weight
    mul_52 = add_96 * l__mod___mobilebert_encoder_layer_6_attention_output_layer_norm_weight_1;  add_96 = l__mod___mobilebert_encoder_layer_6_attention_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_6_attention_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_6_attention_output_LayerNorm_bias
    attention_output_30 = mul_52 + l__mod___mobilebert_encoder_layer_6_attention_output_layer_norm_bias_1;  mul_52 = l__mod___mobilebert_encoder_layer_6_attention_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_54 = self.L__mod___mobilebert_encoder_layer_6_ffn_0_intermediate_dense(attention_output_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_24 = self.L__mod___mobilebert_encoder_layer_6_ffn_0_intermediate_intermediate_act_fn(hidden_states_54);  hidden_states_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_86 = self.L__mod___mobilebert_encoder_layer_6_ffn_0_output_dense(intermediate_output_24);  intermediate_output_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_98 = layer_outputs_86 + attention_output_30;  layer_outputs_86 = attention_output_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_6_ffn_0_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_6_ffn_0_output_LayerNorm_weight
    mul_53 = add_98 * l__mod___mobilebert_encoder_layer_6_ffn_0_output_layer_norm_weight_1;  add_98 = l__mod___mobilebert_encoder_layer_6_ffn_0_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_6_ffn_0_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_6_ffn_0_output_LayerNorm_bias
    attention_output_31 = mul_53 + l__mod___mobilebert_encoder_layer_6_ffn_0_output_layer_norm_bias_1;  mul_53 = l__mod___mobilebert_encoder_layer_6_ffn_0_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_56 = self.L__mod___mobilebert_encoder_layer_6_ffn_1_intermediate_dense(attention_output_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_25 = self.L__mod___mobilebert_encoder_layer_6_ffn_1_intermediate_intermediate_act_fn(hidden_states_56);  hidden_states_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_89 = self.L__mod___mobilebert_encoder_layer_6_ffn_1_output_dense(intermediate_output_25);  intermediate_output_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_100 = layer_outputs_89 + attention_output_31;  layer_outputs_89 = attention_output_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_6_ffn_1_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_6_ffn_1_output_LayerNorm_weight
    mul_54 = add_100 * l__mod___mobilebert_encoder_layer_6_ffn_1_output_layer_norm_weight_1;  add_100 = l__mod___mobilebert_encoder_layer_6_ffn_1_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_6_ffn_1_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_6_ffn_1_output_LayerNorm_bias
    attention_output_32 = mul_54 + l__mod___mobilebert_encoder_layer_6_ffn_1_output_layer_norm_bias_1;  mul_54 = l__mod___mobilebert_encoder_layer_6_ffn_1_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_58 = self.L__mod___mobilebert_encoder_layer_6_ffn_2_intermediate_dense(attention_output_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_26 = self.L__mod___mobilebert_encoder_layer_6_ffn_2_intermediate_intermediate_act_fn(hidden_states_58);  hidden_states_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_92 = self.L__mod___mobilebert_encoder_layer_6_ffn_2_output_dense(intermediate_output_26);  intermediate_output_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_102 = layer_outputs_92 + attention_output_32;  layer_outputs_92 = attention_output_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_6_ffn_2_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_6_ffn_2_output_LayerNorm_weight
    mul_55 = add_102 * l__mod___mobilebert_encoder_layer_6_ffn_2_output_layer_norm_weight_1;  add_102 = l__mod___mobilebert_encoder_layer_6_ffn_2_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_6_ffn_2_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_6_ffn_2_output_LayerNorm_bias
    attention_output_33 = mul_55 + l__mod___mobilebert_encoder_layer_6_ffn_2_output_layer_norm_bias_1;  mul_55 = l__mod___mobilebert_encoder_layer_6_ffn_2_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_60 = self.L__mod___mobilebert_encoder_layer_6_intermediate_dense(attention_output_33)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_27 = self.L__mod___mobilebert_encoder_layer_6_intermediate_intermediate_act_fn(hidden_states_60);  hidden_states_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    layer_output_24 = self.L__mod___mobilebert_encoder_layer_6_output_dense(intermediate_output_27);  intermediate_output_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_104 = layer_output_24 + attention_output_33;  layer_output_24 = attention_output_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_6_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_6_output_LayerNorm_weight
    mul_56 = add_104 * l__mod___mobilebert_encoder_layer_6_output_layer_norm_weight_1;  add_104 = l__mod___mobilebert_encoder_layer_6_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_6_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_6_output_LayerNorm_bias
    layer_output_25 = mul_56 + l__mod___mobilebert_encoder_layer_6_output_layer_norm_bias_1;  mul_56 = l__mod___mobilebert_encoder_layer_6_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_95 = self.L__mod___mobilebert_encoder_layer_6_output_bottleneck_dense(layer_output_25);  layer_output_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    layer_outputs_96 = self.L__mod___mobilebert_encoder_layer_6_output_bottleneck_dropout(layer_outputs_95);  layer_outputs_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_106 = layer_outputs_96 + value_tensor_6;  layer_outputs_96 = value_tensor_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_6_output_bottleneck_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_6_output_bottleneck_LayerNorm_weight
    mul_57 = add_106 * l__mod___mobilebert_encoder_layer_6_output_bottleneck_layer_norm_weight_1;  add_106 = l__mod___mobilebert_encoder_layer_6_output_bottleneck_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_6_output_bottleneck_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_6_output_bottleneck_LayerNorm_bias
    value_tensor_7 = mul_57 + l__mod___mobilebert_encoder_layer_6_output_bottleneck_layer_norm_bias_1;  mul_57 = l__mod___mobilebert_encoder_layer_6_output_bottleneck_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:549, code: torch.tensor(1000),
    tensor_6 = torch.tensor(1000)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_35 = self.L__mod___mobilebert_encoder_layer_7_bottleneck_input_dense(value_tensor_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_7_bottleneck_input_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_7_bottleneck_input_LayerNorm_weight
    mul_58 = layer_input_35 * l__mod___mobilebert_encoder_layer_7_bottleneck_input_layer_norm_weight_1;  layer_input_35 = l__mod___mobilebert_encoder_layer_7_bottleneck_input_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_7_bottleneck_input_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_7_bottleneck_input_LayerNorm_bias
    layer_input_39 = mul_58 + l__mod___mobilebert_encoder_layer_7_bottleneck_input_layer_norm_bias_1;  mul_58 = l__mod___mobilebert_encoder_layer_7_bottleneck_input_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_37 = self.L__mod___mobilebert_encoder_layer_7_bottleneck_attention_dense(value_tensor_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_7_bottleneck_attention_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_7_bottleneck_attention_LayerNorm_weight
    mul_59 = layer_input_37 * l__mod___mobilebert_encoder_layer_7_bottleneck_attention_layer_norm_weight_1;  layer_input_37 = l__mod___mobilebert_encoder_layer_7_bottleneck_attention_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_7_bottleneck_attention_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_7_bottleneck_attention_LayerNorm_bias
    key_tensor_7 = mul_59 + l__mod___mobilebert_encoder_layer_7_bottleneck_attention_layer_norm_bias_1;  mul_59 = l__mod___mobilebert_encoder_layer_7_bottleneck_attention_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    mixed_query_layer_7 = self.L__mod___mobilebert_encoder_layer_7_attention_self_query(key_tensor_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    mixed_key_layer_7 = self.L__mod___mobilebert_encoder_layer_7_attention_self_key(key_tensor_7);  key_tensor_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    mixed_value_layer_7 = self.L__mod___mobilebert_encoder_layer_7_attention_self_value(value_tensor_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_21 = mixed_query_layer_7.view((1, 128, 4, 32));  mixed_query_layer_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    query_layer_7 = x_21.permute(0, 2, 1, 3);  x_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_22 = mixed_key_layer_7.view((1, 128, 4, 32));  mixed_key_layer_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    key_layer_7 = x_22.permute(0, 2, 1, 3);  x_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_23 = mixed_value_layer_7.view((1, 128, 4, 32));  mixed_value_layer_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    value_layer_7 = x_23.permute(0, 2, 1, 3);  x_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:286, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_7 = key_layer_7.transpose(-1, -2);  key_layer_7 = None
    attention_scores_21 = torch.matmul(query_layer_7, transpose_7);  query_layer_7 = transpose_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:287, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_22 = attention_scores_21 / 5.656854249492381;  attention_scores_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:290, code: attention_scores = attention_scores + attention_mask
    attention_scores_23 = attention_scores_22 + extended_attention_mask_3;  attention_scores_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:292, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_14 = torch.nn.functional.softmax(attention_scores_23, dim = -1);  attention_scores_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295, code: attention_probs = self.dropout(attention_probs)
    attention_probs_15 = self.L__mod___mobilebert_encoder_layer_7_attention_self_dropout(attention_probs_14);  attention_probs_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:299, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_21 = torch.matmul(attention_probs_15, value_layer_7);  attention_probs_15 = value_layer_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_31 = context_layer_21.permute(0, 2, 1, 3);  context_layer_21 = None
    context_layer_22 = permute_31.contiguous();  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    context_layer_23 = context_layer_22.view((1, 128, 128));  context_layer_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_98 = self.L__mod___mobilebert_encoder_layer_7_attention_output_dense(context_layer_23);  context_layer_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_111 = layer_outputs_98 + layer_input_39;  layer_outputs_98 = layer_input_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_7_attention_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_7_attention_output_LayerNorm_weight
    mul_60 = add_111 * l__mod___mobilebert_encoder_layer_7_attention_output_layer_norm_weight_1;  add_111 = l__mod___mobilebert_encoder_layer_7_attention_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_7_attention_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_7_attention_output_LayerNorm_bias
    attention_output_35 = mul_60 + l__mod___mobilebert_encoder_layer_7_attention_output_layer_norm_bias_1;  mul_60 = l__mod___mobilebert_encoder_layer_7_attention_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_63 = self.L__mod___mobilebert_encoder_layer_7_ffn_0_intermediate_dense(attention_output_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_28 = self.L__mod___mobilebert_encoder_layer_7_ffn_0_intermediate_intermediate_act_fn(hidden_states_63);  hidden_states_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_100 = self.L__mod___mobilebert_encoder_layer_7_ffn_0_output_dense(intermediate_output_28);  intermediate_output_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_113 = layer_outputs_100 + attention_output_35;  layer_outputs_100 = attention_output_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_7_ffn_0_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_7_ffn_0_output_LayerNorm_weight
    mul_61 = add_113 * l__mod___mobilebert_encoder_layer_7_ffn_0_output_layer_norm_weight_1;  add_113 = l__mod___mobilebert_encoder_layer_7_ffn_0_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_7_ffn_0_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_7_ffn_0_output_LayerNorm_bias
    attention_output_36 = mul_61 + l__mod___mobilebert_encoder_layer_7_ffn_0_output_layer_norm_bias_1;  mul_61 = l__mod___mobilebert_encoder_layer_7_ffn_0_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_65 = self.L__mod___mobilebert_encoder_layer_7_ffn_1_intermediate_dense(attention_output_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_29 = self.L__mod___mobilebert_encoder_layer_7_ffn_1_intermediate_intermediate_act_fn(hidden_states_65);  hidden_states_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_103 = self.L__mod___mobilebert_encoder_layer_7_ffn_1_output_dense(intermediate_output_29);  intermediate_output_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_115 = layer_outputs_103 + attention_output_36;  layer_outputs_103 = attention_output_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_7_ffn_1_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_7_ffn_1_output_LayerNorm_weight
    mul_62 = add_115 * l__mod___mobilebert_encoder_layer_7_ffn_1_output_layer_norm_weight_1;  add_115 = l__mod___mobilebert_encoder_layer_7_ffn_1_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_7_ffn_1_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_7_ffn_1_output_LayerNorm_bias
    attention_output_37 = mul_62 + l__mod___mobilebert_encoder_layer_7_ffn_1_output_layer_norm_bias_1;  mul_62 = l__mod___mobilebert_encoder_layer_7_ffn_1_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_67 = self.L__mod___mobilebert_encoder_layer_7_ffn_2_intermediate_dense(attention_output_37)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_30 = self.L__mod___mobilebert_encoder_layer_7_ffn_2_intermediate_intermediate_act_fn(hidden_states_67);  hidden_states_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_106 = self.L__mod___mobilebert_encoder_layer_7_ffn_2_output_dense(intermediate_output_30);  intermediate_output_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_117 = layer_outputs_106 + attention_output_37;  layer_outputs_106 = attention_output_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_7_ffn_2_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_7_ffn_2_output_LayerNorm_weight
    mul_63 = add_117 * l__mod___mobilebert_encoder_layer_7_ffn_2_output_layer_norm_weight_1;  add_117 = l__mod___mobilebert_encoder_layer_7_ffn_2_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_7_ffn_2_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_7_ffn_2_output_LayerNorm_bias
    attention_output_38 = mul_63 + l__mod___mobilebert_encoder_layer_7_ffn_2_output_layer_norm_bias_1;  mul_63 = l__mod___mobilebert_encoder_layer_7_ffn_2_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_69 = self.L__mod___mobilebert_encoder_layer_7_intermediate_dense(attention_output_38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_31 = self.L__mod___mobilebert_encoder_layer_7_intermediate_intermediate_act_fn(hidden_states_69);  hidden_states_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    layer_output_28 = self.L__mod___mobilebert_encoder_layer_7_output_dense(intermediate_output_31);  intermediate_output_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_119 = layer_output_28 + attention_output_38;  layer_output_28 = attention_output_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_7_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_7_output_LayerNorm_weight
    mul_64 = add_119 * l__mod___mobilebert_encoder_layer_7_output_layer_norm_weight_1;  add_119 = l__mod___mobilebert_encoder_layer_7_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_7_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_7_output_LayerNorm_bias
    layer_output_29 = mul_64 + l__mod___mobilebert_encoder_layer_7_output_layer_norm_bias_1;  mul_64 = l__mod___mobilebert_encoder_layer_7_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_109 = self.L__mod___mobilebert_encoder_layer_7_output_bottleneck_dense(layer_output_29);  layer_output_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    layer_outputs_110 = self.L__mod___mobilebert_encoder_layer_7_output_bottleneck_dropout(layer_outputs_109);  layer_outputs_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_121 = layer_outputs_110 + value_tensor_7;  layer_outputs_110 = value_tensor_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_7_output_bottleneck_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_7_output_bottleneck_LayerNorm_weight
    mul_65 = add_121 * l__mod___mobilebert_encoder_layer_7_output_bottleneck_layer_norm_weight_1;  add_121 = l__mod___mobilebert_encoder_layer_7_output_bottleneck_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_7_output_bottleneck_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_7_output_bottleneck_LayerNorm_bias
    value_tensor_8 = mul_65 + l__mod___mobilebert_encoder_layer_7_output_bottleneck_layer_norm_bias_1;  mul_65 = l__mod___mobilebert_encoder_layer_7_output_bottleneck_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:549, code: torch.tensor(1000),
    tensor_7 = torch.tensor(1000)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_40 = self.L__mod___mobilebert_encoder_layer_8_bottleneck_input_dense(value_tensor_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_8_bottleneck_input_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_8_bottleneck_input_LayerNorm_weight
    mul_66 = layer_input_40 * l__mod___mobilebert_encoder_layer_8_bottleneck_input_layer_norm_weight_1;  layer_input_40 = l__mod___mobilebert_encoder_layer_8_bottleneck_input_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_8_bottleneck_input_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_8_bottleneck_input_LayerNorm_bias
    layer_input_44 = mul_66 + l__mod___mobilebert_encoder_layer_8_bottleneck_input_layer_norm_bias_1;  mul_66 = l__mod___mobilebert_encoder_layer_8_bottleneck_input_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_42 = self.L__mod___mobilebert_encoder_layer_8_bottleneck_attention_dense(value_tensor_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_8_bottleneck_attention_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_8_bottleneck_attention_LayerNorm_weight
    mul_67 = layer_input_42 * l__mod___mobilebert_encoder_layer_8_bottleneck_attention_layer_norm_weight_1;  layer_input_42 = l__mod___mobilebert_encoder_layer_8_bottleneck_attention_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_8_bottleneck_attention_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_8_bottleneck_attention_LayerNorm_bias
    key_tensor_8 = mul_67 + l__mod___mobilebert_encoder_layer_8_bottleneck_attention_layer_norm_bias_1;  mul_67 = l__mod___mobilebert_encoder_layer_8_bottleneck_attention_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    mixed_query_layer_8 = self.L__mod___mobilebert_encoder_layer_8_attention_self_query(key_tensor_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    mixed_key_layer_8 = self.L__mod___mobilebert_encoder_layer_8_attention_self_key(key_tensor_8);  key_tensor_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    mixed_value_layer_8 = self.L__mod___mobilebert_encoder_layer_8_attention_self_value(value_tensor_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_24 = mixed_query_layer_8.view((1, 128, 4, 32));  mixed_query_layer_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    query_layer_8 = x_24.permute(0, 2, 1, 3);  x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_25 = mixed_key_layer_8.view((1, 128, 4, 32));  mixed_key_layer_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    key_layer_8 = x_25.permute(0, 2, 1, 3);  x_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_26 = mixed_value_layer_8.view((1, 128, 4, 32));  mixed_value_layer_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    value_layer_8 = x_26.permute(0, 2, 1, 3);  x_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:286, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_8 = key_layer_8.transpose(-1, -2);  key_layer_8 = None
    attention_scores_24 = torch.matmul(query_layer_8, transpose_8);  query_layer_8 = transpose_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:287, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_25 = attention_scores_24 / 5.656854249492381;  attention_scores_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:290, code: attention_scores = attention_scores + attention_mask
    attention_scores_26 = attention_scores_25 + extended_attention_mask_3;  attention_scores_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:292, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_16 = torch.nn.functional.softmax(attention_scores_26, dim = -1);  attention_scores_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295, code: attention_probs = self.dropout(attention_probs)
    attention_probs_17 = self.L__mod___mobilebert_encoder_layer_8_attention_self_dropout(attention_probs_16);  attention_probs_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:299, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_24 = torch.matmul(attention_probs_17, value_layer_8);  attention_probs_17 = value_layer_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_35 = context_layer_24.permute(0, 2, 1, 3);  context_layer_24 = None
    context_layer_25 = permute_35.contiguous();  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    context_layer_26 = context_layer_25.view((1, 128, 128));  context_layer_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_112 = self.L__mod___mobilebert_encoder_layer_8_attention_output_dense(context_layer_26);  context_layer_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_126 = layer_outputs_112 + layer_input_44;  layer_outputs_112 = layer_input_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_8_attention_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_8_attention_output_LayerNorm_weight
    mul_68 = add_126 * l__mod___mobilebert_encoder_layer_8_attention_output_layer_norm_weight_1;  add_126 = l__mod___mobilebert_encoder_layer_8_attention_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_8_attention_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_8_attention_output_LayerNorm_bias
    attention_output_40 = mul_68 + l__mod___mobilebert_encoder_layer_8_attention_output_layer_norm_bias_1;  mul_68 = l__mod___mobilebert_encoder_layer_8_attention_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_72 = self.L__mod___mobilebert_encoder_layer_8_ffn_0_intermediate_dense(attention_output_40)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_32 = self.L__mod___mobilebert_encoder_layer_8_ffn_0_intermediate_intermediate_act_fn(hidden_states_72);  hidden_states_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_114 = self.L__mod___mobilebert_encoder_layer_8_ffn_0_output_dense(intermediate_output_32);  intermediate_output_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_128 = layer_outputs_114 + attention_output_40;  layer_outputs_114 = attention_output_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_8_ffn_0_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_8_ffn_0_output_LayerNorm_weight
    mul_69 = add_128 * l__mod___mobilebert_encoder_layer_8_ffn_0_output_layer_norm_weight_1;  add_128 = l__mod___mobilebert_encoder_layer_8_ffn_0_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_8_ffn_0_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_8_ffn_0_output_LayerNorm_bias
    attention_output_41 = mul_69 + l__mod___mobilebert_encoder_layer_8_ffn_0_output_layer_norm_bias_1;  mul_69 = l__mod___mobilebert_encoder_layer_8_ffn_0_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_74 = self.L__mod___mobilebert_encoder_layer_8_ffn_1_intermediate_dense(attention_output_41)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_33 = self.L__mod___mobilebert_encoder_layer_8_ffn_1_intermediate_intermediate_act_fn(hidden_states_74);  hidden_states_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_117 = self.L__mod___mobilebert_encoder_layer_8_ffn_1_output_dense(intermediate_output_33);  intermediate_output_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_130 = layer_outputs_117 + attention_output_41;  layer_outputs_117 = attention_output_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_8_ffn_1_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_8_ffn_1_output_LayerNorm_weight
    mul_70 = add_130 * l__mod___mobilebert_encoder_layer_8_ffn_1_output_layer_norm_weight_1;  add_130 = l__mod___mobilebert_encoder_layer_8_ffn_1_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_8_ffn_1_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_8_ffn_1_output_LayerNorm_bias
    attention_output_42 = mul_70 + l__mod___mobilebert_encoder_layer_8_ffn_1_output_layer_norm_bias_1;  mul_70 = l__mod___mobilebert_encoder_layer_8_ffn_1_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_76 = self.L__mod___mobilebert_encoder_layer_8_ffn_2_intermediate_dense(attention_output_42)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_34 = self.L__mod___mobilebert_encoder_layer_8_ffn_2_intermediate_intermediate_act_fn(hidden_states_76);  hidden_states_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_120 = self.L__mod___mobilebert_encoder_layer_8_ffn_2_output_dense(intermediate_output_34);  intermediate_output_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_132 = layer_outputs_120 + attention_output_42;  layer_outputs_120 = attention_output_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_8_ffn_2_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_8_ffn_2_output_LayerNorm_weight
    mul_71 = add_132 * l__mod___mobilebert_encoder_layer_8_ffn_2_output_layer_norm_weight_1;  add_132 = l__mod___mobilebert_encoder_layer_8_ffn_2_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_8_ffn_2_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_8_ffn_2_output_LayerNorm_bias
    attention_output_43 = mul_71 + l__mod___mobilebert_encoder_layer_8_ffn_2_output_layer_norm_bias_1;  mul_71 = l__mod___mobilebert_encoder_layer_8_ffn_2_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_78 = self.L__mod___mobilebert_encoder_layer_8_intermediate_dense(attention_output_43)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_35 = self.L__mod___mobilebert_encoder_layer_8_intermediate_intermediate_act_fn(hidden_states_78);  hidden_states_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    layer_output_32 = self.L__mod___mobilebert_encoder_layer_8_output_dense(intermediate_output_35);  intermediate_output_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_134 = layer_output_32 + attention_output_43;  layer_output_32 = attention_output_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_8_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_8_output_LayerNorm_weight
    mul_72 = add_134 * l__mod___mobilebert_encoder_layer_8_output_layer_norm_weight_1;  add_134 = l__mod___mobilebert_encoder_layer_8_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_8_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_8_output_LayerNorm_bias
    layer_output_33 = mul_72 + l__mod___mobilebert_encoder_layer_8_output_layer_norm_bias_1;  mul_72 = l__mod___mobilebert_encoder_layer_8_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_123 = self.L__mod___mobilebert_encoder_layer_8_output_bottleneck_dense(layer_output_33);  layer_output_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    layer_outputs_124 = self.L__mod___mobilebert_encoder_layer_8_output_bottleneck_dropout(layer_outputs_123);  layer_outputs_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_136 = layer_outputs_124 + value_tensor_8;  layer_outputs_124 = value_tensor_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_8_output_bottleneck_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_8_output_bottleneck_LayerNorm_weight
    mul_73 = add_136 * l__mod___mobilebert_encoder_layer_8_output_bottleneck_layer_norm_weight_1;  add_136 = l__mod___mobilebert_encoder_layer_8_output_bottleneck_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_8_output_bottleneck_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_8_output_bottleneck_LayerNorm_bias
    value_tensor_9 = mul_73 + l__mod___mobilebert_encoder_layer_8_output_bottleneck_layer_norm_bias_1;  mul_73 = l__mod___mobilebert_encoder_layer_8_output_bottleneck_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:549, code: torch.tensor(1000),
    tensor_8 = torch.tensor(1000)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_45 = self.L__mod___mobilebert_encoder_layer_9_bottleneck_input_dense(value_tensor_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_9_bottleneck_input_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_9_bottleneck_input_LayerNorm_weight
    mul_74 = layer_input_45 * l__mod___mobilebert_encoder_layer_9_bottleneck_input_layer_norm_weight_1;  layer_input_45 = l__mod___mobilebert_encoder_layer_9_bottleneck_input_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_9_bottleneck_input_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_9_bottleneck_input_LayerNorm_bias
    layer_input_49 = mul_74 + l__mod___mobilebert_encoder_layer_9_bottleneck_input_layer_norm_bias_1;  mul_74 = l__mod___mobilebert_encoder_layer_9_bottleneck_input_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_47 = self.L__mod___mobilebert_encoder_layer_9_bottleneck_attention_dense(value_tensor_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_9_bottleneck_attention_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_9_bottleneck_attention_LayerNorm_weight
    mul_75 = layer_input_47 * l__mod___mobilebert_encoder_layer_9_bottleneck_attention_layer_norm_weight_1;  layer_input_47 = l__mod___mobilebert_encoder_layer_9_bottleneck_attention_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_9_bottleneck_attention_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_9_bottleneck_attention_LayerNorm_bias
    key_tensor_9 = mul_75 + l__mod___mobilebert_encoder_layer_9_bottleneck_attention_layer_norm_bias_1;  mul_75 = l__mod___mobilebert_encoder_layer_9_bottleneck_attention_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    mixed_query_layer_9 = self.L__mod___mobilebert_encoder_layer_9_attention_self_query(key_tensor_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    mixed_key_layer_9 = self.L__mod___mobilebert_encoder_layer_9_attention_self_key(key_tensor_9);  key_tensor_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    mixed_value_layer_9 = self.L__mod___mobilebert_encoder_layer_9_attention_self_value(value_tensor_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_27 = mixed_query_layer_9.view((1, 128, 4, 32));  mixed_query_layer_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    query_layer_9 = x_27.permute(0, 2, 1, 3);  x_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_28 = mixed_key_layer_9.view((1, 128, 4, 32));  mixed_key_layer_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    key_layer_9 = x_28.permute(0, 2, 1, 3);  x_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_29 = mixed_value_layer_9.view((1, 128, 4, 32));  mixed_value_layer_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    value_layer_9 = x_29.permute(0, 2, 1, 3);  x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:286, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_9 = key_layer_9.transpose(-1, -2);  key_layer_9 = None
    attention_scores_27 = torch.matmul(query_layer_9, transpose_9);  query_layer_9 = transpose_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:287, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_28 = attention_scores_27 / 5.656854249492381;  attention_scores_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:290, code: attention_scores = attention_scores + attention_mask
    attention_scores_29 = attention_scores_28 + extended_attention_mask_3;  attention_scores_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:292, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_18 = torch.nn.functional.softmax(attention_scores_29, dim = -1);  attention_scores_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295, code: attention_probs = self.dropout(attention_probs)
    attention_probs_19 = self.L__mod___mobilebert_encoder_layer_9_attention_self_dropout(attention_probs_18);  attention_probs_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:299, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_27 = torch.matmul(attention_probs_19, value_layer_9);  attention_probs_19 = value_layer_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_39 = context_layer_27.permute(0, 2, 1, 3);  context_layer_27 = None
    context_layer_28 = permute_39.contiguous();  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    context_layer_29 = context_layer_28.view((1, 128, 128));  context_layer_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_126 = self.L__mod___mobilebert_encoder_layer_9_attention_output_dense(context_layer_29);  context_layer_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_141 = layer_outputs_126 + layer_input_49;  layer_outputs_126 = layer_input_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_9_attention_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_9_attention_output_LayerNorm_weight
    mul_76 = add_141 * l__mod___mobilebert_encoder_layer_9_attention_output_layer_norm_weight_1;  add_141 = l__mod___mobilebert_encoder_layer_9_attention_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_9_attention_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_9_attention_output_LayerNorm_bias
    attention_output_45 = mul_76 + l__mod___mobilebert_encoder_layer_9_attention_output_layer_norm_bias_1;  mul_76 = l__mod___mobilebert_encoder_layer_9_attention_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_81 = self.L__mod___mobilebert_encoder_layer_9_ffn_0_intermediate_dense(attention_output_45)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_36 = self.L__mod___mobilebert_encoder_layer_9_ffn_0_intermediate_intermediate_act_fn(hidden_states_81);  hidden_states_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_128 = self.L__mod___mobilebert_encoder_layer_9_ffn_0_output_dense(intermediate_output_36);  intermediate_output_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_143 = layer_outputs_128 + attention_output_45;  layer_outputs_128 = attention_output_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_9_ffn_0_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_9_ffn_0_output_LayerNorm_weight
    mul_77 = add_143 * l__mod___mobilebert_encoder_layer_9_ffn_0_output_layer_norm_weight_1;  add_143 = l__mod___mobilebert_encoder_layer_9_ffn_0_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_9_ffn_0_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_9_ffn_0_output_LayerNorm_bias
    attention_output_46 = mul_77 + l__mod___mobilebert_encoder_layer_9_ffn_0_output_layer_norm_bias_1;  mul_77 = l__mod___mobilebert_encoder_layer_9_ffn_0_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_83 = self.L__mod___mobilebert_encoder_layer_9_ffn_1_intermediate_dense(attention_output_46)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_37 = self.L__mod___mobilebert_encoder_layer_9_ffn_1_intermediate_intermediate_act_fn(hidden_states_83);  hidden_states_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_131 = self.L__mod___mobilebert_encoder_layer_9_ffn_1_output_dense(intermediate_output_37);  intermediate_output_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_145 = layer_outputs_131 + attention_output_46;  layer_outputs_131 = attention_output_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_9_ffn_1_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_9_ffn_1_output_LayerNorm_weight
    mul_78 = add_145 * l__mod___mobilebert_encoder_layer_9_ffn_1_output_layer_norm_weight_1;  add_145 = l__mod___mobilebert_encoder_layer_9_ffn_1_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_9_ffn_1_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_9_ffn_1_output_LayerNorm_bias
    attention_output_47 = mul_78 + l__mod___mobilebert_encoder_layer_9_ffn_1_output_layer_norm_bias_1;  mul_78 = l__mod___mobilebert_encoder_layer_9_ffn_1_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_85 = self.L__mod___mobilebert_encoder_layer_9_ffn_2_intermediate_dense(attention_output_47)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_38 = self.L__mod___mobilebert_encoder_layer_9_ffn_2_intermediate_intermediate_act_fn(hidden_states_85);  hidden_states_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_134 = self.L__mod___mobilebert_encoder_layer_9_ffn_2_output_dense(intermediate_output_38);  intermediate_output_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_147 = layer_outputs_134 + attention_output_47;  layer_outputs_134 = attention_output_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_9_ffn_2_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_9_ffn_2_output_LayerNorm_weight
    mul_79 = add_147 * l__mod___mobilebert_encoder_layer_9_ffn_2_output_layer_norm_weight_1;  add_147 = l__mod___mobilebert_encoder_layer_9_ffn_2_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_9_ffn_2_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_9_ffn_2_output_LayerNorm_bias
    attention_output_48 = mul_79 + l__mod___mobilebert_encoder_layer_9_ffn_2_output_layer_norm_bias_1;  mul_79 = l__mod___mobilebert_encoder_layer_9_ffn_2_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_87 = self.L__mod___mobilebert_encoder_layer_9_intermediate_dense(attention_output_48)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_39 = self.L__mod___mobilebert_encoder_layer_9_intermediate_intermediate_act_fn(hidden_states_87);  hidden_states_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    layer_output_36 = self.L__mod___mobilebert_encoder_layer_9_output_dense(intermediate_output_39);  intermediate_output_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_149 = layer_output_36 + attention_output_48;  layer_output_36 = attention_output_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_9_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_9_output_LayerNorm_weight
    mul_80 = add_149 * l__mod___mobilebert_encoder_layer_9_output_layer_norm_weight_1;  add_149 = l__mod___mobilebert_encoder_layer_9_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_9_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_9_output_LayerNorm_bias
    layer_output_37 = mul_80 + l__mod___mobilebert_encoder_layer_9_output_layer_norm_bias_1;  mul_80 = l__mod___mobilebert_encoder_layer_9_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_137 = self.L__mod___mobilebert_encoder_layer_9_output_bottleneck_dense(layer_output_37);  layer_output_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    layer_outputs_138 = self.L__mod___mobilebert_encoder_layer_9_output_bottleneck_dropout(layer_outputs_137);  layer_outputs_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_151 = layer_outputs_138 + value_tensor_9;  layer_outputs_138 = value_tensor_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_9_output_bottleneck_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_9_output_bottleneck_LayerNorm_weight
    mul_81 = add_151 * l__mod___mobilebert_encoder_layer_9_output_bottleneck_layer_norm_weight_1;  add_151 = l__mod___mobilebert_encoder_layer_9_output_bottleneck_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_9_output_bottleneck_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_9_output_bottleneck_LayerNorm_bias
    value_tensor_10 = mul_81 + l__mod___mobilebert_encoder_layer_9_output_bottleneck_layer_norm_bias_1;  mul_81 = l__mod___mobilebert_encoder_layer_9_output_bottleneck_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:549, code: torch.tensor(1000),
    tensor_9 = torch.tensor(1000)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_50 = self.L__mod___mobilebert_encoder_layer_10_bottleneck_input_dense(value_tensor_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_10_bottleneck_input_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_10_bottleneck_input_LayerNorm_weight
    mul_82 = layer_input_50 * l__mod___mobilebert_encoder_layer_10_bottleneck_input_layer_norm_weight_1;  layer_input_50 = l__mod___mobilebert_encoder_layer_10_bottleneck_input_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_10_bottleneck_input_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_10_bottleneck_input_LayerNorm_bias
    layer_input_54 = mul_82 + l__mod___mobilebert_encoder_layer_10_bottleneck_input_layer_norm_bias_1;  mul_82 = l__mod___mobilebert_encoder_layer_10_bottleneck_input_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_52 = self.L__mod___mobilebert_encoder_layer_10_bottleneck_attention_dense(value_tensor_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_10_bottleneck_attention_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_10_bottleneck_attention_LayerNorm_weight
    mul_83 = layer_input_52 * l__mod___mobilebert_encoder_layer_10_bottleneck_attention_layer_norm_weight_1;  layer_input_52 = l__mod___mobilebert_encoder_layer_10_bottleneck_attention_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_10_bottleneck_attention_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_10_bottleneck_attention_LayerNorm_bias
    key_tensor_10 = mul_83 + l__mod___mobilebert_encoder_layer_10_bottleneck_attention_layer_norm_bias_1;  mul_83 = l__mod___mobilebert_encoder_layer_10_bottleneck_attention_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    mixed_query_layer_10 = self.L__mod___mobilebert_encoder_layer_10_attention_self_query(key_tensor_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    mixed_key_layer_10 = self.L__mod___mobilebert_encoder_layer_10_attention_self_key(key_tensor_10);  key_tensor_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    mixed_value_layer_10 = self.L__mod___mobilebert_encoder_layer_10_attention_self_value(value_tensor_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_30 = mixed_query_layer_10.view((1, 128, 4, 32));  mixed_query_layer_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    query_layer_10 = x_30.permute(0, 2, 1, 3);  x_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_31 = mixed_key_layer_10.view((1, 128, 4, 32));  mixed_key_layer_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    key_layer_10 = x_31.permute(0, 2, 1, 3);  x_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_32 = mixed_value_layer_10.view((1, 128, 4, 32));  mixed_value_layer_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    value_layer_10 = x_32.permute(0, 2, 1, 3);  x_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:286, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_10 = key_layer_10.transpose(-1, -2);  key_layer_10 = None
    attention_scores_30 = torch.matmul(query_layer_10, transpose_10);  query_layer_10 = transpose_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:287, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_31 = attention_scores_30 / 5.656854249492381;  attention_scores_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:290, code: attention_scores = attention_scores + attention_mask
    attention_scores_32 = attention_scores_31 + extended_attention_mask_3;  attention_scores_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:292, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_20 = torch.nn.functional.softmax(attention_scores_32, dim = -1);  attention_scores_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295, code: attention_probs = self.dropout(attention_probs)
    attention_probs_21 = self.L__mod___mobilebert_encoder_layer_10_attention_self_dropout(attention_probs_20);  attention_probs_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:299, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_30 = torch.matmul(attention_probs_21, value_layer_10);  attention_probs_21 = value_layer_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_43 = context_layer_30.permute(0, 2, 1, 3);  context_layer_30 = None
    context_layer_31 = permute_43.contiguous();  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    context_layer_32 = context_layer_31.view((1, 128, 128));  context_layer_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_140 = self.L__mod___mobilebert_encoder_layer_10_attention_output_dense(context_layer_32);  context_layer_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_156 = layer_outputs_140 + layer_input_54;  layer_outputs_140 = layer_input_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_10_attention_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_10_attention_output_LayerNorm_weight
    mul_84 = add_156 * l__mod___mobilebert_encoder_layer_10_attention_output_layer_norm_weight_1;  add_156 = l__mod___mobilebert_encoder_layer_10_attention_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_10_attention_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_10_attention_output_LayerNorm_bias
    attention_output_50 = mul_84 + l__mod___mobilebert_encoder_layer_10_attention_output_layer_norm_bias_1;  mul_84 = l__mod___mobilebert_encoder_layer_10_attention_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_90 = self.L__mod___mobilebert_encoder_layer_10_ffn_0_intermediate_dense(attention_output_50)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_40 = self.L__mod___mobilebert_encoder_layer_10_ffn_0_intermediate_intermediate_act_fn(hidden_states_90);  hidden_states_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_142 = self.L__mod___mobilebert_encoder_layer_10_ffn_0_output_dense(intermediate_output_40);  intermediate_output_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_158 = layer_outputs_142 + attention_output_50;  layer_outputs_142 = attention_output_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_10_ffn_0_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_10_ffn_0_output_LayerNorm_weight
    mul_85 = add_158 * l__mod___mobilebert_encoder_layer_10_ffn_0_output_layer_norm_weight_1;  add_158 = l__mod___mobilebert_encoder_layer_10_ffn_0_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_10_ffn_0_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_10_ffn_0_output_LayerNorm_bias
    attention_output_51 = mul_85 + l__mod___mobilebert_encoder_layer_10_ffn_0_output_layer_norm_bias_1;  mul_85 = l__mod___mobilebert_encoder_layer_10_ffn_0_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_92 = self.L__mod___mobilebert_encoder_layer_10_ffn_1_intermediate_dense(attention_output_51)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_41 = self.L__mod___mobilebert_encoder_layer_10_ffn_1_intermediate_intermediate_act_fn(hidden_states_92);  hidden_states_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_145 = self.L__mod___mobilebert_encoder_layer_10_ffn_1_output_dense(intermediate_output_41);  intermediate_output_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_160 = layer_outputs_145 + attention_output_51;  layer_outputs_145 = attention_output_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_10_ffn_1_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_10_ffn_1_output_LayerNorm_weight
    mul_86 = add_160 * l__mod___mobilebert_encoder_layer_10_ffn_1_output_layer_norm_weight_1;  add_160 = l__mod___mobilebert_encoder_layer_10_ffn_1_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_10_ffn_1_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_10_ffn_1_output_LayerNorm_bias
    attention_output_52 = mul_86 + l__mod___mobilebert_encoder_layer_10_ffn_1_output_layer_norm_bias_1;  mul_86 = l__mod___mobilebert_encoder_layer_10_ffn_1_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_94 = self.L__mod___mobilebert_encoder_layer_10_ffn_2_intermediate_dense(attention_output_52)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_42 = self.L__mod___mobilebert_encoder_layer_10_ffn_2_intermediate_intermediate_act_fn(hidden_states_94);  hidden_states_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_148 = self.L__mod___mobilebert_encoder_layer_10_ffn_2_output_dense(intermediate_output_42);  intermediate_output_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_162 = layer_outputs_148 + attention_output_52;  layer_outputs_148 = attention_output_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_10_ffn_2_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_10_ffn_2_output_LayerNorm_weight
    mul_87 = add_162 * l__mod___mobilebert_encoder_layer_10_ffn_2_output_layer_norm_weight_1;  add_162 = l__mod___mobilebert_encoder_layer_10_ffn_2_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_10_ffn_2_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_10_ffn_2_output_LayerNorm_bias
    attention_output_53 = mul_87 + l__mod___mobilebert_encoder_layer_10_ffn_2_output_layer_norm_bias_1;  mul_87 = l__mod___mobilebert_encoder_layer_10_ffn_2_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_96 = self.L__mod___mobilebert_encoder_layer_10_intermediate_dense(attention_output_53)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_43 = self.L__mod___mobilebert_encoder_layer_10_intermediate_intermediate_act_fn(hidden_states_96);  hidden_states_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    layer_output_40 = self.L__mod___mobilebert_encoder_layer_10_output_dense(intermediate_output_43);  intermediate_output_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_164 = layer_output_40 + attention_output_53;  layer_output_40 = attention_output_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_10_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_10_output_LayerNorm_weight
    mul_88 = add_164 * l__mod___mobilebert_encoder_layer_10_output_layer_norm_weight_1;  add_164 = l__mod___mobilebert_encoder_layer_10_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_10_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_10_output_LayerNorm_bias
    layer_output_41 = mul_88 + l__mod___mobilebert_encoder_layer_10_output_layer_norm_bias_1;  mul_88 = l__mod___mobilebert_encoder_layer_10_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_151 = self.L__mod___mobilebert_encoder_layer_10_output_bottleneck_dense(layer_output_41);  layer_output_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    layer_outputs_152 = self.L__mod___mobilebert_encoder_layer_10_output_bottleneck_dropout(layer_outputs_151);  layer_outputs_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_166 = layer_outputs_152 + value_tensor_10;  layer_outputs_152 = value_tensor_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_10_output_bottleneck_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_10_output_bottleneck_LayerNorm_weight
    mul_89 = add_166 * l__mod___mobilebert_encoder_layer_10_output_bottleneck_layer_norm_weight_1;  add_166 = l__mod___mobilebert_encoder_layer_10_output_bottleneck_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_10_output_bottleneck_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_10_output_bottleneck_LayerNorm_bias
    value_tensor_11 = mul_89 + l__mod___mobilebert_encoder_layer_10_output_bottleneck_layer_norm_bias_1;  mul_89 = l__mod___mobilebert_encoder_layer_10_output_bottleneck_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:549, code: torch.tensor(1000),
    tensor_10 = torch.tensor(1000)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_55 = self.L__mod___mobilebert_encoder_layer_11_bottleneck_input_dense(value_tensor_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_11_bottleneck_input_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_11_bottleneck_input_LayerNorm_weight
    mul_90 = layer_input_55 * l__mod___mobilebert_encoder_layer_11_bottleneck_input_layer_norm_weight_1;  layer_input_55 = l__mod___mobilebert_encoder_layer_11_bottleneck_input_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_11_bottleneck_input_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_11_bottleneck_input_LayerNorm_bias
    layer_input_59 = mul_90 + l__mod___mobilebert_encoder_layer_11_bottleneck_input_layer_norm_bias_1;  mul_90 = l__mod___mobilebert_encoder_layer_11_bottleneck_input_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_57 = self.L__mod___mobilebert_encoder_layer_11_bottleneck_attention_dense(value_tensor_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_11_bottleneck_attention_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_11_bottleneck_attention_LayerNorm_weight
    mul_91 = layer_input_57 * l__mod___mobilebert_encoder_layer_11_bottleneck_attention_layer_norm_weight_1;  layer_input_57 = l__mod___mobilebert_encoder_layer_11_bottleneck_attention_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_11_bottleneck_attention_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_11_bottleneck_attention_LayerNorm_bias
    key_tensor_11 = mul_91 + l__mod___mobilebert_encoder_layer_11_bottleneck_attention_layer_norm_bias_1;  mul_91 = l__mod___mobilebert_encoder_layer_11_bottleneck_attention_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    mixed_query_layer_11 = self.L__mod___mobilebert_encoder_layer_11_attention_self_query(key_tensor_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    mixed_key_layer_11 = self.L__mod___mobilebert_encoder_layer_11_attention_self_key(key_tensor_11);  key_tensor_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    mixed_value_layer_11 = self.L__mod___mobilebert_encoder_layer_11_attention_self_value(value_tensor_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_33 = mixed_query_layer_11.view((1, 128, 4, 32));  mixed_query_layer_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    query_layer_11 = x_33.permute(0, 2, 1, 3);  x_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_34 = mixed_key_layer_11.view((1, 128, 4, 32));  mixed_key_layer_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    key_layer_11 = x_34.permute(0, 2, 1, 3);  x_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_35 = mixed_value_layer_11.view((1, 128, 4, 32));  mixed_value_layer_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    value_layer_11 = x_35.permute(0, 2, 1, 3);  x_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:286, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_11 = key_layer_11.transpose(-1, -2);  key_layer_11 = None
    attention_scores_33 = torch.matmul(query_layer_11, transpose_11);  query_layer_11 = transpose_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:287, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_34 = attention_scores_33 / 5.656854249492381;  attention_scores_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:290, code: attention_scores = attention_scores + attention_mask
    attention_scores_35 = attention_scores_34 + extended_attention_mask_3;  attention_scores_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:292, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_22 = torch.nn.functional.softmax(attention_scores_35, dim = -1);  attention_scores_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295, code: attention_probs = self.dropout(attention_probs)
    attention_probs_23 = self.L__mod___mobilebert_encoder_layer_11_attention_self_dropout(attention_probs_22);  attention_probs_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:299, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_33 = torch.matmul(attention_probs_23, value_layer_11);  attention_probs_23 = value_layer_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_47 = context_layer_33.permute(0, 2, 1, 3);  context_layer_33 = None
    context_layer_34 = permute_47.contiguous();  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    context_layer_35 = context_layer_34.view((1, 128, 128));  context_layer_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_154 = self.L__mod___mobilebert_encoder_layer_11_attention_output_dense(context_layer_35);  context_layer_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_171 = layer_outputs_154 + layer_input_59;  layer_outputs_154 = layer_input_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_11_attention_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_11_attention_output_LayerNorm_weight
    mul_92 = add_171 * l__mod___mobilebert_encoder_layer_11_attention_output_layer_norm_weight_1;  add_171 = l__mod___mobilebert_encoder_layer_11_attention_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_11_attention_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_11_attention_output_LayerNorm_bias
    attention_output_55 = mul_92 + l__mod___mobilebert_encoder_layer_11_attention_output_layer_norm_bias_1;  mul_92 = l__mod___mobilebert_encoder_layer_11_attention_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_99 = self.L__mod___mobilebert_encoder_layer_11_ffn_0_intermediate_dense(attention_output_55)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_44 = self.L__mod___mobilebert_encoder_layer_11_ffn_0_intermediate_intermediate_act_fn(hidden_states_99);  hidden_states_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_156 = self.L__mod___mobilebert_encoder_layer_11_ffn_0_output_dense(intermediate_output_44);  intermediate_output_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_173 = layer_outputs_156 + attention_output_55;  layer_outputs_156 = attention_output_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_11_ffn_0_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_11_ffn_0_output_LayerNorm_weight
    mul_93 = add_173 * l__mod___mobilebert_encoder_layer_11_ffn_0_output_layer_norm_weight_1;  add_173 = l__mod___mobilebert_encoder_layer_11_ffn_0_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_11_ffn_0_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_11_ffn_0_output_LayerNorm_bias
    attention_output_56 = mul_93 + l__mod___mobilebert_encoder_layer_11_ffn_0_output_layer_norm_bias_1;  mul_93 = l__mod___mobilebert_encoder_layer_11_ffn_0_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_101 = self.L__mod___mobilebert_encoder_layer_11_ffn_1_intermediate_dense(attention_output_56)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_45 = self.L__mod___mobilebert_encoder_layer_11_ffn_1_intermediate_intermediate_act_fn(hidden_states_101);  hidden_states_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_159 = self.L__mod___mobilebert_encoder_layer_11_ffn_1_output_dense(intermediate_output_45);  intermediate_output_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_175 = layer_outputs_159 + attention_output_56;  layer_outputs_159 = attention_output_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_11_ffn_1_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_11_ffn_1_output_LayerNorm_weight
    mul_94 = add_175 * l__mod___mobilebert_encoder_layer_11_ffn_1_output_layer_norm_weight_1;  add_175 = l__mod___mobilebert_encoder_layer_11_ffn_1_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_11_ffn_1_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_11_ffn_1_output_LayerNorm_bias
    attention_output_57 = mul_94 + l__mod___mobilebert_encoder_layer_11_ffn_1_output_layer_norm_bias_1;  mul_94 = l__mod___mobilebert_encoder_layer_11_ffn_1_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_103 = self.L__mod___mobilebert_encoder_layer_11_ffn_2_intermediate_dense(attention_output_57)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_46 = self.L__mod___mobilebert_encoder_layer_11_ffn_2_intermediate_intermediate_act_fn(hidden_states_103);  hidden_states_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_162 = self.L__mod___mobilebert_encoder_layer_11_ffn_2_output_dense(intermediate_output_46);  intermediate_output_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_177 = layer_outputs_162 + attention_output_57;  layer_outputs_162 = attention_output_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_11_ffn_2_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_11_ffn_2_output_LayerNorm_weight
    mul_95 = add_177 * l__mod___mobilebert_encoder_layer_11_ffn_2_output_layer_norm_weight_1;  add_177 = l__mod___mobilebert_encoder_layer_11_ffn_2_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_11_ffn_2_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_11_ffn_2_output_LayerNorm_bias
    attention_output_58 = mul_95 + l__mod___mobilebert_encoder_layer_11_ffn_2_output_layer_norm_bias_1;  mul_95 = l__mod___mobilebert_encoder_layer_11_ffn_2_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_105 = self.L__mod___mobilebert_encoder_layer_11_intermediate_dense(attention_output_58)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_47 = self.L__mod___mobilebert_encoder_layer_11_intermediate_intermediate_act_fn(hidden_states_105);  hidden_states_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    layer_output_44 = self.L__mod___mobilebert_encoder_layer_11_output_dense(intermediate_output_47);  intermediate_output_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_179 = layer_output_44 + attention_output_58;  layer_output_44 = attention_output_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_11_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_11_output_LayerNorm_weight
    mul_96 = add_179 * l__mod___mobilebert_encoder_layer_11_output_layer_norm_weight_1;  add_179 = l__mod___mobilebert_encoder_layer_11_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_11_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_11_output_LayerNorm_bias
    layer_output_45 = mul_96 + l__mod___mobilebert_encoder_layer_11_output_layer_norm_bias_1;  mul_96 = l__mod___mobilebert_encoder_layer_11_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_165 = self.L__mod___mobilebert_encoder_layer_11_output_bottleneck_dense(layer_output_45);  layer_output_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    layer_outputs_166 = self.L__mod___mobilebert_encoder_layer_11_output_bottleneck_dropout(layer_outputs_165);  layer_outputs_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_181 = layer_outputs_166 + value_tensor_11;  layer_outputs_166 = value_tensor_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_11_output_bottleneck_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_11_output_bottleneck_LayerNorm_weight
    mul_97 = add_181 * l__mod___mobilebert_encoder_layer_11_output_bottleneck_layer_norm_weight_1;  add_181 = l__mod___mobilebert_encoder_layer_11_output_bottleneck_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_11_output_bottleneck_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_11_output_bottleneck_LayerNorm_bias
    value_tensor_12 = mul_97 + l__mod___mobilebert_encoder_layer_11_output_bottleneck_layer_norm_bias_1;  mul_97 = l__mod___mobilebert_encoder_layer_11_output_bottleneck_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:549, code: torch.tensor(1000),
    tensor_11 = torch.tensor(1000)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_60 = self.L__mod___mobilebert_encoder_layer_12_bottleneck_input_dense(value_tensor_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_12_bottleneck_input_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_12_bottleneck_input_LayerNorm_weight
    mul_98 = layer_input_60 * l__mod___mobilebert_encoder_layer_12_bottleneck_input_layer_norm_weight_1;  layer_input_60 = l__mod___mobilebert_encoder_layer_12_bottleneck_input_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_12_bottleneck_input_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_12_bottleneck_input_LayerNorm_bias
    layer_input_64 = mul_98 + l__mod___mobilebert_encoder_layer_12_bottleneck_input_layer_norm_bias_1;  mul_98 = l__mod___mobilebert_encoder_layer_12_bottleneck_input_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_62 = self.L__mod___mobilebert_encoder_layer_12_bottleneck_attention_dense(value_tensor_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_12_bottleneck_attention_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_12_bottleneck_attention_LayerNorm_weight
    mul_99 = layer_input_62 * l__mod___mobilebert_encoder_layer_12_bottleneck_attention_layer_norm_weight_1;  layer_input_62 = l__mod___mobilebert_encoder_layer_12_bottleneck_attention_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_12_bottleneck_attention_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_12_bottleneck_attention_LayerNorm_bias
    key_tensor_12 = mul_99 + l__mod___mobilebert_encoder_layer_12_bottleneck_attention_layer_norm_bias_1;  mul_99 = l__mod___mobilebert_encoder_layer_12_bottleneck_attention_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    mixed_query_layer_12 = self.L__mod___mobilebert_encoder_layer_12_attention_self_query(key_tensor_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    mixed_key_layer_12 = self.L__mod___mobilebert_encoder_layer_12_attention_self_key(key_tensor_12);  key_tensor_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    mixed_value_layer_12 = self.L__mod___mobilebert_encoder_layer_12_attention_self_value(value_tensor_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_36 = mixed_query_layer_12.view((1, 128, 4, 32));  mixed_query_layer_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    query_layer_12 = x_36.permute(0, 2, 1, 3);  x_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_37 = mixed_key_layer_12.view((1, 128, 4, 32));  mixed_key_layer_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    key_layer_12 = x_37.permute(0, 2, 1, 3);  x_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_38 = mixed_value_layer_12.view((1, 128, 4, 32));  mixed_value_layer_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    value_layer_12 = x_38.permute(0, 2, 1, 3);  x_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:286, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_12 = key_layer_12.transpose(-1, -2);  key_layer_12 = None
    attention_scores_36 = torch.matmul(query_layer_12, transpose_12);  query_layer_12 = transpose_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:287, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_37 = attention_scores_36 / 5.656854249492381;  attention_scores_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:290, code: attention_scores = attention_scores + attention_mask
    attention_scores_38 = attention_scores_37 + extended_attention_mask_3;  attention_scores_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:292, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_24 = torch.nn.functional.softmax(attention_scores_38, dim = -1);  attention_scores_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295, code: attention_probs = self.dropout(attention_probs)
    attention_probs_25 = self.L__mod___mobilebert_encoder_layer_12_attention_self_dropout(attention_probs_24);  attention_probs_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:299, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_36 = torch.matmul(attention_probs_25, value_layer_12);  attention_probs_25 = value_layer_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_51 = context_layer_36.permute(0, 2, 1, 3);  context_layer_36 = None
    context_layer_37 = permute_51.contiguous();  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    context_layer_38 = context_layer_37.view((1, 128, 128));  context_layer_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_168 = self.L__mod___mobilebert_encoder_layer_12_attention_output_dense(context_layer_38);  context_layer_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_186 = layer_outputs_168 + layer_input_64;  layer_outputs_168 = layer_input_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_12_attention_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_12_attention_output_LayerNorm_weight
    mul_100 = add_186 * l__mod___mobilebert_encoder_layer_12_attention_output_layer_norm_weight_1;  add_186 = l__mod___mobilebert_encoder_layer_12_attention_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_12_attention_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_12_attention_output_LayerNorm_bias
    attention_output_60 = mul_100 + l__mod___mobilebert_encoder_layer_12_attention_output_layer_norm_bias_1;  mul_100 = l__mod___mobilebert_encoder_layer_12_attention_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_108 = self.L__mod___mobilebert_encoder_layer_12_ffn_0_intermediate_dense(attention_output_60)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_48 = self.L__mod___mobilebert_encoder_layer_12_ffn_0_intermediate_intermediate_act_fn(hidden_states_108);  hidden_states_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_170 = self.L__mod___mobilebert_encoder_layer_12_ffn_0_output_dense(intermediate_output_48);  intermediate_output_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_188 = layer_outputs_170 + attention_output_60;  layer_outputs_170 = attention_output_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_12_ffn_0_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_12_ffn_0_output_LayerNorm_weight
    mul_101 = add_188 * l__mod___mobilebert_encoder_layer_12_ffn_0_output_layer_norm_weight_1;  add_188 = l__mod___mobilebert_encoder_layer_12_ffn_0_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_12_ffn_0_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_12_ffn_0_output_LayerNorm_bias
    attention_output_61 = mul_101 + l__mod___mobilebert_encoder_layer_12_ffn_0_output_layer_norm_bias_1;  mul_101 = l__mod___mobilebert_encoder_layer_12_ffn_0_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_110 = self.L__mod___mobilebert_encoder_layer_12_ffn_1_intermediate_dense(attention_output_61)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_49 = self.L__mod___mobilebert_encoder_layer_12_ffn_1_intermediate_intermediate_act_fn(hidden_states_110);  hidden_states_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_173 = self.L__mod___mobilebert_encoder_layer_12_ffn_1_output_dense(intermediate_output_49);  intermediate_output_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_190 = layer_outputs_173 + attention_output_61;  layer_outputs_173 = attention_output_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_12_ffn_1_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_12_ffn_1_output_LayerNorm_weight
    mul_102 = add_190 * l__mod___mobilebert_encoder_layer_12_ffn_1_output_layer_norm_weight_1;  add_190 = l__mod___mobilebert_encoder_layer_12_ffn_1_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_12_ffn_1_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_12_ffn_1_output_LayerNorm_bias
    attention_output_62 = mul_102 + l__mod___mobilebert_encoder_layer_12_ffn_1_output_layer_norm_bias_1;  mul_102 = l__mod___mobilebert_encoder_layer_12_ffn_1_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_112 = self.L__mod___mobilebert_encoder_layer_12_ffn_2_intermediate_dense(attention_output_62)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_50 = self.L__mod___mobilebert_encoder_layer_12_ffn_2_intermediate_intermediate_act_fn(hidden_states_112);  hidden_states_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_176 = self.L__mod___mobilebert_encoder_layer_12_ffn_2_output_dense(intermediate_output_50);  intermediate_output_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_192 = layer_outputs_176 + attention_output_62;  layer_outputs_176 = attention_output_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_12_ffn_2_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_12_ffn_2_output_LayerNorm_weight
    mul_103 = add_192 * l__mod___mobilebert_encoder_layer_12_ffn_2_output_layer_norm_weight_1;  add_192 = l__mod___mobilebert_encoder_layer_12_ffn_2_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_12_ffn_2_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_12_ffn_2_output_LayerNorm_bias
    attention_output_63 = mul_103 + l__mod___mobilebert_encoder_layer_12_ffn_2_output_layer_norm_bias_1;  mul_103 = l__mod___mobilebert_encoder_layer_12_ffn_2_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_114 = self.L__mod___mobilebert_encoder_layer_12_intermediate_dense(attention_output_63)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_51 = self.L__mod___mobilebert_encoder_layer_12_intermediate_intermediate_act_fn(hidden_states_114);  hidden_states_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    layer_output_48 = self.L__mod___mobilebert_encoder_layer_12_output_dense(intermediate_output_51);  intermediate_output_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_194 = layer_output_48 + attention_output_63;  layer_output_48 = attention_output_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_12_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_12_output_LayerNorm_weight
    mul_104 = add_194 * l__mod___mobilebert_encoder_layer_12_output_layer_norm_weight_1;  add_194 = l__mod___mobilebert_encoder_layer_12_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_12_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_12_output_LayerNorm_bias
    layer_output_49 = mul_104 + l__mod___mobilebert_encoder_layer_12_output_layer_norm_bias_1;  mul_104 = l__mod___mobilebert_encoder_layer_12_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_179 = self.L__mod___mobilebert_encoder_layer_12_output_bottleneck_dense(layer_output_49);  layer_output_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    layer_outputs_180 = self.L__mod___mobilebert_encoder_layer_12_output_bottleneck_dropout(layer_outputs_179);  layer_outputs_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_196 = layer_outputs_180 + value_tensor_12;  layer_outputs_180 = value_tensor_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_12_output_bottleneck_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_12_output_bottleneck_LayerNorm_weight
    mul_105 = add_196 * l__mod___mobilebert_encoder_layer_12_output_bottleneck_layer_norm_weight_1;  add_196 = l__mod___mobilebert_encoder_layer_12_output_bottleneck_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_12_output_bottleneck_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_12_output_bottleneck_LayerNorm_bias
    value_tensor_13 = mul_105 + l__mod___mobilebert_encoder_layer_12_output_bottleneck_layer_norm_bias_1;  mul_105 = l__mod___mobilebert_encoder_layer_12_output_bottleneck_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:549, code: torch.tensor(1000),
    tensor_12 = torch.tensor(1000)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_65 = self.L__mod___mobilebert_encoder_layer_13_bottleneck_input_dense(value_tensor_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_13_bottleneck_input_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_13_bottleneck_input_LayerNorm_weight
    mul_106 = layer_input_65 * l__mod___mobilebert_encoder_layer_13_bottleneck_input_layer_norm_weight_1;  layer_input_65 = l__mod___mobilebert_encoder_layer_13_bottleneck_input_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_13_bottleneck_input_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_13_bottleneck_input_LayerNorm_bias
    layer_input_69 = mul_106 + l__mod___mobilebert_encoder_layer_13_bottleneck_input_layer_norm_bias_1;  mul_106 = l__mod___mobilebert_encoder_layer_13_bottleneck_input_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_67 = self.L__mod___mobilebert_encoder_layer_13_bottleneck_attention_dense(value_tensor_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_13_bottleneck_attention_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_13_bottleneck_attention_LayerNorm_weight
    mul_107 = layer_input_67 * l__mod___mobilebert_encoder_layer_13_bottleneck_attention_layer_norm_weight_1;  layer_input_67 = l__mod___mobilebert_encoder_layer_13_bottleneck_attention_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_13_bottleneck_attention_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_13_bottleneck_attention_LayerNorm_bias
    key_tensor_13 = mul_107 + l__mod___mobilebert_encoder_layer_13_bottleneck_attention_layer_norm_bias_1;  mul_107 = l__mod___mobilebert_encoder_layer_13_bottleneck_attention_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    mixed_query_layer_13 = self.L__mod___mobilebert_encoder_layer_13_attention_self_query(key_tensor_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    mixed_key_layer_13 = self.L__mod___mobilebert_encoder_layer_13_attention_self_key(key_tensor_13);  key_tensor_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    mixed_value_layer_13 = self.L__mod___mobilebert_encoder_layer_13_attention_self_value(value_tensor_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_39 = mixed_query_layer_13.view((1, 128, 4, 32));  mixed_query_layer_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    query_layer_13 = x_39.permute(0, 2, 1, 3);  x_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_40 = mixed_key_layer_13.view((1, 128, 4, 32));  mixed_key_layer_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    key_layer_13 = x_40.permute(0, 2, 1, 3);  x_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_41 = mixed_value_layer_13.view((1, 128, 4, 32));  mixed_value_layer_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    value_layer_13 = x_41.permute(0, 2, 1, 3);  x_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:286, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_13 = key_layer_13.transpose(-1, -2);  key_layer_13 = None
    attention_scores_39 = torch.matmul(query_layer_13, transpose_13);  query_layer_13 = transpose_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:287, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_40 = attention_scores_39 / 5.656854249492381;  attention_scores_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:290, code: attention_scores = attention_scores + attention_mask
    attention_scores_41 = attention_scores_40 + extended_attention_mask_3;  attention_scores_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:292, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_26 = torch.nn.functional.softmax(attention_scores_41, dim = -1);  attention_scores_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295, code: attention_probs = self.dropout(attention_probs)
    attention_probs_27 = self.L__mod___mobilebert_encoder_layer_13_attention_self_dropout(attention_probs_26);  attention_probs_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:299, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_39 = torch.matmul(attention_probs_27, value_layer_13);  attention_probs_27 = value_layer_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_55 = context_layer_39.permute(0, 2, 1, 3);  context_layer_39 = None
    context_layer_40 = permute_55.contiguous();  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    context_layer_41 = context_layer_40.view((1, 128, 128));  context_layer_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_182 = self.L__mod___mobilebert_encoder_layer_13_attention_output_dense(context_layer_41);  context_layer_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_201 = layer_outputs_182 + layer_input_69;  layer_outputs_182 = layer_input_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_13_attention_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_13_attention_output_LayerNorm_weight
    mul_108 = add_201 * l__mod___mobilebert_encoder_layer_13_attention_output_layer_norm_weight_1;  add_201 = l__mod___mobilebert_encoder_layer_13_attention_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_13_attention_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_13_attention_output_LayerNorm_bias
    attention_output_65 = mul_108 + l__mod___mobilebert_encoder_layer_13_attention_output_layer_norm_bias_1;  mul_108 = l__mod___mobilebert_encoder_layer_13_attention_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_117 = self.L__mod___mobilebert_encoder_layer_13_ffn_0_intermediate_dense(attention_output_65)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_52 = self.L__mod___mobilebert_encoder_layer_13_ffn_0_intermediate_intermediate_act_fn(hidden_states_117);  hidden_states_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_184 = self.L__mod___mobilebert_encoder_layer_13_ffn_0_output_dense(intermediate_output_52);  intermediate_output_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_203 = layer_outputs_184 + attention_output_65;  layer_outputs_184 = attention_output_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_13_ffn_0_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_13_ffn_0_output_LayerNorm_weight
    mul_109 = add_203 * l__mod___mobilebert_encoder_layer_13_ffn_0_output_layer_norm_weight_1;  add_203 = l__mod___mobilebert_encoder_layer_13_ffn_0_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_13_ffn_0_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_13_ffn_0_output_LayerNorm_bias
    attention_output_66 = mul_109 + l__mod___mobilebert_encoder_layer_13_ffn_0_output_layer_norm_bias_1;  mul_109 = l__mod___mobilebert_encoder_layer_13_ffn_0_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_119 = self.L__mod___mobilebert_encoder_layer_13_ffn_1_intermediate_dense(attention_output_66)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_53 = self.L__mod___mobilebert_encoder_layer_13_ffn_1_intermediate_intermediate_act_fn(hidden_states_119);  hidden_states_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_187 = self.L__mod___mobilebert_encoder_layer_13_ffn_1_output_dense(intermediate_output_53);  intermediate_output_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_205 = layer_outputs_187 + attention_output_66;  layer_outputs_187 = attention_output_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_13_ffn_1_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_13_ffn_1_output_LayerNorm_weight
    mul_110 = add_205 * l__mod___mobilebert_encoder_layer_13_ffn_1_output_layer_norm_weight_1;  add_205 = l__mod___mobilebert_encoder_layer_13_ffn_1_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_13_ffn_1_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_13_ffn_1_output_LayerNorm_bias
    attention_output_67 = mul_110 + l__mod___mobilebert_encoder_layer_13_ffn_1_output_layer_norm_bias_1;  mul_110 = l__mod___mobilebert_encoder_layer_13_ffn_1_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_121 = self.L__mod___mobilebert_encoder_layer_13_ffn_2_intermediate_dense(attention_output_67)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_54 = self.L__mod___mobilebert_encoder_layer_13_ffn_2_intermediate_intermediate_act_fn(hidden_states_121);  hidden_states_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_190 = self.L__mod___mobilebert_encoder_layer_13_ffn_2_output_dense(intermediate_output_54);  intermediate_output_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_207 = layer_outputs_190 + attention_output_67;  layer_outputs_190 = attention_output_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_13_ffn_2_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_13_ffn_2_output_LayerNorm_weight
    mul_111 = add_207 * l__mod___mobilebert_encoder_layer_13_ffn_2_output_layer_norm_weight_1;  add_207 = l__mod___mobilebert_encoder_layer_13_ffn_2_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_13_ffn_2_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_13_ffn_2_output_LayerNorm_bias
    attention_output_68 = mul_111 + l__mod___mobilebert_encoder_layer_13_ffn_2_output_layer_norm_bias_1;  mul_111 = l__mod___mobilebert_encoder_layer_13_ffn_2_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_123 = self.L__mod___mobilebert_encoder_layer_13_intermediate_dense(attention_output_68)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_55 = self.L__mod___mobilebert_encoder_layer_13_intermediate_intermediate_act_fn(hidden_states_123);  hidden_states_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    layer_output_52 = self.L__mod___mobilebert_encoder_layer_13_output_dense(intermediate_output_55);  intermediate_output_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_209 = layer_output_52 + attention_output_68;  layer_output_52 = attention_output_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_13_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_13_output_LayerNorm_weight
    mul_112 = add_209 * l__mod___mobilebert_encoder_layer_13_output_layer_norm_weight_1;  add_209 = l__mod___mobilebert_encoder_layer_13_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_13_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_13_output_LayerNorm_bias
    layer_output_53 = mul_112 + l__mod___mobilebert_encoder_layer_13_output_layer_norm_bias_1;  mul_112 = l__mod___mobilebert_encoder_layer_13_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_193 = self.L__mod___mobilebert_encoder_layer_13_output_bottleneck_dense(layer_output_53);  layer_output_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    layer_outputs_194 = self.L__mod___mobilebert_encoder_layer_13_output_bottleneck_dropout(layer_outputs_193);  layer_outputs_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_211 = layer_outputs_194 + value_tensor_13;  layer_outputs_194 = value_tensor_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_13_output_bottleneck_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_13_output_bottleneck_LayerNorm_weight
    mul_113 = add_211 * l__mod___mobilebert_encoder_layer_13_output_bottleneck_layer_norm_weight_1;  add_211 = l__mod___mobilebert_encoder_layer_13_output_bottleneck_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_13_output_bottleneck_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_13_output_bottleneck_LayerNorm_bias
    value_tensor_14 = mul_113 + l__mod___mobilebert_encoder_layer_13_output_bottleneck_layer_norm_bias_1;  mul_113 = l__mod___mobilebert_encoder_layer_13_output_bottleneck_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:549, code: torch.tensor(1000),
    tensor_13 = torch.tensor(1000)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_70 = self.L__mod___mobilebert_encoder_layer_14_bottleneck_input_dense(value_tensor_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_14_bottleneck_input_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_14_bottleneck_input_LayerNorm_weight
    mul_114 = layer_input_70 * l__mod___mobilebert_encoder_layer_14_bottleneck_input_layer_norm_weight_1;  layer_input_70 = l__mod___mobilebert_encoder_layer_14_bottleneck_input_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_14_bottleneck_input_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_14_bottleneck_input_LayerNorm_bias
    layer_input_74 = mul_114 + l__mod___mobilebert_encoder_layer_14_bottleneck_input_layer_norm_bias_1;  mul_114 = l__mod___mobilebert_encoder_layer_14_bottleneck_input_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_72 = self.L__mod___mobilebert_encoder_layer_14_bottleneck_attention_dense(value_tensor_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_14_bottleneck_attention_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_14_bottleneck_attention_LayerNorm_weight
    mul_115 = layer_input_72 * l__mod___mobilebert_encoder_layer_14_bottleneck_attention_layer_norm_weight_1;  layer_input_72 = l__mod___mobilebert_encoder_layer_14_bottleneck_attention_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_14_bottleneck_attention_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_14_bottleneck_attention_LayerNorm_bias
    key_tensor_14 = mul_115 + l__mod___mobilebert_encoder_layer_14_bottleneck_attention_layer_norm_bias_1;  mul_115 = l__mod___mobilebert_encoder_layer_14_bottleneck_attention_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    mixed_query_layer_14 = self.L__mod___mobilebert_encoder_layer_14_attention_self_query(key_tensor_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    mixed_key_layer_14 = self.L__mod___mobilebert_encoder_layer_14_attention_self_key(key_tensor_14);  key_tensor_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    mixed_value_layer_14 = self.L__mod___mobilebert_encoder_layer_14_attention_self_value(value_tensor_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_42 = mixed_query_layer_14.view((1, 128, 4, 32));  mixed_query_layer_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    query_layer_14 = x_42.permute(0, 2, 1, 3);  x_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_43 = mixed_key_layer_14.view((1, 128, 4, 32));  mixed_key_layer_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    key_layer_14 = x_43.permute(0, 2, 1, 3);  x_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_44 = mixed_value_layer_14.view((1, 128, 4, 32));  mixed_value_layer_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    value_layer_14 = x_44.permute(0, 2, 1, 3);  x_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:286, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_14 = key_layer_14.transpose(-1, -2);  key_layer_14 = None
    attention_scores_42 = torch.matmul(query_layer_14, transpose_14);  query_layer_14 = transpose_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:287, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_43 = attention_scores_42 / 5.656854249492381;  attention_scores_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:290, code: attention_scores = attention_scores + attention_mask
    attention_scores_44 = attention_scores_43 + extended_attention_mask_3;  attention_scores_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:292, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_28 = torch.nn.functional.softmax(attention_scores_44, dim = -1);  attention_scores_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295, code: attention_probs = self.dropout(attention_probs)
    attention_probs_29 = self.L__mod___mobilebert_encoder_layer_14_attention_self_dropout(attention_probs_28);  attention_probs_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:299, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_42 = torch.matmul(attention_probs_29, value_layer_14);  attention_probs_29 = value_layer_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_59 = context_layer_42.permute(0, 2, 1, 3);  context_layer_42 = None
    context_layer_43 = permute_59.contiguous();  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    context_layer_44 = context_layer_43.view((1, 128, 128));  context_layer_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_196 = self.L__mod___mobilebert_encoder_layer_14_attention_output_dense(context_layer_44);  context_layer_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_216 = layer_outputs_196 + layer_input_74;  layer_outputs_196 = layer_input_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_14_attention_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_14_attention_output_LayerNorm_weight
    mul_116 = add_216 * l__mod___mobilebert_encoder_layer_14_attention_output_layer_norm_weight_1;  add_216 = l__mod___mobilebert_encoder_layer_14_attention_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_14_attention_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_14_attention_output_LayerNorm_bias
    attention_output_70 = mul_116 + l__mod___mobilebert_encoder_layer_14_attention_output_layer_norm_bias_1;  mul_116 = l__mod___mobilebert_encoder_layer_14_attention_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_126 = self.L__mod___mobilebert_encoder_layer_14_ffn_0_intermediate_dense(attention_output_70)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_56 = self.L__mod___mobilebert_encoder_layer_14_ffn_0_intermediate_intermediate_act_fn(hidden_states_126);  hidden_states_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_198 = self.L__mod___mobilebert_encoder_layer_14_ffn_0_output_dense(intermediate_output_56);  intermediate_output_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_218 = layer_outputs_198 + attention_output_70;  layer_outputs_198 = attention_output_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_14_ffn_0_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_14_ffn_0_output_LayerNorm_weight
    mul_117 = add_218 * l__mod___mobilebert_encoder_layer_14_ffn_0_output_layer_norm_weight_1;  add_218 = l__mod___mobilebert_encoder_layer_14_ffn_0_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_14_ffn_0_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_14_ffn_0_output_LayerNorm_bias
    attention_output_71 = mul_117 + l__mod___mobilebert_encoder_layer_14_ffn_0_output_layer_norm_bias_1;  mul_117 = l__mod___mobilebert_encoder_layer_14_ffn_0_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_128 = self.L__mod___mobilebert_encoder_layer_14_ffn_1_intermediate_dense(attention_output_71)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_57 = self.L__mod___mobilebert_encoder_layer_14_ffn_1_intermediate_intermediate_act_fn(hidden_states_128);  hidden_states_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_201 = self.L__mod___mobilebert_encoder_layer_14_ffn_1_output_dense(intermediate_output_57);  intermediate_output_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_220 = layer_outputs_201 + attention_output_71;  layer_outputs_201 = attention_output_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_14_ffn_1_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_14_ffn_1_output_LayerNorm_weight
    mul_118 = add_220 * l__mod___mobilebert_encoder_layer_14_ffn_1_output_layer_norm_weight_1;  add_220 = l__mod___mobilebert_encoder_layer_14_ffn_1_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_14_ffn_1_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_14_ffn_1_output_LayerNorm_bias
    attention_output_72 = mul_118 + l__mod___mobilebert_encoder_layer_14_ffn_1_output_layer_norm_bias_1;  mul_118 = l__mod___mobilebert_encoder_layer_14_ffn_1_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_130 = self.L__mod___mobilebert_encoder_layer_14_ffn_2_intermediate_dense(attention_output_72)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_58 = self.L__mod___mobilebert_encoder_layer_14_ffn_2_intermediate_intermediate_act_fn(hidden_states_130);  hidden_states_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_204 = self.L__mod___mobilebert_encoder_layer_14_ffn_2_output_dense(intermediate_output_58);  intermediate_output_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_222 = layer_outputs_204 + attention_output_72;  layer_outputs_204 = attention_output_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_14_ffn_2_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_14_ffn_2_output_LayerNorm_weight
    mul_119 = add_222 * l__mod___mobilebert_encoder_layer_14_ffn_2_output_layer_norm_weight_1;  add_222 = l__mod___mobilebert_encoder_layer_14_ffn_2_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_14_ffn_2_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_14_ffn_2_output_LayerNorm_bias
    attention_output_73 = mul_119 + l__mod___mobilebert_encoder_layer_14_ffn_2_output_layer_norm_bias_1;  mul_119 = l__mod___mobilebert_encoder_layer_14_ffn_2_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_132 = self.L__mod___mobilebert_encoder_layer_14_intermediate_dense(attention_output_73)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_59 = self.L__mod___mobilebert_encoder_layer_14_intermediate_intermediate_act_fn(hidden_states_132);  hidden_states_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    layer_output_56 = self.L__mod___mobilebert_encoder_layer_14_output_dense(intermediate_output_59);  intermediate_output_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_224 = layer_output_56 + attention_output_73;  layer_output_56 = attention_output_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_14_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_14_output_LayerNorm_weight
    mul_120 = add_224 * l__mod___mobilebert_encoder_layer_14_output_layer_norm_weight_1;  add_224 = l__mod___mobilebert_encoder_layer_14_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_14_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_14_output_LayerNorm_bias
    layer_output_57 = mul_120 + l__mod___mobilebert_encoder_layer_14_output_layer_norm_bias_1;  mul_120 = l__mod___mobilebert_encoder_layer_14_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_207 = self.L__mod___mobilebert_encoder_layer_14_output_bottleneck_dense(layer_output_57);  layer_output_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    layer_outputs_208 = self.L__mod___mobilebert_encoder_layer_14_output_bottleneck_dropout(layer_outputs_207);  layer_outputs_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_226 = layer_outputs_208 + value_tensor_14;  layer_outputs_208 = value_tensor_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_14_output_bottleneck_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_14_output_bottleneck_LayerNorm_weight
    mul_121 = add_226 * l__mod___mobilebert_encoder_layer_14_output_bottleneck_layer_norm_weight_1;  add_226 = l__mod___mobilebert_encoder_layer_14_output_bottleneck_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_14_output_bottleneck_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_14_output_bottleneck_LayerNorm_bias
    value_tensor_15 = mul_121 + l__mod___mobilebert_encoder_layer_14_output_bottleneck_layer_norm_bias_1;  mul_121 = l__mod___mobilebert_encoder_layer_14_output_bottleneck_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:549, code: torch.tensor(1000),
    tensor_14 = torch.tensor(1000)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_75 = self.L__mod___mobilebert_encoder_layer_15_bottleneck_input_dense(value_tensor_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_15_bottleneck_input_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_15_bottleneck_input_LayerNorm_weight
    mul_122 = layer_input_75 * l__mod___mobilebert_encoder_layer_15_bottleneck_input_layer_norm_weight_1;  layer_input_75 = l__mod___mobilebert_encoder_layer_15_bottleneck_input_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_15_bottleneck_input_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_15_bottleneck_input_LayerNorm_bias
    layer_input_79 = mul_122 + l__mod___mobilebert_encoder_layer_15_bottleneck_input_layer_norm_bias_1;  mul_122 = l__mod___mobilebert_encoder_layer_15_bottleneck_input_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_77 = self.L__mod___mobilebert_encoder_layer_15_bottleneck_attention_dense(value_tensor_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_15_bottleneck_attention_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_15_bottleneck_attention_LayerNorm_weight
    mul_123 = layer_input_77 * l__mod___mobilebert_encoder_layer_15_bottleneck_attention_layer_norm_weight_1;  layer_input_77 = l__mod___mobilebert_encoder_layer_15_bottleneck_attention_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_15_bottleneck_attention_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_15_bottleneck_attention_LayerNorm_bias
    key_tensor_15 = mul_123 + l__mod___mobilebert_encoder_layer_15_bottleneck_attention_layer_norm_bias_1;  mul_123 = l__mod___mobilebert_encoder_layer_15_bottleneck_attention_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    mixed_query_layer_15 = self.L__mod___mobilebert_encoder_layer_15_attention_self_query(key_tensor_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    mixed_key_layer_15 = self.L__mod___mobilebert_encoder_layer_15_attention_self_key(key_tensor_15);  key_tensor_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    mixed_value_layer_15 = self.L__mod___mobilebert_encoder_layer_15_attention_self_value(value_tensor_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_45 = mixed_query_layer_15.view((1, 128, 4, 32));  mixed_query_layer_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    query_layer_15 = x_45.permute(0, 2, 1, 3);  x_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_46 = mixed_key_layer_15.view((1, 128, 4, 32));  mixed_key_layer_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    key_layer_15 = x_46.permute(0, 2, 1, 3);  x_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_47 = mixed_value_layer_15.view((1, 128, 4, 32));  mixed_value_layer_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    value_layer_15 = x_47.permute(0, 2, 1, 3);  x_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:286, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_15 = key_layer_15.transpose(-1, -2);  key_layer_15 = None
    attention_scores_45 = torch.matmul(query_layer_15, transpose_15);  query_layer_15 = transpose_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:287, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_46 = attention_scores_45 / 5.656854249492381;  attention_scores_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:290, code: attention_scores = attention_scores + attention_mask
    attention_scores_47 = attention_scores_46 + extended_attention_mask_3;  attention_scores_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:292, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_30 = torch.nn.functional.softmax(attention_scores_47, dim = -1);  attention_scores_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295, code: attention_probs = self.dropout(attention_probs)
    attention_probs_31 = self.L__mod___mobilebert_encoder_layer_15_attention_self_dropout(attention_probs_30);  attention_probs_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:299, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_45 = torch.matmul(attention_probs_31, value_layer_15);  attention_probs_31 = value_layer_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_63 = context_layer_45.permute(0, 2, 1, 3);  context_layer_45 = None
    context_layer_46 = permute_63.contiguous();  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    context_layer_47 = context_layer_46.view((1, 128, 128));  context_layer_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_210 = self.L__mod___mobilebert_encoder_layer_15_attention_output_dense(context_layer_47);  context_layer_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_231 = layer_outputs_210 + layer_input_79;  layer_outputs_210 = layer_input_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_15_attention_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_15_attention_output_LayerNorm_weight
    mul_124 = add_231 * l__mod___mobilebert_encoder_layer_15_attention_output_layer_norm_weight_1;  add_231 = l__mod___mobilebert_encoder_layer_15_attention_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_15_attention_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_15_attention_output_LayerNorm_bias
    attention_output_75 = mul_124 + l__mod___mobilebert_encoder_layer_15_attention_output_layer_norm_bias_1;  mul_124 = l__mod___mobilebert_encoder_layer_15_attention_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_135 = self.L__mod___mobilebert_encoder_layer_15_ffn_0_intermediate_dense(attention_output_75)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_60 = self.L__mod___mobilebert_encoder_layer_15_ffn_0_intermediate_intermediate_act_fn(hidden_states_135);  hidden_states_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_212 = self.L__mod___mobilebert_encoder_layer_15_ffn_0_output_dense(intermediate_output_60);  intermediate_output_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_233 = layer_outputs_212 + attention_output_75;  layer_outputs_212 = attention_output_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_15_ffn_0_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_15_ffn_0_output_LayerNorm_weight
    mul_125 = add_233 * l__mod___mobilebert_encoder_layer_15_ffn_0_output_layer_norm_weight_1;  add_233 = l__mod___mobilebert_encoder_layer_15_ffn_0_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_15_ffn_0_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_15_ffn_0_output_LayerNorm_bias
    attention_output_76 = mul_125 + l__mod___mobilebert_encoder_layer_15_ffn_0_output_layer_norm_bias_1;  mul_125 = l__mod___mobilebert_encoder_layer_15_ffn_0_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_137 = self.L__mod___mobilebert_encoder_layer_15_ffn_1_intermediate_dense(attention_output_76)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_61 = self.L__mod___mobilebert_encoder_layer_15_ffn_1_intermediate_intermediate_act_fn(hidden_states_137);  hidden_states_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_215 = self.L__mod___mobilebert_encoder_layer_15_ffn_1_output_dense(intermediate_output_61);  intermediate_output_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_235 = layer_outputs_215 + attention_output_76;  layer_outputs_215 = attention_output_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_15_ffn_1_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_15_ffn_1_output_LayerNorm_weight
    mul_126 = add_235 * l__mod___mobilebert_encoder_layer_15_ffn_1_output_layer_norm_weight_1;  add_235 = l__mod___mobilebert_encoder_layer_15_ffn_1_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_15_ffn_1_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_15_ffn_1_output_LayerNorm_bias
    attention_output_77 = mul_126 + l__mod___mobilebert_encoder_layer_15_ffn_1_output_layer_norm_bias_1;  mul_126 = l__mod___mobilebert_encoder_layer_15_ffn_1_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_139 = self.L__mod___mobilebert_encoder_layer_15_ffn_2_intermediate_dense(attention_output_77)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_62 = self.L__mod___mobilebert_encoder_layer_15_ffn_2_intermediate_intermediate_act_fn(hidden_states_139);  hidden_states_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_218 = self.L__mod___mobilebert_encoder_layer_15_ffn_2_output_dense(intermediate_output_62);  intermediate_output_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_237 = layer_outputs_218 + attention_output_77;  layer_outputs_218 = attention_output_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_15_ffn_2_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_15_ffn_2_output_LayerNorm_weight
    mul_127 = add_237 * l__mod___mobilebert_encoder_layer_15_ffn_2_output_layer_norm_weight_1;  add_237 = l__mod___mobilebert_encoder_layer_15_ffn_2_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_15_ffn_2_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_15_ffn_2_output_LayerNorm_bias
    attention_output_78 = mul_127 + l__mod___mobilebert_encoder_layer_15_ffn_2_output_layer_norm_bias_1;  mul_127 = l__mod___mobilebert_encoder_layer_15_ffn_2_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_141 = self.L__mod___mobilebert_encoder_layer_15_intermediate_dense(attention_output_78)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_63 = self.L__mod___mobilebert_encoder_layer_15_intermediate_intermediate_act_fn(hidden_states_141);  hidden_states_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    layer_output_60 = self.L__mod___mobilebert_encoder_layer_15_output_dense(intermediate_output_63);  intermediate_output_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_239 = layer_output_60 + attention_output_78;  layer_output_60 = attention_output_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_15_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_15_output_LayerNorm_weight
    mul_128 = add_239 * l__mod___mobilebert_encoder_layer_15_output_layer_norm_weight_1;  add_239 = l__mod___mobilebert_encoder_layer_15_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_15_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_15_output_LayerNorm_bias
    layer_output_61 = mul_128 + l__mod___mobilebert_encoder_layer_15_output_layer_norm_bias_1;  mul_128 = l__mod___mobilebert_encoder_layer_15_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_221 = self.L__mod___mobilebert_encoder_layer_15_output_bottleneck_dense(layer_output_61);  layer_output_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    layer_outputs_222 = self.L__mod___mobilebert_encoder_layer_15_output_bottleneck_dropout(layer_outputs_221);  layer_outputs_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_241 = layer_outputs_222 + value_tensor_15;  layer_outputs_222 = value_tensor_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_15_output_bottleneck_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_15_output_bottleneck_LayerNorm_weight
    mul_129 = add_241 * l__mod___mobilebert_encoder_layer_15_output_bottleneck_layer_norm_weight_1;  add_241 = l__mod___mobilebert_encoder_layer_15_output_bottleneck_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_15_output_bottleneck_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_15_output_bottleneck_LayerNorm_bias
    value_tensor_16 = mul_129 + l__mod___mobilebert_encoder_layer_15_output_bottleneck_layer_norm_bias_1;  mul_129 = l__mod___mobilebert_encoder_layer_15_output_bottleneck_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:549, code: torch.tensor(1000),
    tensor_15 = torch.tensor(1000)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_80 = self.L__mod___mobilebert_encoder_layer_16_bottleneck_input_dense(value_tensor_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_16_bottleneck_input_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_16_bottleneck_input_LayerNorm_weight
    mul_130 = layer_input_80 * l__mod___mobilebert_encoder_layer_16_bottleneck_input_layer_norm_weight_1;  layer_input_80 = l__mod___mobilebert_encoder_layer_16_bottleneck_input_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_16_bottleneck_input_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_16_bottleneck_input_LayerNorm_bias
    layer_input_84 = mul_130 + l__mod___mobilebert_encoder_layer_16_bottleneck_input_layer_norm_bias_1;  mul_130 = l__mod___mobilebert_encoder_layer_16_bottleneck_input_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_82 = self.L__mod___mobilebert_encoder_layer_16_bottleneck_attention_dense(value_tensor_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_16_bottleneck_attention_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_16_bottleneck_attention_LayerNorm_weight
    mul_131 = layer_input_82 * l__mod___mobilebert_encoder_layer_16_bottleneck_attention_layer_norm_weight_1;  layer_input_82 = l__mod___mobilebert_encoder_layer_16_bottleneck_attention_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_16_bottleneck_attention_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_16_bottleneck_attention_LayerNorm_bias
    key_tensor_16 = mul_131 + l__mod___mobilebert_encoder_layer_16_bottleneck_attention_layer_norm_bias_1;  mul_131 = l__mod___mobilebert_encoder_layer_16_bottleneck_attention_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    mixed_query_layer_16 = self.L__mod___mobilebert_encoder_layer_16_attention_self_query(key_tensor_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    mixed_key_layer_16 = self.L__mod___mobilebert_encoder_layer_16_attention_self_key(key_tensor_16);  key_tensor_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    mixed_value_layer_16 = self.L__mod___mobilebert_encoder_layer_16_attention_self_value(value_tensor_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_48 = mixed_query_layer_16.view((1, 128, 4, 32));  mixed_query_layer_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    query_layer_16 = x_48.permute(0, 2, 1, 3);  x_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_49 = mixed_key_layer_16.view((1, 128, 4, 32));  mixed_key_layer_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    key_layer_16 = x_49.permute(0, 2, 1, 3);  x_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_50 = mixed_value_layer_16.view((1, 128, 4, 32));  mixed_value_layer_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    value_layer_16 = x_50.permute(0, 2, 1, 3);  x_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:286, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_16 = key_layer_16.transpose(-1, -2);  key_layer_16 = None
    attention_scores_48 = torch.matmul(query_layer_16, transpose_16);  query_layer_16 = transpose_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:287, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_49 = attention_scores_48 / 5.656854249492381;  attention_scores_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:290, code: attention_scores = attention_scores + attention_mask
    attention_scores_50 = attention_scores_49 + extended_attention_mask_3;  attention_scores_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:292, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_32 = torch.nn.functional.softmax(attention_scores_50, dim = -1);  attention_scores_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295, code: attention_probs = self.dropout(attention_probs)
    attention_probs_33 = self.L__mod___mobilebert_encoder_layer_16_attention_self_dropout(attention_probs_32);  attention_probs_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:299, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_48 = torch.matmul(attention_probs_33, value_layer_16);  attention_probs_33 = value_layer_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_67 = context_layer_48.permute(0, 2, 1, 3);  context_layer_48 = None
    context_layer_49 = permute_67.contiguous();  permute_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    context_layer_50 = context_layer_49.view((1, 128, 128));  context_layer_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_224 = self.L__mod___mobilebert_encoder_layer_16_attention_output_dense(context_layer_50);  context_layer_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_246 = layer_outputs_224 + layer_input_84;  layer_outputs_224 = layer_input_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_16_attention_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_16_attention_output_LayerNorm_weight
    mul_132 = add_246 * l__mod___mobilebert_encoder_layer_16_attention_output_layer_norm_weight_1;  add_246 = l__mod___mobilebert_encoder_layer_16_attention_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_16_attention_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_16_attention_output_LayerNorm_bias
    attention_output_80 = mul_132 + l__mod___mobilebert_encoder_layer_16_attention_output_layer_norm_bias_1;  mul_132 = l__mod___mobilebert_encoder_layer_16_attention_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_144 = self.L__mod___mobilebert_encoder_layer_16_ffn_0_intermediate_dense(attention_output_80)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_64 = self.L__mod___mobilebert_encoder_layer_16_ffn_0_intermediate_intermediate_act_fn(hidden_states_144);  hidden_states_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_226 = self.L__mod___mobilebert_encoder_layer_16_ffn_0_output_dense(intermediate_output_64);  intermediate_output_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_248 = layer_outputs_226 + attention_output_80;  layer_outputs_226 = attention_output_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_16_ffn_0_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_16_ffn_0_output_LayerNorm_weight
    mul_133 = add_248 * l__mod___mobilebert_encoder_layer_16_ffn_0_output_layer_norm_weight_1;  add_248 = l__mod___mobilebert_encoder_layer_16_ffn_0_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_16_ffn_0_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_16_ffn_0_output_LayerNorm_bias
    attention_output_81 = mul_133 + l__mod___mobilebert_encoder_layer_16_ffn_0_output_layer_norm_bias_1;  mul_133 = l__mod___mobilebert_encoder_layer_16_ffn_0_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_146 = self.L__mod___mobilebert_encoder_layer_16_ffn_1_intermediate_dense(attention_output_81)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_65 = self.L__mod___mobilebert_encoder_layer_16_ffn_1_intermediate_intermediate_act_fn(hidden_states_146);  hidden_states_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_229 = self.L__mod___mobilebert_encoder_layer_16_ffn_1_output_dense(intermediate_output_65);  intermediate_output_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_250 = layer_outputs_229 + attention_output_81;  layer_outputs_229 = attention_output_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_16_ffn_1_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_16_ffn_1_output_LayerNorm_weight
    mul_134 = add_250 * l__mod___mobilebert_encoder_layer_16_ffn_1_output_layer_norm_weight_1;  add_250 = l__mod___mobilebert_encoder_layer_16_ffn_1_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_16_ffn_1_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_16_ffn_1_output_LayerNorm_bias
    attention_output_82 = mul_134 + l__mod___mobilebert_encoder_layer_16_ffn_1_output_layer_norm_bias_1;  mul_134 = l__mod___mobilebert_encoder_layer_16_ffn_1_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_148 = self.L__mod___mobilebert_encoder_layer_16_ffn_2_intermediate_dense(attention_output_82)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_66 = self.L__mod___mobilebert_encoder_layer_16_ffn_2_intermediate_intermediate_act_fn(hidden_states_148);  hidden_states_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_232 = self.L__mod___mobilebert_encoder_layer_16_ffn_2_output_dense(intermediate_output_66);  intermediate_output_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_252 = layer_outputs_232 + attention_output_82;  layer_outputs_232 = attention_output_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_16_ffn_2_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_16_ffn_2_output_LayerNorm_weight
    mul_135 = add_252 * l__mod___mobilebert_encoder_layer_16_ffn_2_output_layer_norm_weight_1;  add_252 = l__mod___mobilebert_encoder_layer_16_ffn_2_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_16_ffn_2_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_16_ffn_2_output_LayerNorm_bias
    attention_output_83 = mul_135 + l__mod___mobilebert_encoder_layer_16_ffn_2_output_layer_norm_bias_1;  mul_135 = l__mod___mobilebert_encoder_layer_16_ffn_2_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_150 = self.L__mod___mobilebert_encoder_layer_16_intermediate_dense(attention_output_83)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_67 = self.L__mod___mobilebert_encoder_layer_16_intermediate_intermediate_act_fn(hidden_states_150);  hidden_states_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    layer_output_64 = self.L__mod___mobilebert_encoder_layer_16_output_dense(intermediate_output_67);  intermediate_output_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_254 = layer_output_64 + attention_output_83;  layer_output_64 = attention_output_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_16_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_16_output_LayerNorm_weight
    mul_136 = add_254 * l__mod___mobilebert_encoder_layer_16_output_layer_norm_weight_1;  add_254 = l__mod___mobilebert_encoder_layer_16_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_16_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_16_output_LayerNorm_bias
    layer_output_65 = mul_136 + l__mod___mobilebert_encoder_layer_16_output_layer_norm_bias_1;  mul_136 = l__mod___mobilebert_encoder_layer_16_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_235 = self.L__mod___mobilebert_encoder_layer_16_output_bottleneck_dense(layer_output_65);  layer_output_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    layer_outputs_236 = self.L__mod___mobilebert_encoder_layer_16_output_bottleneck_dropout(layer_outputs_235);  layer_outputs_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_256 = layer_outputs_236 + value_tensor_16;  layer_outputs_236 = value_tensor_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_16_output_bottleneck_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_16_output_bottleneck_LayerNorm_weight
    mul_137 = add_256 * l__mod___mobilebert_encoder_layer_16_output_bottleneck_layer_norm_weight_1;  add_256 = l__mod___mobilebert_encoder_layer_16_output_bottleneck_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_16_output_bottleneck_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_16_output_bottleneck_LayerNorm_bias
    value_tensor_17 = mul_137 + l__mod___mobilebert_encoder_layer_16_output_bottleneck_layer_norm_bias_1;  mul_137 = l__mod___mobilebert_encoder_layer_16_output_bottleneck_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:549, code: torch.tensor(1000),
    tensor_16 = torch.tensor(1000)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_85 = self.L__mod___mobilebert_encoder_layer_17_bottleneck_input_dense(value_tensor_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_17_bottleneck_input_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_17_bottleneck_input_LayerNorm_weight
    mul_138 = layer_input_85 * l__mod___mobilebert_encoder_layer_17_bottleneck_input_layer_norm_weight_1;  layer_input_85 = l__mod___mobilebert_encoder_layer_17_bottleneck_input_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_17_bottleneck_input_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_17_bottleneck_input_LayerNorm_bias
    layer_input_89 = mul_138 + l__mod___mobilebert_encoder_layer_17_bottleneck_input_layer_norm_bias_1;  mul_138 = l__mod___mobilebert_encoder_layer_17_bottleneck_input_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_87 = self.L__mod___mobilebert_encoder_layer_17_bottleneck_attention_dense(value_tensor_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_17_bottleneck_attention_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_17_bottleneck_attention_LayerNorm_weight
    mul_139 = layer_input_87 * l__mod___mobilebert_encoder_layer_17_bottleneck_attention_layer_norm_weight_1;  layer_input_87 = l__mod___mobilebert_encoder_layer_17_bottleneck_attention_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_17_bottleneck_attention_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_17_bottleneck_attention_LayerNorm_bias
    key_tensor_17 = mul_139 + l__mod___mobilebert_encoder_layer_17_bottleneck_attention_layer_norm_bias_1;  mul_139 = l__mod___mobilebert_encoder_layer_17_bottleneck_attention_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    mixed_query_layer_17 = self.L__mod___mobilebert_encoder_layer_17_attention_self_query(key_tensor_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    mixed_key_layer_17 = self.L__mod___mobilebert_encoder_layer_17_attention_self_key(key_tensor_17);  key_tensor_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    mixed_value_layer_17 = self.L__mod___mobilebert_encoder_layer_17_attention_self_value(value_tensor_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_51 = mixed_query_layer_17.view((1, 128, 4, 32));  mixed_query_layer_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    query_layer_17 = x_51.permute(0, 2, 1, 3);  x_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_52 = mixed_key_layer_17.view((1, 128, 4, 32));  mixed_key_layer_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    key_layer_17 = x_52.permute(0, 2, 1, 3);  x_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_53 = mixed_value_layer_17.view((1, 128, 4, 32));  mixed_value_layer_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    value_layer_17 = x_53.permute(0, 2, 1, 3);  x_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:286, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_17 = key_layer_17.transpose(-1, -2);  key_layer_17 = None
    attention_scores_51 = torch.matmul(query_layer_17, transpose_17);  query_layer_17 = transpose_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:287, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_52 = attention_scores_51 / 5.656854249492381;  attention_scores_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:290, code: attention_scores = attention_scores + attention_mask
    attention_scores_53 = attention_scores_52 + extended_attention_mask_3;  attention_scores_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:292, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_34 = torch.nn.functional.softmax(attention_scores_53, dim = -1);  attention_scores_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295, code: attention_probs = self.dropout(attention_probs)
    attention_probs_35 = self.L__mod___mobilebert_encoder_layer_17_attention_self_dropout(attention_probs_34);  attention_probs_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:299, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_51 = torch.matmul(attention_probs_35, value_layer_17);  attention_probs_35 = value_layer_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_71 = context_layer_51.permute(0, 2, 1, 3);  context_layer_51 = None
    context_layer_52 = permute_71.contiguous();  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    context_layer_53 = context_layer_52.view((1, 128, 128));  context_layer_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_238 = self.L__mod___mobilebert_encoder_layer_17_attention_output_dense(context_layer_53);  context_layer_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_261 = layer_outputs_238 + layer_input_89;  layer_outputs_238 = layer_input_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_17_attention_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_17_attention_output_LayerNorm_weight
    mul_140 = add_261 * l__mod___mobilebert_encoder_layer_17_attention_output_layer_norm_weight_1;  add_261 = l__mod___mobilebert_encoder_layer_17_attention_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_17_attention_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_17_attention_output_LayerNorm_bias
    attention_output_85 = mul_140 + l__mod___mobilebert_encoder_layer_17_attention_output_layer_norm_bias_1;  mul_140 = l__mod___mobilebert_encoder_layer_17_attention_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_153 = self.L__mod___mobilebert_encoder_layer_17_ffn_0_intermediate_dense(attention_output_85)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_68 = self.L__mod___mobilebert_encoder_layer_17_ffn_0_intermediate_intermediate_act_fn(hidden_states_153);  hidden_states_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_240 = self.L__mod___mobilebert_encoder_layer_17_ffn_0_output_dense(intermediate_output_68);  intermediate_output_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_263 = layer_outputs_240 + attention_output_85;  layer_outputs_240 = attention_output_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_17_ffn_0_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_17_ffn_0_output_LayerNorm_weight
    mul_141 = add_263 * l__mod___mobilebert_encoder_layer_17_ffn_0_output_layer_norm_weight_1;  add_263 = l__mod___mobilebert_encoder_layer_17_ffn_0_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_17_ffn_0_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_17_ffn_0_output_LayerNorm_bias
    attention_output_86 = mul_141 + l__mod___mobilebert_encoder_layer_17_ffn_0_output_layer_norm_bias_1;  mul_141 = l__mod___mobilebert_encoder_layer_17_ffn_0_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_155 = self.L__mod___mobilebert_encoder_layer_17_ffn_1_intermediate_dense(attention_output_86)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_69 = self.L__mod___mobilebert_encoder_layer_17_ffn_1_intermediate_intermediate_act_fn(hidden_states_155);  hidden_states_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_243 = self.L__mod___mobilebert_encoder_layer_17_ffn_1_output_dense(intermediate_output_69);  intermediate_output_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_265 = layer_outputs_243 + attention_output_86;  layer_outputs_243 = attention_output_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_17_ffn_1_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_17_ffn_1_output_LayerNorm_weight
    mul_142 = add_265 * l__mod___mobilebert_encoder_layer_17_ffn_1_output_layer_norm_weight_1;  add_265 = l__mod___mobilebert_encoder_layer_17_ffn_1_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_17_ffn_1_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_17_ffn_1_output_LayerNorm_bias
    attention_output_87 = mul_142 + l__mod___mobilebert_encoder_layer_17_ffn_1_output_layer_norm_bias_1;  mul_142 = l__mod___mobilebert_encoder_layer_17_ffn_1_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_157 = self.L__mod___mobilebert_encoder_layer_17_ffn_2_intermediate_dense(attention_output_87)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_70 = self.L__mod___mobilebert_encoder_layer_17_ffn_2_intermediate_intermediate_act_fn(hidden_states_157);  hidden_states_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_246 = self.L__mod___mobilebert_encoder_layer_17_ffn_2_output_dense(intermediate_output_70);  intermediate_output_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_267 = layer_outputs_246 + attention_output_87;  layer_outputs_246 = attention_output_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_17_ffn_2_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_17_ffn_2_output_LayerNorm_weight
    mul_143 = add_267 * l__mod___mobilebert_encoder_layer_17_ffn_2_output_layer_norm_weight_1;  add_267 = l__mod___mobilebert_encoder_layer_17_ffn_2_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_17_ffn_2_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_17_ffn_2_output_LayerNorm_bias
    attention_output_88 = mul_143 + l__mod___mobilebert_encoder_layer_17_ffn_2_output_layer_norm_bias_1;  mul_143 = l__mod___mobilebert_encoder_layer_17_ffn_2_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_159 = self.L__mod___mobilebert_encoder_layer_17_intermediate_dense(attention_output_88)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_71 = self.L__mod___mobilebert_encoder_layer_17_intermediate_intermediate_act_fn(hidden_states_159);  hidden_states_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    layer_output_68 = self.L__mod___mobilebert_encoder_layer_17_output_dense(intermediate_output_71);  intermediate_output_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_269 = layer_output_68 + attention_output_88;  layer_output_68 = attention_output_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_17_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_17_output_LayerNorm_weight
    mul_144 = add_269 * l__mod___mobilebert_encoder_layer_17_output_layer_norm_weight_1;  add_269 = l__mod___mobilebert_encoder_layer_17_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_17_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_17_output_LayerNorm_bias
    layer_output_69 = mul_144 + l__mod___mobilebert_encoder_layer_17_output_layer_norm_bias_1;  mul_144 = l__mod___mobilebert_encoder_layer_17_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_249 = self.L__mod___mobilebert_encoder_layer_17_output_bottleneck_dense(layer_output_69);  layer_output_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    layer_outputs_250 = self.L__mod___mobilebert_encoder_layer_17_output_bottleneck_dropout(layer_outputs_249);  layer_outputs_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_271 = layer_outputs_250 + value_tensor_17;  layer_outputs_250 = value_tensor_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_17_output_bottleneck_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_17_output_bottleneck_LayerNorm_weight
    mul_145 = add_271 * l__mod___mobilebert_encoder_layer_17_output_bottleneck_layer_norm_weight_1;  add_271 = l__mod___mobilebert_encoder_layer_17_output_bottleneck_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_17_output_bottleneck_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_17_output_bottleneck_LayerNorm_bias
    value_tensor_18 = mul_145 + l__mod___mobilebert_encoder_layer_17_output_bottleneck_layer_norm_bias_1;  mul_145 = l__mod___mobilebert_encoder_layer_17_output_bottleneck_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:549, code: torch.tensor(1000),
    tensor_17 = torch.tensor(1000)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_90 = self.L__mod___mobilebert_encoder_layer_18_bottleneck_input_dense(value_tensor_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_18_bottleneck_input_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_18_bottleneck_input_LayerNorm_weight
    mul_146 = layer_input_90 * l__mod___mobilebert_encoder_layer_18_bottleneck_input_layer_norm_weight_1;  layer_input_90 = l__mod___mobilebert_encoder_layer_18_bottleneck_input_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_18_bottleneck_input_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_18_bottleneck_input_LayerNorm_bias
    layer_input_94 = mul_146 + l__mod___mobilebert_encoder_layer_18_bottleneck_input_layer_norm_bias_1;  mul_146 = l__mod___mobilebert_encoder_layer_18_bottleneck_input_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_92 = self.L__mod___mobilebert_encoder_layer_18_bottleneck_attention_dense(value_tensor_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_18_bottleneck_attention_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_18_bottleneck_attention_LayerNorm_weight
    mul_147 = layer_input_92 * l__mod___mobilebert_encoder_layer_18_bottleneck_attention_layer_norm_weight_1;  layer_input_92 = l__mod___mobilebert_encoder_layer_18_bottleneck_attention_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_18_bottleneck_attention_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_18_bottleneck_attention_LayerNorm_bias
    key_tensor_18 = mul_147 + l__mod___mobilebert_encoder_layer_18_bottleneck_attention_layer_norm_bias_1;  mul_147 = l__mod___mobilebert_encoder_layer_18_bottleneck_attention_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    mixed_query_layer_18 = self.L__mod___mobilebert_encoder_layer_18_attention_self_query(key_tensor_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    mixed_key_layer_18 = self.L__mod___mobilebert_encoder_layer_18_attention_self_key(key_tensor_18);  key_tensor_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    mixed_value_layer_18 = self.L__mod___mobilebert_encoder_layer_18_attention_self_value(value_tensor_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_54 = mixed_query_layer_18.view((1, 128, 4, 32));  mixed_query_layer_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    query_layer_18 = x_54.permute(0, 2, 1, 3);  x_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_55 = mixed_key_layer_18.view((1, 128, 4, 32));  mixed_key_layer_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    key_layer_18 = x_55.permute(0, 2, 1, 3);  x_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_56 = mixed_value_layer_18.view((1, 128, 4, 32));  mixed_value_layer_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    value_layer_18 = x_56.permute(0, 2, 1, 3);  x_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:286, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_18 = key_layer_18.transpose(-1, -2);  key_layer_18 = None
    attention_scores_54 = torch.matmul(query_layer_18, transpose_18);  query_layer_18 = transpose_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:287, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_55 = attention_scores_54 / 5.656854249492381;  attention_scores_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:290, code: attention_scores = attention_scores + attention_mask
    attention_scores_56 = attention_scores_55 + extended_attention_mask_3;  attention_scores_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:292, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_36 = torch.nn.functional.softmax(attention_scores_56, dim = -1);  attention_scores_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295, code: attention_probs = self.dropout(attention_probs)
    attention_probs_37 = self.L__mod___mobilebert_encoder_layer_18_attention_self_dropout(attention_probs_36);  attention_probs_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:299, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_54 = torch.matmul(attention_probs_37, value_layer_18);  attention_probs_37 = value_layer_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_75 = context_layer_54.permute(0, 2, 1, 3);  context_layer_54 = None
    context_layer_55 = permute_75.contiguous();  permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    context_layer_56 = context_layer_55.view((1, 128, 128));  context_layer_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_252 = self.L__mod___mobilebert_encoder_layer_18_attention_output_dense(context_layer_56);  context_layer_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_276 = layer_outputs_252 + layer_input_94;  layer_outputs_252 = layer_input_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_18_attention_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_18_attention_output_LayerNorm_weight
    mul_148 = add_276 * l__mod___mobilebert_encoder_layer_18_attention_output_layer_norm_weight_1;  add_276 = l__mod___mobilebert_encoder_layer_18_attention_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_18_attention_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_18_attention_output_LayerNorm_bias
    attention_output_90 = mul_148 + l__mod___mobilebert_encoder_layer_18_attention_output_layer_norm_bias_1;  mul_148 = l__mod___mobilebert_encoder_layer_18_attention_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_162 = self.L__mod___mobilebert_encoder_layer_18_ffn_0_intermediate_dense(attention_output_90)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_72 = self.L__mod___mobilebert_encoder_layer_18_ffn_0_intermediate_intermediate_act_fn(hidden_states_162);  hidden_states_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_254 = self.L__mod___mobilebert_encoder_layer_18_ffn_0_output_dense(intermediate_output_72);  intermediate_output_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_278 = layer_outputs_254 + attention_output_90;  layer_outputs_254 = attention_output_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_18_ffn_0_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_18_ffn_0_output_LayerNorm_weight
    mul_149 = add_278 * l__mod___mobilebert_encoder_layer_18_ffn_0_output_layer_norm_weight_1;  add_278 = l__mod___mobilebert_encoder_layer_18_ffn_0_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_18_ffn_0_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_18_ffn_0_output_LayerNorm_bias
    attention_output_91 = mul_149 + l__mod___mobilebert_encoder_layer_18_ffn_0_output_layer_norm_bias_1;  mul_149 = l__mod___mobilebert_encoder_layer_18_ffn_0_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_164 = self.L__mod___mobilebert_encoder_layer_18_ffn_1_intermediate_dense(attention_output_91)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_73 = self.L__mod___mobilebert_encoder_layer_18_ffn_1_intermediate_intermediate_act_fn(hidden_states_164);  hidden_states_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_257 = self.L__mod___mobilebert_encoder_layer_18_ffn_1_output_dense(intermediate_output_73);  intermediate_output_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_280 = layer_outputs_257 + attention_output_91;  layer_outputs_257 = attention_output_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_18_ffn_1_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_18_ffn_1_output_LayerNorm_weight
    mul_150 = add_280 * l__mod___mobilebert_encoder_layer_18_ffn_1_output_layer_norm_weight_1;  add_280 = l__mod___mobilebert_encoder_layer_18_ffn_1_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_18_ffn_1_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_18_ffn_1_output_LayerNorm_bias
    attention_output_92 = mul_150 + l__mod___mobilebert_encoder_layer_18_ffn_1_output_layer_norm_bias_1;  mul_150 = l__mod___mobilebert_encoder_layer_18_ffn_1_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_166 = self.L__mod___mobilebert_encoder_layer_18_ffn_2_intermediate_dense(attention_output_92)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_74 = self.L__mod___mobilebert_encoder_layer_18_ffn_2_intermediate_intermediate_act_fn(hidden_states_166);  hidden_states_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_260 = self.L__mod___mobilebert_encoder_layer_18_ffn_2_output_dense(intermediate_output_74);  intermediate_output_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_282 = layer_outputs_260 + attention_output_92;  layer_outputs_260 = attention_output_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_18_ffn_2_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_18_ffn_2_output_LayerNorm_weight
    mul_151 = add_282 * l__mod___mobilebert_encoder_layer_18_ffn_2_output_layer_norm_weight_1;  add_282 = l__mod___mobilebert_encoder_layer_18_ffn_2_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_18_ffn_2_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_18_ffn_2_output_LayerNorm_bias
    attention_output_93 = mul_151 + l__mod___mobilebert_encoder_layer_18_ffn_2_output_layer_norm_bias_1;  mul_151 = l__mod___mobilebert_encoder_layer_18_ffn_2_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_168 = self.L__mod___mobilebert_encoder_layer_18_intermediate_dense(attention_output_93)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_75 = self.L__mod___mobilebert_encoder_layer_18_intermediate_intermediate_act_fn(hidden_states_168);  hidden_states_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    layer_output_72 = self.L__mod___mobilebert_encoder_layer_18_output_dense(intermediate_output_75);  intermediate_output_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_284 = layer_output_72 + attention_output_93;  layer_output_72 = attention_output_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_18_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_18_output_LayerNorm_weight
    mul_152 = add_284 * l__mod___mobilebert_encoder_layer_18_output_layer_norm_weight_1;  add_284 = l__mod___mobilebert_encoder_layer_18_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_18_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_18_output_LayerNorm_bias
    layer_output_73 = mul_152 + l__mod___mobilebert_encoder_layer_18_output_layer_norm_bias_1;  mul_152 = l__mod___mobilebert_encoder_layer_18_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_263 = self.L__mod___mobilebert_encoder_layer_18_output_bottleneck_dense(layer_output_73);  layer_output_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    layer_outputs_264 = self.L__mod___mobilebert_encoder_layer_18_output_bottleneck_dropout(layer_outputs_263);  layer_outputs_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_286 = layer_outputs_264 + value_tensor_18;  layer_outputs_264 = value_tensor_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_18_output_bottleneck_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_18_output_bottleneck_LayerNorm_weight
    mul_153 = add_286 * l__mod___mobilebert_encoder_layer_18_output_bottleneck_layer_norm_weight_1;  add_286 = l__mod___mobilebert_encoder_layer_18_output_bottleneck_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_18_output_bottleneck_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_18_output_bottleneck_LayerNorm_bias
    value_tensor_19 = mul_153 + l__mod___mobilebert_encoder_layer_18_output_bottleneck_layer_norm_bias_1;  mul_153 = l__mod___mobilebert_encoder_layer_18_output_bottleneck_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:549, code: torch.tensor(1000),
    tensor_18 = torch.tensor(1000)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_95 = self.L__mod___mobilebert_encoder_layer_19_bottleneck_input_dense(value_tensor_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_19_bottleneck_input_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_19_bottleneck_input_LayerNorm_weight
    mul_154 = layer_input_95 * l__mod___mobilebert_encoder_layer_19_bottleneck_input_layer_norm_weight_1;  layer_input_95 = l__mod___mobilebert_encoder_layer_19_bottleneck_input_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_19_bottleneck_input_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_19_bottleneck_input_LayerNorm_bias
    layer_input_99 = mul_154 + l__mod___mobilebert_encoder_layer_19_bottleneck_input_layer_norm_bias_1;  mul_154 = l__mod___mobilebert_encoder_layer_19_bottleneck_input_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_97 = self.L__mod___mobilebert_encoder_layer_19_bottleneck_attention_dense(value_tensor_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_19_bottleneck_attention_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_19_bottleneck_attention_LayerNorm_weight
    mul_155 = layer_input_97 * l__mod___mobilebert_encoder_layer_19_bottleneck_attention_layer_norm_weight_1;  layer_input_97 = l__mod___mobilebert_encoder_layer_19_bottleneck_attention_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_19_bottleneck_attention_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_19_bottleneck_attention_LayerNorm_bias
    key_tensor_19 = mul_155 + l__mod___mobilebert_encoder_layer_19_bottleneck_attention_layer_norm_bias_1;  mul_155 = l__mod___mobilebert_encoder_layer_19_bottleneck_attention_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    mixed_query_layer_19 = self.L__mod___mobilebert_encoder_layer_19_attention_self_query(key_tensor_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    mixed_key_layer_19 = self.L__mod___mobilebert_encoder_layer_19_attention_self_key(key_tensor_19);  key_tensor_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    mixed_value_layer_19 = self.L__mod___mobilebert_encoder_layer_19_attention_self_value(value_tensor_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_57 = mixed_query_layer_19.view((1, 128, 4, 32));  mixed_query_layer_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    query_layer_19 = x_57.permute(0, 2, 1, 3);  x_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_58 = mixed_key_layer_19.view((1, 128, 4, 32));  mixed_key_layer_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    key_layer_19 = x_58.permute(0, 2, 1, 3);  x_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_59 = mixed_value_layer_19.view((1, 128, 4, 32));  mixed_value_layer_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    value_layer_19 = x_59.permute(0, 2, 1, 3);  x_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:286, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_19 = key_layer_19.transpose(-1, -2);  key_layer_19 = None
    attention_scores_57 = torch.matmul(query_layer_19, transpose_19);  query_layer_19 = transpose_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:287, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_58 = attention_scores_57 / 5.656854249492381;  attention_scores_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:290, code: attention_scores = attention_scores + attention_mask
    attention_scores_59 = attention_scores_58 + extended_attention_mask_3;  attention_scores_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:292, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_38 = torch.nn.functional.softmax(attention_scores_59, dim = -1);  attention_scores_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295, code: attention_probs = self.dropout(attention_probs)
    attention_probs_39 = self.L__mod___mobilebert_encoder_layer_19_attention_self_dropout(attention_probs_38);  attention_probs_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:299, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_57 = torch.matmul(attention_probs_39, value_layer_19);  attention_probs_39 = value_layer_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_79 = context_layer_57.permute(0, 2, 1, 3);  context_layer_57 = None
    context_layer_58 = permute_79.contiguous();  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    context_layer_59 = context_layer_58.view((1, 128, 128));  context_layer_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_266 = self.L__mod___mobilebert_encoder_layer_19_attention_output_dense(context_layer_59);  context_layer_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_291 = layer_outputs_266 + layer_input_99;  layer_outputs_266 = layer_input_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_19_attention_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_19_attention_output_LayerNorm_weight
    mul_156 = add_291 * l__mod___mobilebert_encoder_layer_19_attention_output_layer_norm_weight_1;  add_291 = l__mod___mobilebert_encoder_layer_19_attention_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_19_attention_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_19_attention_output_LayerNorm_bias
    attention_output_95 = mul_156 + l__mod___mobilebert_encoder_layer_19_attention_output_layer_norm_bias_1;  mul_156 = l__mod___mobilebert_encoder_layer_19_attention_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_171 = self.L__mod___mobilebert_encoder_layer_19_ffn_0_intermediate_dense(attention_output_95)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_76 = self.L__mod___mobilebert_encoder_layer_19_ffn_0_intermediate_intermediate_act_fn(hidden_states_171);  hidden_states_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_268 = self.L__mod___mobilebert_encoder_layer_19_ffn_0_output_dense(intermediate_output_76);  intermediate_output_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_293 = layer_outputs_268 + attention_output_95;  layer_outputs_268 = attention_output_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_19_ffn_0_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_19_ffn_0_output_LayerNorm_weight
    mul_157 = add_293 * l__mod___mobilebert_encoder_layer_19_ffn_0_output_layer_norm_weight_1;  add_293 = l__mod___mobilebert_encoder_layer_19_ffn_0_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_19_ffn_0_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_19_ffn_0_output_LayerNorm_bias
    attention_output_96 = mul_157 + l__mod___mobilebert_encoder_layer_19_ffn_0_output_layer_norm_bias_1;  mul_157 = l__mod___mobilebert_encoder_layer_19_ffn_0_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_173 = self.L__mod___mobilebert_encoder_layer_19_ffn_1_intermediate_dense(attention_output_96)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_77 = self.L__mod___mobilebert_encoder_layer_19_ffn_1_intermediate_intermediate_act_fn(hidden_states_173);  hidden_states_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_271 = self.L__mod___mobilebert_encoder_layer_19_ffn_1_output_dense(intermediate_output_77);  intermediate_output_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_295 = layer_outputs_271 + attention_output_96;  layer_outputs_271 = attention_output_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_19_ffn_1_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_19_ffn_1_output_LayerNorm_weight
    mul_158 = add_295 * l__mod___mobilebert_encoder_layer_19_ffn_1_output_layer_norm_weight_1;  add_295 = l__mod___mobilebert_encoder_layer_19_ffn_1_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_19_ffn_1_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_19_ffn_1_output_LayerNorm_bias
    attention_output_97 = mul_158 + l__mod___mobilebert_encoder_layer_19_ffn_1_output_layer_norm_bias_1;  mul_158 = l__mod___mobilebert_encoder_layer_19_ffn_1_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_175 = self.L__mod___mobilebert_encoder_layer_19_ffn_2_intermediate_dense(attention_output_97)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_78 = self.L__mod___mobilebert_encoder_layer_19_ffn_2_intermediate_intermediate_act_fn(hidden_states_175);  hidden_states_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_274 = self.L__mod___mobilebert_encoder_layer_19_ffn_2_output_dense(intermediate_output_78);  intermediate_output_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_297 = layer_outputs_274 + attention_output_97;  layer_outputs_274 = attention_output_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_19_ffn_2_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_19_ffn_2_output_LayerNorm_weight
    mul_159 = add_297 * l__mod___mobilebert_encoder_layer_19_ffn_2_output_layer_norm_weight_1;  add_297 = l__mod___mobilebert_encoder_layer_19_ffn_2_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_19_ffn_2_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_19_ffn_2_output_LayerNorm_bias
    attention_output_98 = mul_159 + l__mod___mobilebert_encoder_layer_19_ffn_2_output_layer_norm_bias_1;  mul_159 = l__mod___mobilebert_encoder_layer_19_ffn_2_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_177 = self.L__mod___mobilebert_encoder_layer_19_intermediate_dense(attention_output_98)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_79 = self.L__mod___mobilebert_encoder_layer_19_intermediate_intermediate_act_fn(hidden_states_177);  hidden_states_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    layer_output_76 = self.L__mod___mobilebert_encoder_layer_19_output_dense(intermediate_output_79);  intermediate_output_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_299 = layer_output_76 + attention_output_98;  layer_output_76 = attention_output_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_19_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_19_output_LayerNorm_weight
    mul_160 = add_299 * l__mod___mobilebert_encoder_layer_19_output_layer_norm_weight_1;  add_299 = l__mod___mobilebert_encoder_layer_19_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_19_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_19_output_LayerNorm_bias
    layer_output_77 = mul_160 + l__mod___mobilebert_encoder_layer_19_output_layer_norm_bias_1;  mul_160 = l__mod___mobilebert_encoder_layer_19_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_277 = self.L__mod___mobilebert_encoder_layer_19_output_bottleneck_dense(layer_output_77);  layer_output_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    layer_outputs_278 = self.L__mod___mobilebert_encoder_layer_19_output_bottleneck_dropout(layer_outputs_277);  layer_outputs_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_301 = layer_outputs_278 + value_tensor_19;  layer_outputs_278 = value_tensor_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_19_output_bottleneck_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_19_output_bottleneck_LayerNorm_weight
    mul_161 = add_301 * l__mod___mobilebert_encoder_layer_19_output_bottleneck_layer_norm_weight_1;  add_301 = l__mod___mobilebert_encoder_layer_19_output_bottleneck_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_19_output_bottleneck_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_19_output_bottleneck_LayerNorm_bias
    value_tensor_20 = mul_161 + l__mod___mobilebert_encoder_layer_19_output_bottleneck_layer_norm_bias_1;  mul_161 = l__mod___mobilebert_encoder_layer_19_output_bottleneck_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:549, code: torch.tensor(1000),
    tensor_19 = torch.tensor(1000)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_100 = self.L__mod___mobilebert_encoder_layer_20_bottleneck_input_dense(value_tensor_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_20_bottleneck_input_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_20_bottleneck_input_LayerNorm_weight
    mul_162 = layer_input_100 * l__mod___mobilebert_encoder_layer_20_bottleneck_input_layer_norm_weight_1;  layer_input_100 = l__mod___mobilebert_encoder_layer_20_bottleneck_input_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_20_bottleneck_input_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_20_bottleneck_input_LayerNorm_bias
    layer_input_104 = mul_162 + l__mod___mobilebert_encoder_layer_20_bottleneck_input_layer_norm_bias_1;  mul_162 = l__mod___mobilebert_encoder_layer_20_bottleneck_input_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_102 = self.L__mod___mobilebert_encoder_layer_20_bottleneck_attention_dense(value_tensor_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_20_bottleneck_attention_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_20_bottleneck_attention_LayerNorm_weight
    mul_163 = layer_input_102 * l__mod___mobilebert_encoder_layer_20_bottleneck_attention_layer_norm_weight_1;  layer_input_102 = l__mod___mobilebert_encoder_layer_20_bottleneck_attention_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_20_bottleneck_attention_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_20_bottleneck_attention_LayerNorm_bias
    key_tensor_20 = mul_163 + l__mod___mobilebert_encoder_layer_20_bottleneck_attention_layer_norm_bias_1;  mul_163 = l__mod___mobilebert_encoder_layer_20_bottleneck_attention_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    mixed_query_layer_20 = self.L__mod___mobilebert_encoder_layer_20_attention_self_query(key_tensor_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    mixed_key_layer_20 = self.L__mod___mobilebert_encoder_layer_20_attention_self_key(key_tensor_20);  key_tensor_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    mixed_value_layer_20 = self.L__mod___mobilebert_encoder_layer_20_attention_self_value(value_tensor_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_60 = mixed_query_layer_20.view((1, 128, 4, 32));  mixed_query_layer_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    query_layer_20 = x_60.permute(0, 2, 1, 3);  x_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_61 = mixed_key_layer_20.view((1, 128, 4, 32));  mixed_key_layer_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    key_layer_20 = x_61.permute(0, 2, 1, 3);  x_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_62 = mixed_value_layer_20.view((1, 128, 4, 32));  mixed_value_layer_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    value_layer_20 = x_62.permute(0, 2, 1, 3);  x_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:286, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_20 = key_layer_20.transpose(-1, -2);  key_layer_20 = None
    attention_scores_60 = torch.matmul(query_layer_20, transpose_20);  query_layer_20 = transpose_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:287, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_61 = attention_scores_60 / 5.656854249492381;  attention_scores_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:290, code: attention_scores = attention_scores + attention_mask
    attention_scores_62 = attention_scores_61 + extended_attention_mask_3;  attention_scores_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:292, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_40 = torch.nn.functional.softmax(attention_scores_62, dim = -1);  attention_scores_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295, code: attention_probs = self.dropout(attention_probs)
    attention_probs_41 = self.L__mod___mobilebert_encoder_layer_20_attention_self_dropout(attention_probs_40);  attention_probs_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:299, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_60 = torch.matmul(attention_probs_41, value_layer_20);  attention_probs_41 = value_layer_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_83 = context_layer_60.permute(0, 2, 1, 3);  context_layer_60 = None
    context_layer_61 = permute_83.contiguous();  permute_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    context_layer_62 = context_layer_61.view((1, 128, 128));  context_layer_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_280 = self.L__mod___mobilebert_encoder_layer_20_attention_output_dense(context_layer_62);  context_layer_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_306 = layer_outputs_280 + layer_input_104;  layer_outputs_280 = layer_input_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_20_attention_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_20_attention_output_LayerNorm_weight
    mul_164 = add_306 * l__mod___mobilebert_encoder_layer_20_attention_output_layer_norm_weight_1;  add_306 = l__mod___mobilebert_encoder_layer_20_attention_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_20_attention_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_20_attention_output_LayerNorm_bias
    attention_output_100 = mul_164 + l__mod___mobilebert_encoder_layer_20_attention_output_layer_norm_bias_1;  mul_164 = l__mod___mobilebert_encoder_layer_20_attention_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_180 = self.L__mod___mobilebert_encoder_layer_20_ffn_0_intermediate_dense(attention_output_100)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_80 = self.L__mod___mobilebert_encoder_layer_20_ffn_0_intermediate_intermediate_act_fn(hidden_states_180);  hidden_states_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_282 = self.L__mod___mobilebert_encoder_layer_20_ffn_0_output_dense(intermediate_output_80);  intermediate_output_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_308 = layer_outputs_282 + attention_output_100;  layer_outputs_282 = attention_output_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_20_ffn_0_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_20_ffn_0_output_LayerNorm_weight
    mul_165 = add_308 * l__mod___mobilebert_encoder_layer_20_ffn_0_output_layer_norm_weight_1;  add_308 = l__mod___mobilebert_encoder_layer_20_ffn_0_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_20_ffn_0_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_20_ffn_0_output_LayerNorm_bias
    attention_output_101 = mul_165 + l__mod___mobilebert_encoder_layer_20_ffn_0_output_layer_norm_bias_1;  mul_165 = l__mod___mobilebert_encoder_layer_20_ffn_0_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_182 = self.L__mod___mobilebert_encoder_layer_20_ffn_1_intermediate_dense(attention_output_101)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_81 = self.L__mod___mobilebert_encoder_layer_20_ffn_1_intermediate_intermediate_act_fn(hidden_states_182);  hidden_states_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_285 = self.L__mod___mobilebert_encoder_layer_20_ffn_1_output_dense(intermediate_output_81);  intermediate_output_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_310 = layer_outputs_285 + attention_output_101;  layer_outputs_285 = attention_output_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_20_ffn_1_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_20_ffn_1_output_LayerNorm_weight
    mul_166 = add_310 * l__mod___mobilebert_encoder_layer_20_ffn_1_output_layer_norm_weight_1;  add_310 = l__mod___mobilebert_encoder_layer_20_ffn_1_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_20_ffn_1_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_20_ffn_1_output_LayerNorm_bias
    attention_output_102 = mul_166 + l__mod___mobilebert_encoder_layer_20_ffn_1_output_layer_norm_bias_1;  mul_166 = l__mod___mobilebert_encoder_layer_20_ffn_1_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_184 = self.L__mod___mobilebert_encoder_layer_20_ffn_2_intermediate_dense(attention_output_102)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_82 = self.L__mod___mobilebert_encoder_layer_20_ffn_2_intermediate_intermediate_act_fn(hidden_states_184);  hidden_states_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_288 = self.L__mod___mobilebert_encoder_layer_20_ffn_2_output_dense(intermediate_output_82);  intermediate_output_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_312 = layer_outputs_288 + attention_output_102;  layer_outputs_288 = attention_output_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_20_ffn_2_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_20_ffn_2_output_LayerNorm_weight
    mul_167 = add_312 * l__mod___mobilebert_encoder_layer_20_ffn_2_output_layer_norm_weight_1;  add_312 = l__mod___mobilebert_encoder_layer_20_ffn_2_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_20_ffn_2_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_20_ffn_2_output_LayerNorm_bias
    attention_output_103 = mul_167 + l__mod___mobilebert_encoder_layer_20_ffn_2_output_layer_norm_bias_1;  mul_167 = l__mod___mobilebert_encoder_layer_20_ffn_2_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_186 = self.L__mod___mobilebert_encoder_layer_20_intermediate_dense(attention_output_103)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_83 = self.L__mod___mobilebert_encoder_layer_20_intermediate_intermediate_act_fn(hidden_states_186);  hidden_states_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    layer_output_80 = self.L__mod___mobilebert_encoder_layer_20_output_dense(intermediate_output_83);  intermediate_output_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_314 = layer_output_80 + attention_output_103;  layer_output_80 = attention_output_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_20_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_20_output_LayerNorm_weight
    mul_168 = add_314 * l__mod___mobilebert_encoder_layer_20_output_layer_norm_weight_1;  add_314 = l__mod___mobilebert_encoder_layer_20_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_20_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_20_output_LayerNorm_bias
    layer_output_81 = mul_168 + l__mod___mobilebert_encoder_layer_20_output_layer_norm_bias_1;  mul_168 = l__mod___mobilebert_encoder_layer_20_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_291 = self.L__mod___mobilebert_encoder_layer_20_output_bottleneck_dense(layer_output_81);  layer_output_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    layer_outputs_292 = self.L__mod___mobilebert_encoder_layer_20_output_bottleneck_dropout(layer_outputs_291);  layer_outputs_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_316 = layer_outputs_292 + value_tensor_20;  layer_outputs_292 = value_tensor_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_20_output_bottleneck_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_20_output_bottleneck_LayerNorm_weight
    mul_169 = add_316 * l__mod___mobilebert_encoder_layer_20_output_bottleneck_layer_norm_weight_1;  add_316 = l__mod___mobilebert_encoder_layer_20_output_bottleneck_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_20_output_bottleneck_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_20_output_bottleneck_LayerNorm_bias
    value_tensor_21 = mul_169 + l__mod___mobilebert_encoder_layer_20_output_bottleneck_layer_norm_bias_1;  mul_169 = l__mod___mobilebert_encoder_layer_20_output_bottleneck_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:549, code: torch.tensor(1000),
    tensor_20 = torch.tensor(1000)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_105 = self.L__mod___mobilebert_encoder_layer_21_bottleneck_input_dense(value_tensor_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_21_bottleneck_input_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_21_bottleneck_input_LayerNorm_weight
    mul_170 = layer_input_105 * l__mod___mobilebert_encoder_layer_21_bottleneck_input_layer_norm_weight_1;  layer_input_105 = l__mod___mobilebert_encoder_layer_21_bottleneck_input_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_21_bottleneck_input_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_21_bottleneck_input_LayerNorm_bias
    layer_input_109 = mul_170 + l__mod___mobilebert_encoder_layer_21_bottleneck_input_layer_norm_bias_1;  mul_170 = l__mod___mobilebert_encoder_layer_21_bottleneck_input_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_107 = self.L__mod___mobilebert_encoder_layer_21_bottleneck_attention_dense(value_tensor_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_21_bottleneck_attention_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_21_bottleneck_attention_LayerNorm_weight
    mul_171 = layer_input_107 * l__mod___mobilebert_encoder_layer_21_bottleneck_attention_layer_norm_weight_1;  layer_input_107 = l__mod___mobilebert_encoder_layer_21_bottleneck_attention_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_21_bottleneck_attention_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_21_bottleneck_attention_LayerNorm_bias
    key_tensor_21 = mul_171 + l__mod___mobilebert_encoder_layer_21_bottleneck_attention_layer_norm_bias_1;  mul_171 = l__mod___mobilebert_encoder_layer_21_bottleneck_attention_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    mixed_query_layer_21 = self.L__mod___mobilebert_encoder_layer_21_attention_self_query(key_tensor_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    mixed_key_layer_21 = self.L__mod___mobilebert_encoder_layer_21_attention_self_key(key_tensor_21);  key_tensor_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    mixed_value_layer_21 = self.L__mod___mobilebert_encoder_layer_21_attention_self_value(value_tensor_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_63 = mixed_query_layer_21.view((1, 128, 4, 32));  mixed_query_layer_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    query_layer_21 = x_63.permute(0, 2, 1, 3);  x_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_64 = mixed_key_layer_21.view((1, 128, 4, 32));  mixed_key_layer_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    key_layer_21 = x_64.permute(0, 2, 1, 3);  x_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_65 = mixed_value_layer_21.view((1, 128, 4, 32));  mixed_value_layer_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    value_layer_21 = x_65.permute(0, 2, 1, 3);  x_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:286, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_21 = key_layer_21.transpose(-1, -2);  key_layer_21 = None
    attention_scores_63 = torch.matmul(query_layer_21, transpose_21);  query_layer_21 = transpose_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:287, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_64 = attention_scores_63 / 5.656854249492381;  attention_scores_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:290, code: attention_scores = attention_scores + attention_mask
    attention_scores_65 = attention_scores_64 + extended_attention_mask_3;  attention_scores_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:292, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_42 = torch.nn.functional.softmax(attention_scores_65, dim = -1);  attention_scores_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295, code: attention_probs = self.dropout(attention_probs)
    attention_probs_43 = self.L__mod___mobilebert_encoder_layer_21_attention_self_dropout(attention_probs_42);  attention_probs_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:299, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_63 = torch.matmul(attention_probs_43, value_layer_21);  attention_probs_43 = value_layer_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_87 = context_layer_63.permute(0, 2, 1, 3);  context_layer_63 = None
    context_layer_64 = permute_87.contiguous();  permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    context_layer_65 = context_layer_64.view((1, 128, 128));  context_layer_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_294 = self.L__mod___mobilebert_encoder_layer_21_attention_output_dense(context_layer_65);  context_layer_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_321 = layer_outputs_294 + layer_input_109;  layer_outputs_294 = layer_input_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_21_attention_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_21_attention_output_LayerNorm_weight
    mul_172 = add_321 * l__mod___mobilebert_encoder_layer_21_attention_output_layer_norm_weight_1;  add_321 = l__mod___mobilebert_encoder_layer_21_attention_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_21_attention_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_21_attention_output_LayerNorm_bias
    attention_output_105 = mul_172 + l__mod___mobilebert_encoder_layer_21_attention_output_layer_norm_bias_1;  mul_172 = l__mod___mobilebert_encoder_layer_21_attention_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_189 = self.L__mod___mobilebert_encoder_layer_21_ffn_0_intermediate_dense(attention_output_105)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_84 = self.L__mod___mobilebert_encoder_layer_21_ffn_0_intermediate_intermediate_act_fn(hidden_states_189);  hidden_states_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_296 = self.L__mod___mobilebert_encoder_layer_21_ffn_0_output_dense(intermediate_output_84);  intermediate_output_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_323 = layer_outputs_296 + attention_output_105;  layer_outputs_296 = attention_output_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_21_ffn_0_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_21_ffn_0_output_LayerNorm_weight
    mul_173 = add_323 * l__mod___mobilebert_encoder_layer_21_ffn_0_output_layer_norm_weight_1;  add_323 = l__mod___mobilebert_encoder_layer_21_ffn_0_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_21_ffn_0_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_21_ffn_0_output_LayerNorm_bias
    attention_output_106 = mul_173 + l__mod___mobilebert_encoder_layer_21_ffn_0_output_layer_norm_bias_1;  mul_173 = l__mod___mobilebert_encoder_layer_21_ffn_0_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_191 = self.L__mod___mobilebert_encoder_layer_21_ffn_1_intermediate_dense(attention_output_106)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_85 = self.L__mod___mobilebert_encoder_layer_21_ffn_1_intermediate_intermediate_act_fn(hidden_states_191);  hidden_states_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_299 = self.L__mod___mobilebert_encoder_layer_21_ffn_1_output_dense(intermediate_output_85);  intermediate_output_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_325 = layer_outputs_299 + attention_output_106;  layer_outputs_299 = attention_output_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_21_ffn_1_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_21_ffn_1_output_LayerNorm_weight
    mul_174 = add_325 * l__mod___mobilebert_encoder_layer_21_ffn_1_output_layer_norm_weight_1;  add_325 = l__mod___mobilebert_encoder_layer_21_ffn_1_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_21_ffn_1_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_21_ffn_1_output_LayerNorm_bias
    attention_output_107 = mul_174 + l__mod___mobilebert_encoder_layer_21_ffn_1_output_layer_norm_bias_1;  mul_174 = l__mod___mobilebert_encoder_layer_21_ffn_1_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_193 = self.L__mod___mobilebert_encoder_layer_21_ffn_2_intermediate_dense(attention_output_107)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_86 = self.L__mod___mobilebert_encoder_layer_21_ffn_2_intermediate_intermediate_act_fn(hidden_states_193);  hidden_states_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_302 = self.L__mod___mobilebert_encoder_layer_21_ffn_2_output_dense(intermediate_output_86);  intermediate_output_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_327 = layer_outputs_302 + attention_output_107;  layer_outputs_302 = attention_output_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_21_ffn_2_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_21_ffn_2_output_LayerNorm_weight
    mul_175 = add_327 * l__mod___mobilebert_encoder_layer_21_ffn_2_output_layer_norm_weight_1;  add_327 = l__mod___mobilebert_encoder_layer_21_ffn_2_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_21_ffn_2_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_21_ffn_2_output_LayerNorm_bias
    attention_output_108 = mul_175 + l__mod___mobilebert_encoder_layer_21_ffn_2_output_layer_norm_bias_1;  mul_175 = l__mod___mobilebert_encoder_layer_21_ffn_2_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_195 = self.L__mod___mobilebert_encoder_layer_21_intermediate_dense(attention_output_108)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_87 = self.L__mod___mobilebert_encoder_layer_21_intermediate_intermediate_act_fn(hidden_states_195);  hidden_states_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    layer_output_84 = self.L__mod___mobilebert_encoder_layer_21_output_dense(intermediate_output_87);  intermediate_output_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_329 = layer_output_84 + attention_output_108;  layer_output_84 = attention_output_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_21_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_21_output_LayerNorm_weight
    mul_176 = add_329 * l__mod___mobilebert_encoder_layer_21_output_layer_norm_weight_1;  add_329 = l__mod___mobilebert_encoder_layer_21_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_21_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_21_output_LayerNorm_bias
    layer_output_85 = mul_176 + l__mod___mobilebert_encoder_layer_21_output_layer_norm_bias_1;  mul_176 = l__mod___mobilebert_encoder_layer_21_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_305 = self.L__mod___mobilebert_encoder_layer_21_output_bottleneck_dense(layer_output_85);  layer_output_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    layer_outputs_306 = self.L__mod___mobilebert_encoder_layer_21_output_bottleneck_dropout(layer_outputs_305);  layer_outputs_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_331 = layer_outputs_306 + value_tensor_21;  layer_outputs_306 = value_tensor_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_21_output_bottleneck_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_21_output_bottleneck_LayerNorm_weight
    mul_177 = add_331 * l__mod___mobilebert_encoder_layer_21_output_bottleneck_layer_norm_weight_1;  add_331 = l__mod___mobilebert_encoder_layer_21_output_bottleneck_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_21_output_bottleneck_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_21_output_bottleneck_LayerNorm_bias
    value_tensor_22 = mul_177 + l__mod___mobilebert_encoder_layer_21_output_bottleneck_layer_norm_bias_1;  mul_177 = l__mod___mobilebert_encoder_layer_21_output_bottleneck_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:549, code: torch.tensor(1000),
    tensor_21 = torch.tensor(1000)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_110 = self.L__mod___mobilebert_encoder_layer_22_bottleneck_input_dense(value_tensor_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_22_bottleneck_input_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_22_bottleneck_input_LayerNorm_weight
    mul_178 = layer_input_110 * l__mod___mobilebert_encoder_layer_22_bottleneck_input_layer_norm_weight_1;  layer_input_110 = l__mod___mobilebert_encoder_layer_22_bottleneck_input_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_22_bottleneck_input_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_22_bottleneck_input_LayerNorm_bias
    layer_input_114 = mul_178 + l__mod___mobilebert_encoder_layer_22_bottleneck_input_layer_norm_bias_1;  mul_178 = l__mod___mobilebert_encoder_layer_22_bottleneck_input_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_112 = self.L__mod___mobilebert_encoder_layer_22_bottleneck_attention_dense(value_tensor_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_22_bottleneck_attention_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_22_bottleneck_attention_LayerNorm_weight
    mul_179 = layer_input_112 * l__mod___mobilebert_encoder_layer_22_bottleneck_attention_layer_norm_weight_1;  layer_input_112 = l__mod___mobilebert_encoder_layer_22_bottleneck_attention_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_22_bottleneck_attention_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_22_bottleneck_attention_LayerNorm_bias
    key_tensor_22 = mul_179 + l__mod___mobilebert_encoder_layer_22_bottleneck_attention_layer_norm_bias_1;  mul_179 = l__mod___mobilebert_encoder_layer_22_bottleneck_attention_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    mixed_query_layer_22 = self.L__mod___mobilebert_encoder_layer_22_attention_self_query(key_tensor_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    mixed_key_layer_22 = self.L__mod___mobilebert_encoder_layer_22_attention_self_key(key_tensor_22);  key_tensor_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    mixed_value_layer_22 = self.L__mod___mobilebert_encoder_layer_22_attention_self_value(value_tensor_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_66 = mixed_query_layer_22.view((1, 128, 4, 32));  mixed_query_layer_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    query_layer_22 = x_66.permute(0, 2, 1, 3);  x_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_67 = mixed_key_layer_22.view((1, 128, 4, 32));  mixed_key_layer_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    key_layer_22 = x_67.permute(0, 2, 1, 3);  x_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_68 = mixed_value_layer_22.view((1, 128, 4, 32));  mixed_value_layer_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    value_layer_22 = x_68.permute(0, 2, 1, 3);  x_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:286, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_22 = key_layer_22.transpose(-1, -2);  key_layer_22 = None
    attention_scores_66 = torch.matmul(query_layer_22, transpose_22);  query_layer_22 = transpose_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:287, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_67 = attention_scores_66 / 5.656854249492381;  attention_scores_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:290, code: attention_scores = attention_scores + attention_mask
    attention_scores_68 = attention_scores_67 + extended_attention_mask_3;  attention_scores_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:292, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_44 = torch.nn.functional.softmax(attention_scores_68, dim = -1);  attention_scores_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295, code: attention_probs = self.dropout(attention_probs)
    attention_probs_45 = self.L__mod___mobilebert_encoder_layer_22_attention_self_dropout(attention_probs_44);  attention_probs_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:299, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_66 = torch.matmul(attention_probs_45, value_layer_22);  attention_probs_45 = value_layer_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_91 = context_layer_66.permute(0, 2, 1, 3);  context_layer_66 = None
    context_layer_67 = permute_91.contiguous();  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    context_layer_68 = context_layer_67.view((1, 128, 128));  context_layer_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_308 = self.L__mod___mobilebert_encoder_layer_22_attention_output_dense(context_layer_68);  context_layer_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_336 = layer_outputs_308 + layer_input_114;  layer_outputs_308 = layer_input_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_22_attention_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_22_attention_output_LayerNorm_weight
    mul_180 = add_336 * l__mod___mobilebert_encoder_layer_22_attention_output_layer_norm_weight_1;  add_336 = l__mod___mobilebert_encoder_layer_22_attention_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_22_attention_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_22_attention_output_LayerNorm_bias
    attention_output_110 = mul_180 + l__mod___mobilebert_encoder_layer_22_attention_output_layer_norm_bias_1;  mul_180 = l__mod___mobilebert_encoder_layer_22_attention_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_198 = self.L__mod___mobilebert_encoder_layer_22_ffn_0_intermediate_dense(attention_output_110)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_88 = self.L__mod___mobilebert_encoder_layer_22_ffn_0_intermediate_intermediate_act_fn(hidden_states_198);  hidden_states_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_310 = self.L__mod___mobilebert_encoder_layer_22_ffn_0_output_dense(intermediate_output_88);  intermediate_output_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_338 = layer_outputs_310 + attention_output_110;  layer_outputs_310 = attention_output_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_22_ffn_0_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_22_ffn_0_output_LayerNorm_weight
    mul_181 = add_338 * l__mod___mobilebert_encoder_layer_22_ffn_0_output_layer_norm_weight_1;  add_338 = l__mod___mobilebert_encoder_layer_22_ffn_0_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_22_ffn_0_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_22_ffn_0_output_LayerNorm_bias
    attention_output_111 = mul_181 + l__mod___mobilebert_encoder_layer_22_ffn_0_output_layer_norm_bias_1;  mul_181 = l__mod___mobilebert_encoder_layer_22_ffn_0_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_200 = self.L__mod___mobilebert_encoder_layer_22_ffn_1_intermediate_dense(attention_output_111)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_89 = self.L__mod___mobilebert_encoder_layer_22_ffn_1_intermediate_intermediate_act_fn(hidden_states_200);  hidden_states_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_313 = self.L__mod___mobilebert_encoder_layer_22_ffn_1_output_dense(intermediate_output_89);  intermediate_output_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_340 = layer_outputs_313 + attention_output_111;  layer_outputs_313 = attention_output_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_22_ffn_1_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_22_ffn_1_output_LayerNorm_weight
    mul_182 = add_340 * l__mod___mobilebert_encoder_layer_22_ffn_1_output_layer_norm_weight_1;  add_340 = l__mod___mobilebert_encoder_layer_22_ffn_1_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_22_ffn_1_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_22_ffn_1_output_LayerNorm_bias
    attention_output_112 = mul_182 + l__mod___mobilebert_encoder_layer_22_ffn_1_output_layer_norm_bias_1;  mul_182 = l__mod___mobilebert_encoder_layer_22_ffn_1_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_202 = self.L__mod___mobilebert_encoder_layer_22_ffn_2_intermediate_dense(attention_output_112)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_90 = self.L__mod___mobilebert_encoder_layer_22_ffn_2_intermediate_intermediate_act_fn(hidden_states_202);  hidden_states_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_316 = self.L__mod___mobilebert_encoder_layer_22_ffn_2_output_dense(intermediate_output_90);  intermediate_output_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_342 = layer_outputs_316 + attention_output_112;  layer_outputs_316 = attention_output_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_22_ffn_2_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_22_ffn_2_output_LayerNorm_weight
    mul_183 = add_342 * l__mod___mobilebert_encoder_layer_22_ffn_2_output_layer_norm_weight_1;  add_342 = l__mod___mobilebert_encoder_layer_22_ffn_2_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_22_ffn_2_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_22_ffn_2_output_LayerNorm_bias
    attention_output_113 = mul_183 + l__mod___mobilebert_encoder_layer_22_ffn_2_output_layer_norm_bias_1;  mul_183 = l__mod___mobilebert_encoder_layer_22_ffn_2_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_204 = self.L__mod___mobilebert_encoder_layer_22_intermediate_dense(attention_output_113)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_91 = self.L__mod___mobilebert_encoder_layer_22_intermediate_intermediate_act_fn(hidden_states_204);  hidden_states_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    layer_output_88 = self.L__mod___mobilebert_encoder_layer_22_output_dense(intermediate_output_91);  intermediate_output_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_344 = layer_output_88 + attention_output_113;  layer_output_88 = attention_output_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_22_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_22_output_LayerNorm_weight
    mul_184 = add_344 * l__mod___mobilebert_encoder_layer_22_output_layer_norm_weight_1;  add_344 = l__mod___mobilebert_encoder_layer_22_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_22_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_22_output_LayerNorm_bias
    layer_output_89 = mul_184 + l__mod___mobilebert_encoder_layer_22_output_layer_norm_bias_1;  mul_184 = l__mod___mobilebert_encoder_layer_22_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_319 = self.L__mod___mobilebert_encoder_layer_22_output_bottleneck_dense(layer_output_89);  layer_output_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    layer_outputs_320 = self.L__mod___mobilebert_encoder_layer_22_output_bottleneck_dropout(layer_outputs_319);  layer_outputs_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_346 = layer_outputs_320 + value_tensor_22;  layer_outputs_320 = value_tensor_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_22_output_bottleneck_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_22_output_bottleneck_LayerNorm_weight
    mul_185 = add_346 * l__mod___mobilebert_encoder_layer_22_output_bottleneck_layer_norm_weight_1;  add_346 = l__mod___mobilebert_encoder_layer_22_output_bottleneck_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_22_output_bottleneck_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_22_output_bottleneck_LayerNorm_bias
    value_tensor_23 = mul_185 + l__mod___mobilebert_encoder_layer_22_output_bottleneck_layer_norm_bias_1;  mul_185 = l__mod___mobilebert_encoder_layer_22_output_bottleneck_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:549, code: torch.tensor(1000),
    tensor_22 = torch.tensor(1000)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_115 = self.L__mod___mobilebert_encoder_layer_23_bottleneck_input_dense(value_tensor_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_23_bottleneck_input_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_23_bottleneck_input_LayerNorm_weight
    mul_186 = layer_input_115 * l__mod___mobilebert_encoder_layer_23_bottleneck_input_layer_norm_weight_1;  layer_input_115 = l__mod___mobilebert_encoder_layer_23_bottleneck_input_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_23_bottleneck_input_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_23_bottleneck_input_LayerNorm_bias
    layer_input_119 = mul_186 + l__mod___mobilebert_encoder_layer_23_bottleneck_input_layer_norm_bias_1;  mul_186 = l__mod___mobilebert_encoder_layer_23_bottleneck_input_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    layer_input_117 = self.L__mod___mobilebert_encoder_layer_23_bottleneck_attention_dense(value_tensor_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_23_bottleneck_attention_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_23_bottleneck_attention_LayerNorm_weight
    mul_187 = layer_input_117 * l__mod___mobilebert_encoder_layer_23_bottleneck_attention_layer_norm_weight_1;  layer_input_117 = l__mod___mobilebert_encoder_layer_23_bottleneck_attention_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_23_bottleneck_attention_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_23_bottleneck_attention_LayerNorm_bias
    key_tensor_23 = mul_187 + l__mod___mobilebert_encoder_layer_23_bottleneck_attention_layer_norm_bias_1;  mul_187 = l__mod___mobilebert_encoder_layer_23_bottleneck_attention_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    mixed_query_layer_23 = self.L__mod___mobilebert_encoder_layer_23_attention_self_query(key_tensor_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    mixed_key_layer_23 = self.L__mod___mobilebert_encoder_layer_23_attention_self_key(key_tensor_23);  key_tensor_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    mixed_value_layer_23 = self.L__mod___mobilebert_encoder_layer_23_attention_self_value(value_tensor_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_69 = mixed_query_layer_23.view((1, 128, 4, 32));  mixed_query_layer_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    query_layer_23 = x_69.permute(0, 2, 1, 3);  x_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_70 = mixed_key_layer_23.view((1, 128, 4, 32));  mixed_key_layer_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    key_layer_23 = x_70.permute(0, 2, 1, 3);  x_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    x_71 = mixed_value_layer_23.view((1, 128, 4, 32));  mixed_value_layer_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    value_layer_23 = x_71.permute(0, 2, 1, 3);  x_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:286, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_23 = key_layer_23.transpose(-1, -2);  key_layer_23 = None
    attention_scores_69 = torch.matmul(query_layer_23, transpose_23);  query_layer_23 = transpose_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:287, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_70 = attention_scores_69 / 5.656854249492381;  attention_scores_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:290, code: attention_scores = attention_scores + attention_mask
    attention_scores_71 = attention_scores_70 + extended_attention_mask_3;  attention_scores_70 = extended_attention_mask_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:292, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_46 = torch.nn.functional.softmax(attention_scores_71, dim = -1);  attention_scores_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295, code: attention_probs = self.dropout(attention_probs)
    attention_probs_47 = self.L__mod___mobilebert_encoder_layer_23_attention_self_dropout(attention_probs_46);  attention_probs_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:299, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_69 = torch.matmul(attention_probs_47, value_layer_23);  attention_probs_47 = value_layer_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_95 = context_layer_69.permute(0, 2, 1, 3);  context_layer_69 = None
    context_layer_70 = permute_95.contiguous();  permute_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    context_layer_71 = context_layer_70.view((1, 128, 128));  context_layer_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_322 = self.L__mod___mobilebert_encoder_layer_23_attention_output_dense(context_layer_71);  context_layer_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_351 = layer_outputs_322 + layer_input_119;  layer_outputs_322 = layer_input_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_23_attention_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_23_attention_output_LayerNorm_weight
    mul_188 = add_351 * l__mod___mobilebert_encoder_layer_23_attention_output_layer_norm_weight_1;  add_351 = l__mod___mobilebert_encoder_layer_23_attention_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_23_attention_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_23_attention_output_LayerNorm_bias
    attention_output_115 = mul_188 + l__mod___mobilebert_encoder_layer_23_attention_output_layer_norm_bias_1;  mul_188 = l__mod___mobilebert_encoder_layer_23_attention_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_207 = self.L__mod___mobilebert_encoder_layer_23_ffn_0_intermediate_dense(attention_output_115)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_92 = self.L__mod___mobilebert_encoder_layer_23_ffn_0_intermediate_intermediate_act_fn(hidden_states_207);  hidden_states_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_324 = self.L__mod___mobilebert_encoder_layer_23_ffn_0_output_dense(intermediate_output_92);  intermediate_output_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_353 = layer_outputs_324 + attention_output_115;  layer_outputs_324 = attention_output_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_23_ffn_0_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_23_ffn_0_output_LayerNorm_weight
    mul_189 = add_353 * l__mod___mobilebert_encoder_layer_23_ffn_0_output_layer_norm_weight_1;  add_353 = l__mod___mobilebert_encoder_layer_23_ffn_0_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_23_ffn_0_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_23_ffn_0_output_LayerNorm_bias
    attention_output_116 = mul_189 + l__mod___mobilebert_encoder_layer_23_ffn_0_output_layer_norm_bias_1;  mul_189 = l__mod___mobilebert_encoder_layer_23_ffn_0_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_209 = self.L__mod___mobilebert_encoder_layer_23_ffn_1_intermediate_dense(attention_output_116)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_93 = self.L__mod___mobilebert_encoder_layer_23_ffn_1_intermediate_intermediate_act_fn(hidden_states_209);  hidden_states_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_327 = self.L__mod___mobilebert_encoder_layer_23_ffn_1_output_dense(intermediate_output_93);  intermediate_output_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_355 = layer_outputs_327 + attention_output_116;  layer_outputs_327 = attention_output_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_23_ffn_1_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_23_ffn_1_output_LayerNorm_weight
    mul_190 = add_355 * l__mod___mobilebert_encoder_layer_23_ffn_1_output_layer_norm_weight_1;  add_355 = l__mod___mobilebert_encoder_layer_23_ffn_1_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_23_ffn_1_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_23_ffn_1_output_LayerNorm_bias
    attention_output_117 = mul_190 + l__mod___mobilebert_encoder_layer_23_ffn_1_output_layer_norm_bias_1;  mul_190 = l__mod___mobilebert_encoder_layer_23_ffn_1_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_211 = self.L__mod___mobilebert_encoder_layer_23_ffn_2_intermediate_dense(attention_output_117)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_94 = self.L__mod___mobilebert_encoder_layer_23_ffn_2_intermediate_intermediate_act_fn(hidden_states_211);  hidden_states_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_330 = self.L__mod___mobilebert_encoder_layer_23_ffn_2_output_dense(intermediate_output_94);  intermediate_output_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_357 = layer_outputs_330 + attention_output_117;  layer_outputs_330 = attention_output_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_23_ffn_2_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_23_ffn_2_output_LayerNorm_weight
    mul_191 = add_357 * l__mod___mobilebert_encoder_layer_23_ffn_2_output_layer_norm_weight_1;  add_357 = l__mod___mobilebert_encoder_layer_23_ffn_2_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_23_ffn_2_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_23_ffn_2_output_LayerNorm_bias
    attention_output_118 = mul_191 + l__mod___mobilebert_encoder_layer_23_ffn_2_output_layer_norm_bias_1;  mul_191 = l__mod___mobilebert_encoder_layer_23_ffn_2_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    hidden_states_213 = self.L__mod___mobilebert_encoder_layer_23_intermediate_dense(attention_output_118)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    intermediate_output_95 = self.L__mod___mobilebert_encoder_layer_23_intermediate_intermediate_act_fn(hidden_states_213);  hidden_states_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    layer_output_92 = self.L__mod___mobilebert_encoder_layer_23_output_dense(intermediate_output_95);  intermediate_output_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_359 = layer_output_92 + attention_output_118;  layer_output_92 = attention_output_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_23_output_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_23_output_LayerNorm_weight
    mul_192 = add_359 * l__mod___mobilebert_encoder_layer_23_output_layer_norm_weight_1;  add_359 = l__mod___mobilebert_encoder_layer_23_output_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_23_output_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_23_output_LayerNorm_bias
    layer_output_93 = mul_192 + l__mod___mobilebert_encoder_layer_23_output_layer_norm_bias_1;  mul_192 = l__mod___mobilebert_encoder_layer_23_output_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    layer_outputs_333 = self.L__mod___mobilebert_encoder_layer_23_output_bottleneck_dense(layer_output_93);  layer_output_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    layer_outputs_334 = self.L__mod___mobilebert_encoder_layer_23_output_bottleneck_dropout(layer_outputs_333);  layer_outputs_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_361 = layer_outputs_334 + value_tensor_23;  layer_outputs_334 = value_tensor_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    l__mod___mobilebert_encoder_layer_23_output_bottleneck_layer_norm_weight_1 = self.L__mod___mobilebert_encoder_layer_23_output_bottleneck_LayerNorm_weight
    mul_193 = add_361 * l__mod___mobilebert_encoder_layer_23_output_bottleneck_layer_norm_weight_1;  add_361 = l__mod___mobilebert_encoder_layer_23_output_bottleneck_layer_norm_weight_1 = None
    l__mod___mobilebert_encoder_layer_23_output_bottleneck_layer_norm_bias_1 = self.L__mod___mobilebert_encoder_layer_23_output_bottleneck_LayerNorm_bias
    sequence_output = mul_193 + l__mod___mobilebert_encoder_layer_23_output_bottleneck_layer_norm_bias_1;  mul_193 = l__mod___mobilebert_encoder_layer_23_output_bottleneck_layer_norm_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:549, code: torch.tensor(1000),
    tensor_23 = torch.tensor(1000)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:1404, code: logits = self.qa_outputs(sequence_output)
    logits = self.L__mod___qa_outputs(sequence_output);  sequence_output = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:1405, code: start_logits, end_logits = logits.split(1, dim=-1)
    split = logits.split(1, dim = -1);  logits = None
    start_logits = split[0]
    end_logits = split[1];  split = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:1406, code: start_logits = start_logits.squeeze(-1).contiguous()
    squeeze = start_logits.squeeze(-1);  start_logits = None
    start_logits_1 = squeeze.contiguous();  squeeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:1407, code: end_logits = end_logits.squeeze(-1).contiguous()
    squeeze_1 = end_logits.squeeze(-1);  end_logits = None
    end_logits_1 = squeeze_1.contiguous();  squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:1418, code: start_positions = start_positions.clamp(0, ignored_index)
    start_positions = l_cloned_inputs_start_positions_.clamp(0, 128);  l_cloned_inputs_start_positions_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:1419, code: end_positions = end_positions.clamp(0, ignored_index)
    end_positions = l_cloned_inputs_end_positions_.clamp(0, 128);  l_cloned_inputs_end_positions_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:1422, code: start_loss = loss_fct(start_logits, start_positions)
    start_loss = torch.nn.functional.cross_entropy(start_logits_1, start_positions, None, None, 128, None, 'mean', 0.0);  start_positions = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:1423, code: end_loss = loss_fct(end_logits, end_positions)
    end_loss = torch.nn.functional.cross_entropy(end_logits_1, end_positions, None, None, 128, None, 'mean', 0.0);  end_positions = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:1424, code: total_loss = (start_loss + end_loss) / 2
    add_363 = start_loss + end_loss;  start_loss = end_loss = None
    loss = add_363 / 2;  add_363 = None
    return (loss, start_logits_1, end_logits_1)
    