from __future__ import annotations



def forward(self, L_inputs_input_ids_ : torch.Tensor, L_inputs_labels_ : torch.Tensor):
    l_inputs_input_ids_ = L_inputs_input_ids_
    l_inputs_labels_ = L_inputs_labels_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:715, code: attention_mask = torch.ones(input_shape, device=device)
    attention_mask = torch.ones((1, 512), device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:718, code: buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
    l__mod___albert_embeddings_token_type_ids = self.L__mod___albert_embeddings_token_type_ids
    buffered_token_type_ids = l__mod___albert_embeddings_token_type_ids[(slice(None, None, None), slice(None, 512, None))];  l__mod___albert_embeddings_token_type_ids = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:719, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
    token_type_ids = buffered_token_type_ids.expand(1, 512);  buffered_token_type_ids = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:724, code: extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    unsqueeze = attention_mask.unsqueeze(1);  attention_mask = None
    extended_attention_mask = unsqueeze.unsqueeze(2);  unsqueeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:725, code: extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
    extended_attention_mask_1 = extended_attention_mask.to(dtype = torch.float32);  extended_attention_mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:726, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min
    sub = 1.0 - extended_attention_mask_1;  extended_attention_mask_1 = None
    extended_attention_mask_2 = sub * -3.4028234663852886e+38;  sub = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:236, code: position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
    l__mod___albert_embeddings_position_ids = self.L__mod___albert_embeddings_position_ids
    position_ids = l__mod___albert_embeddings_position_ids[(slice(None, None, None), slice(0, 512, None))];  l__mod___albert_embeddings_position_ids = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:250, code: inputs_embeds = self.word_embeddings(input_ids)
    inputs_embeds = self.L__mod___albert_embeddings_word_embeddings(l_inputs_input_ids_);  l_inputs_input_ids_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:251, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    token_type_embeddings = self.L__mod___albert_embeddings_token_type_embeddings(token_type_ids);  token_type_ids = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:253, code: embeddings = inputs_embeds + token_type_embeddings
    embeddings = inputs_embeds + token_type_embeddings;  inputs_embeds = token_type_embeddings = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:255, code: position_embeddings = self.position_embeddings(position_ids)
    position_embeddings = self.L__mod___albert_embeddings_position_embeddings(position_ids);  position_ids = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:256, code: embeddings += position_embeddings
    embeddings += position_embeddings;  embeddings_1 = embeddings;  embeddings = position_embeddings = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:257, code: embeddings = self.LayerNorm(embeddings)
    embeddings_2 = self.L__mod___albert_embeddings_LayerNorm(embeddings_1);  embeddings_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:258, code: embeddings = self.dropout(embeddings)
    embedding_output = self.L__mod___albert_embeddings_dropout(embeddings_2);  embeddings_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:467, code: hidden_states = self.embedding_hidden_mapping_in(hidden_states)
    hidden_states = self.L__mod___albert_encoder_embedding_hidden_mapping_in(embedding_output);  embedding_output = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    mixed_query_layer = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_query(hidden_states)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    mixed_key_layer = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_key(hidden_states)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    mixed_value_layer = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_value(hidden_states)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    x = mixed_query_layer.view((1, 512, 64, 64));  mixed_query_layer = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    query_layer = x.permute(0, 2, 1, 3);  x = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    x_1 = mixed_key_layer.view((1, 512, 64, 64));  mixed_key_layer = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    key_layer = x_1.permute(0, 2, 1, 3);  x_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    x_2 = mixed_value_layer.view((1, 512, 64, 64));  mixed_value_layer = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    value_layer = x_2.permute(0, 2, 1, 3);  x_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose = key_layer.transpose(-1, -2);  key_layer = None
    attention_scores = torch.matmul(query_layer, transpose);  query_layer = transpose = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_1 = attention_scores / 8.0;  attention_scores = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:336, code: attention_scores = attention_scores + attention_mask
    attention_scores_2 = attention_scores_1 + extended_attention_mask_2;  attention_scores_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs = torch.nn.functional.softmax(attention_scores_2, dim = -1);  attention_scores_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    attention_probs_1 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_attention_dropout(attention_probs);  attention_probs = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer = torch.matmul(attention_probs_1, value_layer);  attention_probs_1 = value_layer = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    transpose_1 = context_layer.transpose(2, 1);  context_layer = None
    context_layer_1 = transpose_1.flatten(2);  transpose_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    projected_context_layer = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_dense(context_layer_1);  context_layer_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:369, code: projected_context_layer_dropout = self.output_dropout(projected_context_layer)
    projected_context_layer_dropout = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_output_dropout(projected_context_layer);  projected_context_layer = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_2 = hidden_states + projected_context_layer_dropout;  hidden_states = projected_context_layer_dropout = None
    layernormed_context_layer = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_LayerNorm(add_2);  add_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    ffn_output = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_ffn(layernormed_context_layer)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_1 = 0.5 * ffn_output
    pow_1 = torch.pow(ffn_output, 3.0)
    mul_2 = 0.044715 * pow_1;  pow_1 = None
    add_3 = ffn_output + mul_2;  ffn_output = mul_2 = None
    mul_3 = 0.7978845608028654 * add_3;  add_3 = None
    tanh = torch.tanh(mul_3);  mul_3 = None
    add_4 = 1.0 + tanh;  tanh = None
    ffn_output_1 = mul_1 * add_4;  mul_1 = add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    ffn_output_3 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_ffn_output(ffn_output_1);  ffn_output_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_5 = ffn_output_3 + layernormed_context_layer;  ffn_output_3 = layernormed_context_layer = None
    hidden_states_3 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_full_layer_layer_norm(add_5);  add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    mixed_query_layer_1 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_query(hidden_states_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    mixed_key_layer_1 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_key(hidden_states_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    mixed_value_layer_1 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_value(hidden_states_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    x_3 = mixed_query_layer_1.view((1, 512, 64, 64));  mixed_query_layer_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    query_layer_1 = x_3.permute(0, 2, 1, 3);  x_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    x_4 = mixed_key_layer_1.view((1, 512, 64, 64));  mixed_key_layer_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    key_layer_1 = x_4.permute(0, 2, 1, 3);  x_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    x_5 = mixed_value_layer_1.view((1, 512, 64, 64));  mixed_value_layer_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    value_layer_1 = x_5.permute(0, 2, 1, 3);  x_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_2 = key_layer_1.transpose(-1, -2);  key_layer_1 = None
    attention_scores_3 = torch.matmul(query_layer_1, transpose_2);  query_layer_1 = transpose_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_4 = attention_scores_3 / 8.0;  attention_scores_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:336, code: attention_scores = attention_scores + attention_mask
    attention_scores_5 = attention_scores_4 + extended_attention_mask_2;  attention_scores_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_2 = torch.nn.functional.softmax(attention_scores_5, dim = -1);  attention_scores_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    attention_probs_3 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_attention_dropout(attention_probs_2);  attention_probs_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_2 = torch.matmul(attention_probs_3, value_layer_1);  attention_probs_3 = value_layer_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    transpose_3 = context_layer_2.transpose(2, 1);  context_layer_2 = None
    context_layer_3 = transpose_3.flatten(2);  transpose_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    projected_context_layer_1 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_dense(context_layer_3);  context_layer_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:369, code: projected_context_layer_dropout = self.output_dropout(projected_context_layer)
    projected_context_layer_dropout_1 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_output_dropout(projected_context_layer_1);  projected_context_layer_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_7 = hidden_states_3 + projected_context_layer_dropout_1;  hidden_states_3 = projected_context_layer_dropout_1 = None
    layernormed_context_layer_1 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_LayerNorm(add_7);  add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    ffn_output_4 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_ffn(layernormed_context_layer_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_5 = 0.5 * ffn_output_4
    pow_2 = torch.pow(ffn_output_4, 3.0)
    mul_6 = 0.044715 * pow_2;  pow_2 = None
    add_8 = ffn_output_4 + mul_6;  ffn_output_4 = mul_6 = None
    mul_7 = 0.7978845608028654 * add_8;  add_8 = None
    tanh_1 = torch.tanh(mul_7);  mul_7 = None
    add_9 = 1.0 + tanh_1;  tanh_1 = None
    ffn_output_5 = mul_5 * add_9;  mul_5 = add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    ffn_output_7 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_ffn_output(ffn_output_5);  ffn_output_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_10 = ffn_output_7 + layernormed_context_layer_1;  ffn_output_7 = layernormed_context_layer_1 = None
    hidden_states_6 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_full_layer_layer_norm(add_10);  add_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    mixed_query_layer_2 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_query(hidden_states_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    mixed_key_layer_2 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_key(hidden_states_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    mixed_value_layer_2 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_value(hidden_states_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    x_6 = mixed_query_layer_2.view((1, 512, 64, 64));  mixed_query_layer_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    query_layer_2 = x_6.permute(0, 2, 1, 3);  x_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    x_7 = mixed_key_layer_2.view((1, 512, 64, 64));  mixed_key_layer_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    key_layer_2 = x_7.permute(0, 2, 1, 3);  x_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    x_8 = mixed_value_layer_2.view((1, 512, 64, 64));  mixed_value_layer_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    value_layer_2 = x_8.permute(0, 2, 1, 3);  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_4 = key_layer_2.transpose(-1, -2);  key_layer_2 = None
    attention_scores_6 = torch.matmul(query_layer_2, transpose_4);  query_layer_2 = transpose_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_7 = attention_scores_6 / 8.0;  attention_scores_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:336, code: attention_scores = attention_scores + attention_mask
    attention_scores_8 = attention_scores_7 + extended_attention_mask_2;  attention_scores_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_4 = torch.nn.functional.softmax(attention_scores_8, dim = -1);  attention_scores_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    attention_probs_5 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_attention_dropout(attention_probs_4);  attention_probs_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_4 = torch.matmul(attention_probs_5, value_layer_2);  attention_probs_5 = value_layer_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    transpose_5 = context_layer_4.transpose(2, 1);  context_layer_4 = None
    context_layer_5 = transpose_5.flatten(2);  transpose_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    projected_context_layer_2 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_dense(context_layer_5);  context_layer_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:369, code: projected_context_layer_dropout = self.output_dropout(projected_context_layer)
    projected_context_layer_dropout_2 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_output_dropout(projected_context_layer_2);  projected_context_layer_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_12 = hidden_states_6 + projected_context_layer_dropout_2;  hidden_states_6 = projected_context_layer_dropout_2 = None
    layernormed_context_layer_2 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_LayerNorm(add_12);  add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    ffn_output_8 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_ffn(layernormed_context_layer_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_9 = 0.5 * ffn_output_8
    pow_3 = torch.pow(ffn_output_8, 3.0)
    mul_10 = 0.044715 * pow_3;  pow_3 = None
    add_13 = ffn_output_8 + mul_10;  ffn_output_8 = mul_10 = None
    mul_11 = 0.7978845608028654 * add_13;  add_13 = None
    tanh_2 = torch.tanh(mul_11);  mul_11 = None
    add_14 = 1.0 + tanh_2;  tanh_2 = None
    ffn_output_9 = mul_9 * add_14;  mul_9 = add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    ffn_output_11 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_ffn_output(ffn_output_9);  ffn_output_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_15 = ffn_output_11 + layernormed_context_layer_2;  ffn_output_11 = layernormed_context_layer_2 = None
    hidden_states_9 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_full_layer_layer_norm(add_15);  add_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    mixed_query_layer_3 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_query(hidden_states_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    mixed_key_layer_3 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_key(hidden_states_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    mixed_value_layer_3 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_value(hidden_states_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    x_9 = mixed_query_layer_3.view((1, 512, 64, 64));  mixed_query_layer_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    query_layer_3 = x_9.permute(0, 2, 1, 3);  x_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    x_10 = mixed_key_layer_3.view((1, 512, 64, 64));  mixed_key_layer_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    key_layer_3 = x_10.permute(0, 2, 1, 3);  x_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    x_11 = mixed_value_layer_3.view((1, 512, 64, 64));  mixed_value_layer_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    value_layer_3 = x_11.permute(0, 2, 1, 3);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_6 = key_layer_3.transpose(-1, -2);  key_layer_3 = None
    attention_scores_9 = torch.matmul(query_layer_3, transpose_6);  query_layer_3 = transpose_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_10 = attention_scores_9 / 8.0;  attention_scores_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:336, code: attention_scores = attention_scores + attention_mask
    attention_scores_11 = attention_scores_10 + extended_attention_mask_2;  attention_scores_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_6 = torch.nn.functional.softmax(attention_scores_11, dim = -1);  attention_scores_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    attention_probs_7 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_attention_dropout(attention_probs_6);  attention_probs_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_6 = torch.matmul(attention_probs_7, value_layer_3);  attention_probs_7 = value_layer_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    transpose_7 = context_layer_6.transpose(2, 1);  context_layer_6 = None
    context_layer_7 = transpose_7.flatten(2);  transpose_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    projected_context_layer_3 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_dense(context_layer_7);  context_layer_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:369, code: projected_context_layer_dropout = self.output_dropout(projected_context_layer)
    projected_context_layer_dropout_3 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_output_dropout(projected_context_layer_3);  projected_context_layer_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_17 = hidden_states_9 + projected_context_layer_dropout_3;  hidden_states_9 = projected_context_layer_dropout_3 = None
    layernormed_context_layer_3 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_LayerNorm(add_17);  add_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    ffn_output_12 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_ffn(layernormed_context_layer_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_13 = 0.5 * ffn_output_12
    pow_4 = torch.pow(ffn_output_12, 3.0)
    mul_14 = 0.044715 * pow_4;  pow_4 = None
    add_18 = ffn_output_12 + mul_14;  ffn_output_12 = mul_14 = None
    mul_15 = 0.7978845608028654 * add_18;  add_18 = None
    tanh_3 = torch.tanh(mul_15);  mul_15 = None
    add_19 = 1.0 + tanh_3;  tanh_3 = None
    ffn_output_13 = mul_13 * add_19;  mul_13 = add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    ffn_output_15 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_ffn_output(ffn_output_13);  ffn_output_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_20 = ffn_output_15 + layernormed_context_layer_3;  ffn_output_15 = layernormed_context_layer_3 = None
    hidden_states_12 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_full_layer_layer_norm(add_20);  add_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    mixed_query_layer_4 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_query(hidden_states_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    mixed_key_layer_4 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_key(hidden_states_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    mixed_value_layer_4 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_value(hidden_states_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    x_12 = mixed_query_layer_4.view((1, 512, 64, 64));  mixed_query_layer_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    query_layer_4 = x_12.permute(0, 2, 1, 3);  x_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    x_13 = mixed_key_layer_4.view((1, 512, 64, 64));  mixed_key_layer_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    key_layer_4 = x_13.permute(0, 2, 1, 3);  x_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    x_14 = mixed_value_layer_4.view((1, 512, 64, 64));  mixed_value_layer_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    value_layer_4 = x_14.permute(0, 2, 1, 3);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_8 = key_layer_4.transpose(-1, -2);  key_layer_4 = None
    attention_scores_12 = torch.matmul(query_layer_4, transpose_8);  query_layer_4 = transpose_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_13 = attention_scores_12 / 8.0;  attention_scores_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:336, code: attention_scores = attention_scores + attention_mask
    attention_scores_14 = attention_scores_13 + extended_attention_mask_2;  attention_scores_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_8 = torch.nn.functional.softmax(attention_scores_14, dim = -1);  attention_scores_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    attention_probs_9 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_attention_dropout(attention_probs_8);  attention_probs_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_8 = torch.matmul(attention_probs_9, value_layer_4);  attention_probs_9 = value_layer_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    transpose_9 = context_layer_8.transpose(2, 1);  context_layer_8 = None
    context_layer_9 = transpose_9.flatten(2);  transpose_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    projected_context_layer_4 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_dense(context_layer_9);  context_layer_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:369, code: projected_context_layer_dropout = self.output_dropout(projected_context_layer)
    projected_context_layer_dropout_4 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_output_dropout(projected_context_layer_4);  projected_context_layer_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_22 = hidden_states_12 + projected_context_layer_dropout_4;  hidden_states_12 = projected_context_layer_dropout_4 = None
    layernormed_context_layer_4 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_LayerNorm(add_22);  add_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    ffn_output_16 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_ffn(layernormed_context_layer_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_17 = 0.5 * ffn_output_16
    pow_5 = torch.pow(ffn_output_16, 3.0)
    mul_18 = 0.044715 * pow_5;  pow_5 = None
    add_23 = ffn_output_16 + mul_18;  ffn_output_16 = mul_18 = None
    mul_19 = 0.7978845608028654 * add_23;  add_23 = None
    tanh_4 = torch.tanh(mul_19);  mul_19 = None
    add_24 = 1.0 + tanh_4;  tanh_4 = None
    ffn_output_17 = mul_17 * add_24;  mul_17 = add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    ffn_output_19 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_ffn_output(ffn_output_17);  ffn_output_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_25 = ffn_output_19 + layernormed_context_layer_4;  ffn_output_19 = layernormed_context_layer_4 = None
    hidden_states_15 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_full_layer_layer_norm(add_25);  add_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    mixed_query_layer_5 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_query(hidden_states_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    mixed_key_layer_5 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_key(hidden_states_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    mixed_value_layer_5 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_value(hidden_states_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    x_15 = mixed_query_layer_5.view((1, 512, 64, 64));  mixed_query_layer_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    query_layer_5 = x_15.permute(0, 2, 1, 3);  x_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    x_16 = mixed_key_layer_5.view((1, 512, 64, 64));  mixed_key_layer_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    key_layer_5 = x_16.permute(0, 2, 1, 3);  x_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    x_17 = mixed_value_layer_5.view((1, 512, 64, 64));  mixed_value_layer_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    value_layer_5 = x_17.permute(0, 2, 1, 3);  x_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_10 = key_layer_5.transpose(-1, -2);  key_layer_5 = None
    attention_scores_15 = torch.matmul(query_layer_5, transpose_10);  query_layer_5 = transpose_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_16 = attention_scores_15 / 8.0;  attention_scores_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:336, code: attention_scores = attention_scores + attention_mask
    attention_scores_17 = attention_scores_16 + extended_attention_mask_2;  attention_scores_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_10 = torch.nn.functional.softmax(attention_scores_17, dim = -1);  attention_scores_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    attention_probs_11 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_attention_dropout(attention_probs_10);  attention_probs_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_10 = torch.matmul(attention_probs_11, value_layer_5);  attention_probs_11 = value_layer_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    transpose_11 = context_layer_10.transpose(2, 1);  context_layer_10 = None
    context_layer_11 = transpose_11.flatten(2);  transpose_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    projected_context_layer_5 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_dense(context_layer_11);  context_layer_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:369, code: projected_context_layer_dropout = self.output_dropout(projected_context_layer)
    projected_context_layer_dropout_5 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_output_dropout(projected_context_layer_5);  projected_context_layer_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_27 = hidden_states_15 + projected_context_layer_dropout_5;  hidden_states_15 = projected_context_layer_dropout_5 = None
    layernormed_context_layer_5 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_LayerNorm(add_27);  add_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    ffn_output_20 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_ffn(layernormed_context_layer_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_21 = 0.5 * ffn_output_20
    pow_6 = torch.pow(ffn_output_20, 3.0)
    mul_22 = 0.044715 * pow_6;  pow_6 = None
    add_28 = ffn_output_20 + mul_22;  ffn_output_20 = mul_22 = None
    mul_23 = 0.7978845608028654 * add_28;  add_28 = None
    tanh_5 = torch.tanh(mul_23);  mul_23 = None
    add_29 = 1.0 + tanh_5;  tanh_5 = None
    ffn_output_21 = mul_21 * add_29;  mul_21 = add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    ffn_output_23 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_ffn_output(ffn_output_21);  ffn_output_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_30 = ffn_output_23 + layernormed_context_layer_5;  ffn_output_23 = layernormed_context_layer_5 = None
    hidden_states_18 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_full_layer_layer_norm(add_30);  add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    mixed_query_layer_6 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_query(hidden_states_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    mixed_key_layer_6 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_key(hidden_states_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    mixed_value_layer_6 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_value(hidden_states_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    x_18 = mixed_query_layer_6.view((1, 512, 64, 64));  mixed_query_layer_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    query_layer_6 = x_18.permute(0, 2, 1, 3);  x_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    x_19 = mixed_key_layer_6.view((1, 512, 64, 64));  mixed_key_layer_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    key_layer_6 = x_19.permute(0, 2, 1, 3);  x_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    x_20 = mixed_value_layer_6.view((1, 512, 64, 64));  mixed_value_layer_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    value_layer_6 = x_20.permute(0, 2, 1, 3);  x_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_12 = key_layer_6.transpose(-1, -2);  key_layer_6 = None
    attention_scores_18 = torch.matmul(query_layer_6, transpose_12);  query_layer_6 = transpose_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_19 = attention_scores_18 / 8.0;  attention_scores_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:336, code: attention_scores = attention_scores + attention_mask
    attention_scores_20 = attention_scores_19 + extended_attention_mask_2;  attention_scores_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_12 = torch.nn.functional.softmax(attention_scores_20, dim = -1);  attention_scores_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    attention_probs_13 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_attention_dropout(attention_probs_12);  attention_probs_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_12 = torch.matmul(attention_probs_13, value_layer_6);  attention_probs_13 = value_layer_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    transpose_13 = context_layer_12.transpose(2, 1);  context_layer_12 = None
    context_layer_13 = transpose_13.flatten(2);  transpose_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    projected_context_layer_6 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_dense(context_layer_13);  context_layer_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:369, code: projected_context_layer_dropout = self.output_dropout(projected_context_layer)
    projected_context_layer_dropout_6 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_output_dropout(projected_context_layer_6);  projected_context_layer_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_32 = hidden_states_18 + projected_context_layer_dropout_6;  hidden_states_18 = projected_context_layer_dropout_6 = None
    layernormed_context_layer_6 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_LayerNorm(add_32);  add_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    ffn_output_24 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_ffn(layernormed_context_layer_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_25 = 0.5 * ffn_output_24
    pow_7 = torch.pow(ffn_output_24, 3.0)
    mul_26 = 0.044715 * pow_7;  pow_7 = None
    add_33 = ffn_output_24 + mul_26;  ffn_output_24 = mul_26 = None
    mul_27 = 0.7978845608028654 * add_33;  add_33 = None
    tanh_6 = torch.tanh(mul_27);  mul_27 = None
    add_34 = 1.0 + tanh_6;  tanh_6 = None
    ffn_output_25 = mul_25 * add_34;  mul_25 = add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    ffn_output_27 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_ffn_output(ffn_output_25);  ffn_output_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_35 = ffn_output_27 + layernormed_context_layer_6;  ffn_output_27 = layernormed_context_layer_6 = None
    hidden_states_21 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_full_layer_layer_norm(add_35);  add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    mixed_query_layer_7 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_query(hidden_states_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    mixed_key_layer_7 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_key(hidden_states_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    mixed_value_layer_7 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_value(hidden_states_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    x_21 = mixed_query_layer_7.view((1, 512, 64, 64));  mixed_query_layer_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    query_layer_7 = x_21.permute(0, 2, 1, 3);  x_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    x_22 = mixed_key_layer_7.view((1, 512, 64, 64));  mixed_key_layer_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    key_layer_7 = x_22.permute(0, 2, 1, 3);  x_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    x_23 = mixed_value_layer_7.view((1, 512, 64, 64));  mixed_value_layer_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    value_layer_7 = x_23.permute(0, 2, 1, 3);  x_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_14 = key_layer_7.transpose(-1, -2);  key_layer_7 = None
    attention_scores_21 = torch.matmul(query_layer_7, transpose_14);  query_layer_7 = transpose_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_22 = attention_scores_21 / 8.0;  attention_scores_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:336, code: attention_scores = attention_scores + attention_mask
    attention_scores_23 = attention_scores_22 + extended_attention_mask_2;  attention_scores_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_14 = torch.nn.functional.softmax(attention_scores_23, dim = -1);  attention_scores_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    attention_probs_15 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_attention_dropout(attention_probs_14);  attention_probs_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_14 = torch.matmul(attention_probs_15, value_layer_7);  attention_probs_15 = value_layer_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    transpose_15 = context_layer_14.transpose(2, 1);  context_layer_14 = None
    context_layer_15 = transpose_15.flatten(2);  transpose_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    projected_context_layer_7 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_dense(context_layer_15);  context_layer_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:369, code: projected_context_layer_dropout = self.output_dropout(projected_context_layer)
    projected_context_layer_dropout_7 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_output_dropout(projected_context_layer_7);  projected_context_layer_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_37 = hidden_states_21 + projected_context_layer_dropout_7;  hidden_states_21 = projected_context_layer_dropout_7 = None
    layernormed_context_layer_7 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_LayerNorm(add_37);  add_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    ffn_output_28 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_ffn(layernormed_context_layer_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_29 = 0.5 * ffn_output_28
    pow_8 = torch.pow(ffn_output_28, 3.0)
    mul_30 = 0.044715 * pow_8;  pow_8 = None
    add_38 = ffn_output_28 + mul_30;  ffn_output_28 = mul_30 = None
    mul_31 = 0.7978845608028654 * add_38;  add_38 = None
    tanh_7 = torch.tanh(mul_31);  mul_31 = None
    add_39 = 1.0 + tanh_7;  tanh_7 = None
    ffn_output_29 = mul_29 * add_39;  mul_29 = add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    ffn_output_31 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_ffn_output(ffn_output_29);  ffn_output_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_40 = ffn_output_31 + layernormed_context_layer_7;  ffn_output_31 = layernormed_context_layer_7 = None
    hidden_states_24 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_full_layer_layer_norm(add_40);  add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    mixed_query_layer_8 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_query(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    mixed_key_layer_8 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_key(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    mixed_value_layer_8 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_value(hidden_states_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    x_24 = mixed_query_layer_8.view((1, 512, 64, 64));  mixed_query_layer_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    query_layer_8 = x_24.permute(0, 2, 1, 3);  x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    x_25 = mixed_key_layer_8.view((1, 512, 64, 64));  mixed_key_layer_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    key_layer_8 = x_25.permute(0, 2, 1, 3);  x_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    x_26 = mixed_value_layer_8.view((1, 512, 64, 64));  mixed_value_layer_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    value_layer_8 = x_26.permute(0, 2, 1, 3);  x_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_16 = key_layer_8.transpose(-1, -2);  key_layer_8 = None
    attention_scores_24 = torch.matmul(query_layer_8, transpose_16);  query_layer_8 = transpose_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_25 = attention_scores_24 / 8.0;  attention_scores_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:336, code: attention_scores = attention_scores + attention_mask
    attention_scores_26 = attention_scores_25 + extended_attention_mask_2;  attention_scores_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_16 = torch.nn.functional.softmax(attention_scores_26, dim = -1);  attention_scores_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    attention_probs_17 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_attention_dropout(attention_probs_16);  attention_probs_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_16 = torch.matmul(attention_probs_17, value_layer_8);  attention_probs_17 = value_layer_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    transpose_17 = context_layer_16.transpose(2, 1);  context_layer_16 = None
    context_layer_17 = transpose_17.flatten(2);  transpose_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    projected_context_layer_8 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_dense(context_layer_17);  context_layer_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:369, code: projected_context_layer_dropout = self.output_dropout(projected_context_layer)
    projected_context_layer_dropout_8 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_output_dropout(projected_context_layer_8);  projected_context_layer_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_42 = hidden_states_24 + projected_context_layer_dropout_8;  hidden_states_24 = projected_context_layer_dropout_8 = None
    layernormed_context_layer_8 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_LayerNorm(add_42);  add_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    ffn_output_32 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_ffn(layernormed_context_layer_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_33 = 0.5 * ffn_output_32
    pow_9 = torch.pow(ffn_output_32, 3.0)
    mul_34 = 0.044715 * pow_9;  pow_9 = None
    add_43 = ffn_output_32 + mul_34;  ffn_output_32 = mul_34 = None
    mul_35 = 0.7978845608028654 * add_43;  add_43 = None
    tanh_8 = torch.tanh(mul_35);  mul_35 = None
    add_44 = 1.0 + tanh_8;  tanh_8 = None
    ffn_output_33 = mul_33 * add_44;  mul_33 = add_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    ffn_output_35 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_ffn_output(ffn_output_33);  ffn_output_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_45 = ffn_output_35 + layernormed_context_layer_8;  ffn_output_35 = layernormed_context_layer_8 = None
    hidden_states_27 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_full_layer_layer_norm(add_45);  add_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    mixed_query_layer_9 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_query(hidden_states_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    mixed_key_layer_9 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_key(hidden_states_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    mixed_value_layer_9 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_value(hidden_states_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    x_27 = mixed_query_layer_9.view((1, 512, 64, 64));  mixed_query_layer_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    query_layer_9 = x_27.permute(0, 2, 1, 3);  x_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    x_28 = mixed_key_layer_9.view((1, 512, 64, 64));  mixed_key_layer_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    key_layer_9 = x_28.permute(0, 2, 1, 3);  x_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    x_29 = mixed_value_layer_9.view((1, 512, 64, 64));  mixed_value_layer_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    value_layer_9 = x_29.permute(0, 2, 1, 3);  x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_18 = key_layer_9.transpose(-1, -2);  key_layer_9 = None
    attention_scores_27 = torch.matmul(query_layer_9, transpose_18);  query_layer_9 = transpose_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_28 = attention_scores_27 / 8.0;  attention_scores_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:336, code: attention_scores = attention_scores + attention_mask
    attention_scores_29 = attention_scores_28 + extended_attention_mask_2;  attention_scores_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_18 = torch.nn.functional.softmax(attention_scores_29, dim = -1);  attention_scores_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    attention_probs_19 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_attention_dropout(attention_probs_18);  attention_probs_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_18 = torch.matmul(attention_probs_19, value_layer_9);  attention_probs_19 = value_layer_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    transpose_19 = context_layer_18.transpose(2, 1);  context_layer_18 = None
    context_layer_19 = transpose_19.flatten(2);  transpose_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    projected_context_layer_9 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_dense(context_layer_19);  context_layer_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:369, code: projected_context_layer_dropout = self.output_dropout(projected_context_layer)
    projected_context_layer_dropout_9 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_output_dropout(projected_context_layer_9);  projected_context_layer_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_47 = hidden_states_27 + projected_context_layer_dropout_9;  hidden_states_27 = projected_context_layer_dropout_9 = None
    layernormed_context_layer_9 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_LayerNorm(add_47);  add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    ffn_output_36 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_ffn(layernormed_context_layer_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_37 = 0.5 * ffn_output_36
    pow_10 = torch.pow(ffn_output_36, 3.0)
    mul_38 = 0.044715 * pow_10;  pow_10 = None
    add_48 = ffn_output_36 + mul_38;  ffn_output_36 = mul_38 = None
    mul_39 = 0.7978845608028654 * add_48;  add_48 = None
    tanh_9 = torch.tanh(mul_39);  mul_39 = None
    add_49 = 1.0 + tanh_9;  tanh_9 = None
    ffn_output_37 = mul_37 * add_49;  mul_37 = add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    ffn_output_39 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_ffn_output(ffn_output_37);  ffn_output_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_50 = ffn_output_39 + layernormed_context_layer_9;  ffn_output_39 = layernormed_context_layer_9 = None
    hidden_states_30 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_full_layer_layer_norm(add_50);  add_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    mixed_query_layer_10 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_query(hidden_states_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    mixed_key_layer_10 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_key(hidden_states_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    mixed_value_layer_10 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_value(hidden_states_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    x_30 = mixed_query_layer_10.view((1, 512, 64, 64));  mixed_query_layer_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    query_layer_10 = x_30.permute(0, 2, 1, 3);  x_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    x_31 = mixed_key_layer_10.view((1, 512, 64, 64));  mixed_key_layer_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    key_layer_10 = x_31.permute(0, 2, 1, 3);  x_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    x_32 = mixed_value_layer_10.view((1, 512, 64, 64));  mixed_value_layer_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    value_layer_10 = x_32.permute(0, 2, 1, 3);  x_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_20 = key_layer_10.transpose(-1, -2);  key_layer_10 = None
    attention_scores_30 = torch.matmul(query_layer_10, transpose_20);  query_layer_10 = transpose_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_31 = attention_scores_30 / 8.0;  attention_scores_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:336, code: attention_scores = attention_scores + attention_mask
    attention_scores_32 = attention_scores_31 + extended_attention_mask_2;  attention_scores_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_20 = torch.nn.functional.softmax(attention_scores_32, dim = -1);  attention_scores_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    attention_probs_21 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_attention_dropout(attention_probs_20);  attention_probs_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_20 = torch.matmul(attention_probs_21, value_layer_10);  attention_probs_21 = value_layer_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    transpose_21 = context_layer_20.transpose(2, 1);  context_layer_20 = None
    context_layer_21 = transpose_21.flatten(2);  transpose_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    projected_context_layer_10 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_dense(context_layer_21);  context_layer_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:369, code: projected_context_layer_dropout = self.output_dropout(projected_context_layer)
    projected_context_layer_dropout_10 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_output_dropout(projected_context_layer_10);  projected_context_layer_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_52 = hidden_states_30 + projected_context_layer_dropout_10;  hidden_states_30 = projected_context_layer_dropout_10 = None
    layernormed_context_layer_10 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_LayerNorm(add_52);  add_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    ffn_output_40 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_ffn(layernormed_context_layer_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_41 = 0.5 * ffn_output_40
    pow_11 = torch.pow(ffn_output_40, 3.0)
    mul_42 = 0.044715 * pow_11;  pow_11 = None
    add_53 = ffn_output_40 + mul_42;  ffn_output_40 = mul_42 = None
    mul_43 = 0.7978845608028654 * add_53;  add_53 = None
    tanh_10 = torch.tanh(mul_43);  mul_43 = None
    add_54 = 1.0 + tanh_10;  tanh_10 = None
    ffn_output_41 = mul_41 * add_54;  mul_41 = add_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    ffn_output_43 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_ffn_output(ffn_output_41);  ffn_output_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_55 = ffn_output_43 + layernormed_context_layer_10;  ffn_output_43 = layernormed_context_layer_10 = None
    hidden_states_33 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_full_layer_layer_norm(add_55);  add_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    mixed_query_layer_11 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_query(hidden_states_33)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    mixed_key_layer_11 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_key(hidden_states_33)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    mixed_value_layer_11 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_value(hidden_states_33)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    x_33 = mixed_query_layer_11.view((1, 512, 64, 64));  mixed_query_layer_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    query_layer_11 = x_33.permute(0, 2, 1, 3);  x_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    x_34 = mixed_key_layer_11.view((1, 512, 64, 64));  mixed_key_layer_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    key_layer_11 = x_34.permute(0, 2, 1, 3);  x_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    x_35 = mixed_value_layer_11.view((1, 512, 64, 64));  mixed_value_layer_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    value_layer_11 = x_35.permute(0, 2, 1, 3);  x_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_22 = key_layer_11.transpose(-1, -2);  key_layer_11 = None
    attention_scores_33 = torch.matmul(query_layer_11, transpose_22);  query_layer_11 = transpose_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_34 = attention_scores_33 / 8.0;  attention_scores_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:336, code: attention_scores = attention_scores + attention_mask
    attention_scores_35 = attention_scores_34 + extended_attention_mask_2;  attention_scores_34 = extended_attention_mask_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_22 = torch.nn.functional.softmax(attention_scores_35, dim = -1);  attention_scores_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    attention_probs_23 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_attention_dropout(attention_probs_22);  attention_probs_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_22 = torch.matmul(attention_probs_23, value_layer_11);  attention_probs_23 = value_layer_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    transpose_23 = context_layer_22.transpose(2, 1);  context_layer_22 = None
    context_layer_23 = transpose_23.flatten(2);  transpose_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    projected_context_layer_11 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_dense(context_layer_23);  context_layer_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:369, code: projected_context_layer_dropout = self.output_dropout(projected_context_layer)
    projected_context_layer_dropout_11 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_output_dropout(projected_context_layer_11);  projected_context_layer_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_57 = hidden_states_33 + projected_context_layer_dropout_11;  hidden_states_33 = projected_context_layer_dropout_11 = None
    layernormed_context_layer_11 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_LayerNorm(add_57);  add_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    ffn_output_44 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_ffn(layernormed_context_layer_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_45 = 0.5 * ffn_output_44
    pow_12 = torch.pow(ffn_output_44, 3.0)
    mul_46 = 0.044715 * pow_12;  pow_12 = None
    add_58 = ffn_output_44 + mul_46;  ffn_output_44 = mul_46 = None
    mul_47 = 0.7978845608028654 * add_58;  add_58 = None
    tanh_11 = torch.tanh(mul_47);  mul_47 = None
    add_59 = 1.0 + tanh_11;  tanh_11 = None
    ffn_output_45 = mul_45 * add_59;  mul_45 = add_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    ffn_output_47 = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_ffn_output(ffn_output_45);  ffn_output_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_60 = ffn_output_47 + layernormed_context_layer_11;  ffn_output_47 = layernormed_context_layer_11 = None
    sequence_outputs = self.L__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_full_layer_layer_norm(add_60);  add_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:880, code: hidden_states = self.dense(hidden_states)
    hidden_states_37 = self.L__mod___predictions_dense(sequence_outputs);  sequence_outputs = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_49 = 0.5 * hidden_states_37
    pow_13 = torch.pow(hidden_states_37, 3.0)
    mul_50 = 0.044715 * pow_13;  pow_13 = None
    add_61 = hidden_states_37 + mul_50;  hidden_states_37 = mul_50 = None
    mul_51 = 0.7978845608028654 * add_61;  add_61 = None
    tanh_12 = torch.tanh(mul_51);  mul_51 = None
    add_62 = 1.0 + tanh_12;  tanh_12 = None
    hidden_states_38 = mul_49 * add_62;  mul_49 = add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:882, code: hidden_states = self.LayerNorm(hidden_states)
    hidden_states_39 = self.L__mod___predictions_LayerNorm(hidden_states_38);  hidden_states_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:883, code: hidden_states = self.decoder(hidden_states)
    prediction_scores = self.L__mod___predictions_decoder(hidden_states_39);  hidden_states_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:1004, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    view_36 = prediction_scores.view(-1, 30000)
    view_37 = l_inputs_labels_.view(-1);  l_inputs_labels_ = None
    masked_lm_loss = torch.nn.functional.cross_entropy(view_36, view_37, None, None, -100, None, 'mean', 0.0);  view_36 = view_37 = None
    return (masked_lm_loss, prediction_scores)
    