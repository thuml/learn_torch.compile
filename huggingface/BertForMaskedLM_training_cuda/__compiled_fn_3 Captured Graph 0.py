from __future__ import annotations



def forward(self, L_cloned_inputs_input_ids_ : torch.Tensor, L_cloned_inputs_labels_ : torch.Tensor):
    l_cloned_inputs_input_ids_ = L_cloned_inputs_input_ids_
    l_cloned_inputs_labels_ = L_cloned_inputs_labels_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:983, code: attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
    attention_mask = torch.ones((1, 512), device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:987, code: buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
    l__mod___bert_embeddings_token_type_ids = self.L__mod___bert_embeddings_token_type_ids
    buffered_token_type_ids = l__mod___bert_embeddings_token_type_ids[(slice(None, None, None), slice(None, 512, None))];  l__mod___bert_embeddings_token_type_ids = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:988, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
    token_type_ids = buffered_token_type_ids.expand(1, 512);  buffered_token_type_ids = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:916, code: extended_attention_mask = attention_mask[:, None, None, :]
    extended_attention_mask = attention_mask[(slice(None, None, None), None, None, slice(None, None, None))];  attention_mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:927, code: extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask_1 = extended_attention_mask.to(dtype = torch.float32);  extended_attention_mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:928, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    sub = 1.0 - extended_attention_mask_1;  extended_attention_mask_1 = None
    extended_attention_mask_3 = sub * -3.4028234663852886e+38;  sub = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:218, code: position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
    l__mod___bert_embeddings_position_ids = self.L__mod___bert_embeddings_position_ids
    position_ids = l__mod___bert_embeddings_position_ids[(slice(None, None, None), slice(0, 512, None))];  l__mod___bert_embeddings_position_ids = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:232, code: inputs_embeds = self.word_embeddings(input_ids)
    inputs_embeds = self.L__mod___bert_embeddings_word_embeddings(l_cloned_inputs_input_ids_);  l_cloned_inputs_input_ids_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:233, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    token_type_embeddings = self.L__mod___bert_embeddings_token_type_embeddings(token_type_ids);  token_type_ids = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:235, code: embeddings = inputs_embeds + token_type_embeddings
    embeddings = inputs_embeds + token_type_embeddings;  inputs_embeds = token_type_embeddings = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:237, code: position_embeddings = self.position_embeddings(position_ids)
    position_embeddings = self.L__mod___bert_embeddings_position_embeddings(position_ids);  position_ids = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:238, code: embeddings += position_embeddings
    embeddings += position_embeddings;  embeddings_1 = embeddings;  embeddings = position_embeddings = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:239, code: embeddings = self.LayerNorm(embeddings)
    embeddings_2 = self.L__mod___bert_embeddings_LayerNorm(embeddings_1);  embeddings_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:240, code: embeddings = self.dropout(embeddings)
    embedding_output = self.L__mod___bert_embeddings_dropout(embeddings_2);  embeddings_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    mixed_query_layer = self.L__mod___bert_encoder_layer_0_attention_self_query(embedding_output)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    l__mod___bert_encoder_layer_0_attention_self_key = self.L__mod___bert_encoder_layer_0_attention_self_key(embedding_output)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    x = l__mod___bert_encoder_layer_0_attention_self_key.view((1, 512, 12, 64));  l__mod___bert_encoder_layer_0_attention_self_key = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    key_layer = x.permute(0, 2, 1, 3);  x = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    l__mod___bert_encoder_layer_0_attention_self_value = self.L__mod___bert_encoder_layer_0_attention_self_value(embedding_output)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    x_1 = l__mod___bert_encoder_layer_0_attention_self_value.view((1, 512, 12, 64));  l__mod___bert_encoder_layer_0_attention_self_value = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    value_layer = x_1.permute(0, 2, 1, 3);  x_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    x_2 = mixed_query_layer.view((1, 512, 12, 64));  mixed_query_layer = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    query_layer = x_2.permute(0, 2, 1, 3);  x_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose = key_layer.transpose(-1, -2);  key_layer = None
    attention_scores = torch.matmul(query_layer, transpose);  query_layer = transpose = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:349, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_1 = attention_scores / 8.0;  attention_scores = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:352, code: attention_scores = attention_scores + attention_mask
    attention_scores_2 = attention_scores_1 + extended_attention_mask_3;  attention_scores_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs = torch.nn.functional.softmax(attention_scores_2, dim = -1);  attention_scores_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: attention_probs = self.dropout(attention_probs)
    attention_probs_1 = self.L__mod___bert_encoder_layer_0_attention_self_dropout(attention_probs);  attention_probs = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer = torch.matmul(attention_probs_1, value_layer);  attention_probs_1 = value_layer = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_3 = context_layer.permute(0, 2, 1, 3);  context_layer = None
    context_layer_1 = permute_3.contiguous();  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    context_layer_2 = context_layer_1.view((1, 512, 768));  context_layer_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    hidden_states = self.L__mod___bert_encoder_layer_0_attention_output_dense(context_layer_2);  context_layer_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:387, code: hidden_states = self.dropout(hidden_states)
    hidden_states_1 = self.L__mod___bert_encoder_layer_0_attention_output_dropout(hidden_states);  hidden_states = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_2 = hidden_states_1 + embedding_output;  hidden_states_1 = embedding_output = None
    attention_output = self.L__mod___bert_encoder_layer_0_attention_output_LayerNorm(add_2);  add_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    hidden_states_3 = self.L__mod___bert_encoder_layer_0_intermediate_dense(attention_output)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output = torch._C._nn.gelu(hidden_states_3);  hidden_states_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    hidden_states_5 = self.L__mod___bert_encoder_layer_0_output_dense(intermediate_output);  intermediate_output = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:465, code: hidden_states = self.dropout(hidden_states)
    hidden_states_6 = self.L__mod___bert_encoder_layer_0_output_dropout(hidden_states_5);  hidden_states_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_3 = hidden_states_6 + attention_output;  hidden_states_6 = attention_output = None
    hidden_states_8 = self.L__mod___bert_encoder_layer_0_output_LayerNorm(add_3);  add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    mixed_query_layer_1 = self.L__mod___bert_encoder_layer_1_attention_self_query(hidden_states_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    l__mod___bert_encoder_layer_1_attention_self_key = self.L__mod___bert_encoder_layer_1_attention_self_key(hidden_states_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    x_3 = l__mod___bert_encoder_layer_1_attention_self_key.view((1, 512, 12, 64));  l__mod___bert_encoder_layer_1_attention_self_key = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    key_layer_1 = x_3.permute(0, 2, 1, 3);  x_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    l__mod___bert_encoder_layer_1_attention_self_value = self.L__mod___bert_encoder_layer_1_attention_self_value(hidden_states_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    x_4 = l__mod___bert_encoder_layer_1_attention_self_value.view((1, 512, 12, 64));  l__mod___bert_encoder_layer_1_attention_self_value = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    value_layer_1 = x_4.permute(0, 2, 1, 3);  x_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    x_5 = mixed_query_layer_1.view((1, 512, 12, 64));  mixed_query_layer_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    query_layer_1 = x_5.permute(0, 2, 1, 3);  x_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_1 = key_layer_1.transpose(-1, -2);  key_layer_1 = None
    attention_scores_3 = torch.matmul(query_layer_1, transpose_1);  query_layer_1 = transpose_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:349, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_4 = attention_scores_3 / 8.0;  attention_scores_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:352, code: attention_scores = attention_scores + attention_mask
    attention_scores_5 = attention_scores_4 + extended_attention_mask_3;  attention_scores_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_2 = torch.nn.functional.softmax(attention_scores_5, dim = -1);  attention_scores_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: attention_probs = self.dropout(attention_probs)
    attention_probs_3 = self.L__mod___bert_encoder_layer_1_attention_self_dropout(attention_probs_2);  attention_probs_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_3 = torch.matmul(attention_probs_3, value_layer_1);  attention_probs_3 = value_layer_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_7 = context_layer_3.permute(0, 2, 1, 3);  context_layer_3 = None
    context_layer_4 = permute_7.contiguous();  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    context_layer_5 = context_layer_4.view((1, 512, 768));  context_layer_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    hidden_states_9 = self.L__mod___bert_encoder_layer_1_attention_output_dense(context_layer_5);  context_layer_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:387, code: hidden_states = self.dropout(hidden_states)
    hidden_states_10 = self.L__mod___bert_encoder_layer_1_attention_output_dropout(hidden_states_9);  hidden_states_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_5 = hidden_states_10 + hidden_states_8;  hidden_states_10 = hidden_states_8 = None
    attention_output_2 = self.L__mod___bert_encoder_layer_1_attention_output_LayerNorm(add_5);  add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    hidden_states_12 = self.L__mod___bert_encoder_layer_1_intermediate_dense(attention_output_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_1 = torch._C._nn.gelu(hidden_states_12);  hidden_states_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    hidden_states_14 = self.L__mod___bert_encoder_layer_1_output_dense(intermediate_output_1);  intermediate_output_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:465, code: hidden_states = self.dropout(hidden_states)
    hidden_states_15 = self.L__mod___bert_encoder_layer_1_output_dropout(hidden_states_14);  hidden_states_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_6 = hidden_states_15 + attention_output_2;  hidden_states_15 = attention_output_2 = None
    hidden_states_17 = self.L__mod___bert_encoder_layer_1_output_LayerNorm(add_6);  add_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    mixed_query_layer_2 = self.L__mod___bert_encoder_layer_2_attention_self_query(hidden_states_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    l__mod___bert_encoder_layer_2_attention_self_key = self.L__mod___bert_encoder_layer_2_attention_self_key(hidden_states_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    x_6 = l__mod___bert_encoder_layer_2_attention_self_key.view((1, 512, 12, 64));  l__mod___bert_encoder_layer_2_attention_self_key = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    key_layer_2 = x_6.permute(0, 2, 1, 3);  x_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    l__mod___bert_encoder_layer_2_attention_self_value = self.L__mod___bert_encoder_layer_2_attention_self_value(hidden_states_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    x_7 = l__mod___bert_encoder_layer_2_attention_self_value.view((1, 512, 12, 64));  l__mod___bert_encoder_layer_2_attention_self_value = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    value_layer_2 = x_7.permute(0, 2, 1, 3);  x_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    x_8 = mixed_query_layer_2.view((1, 512, 12, 64));  mixed_query_layer_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    query_layer_2 = x_8.permute(0, 2, 1, 3);  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_2 = key_layer_2.transpose(-1, -2);  key_layer_2 = None
    attention_scores_6 = torch.matmul(query_layer_2, transpose_2);  query_layer_2 = transpose_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:349, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_7 = attention_scores_6 / 8.0;  attention_scores_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:352, code: attention_scores = attention_scores + attention_mask
    attention_scores_8 = attention_scores_7 + extended_attention_mask_3;  attention_scores_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_4 = torch.nn.functional.softmax(attention_scores_8, dim = -1);  attention_scores_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: attention_probs = self.dropout(attention_probs)
    attention_probs_5 = self.L__mod___bert_encoder_layer_2_attention_self_dropout(attention_probs_4);  attention_probs_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_6 = torch.matmul(attention_probs_5, value_layer_2);  attention_probs_5 = value_layer_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_11 = context_layer_6.permute(0, 2, 1, 3);  context_layer_6 = None
    context_layer_7 = permute_11.contiguous();  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    context_layer_8 = context_layer_7.view((1, 512, 768));  context_layer_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    hidden_states_18 = self.L__mod___bert_encoder_layer_2_attention_output_dense(context_layer_8);  context_layer_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:387, code: hidden_states = self.dropout(hidden_states)
    hidden_states_19 = self.L__mod___bert_encoder_layer_2_attention_output_dropout(hidden_states_18);  hidden_states_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_8 = hidden_states_19 + hidden_states_17;  hidden_states_19 = hidden_states_17 = None
    attention_output_4 = self.L__mod___bert_encoder_layer_2_attention_output_LayerNorm(add_8);  add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    hidden_states_21 = self.L__mod___bert_encoder_layer_2_intermediate_dense(attention_output_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_2 = torch._C._nn.gelu(hidden_states_21);  hidden_states_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    hidden_states_23 = self.L__mod___bert_encoder_layer_2_output_dense(intermediate_output_2);  intermediate_output_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:465, code: hidden_states = self.dropout(hidden_states)
    hidden_states_24 = self.L__mod___bert_encoder_layer_2_output_dropout(hidden_states_23);  hidden_states_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_9 = hidden_states_24 + attention_output_4;  hidden_states_24 = attention_output_4 = None
    hidden_states_26 = self.L__mod___bert_encoder_layer_2_output_LayerNorm(add_9);  add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    mixed_query_layer_3 = self.L__mod___bert_encoder_layer_3_attention_self_query(hidden_states_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    l__mod___bert_encoder_layer_3_attention_self_key = self.L__mod___bert_encoder_layer_3_attention_self_key(hidden_states_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    x_9 = l__mod___bert_encoder_layer_3_attention_self_key.view((1, 512, 12, 64));  l__mod___bert_encoder_layer_3_attention_self_key = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    key_layer_3 = x_9.permute(0, 2, 1, 3);  x_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    l__mod___bert_encoder_layer_3_attention_self_value = self.L__mod___bert_encoder_layer_3_attention_self_value(hidden_states_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    x_10 = l__mod___bert_encoder_layer_3_attention_self_value.view((1, 512, 12, 64));  l__mod___bert_encoder_layer_3_attention_self_value = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    value_layer_3 = x_10.permute(0, 2, 1, 3);  x_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    x_11 = mixed_query_layer_3.view((1, 512, 12, 64));  mixed_query_layer_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    query_layer_3 = x_11.permute(0, 2, 1, 3);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_3 = key_layer_3.transpose(-1, -2);  key_layer_3 = None
    attention_scores_9 = torch.matmul(query_layer_3, transpose_3);  query_layer_3 = transpose_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:349, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_10 = attention_scores_9 / 8.0;  attention_scores_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:352, code: attention_scores = attention_scores + attention_mask
    attention_scores_11 = attention_scores_10 + extended_attention_mask_3;  attention_scores_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_6 = torch.nn.functional.softmax(attention_scores_11, dim = -1);  attention_scores_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: attention_probs = self.dropout(attention_probs)
    attention_probs_7 = self.L__mod___bert_encoder_layer_3_attention_self_dropout(attention_probs_6);  attention_probs_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_9 = torch.matmul(attention_probs_7, value_layer_3);  attention_probs_7 = value_layer_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_15 = context_layer_9.permute(0, 2, 1, 3);  context_layer_9 = None
    context_layer_10 = permute_15.contiguous();  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    context_layer_11 = context_layer_10.view((1, 512, 768));  context_layer_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    hidden_states_27 = self.L__mod___bert_encoder_layer_3_attention_output_dense(context_layer_11);  context_layer_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:387, code: hidden_states = self.dropout(hidden_states)
    hidden_states_28 = self.L__mod___bert_encoder_layer_3_attention_output_dropout(hidden_states_27);  hidden_states_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_11 = hidden_states_28 + hidden_states_26;  hidden_states_28 = hidden_states_26 = None
    attention_output_6 = self.L__mod___bert_encoder_layer_3_attention_output_LayerNorm(add_11);  add_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    hidden_states_30 = self.L__mod___bert_encoder_layer_3_intermediate_dense(attention_output_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_3 = torch._C._nn.gelu(hidden_states_30);  hidden_states_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    hidden_states_32 = self.L__mod___bert_encoder_layer_3_output_dense(intermediate_output_3);  intermediate_output_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:465, code: hidden_states = self.dropout(hidden_states)
    hidden_states_33 = self.L__mod___bert_encoder_layer_3_output_dropout(hidden_states_32);  hidden_states_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_12 = hidden_states_33 + attention_output_6;  hidden_states_33 = attention_output_6 = None
    hidden_states_35 = self.L__mod___bert_encoder_layer_3_output_LayerNorm(add_12);  add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    mixed_query_layer_4 = self.L__mod___bert_encoder_layer_4_attention_self_query(hidden_states_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    l__mod___bert_encoder_layer_4_attention_self_key = self.L__mod___bert_encoder_layer_4_attention_self_key(hidden_states_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    x_12 = l__mod___bert_encoder_layer_4_attention_self_key.view((1, 512, 12, 64));  l__mod___bert_encoder_layer_4_attention_self_key = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    key_layer_4 = x_12.permute(0, 2, 1, 3);  x_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    l__mod___bert_encoder_layer_4_attention_self_value = self.L__mod___bert_encoder_layer_4_attention_self_value(hidden_states_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    x_13 = l__mod___bert_encoder_layer_4_attention_self_value.view((1, 512, 12, 64));  l__mod___bert_encoder_layer_4_attention_self_value = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    value_layer_4 = x_13.permute(0, 2, 1, 3);  x_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    x_14 = mixed_query_layer_4.view((1, 512, 12, 64));  mixed_query_layer_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    query_layer_4 = x_14.permute(0, 2, 1, 3);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_4 = key_layer_4.transpose(-1, -2);  key_layer_4 = None
    attention_scores_12 = torch.matmul(query_layer_4, transpose_4);  query_layer_4 = transpose_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:349, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_13 = attention_scores_12 / 8.0;  attention_scores_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:352, code: attention_scores = attention_scores + attention_mask
    attention_scores_14 = attention_scores_13 + extended_attention_mask_3;  attention_scores_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_8 = torch.nn.functional.softmax(attention_scores_14, dim = -1);  attention_scores_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: attention_probs = self.dropout(attention_probs)
    attention_probs_9 = self.L__mod___bert_encoder_layer_4_attention_self_dropout(attention_probs_8);  attention_probs_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_12 = torch.matmul(attention_probs_9, value_layer_4);  attention_probs_9 = value_layer_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_19 = context_layer_12.permute(0, 2, 1, 3);  context_layer_12 = None
    context_layer_13 = permute_19.contiguous();  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    context_layer_14 = context_layer_13.view((1, 512, 768));  context_layer_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    hidden_states_36 = self.L__mod___bert_encoder_layer_4_attention_output_dense(context_layer_14);  context_layer_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:387, code: hidden_states = self.dropout(hidden_states)
    hidden_states_37 = self.L__mod___bert_encoder_layer_4_attention_output_dropout(hidden_states_36);  hidden_states_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_14 = hidden_states_37 + hidden_states_35;  hidden_states_37 = hidden_states_35 = None
    attention_output_8 = self.L__mod___bert_encoder_layer_4_attention_output_LayerNorm(add_14);  add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    hidden_states_39 = self.L__mod___bert_encoder_layer_4_intermediate_dense(attention_output_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_4 = torch._C._nn.gelu(hidden_states_39);  hidden_states_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    hidden_states_41 = self.L__mod___bert_encoder_layer_4_output_dense(intermediate_output_4);  intermediate_output_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:465, code: hidden_states = self.dropout(hidden_states)
    hidden_states_42 = self.L__mod___bert_encoder_layer_4_output_dropout(hidden_states_41);  hidden_states_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_15 = hidden_states_42 + attention_output_8;  hidden_states_42 = attention_output_8 = None
    hidden_states_44 = self.L__mod___bert_encoder_layer_4_output_LayerNorm(add_15);  add_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    mixed_query_layer_5 = self.L__mod___bert_encoder_layer_5_attention_self_query(hidden_states_44)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    l__mod___bert_encoder_layer_5_attention_self_key = self.L__mod___bert_encoder_layer_5_attention_self_key(hidden_states_44)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    x_15 = l__mod___bert_encoder_layer_5_attention_self_key.view((1, 512, 12, 64));  l__mod___bert_encoder_layer_5_attention_self_key = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    key_layer_5 = x_15.permute(0, 2, 1, 3);  x_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    l__mod___bert_encoder_layer_5_attention_self_value = self.L__mod___bert_encoder_layer_5_attention_self_value(hidden_states_44)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    x_16 = l__mod___bert_encoder_layer_5_attention_self_value.view((1, 512, 12, 64));  l__mod___bert_encoder_layer_5_attention_self_value = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    value_layer_5 = x_16.permute(0, 2, 1, 3);  x_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    x_17 = mixed_query_layer_5.view((1, 512, 12, 64));  mixed_query_layer_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    query_layer_5 = x_17.permute(0, 2, 1, 3);  x_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_5 = key_layer_5.transpose(-1, -2);  key_layer_5 = None
    attention_scores_15 = torch.matmul(query_layer_5, transpose_5);  query_layer_5 = transpose_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:349, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_16 = attention_scores_15 / 8.0;  attention_scores_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:352, code: attention_scores = attention_scores + attention_mask
    attention_scores_17 = attention_scores_16 + extended_attention_mask_3;  attention_scores_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_10 = torch.nn.functional.softmax(attention_scores_17, dim = -1);  attention_scores_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: attention_probs = self.dropout(attention_probs)
    attention_probs_11 = self.L__mod___bert_encoder_layer_5_attention_self_dropout(attention_probs_10);  attention_probs_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_15 = torch.matmul(attention_probs_11, value_layer_5);  attention_probs_11 = value_layer_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_23 = context_layer_15.permute(0, 2, 1, 3);  context_layer_15 = None
    context_layer_16 = permute_23.contiguous();  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    context_layer_17 = context_layer_16.view((1, 512, 768));  context_layer_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    hidden_states_45 = self.L__mod___bert_encoder_layer_5_attention_output_dense(context_layer_17);  context_layer_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:387, code: hidden_states = self.dropout(hidden_states)
    hidden_states_46 = self.L__mod___bert_encoder_layer_5_attention_output_dropout(hidden_states_45);  hidden_states_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_17 = hidden_states_46 + hidden_states_44;  hidden_states_46 = hidden_states_44 = None
    attention_output_10 = self.L__mod___bert_encoder_layer_5_attention_output_LayerNorm(add_17);  add_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    hidden_states_48 = self.L__mod___bert_encoder_layer_5_intermediate_dense(attention_output_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_5 = torch._C._nn.gelu(hidden_states_48);  hidden_states_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    hidden_states_50 = self.L__mod___bert_encoder_layer_5_output_dense(intermediate_output_5);  intermediate_output_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:465, code: hidden_states = self.dropout(hidden_states)
    hidden_states_51 = self.L__mod___bert_encoder_layer_5_output_dropout(hidden_states_50);  hidden_states_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_18 = hidden_states_51 + attention_output_10;  hidden_states_51 = attention_output_10 = None
    hidden_states_53 = self.L__mod___bert_encoder_layer_5_output_LayerNorm(add_18);  add_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    mixed_query_layer_6 = self.L__mod___bert_encoder_layer_6_attention_self_query(hidden_states_53)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    l__mod___bert_encoder_layer_6_attention_self_key = self.L__mod___bert_encoder_layer_6_attention_self_key(hidden_states_53)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    x_18 = l__mod___bert_encoder_layer_6_attention_self_key.view((1, 512, 12, 64));  l__mod___bert_encoder_layer_6_attention_self_key = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    key_layer_6 = x_18.permute(0, 2, 1, 3);  x_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    l__mod___bert_encoder_layer_6_attention_self_value = self.L__mod___bert_encoder_layer_6_attention_self_value(hidden_states_53)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    x_19 = l__mod___bert_encoder_layer_6_attention_self_value.view((1, 512, 12, 64));  l__mod___bert_encoder_layer_6_attention_self_value = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    value_layer_6 = x_19.permute(0, 2, 1, 3);  x_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    x_20 = mixed_query_layer_6.view((1, 512, 12, 64));  mixed_query_layer_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    query_layer_6 = x_20.permute(0, 2, 1, 3);  x_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_6 = key_layer_6.transpose(-1, -2);  key_layer_6 = None
    attention_scores_18 = torch.matmul(query_layer_6, transpose_6);  query_layer_6 = transpose_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:349, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_19 = attention_scores_18 / 8.0;  attention_scores_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:352, code: attention_scores = attention_scores + attention_mask
    attention_scores_20 = attention_scores_19 + extended_attention_mask_3;  attention_scores_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_12 = torch.nn.functional.softmax(attention_scores_20, dim = -1);  attention_scores_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: attention_probs = self.dropout(attention_probs)
    attention_probs_13 = self.L__mod___bert_encoder_layer_6_attention_self_dropout(attention_probs_12);  attention_probs_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_18 = torch.matmul(attention_probs_13, value_layer_6);  attention_probs_13 = value_layer_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_27 = context_layer_18.permute(0, 2, 1, 3);  context_layer_18 = None
    context_layer_19 = permute_27.contiguous();  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    context_layer_20 = context_layer_19.view((1, 512, 768));  context_layer_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    hidden_states_54 = self.L__mod___bert_encoder_layer_6_attention_output_dense(context_layer_20);  context_layer_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:387, code: hidden_states = self.dropout(hidden_states)
    hidden_states_55 = self.L__mod___bert_encoder_layer_6_attention_output_dropout(hidden_states_54);  hidden_states_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_20 = hidden_states_55 + hidden_states_53;  hidden_states_55 = hidden_states_53 = None
    attention_output_12 = self.L__mod___bert_encoder_layer_6_attention_output_LayerNorm(add_20);  add_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    hidden_states_57 = self.L__mod___bert_encoder_layer_6_intermediate_dense(attention_output_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_6 = torch._C._nn.gelu(hidden_states_57);  hidden_states_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    hidden_states_59 = self.L__mod___bert_encoder_layer_6_output_dense(intermediate_output_6);  intermediate_output_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:465, code: hidden_states = self.dropout(hidden_states)
    hidden_states_60 = self.L__mod___bert_encoder_layer_6_output_dropout(hidden_states_59);  hidden_states_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_21 = hidden_states_60 + attention_output_12;  hidden_states_60 = attention_output_12 = None
    hidden_states_62 = self.L__mod___bert_encoder_layer_6_output_LayerNorm(add_21);  add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    mixed_query_layer_7 = self.L__mod___bert_encoder_layer_7_attention_self_query(hidden_states_62)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    l__mod___bert_encoder_layer_7_attention_self_key = self.L__mod___bert_encoder_layer_7_attention_self_key(hidden_states_62)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    x_21 = l__mod___bert_encoder_layer_7_attention_self_key.view((1, 512, 12, 64));  l__mod___bert_encoder_layer_7_attention_self_key = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    key_layer_7 = x_21.permute(0, 2, 1, 3);  x_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    l__mod___bert_encoder_layer_7_attention_self_value = self.L__mod___bert_encoder_layer_7_attention_self_value(hidden_states_62)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    x_22 = l__mod___bert_encoder_layer_7_attention_self_value.view((1, 512, 12, 64));  l__mod___bert_encoder_layer_7_attention_self_value = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    value_layer_7 = x_22.permute(0, 2, 1, 3);  x_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    x_23 = mixed_query_layer_7.view((1, 512, 12, 64));  mixed_query_layer_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    query_layer_7 = x_23.permute(0, 2, 1, 3);  x_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_7 = key_layer_7.transpose(-1, -2);  key_layer_7 = None
    attention_scores_21 = torch.matmul(query_layer_7, transpose_7);  query_layer_7 = transpose_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:349, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_22 = attention_scores_21 / 8.0;  attention_scores_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:352, code: attention_scores = attention_scores + attention_mask
    attention_scores_23 = attention_scores_22 + extended_attention_mask_3;  attention_scores_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_14 = torch.nn.functional.softmax(attention_scores_23, dim = -1);  attention_scores_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: attention_probs = self.dropout(attention_probs)
    attention_probs_15 = self.L__mod___bert_encoder_layer_7_attention_self_dropout(attention_probs_14);  attention_probs_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_21 = torch.matmul(attention_probs_15, value_layer_7);  attention_probs_15 = value_layer_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_31 = context_layer_21.permute(0, 2, 1, 3);  context_layer_21 = None
    context_layer_22 = permute_31.contiguous();  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    context_layer_23 = context_layer_22.view((1, 512, 768));  context_layer_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    hidden_states_63 = self.L__mod___bert_encoder_layer_7_attention_output_dense(context_layer_23);  context_layer_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:387, code: hidden_states = self.dropout(hidden_states)
    hidden_states_64 = self.L__mod___bert_encoder_layer_7_attention_output_dropout(hidden_states_63);  hidden_states_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_23 = hidden_states_64 + hidden_states_62;  hidden_states_64 = hidden_states_62 = None
    attention_output_14 = self.L__mod___bert_encoder_layer_7_attention_output_LayerNorm(add_23);  add_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    hidden_states_66 = self.L__mod___bert_encoder_layer_7_intermediate_dense(attention_output_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_7 = torch._C._nn.gelu(hidden_states_66);  hidden_states_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    hidden_states_68 = self.L__mod___bert_encoder_layer_7_output_dense(intermediate_output_7);  intermediate_output_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:465, code: hidden_states = self.dropout(hidden_states)
    hidden_states_69 = self.L__mod___bert_encoder_layer_7_output_dropout(hidden_states_68);  hidden_states_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_24 = hidden_states_69 + attention_output_14;  hidden_states_69 = attention_output_14 = None
    hidden_states_71 = self.L__mod___bert_encoder_layer_7_output_LayerNorm(add_24);  add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    mixed_query_layer_8 = self.L__mod___bert_encoder_layer_8_attention_self_query(hidden_states_71)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    l__mod___bert_encoder_layer_8_attention_self_key = self.L__mod___bert_encoder_layer_8_attention_self_key(hidden_states_71)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    x_24 = l__mod___bert_encoder_layer_8_attention_self_key.view((1, 512, 12, 64));  l__mod___bert_encoder_layer_8_attention_self_key = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    key_layer_8 = x_24.permute(0, 2, 1, 3);  x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    l__mod___bert_encoder_layer_8_attention_self_value = self.L__mod___bert_encoder_layer_8_attention_self_value(hidden_states_71)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    x_25 = l__mod___bert_encoder_layer_8_attention_self_value.view((1, 512, 12, 64));  l__mod___bert_encoder_layer_8_attention_self_value = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    value_layer_8 = x_25.permute(0, 2, 1, 3);  x_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    x_26 = mixed_query_layer_8.view((1, 512, 12, 64));  mixed_query_layer_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    query_layer_8 = x_26.permute(0, 2, 1, 3);  x_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_8 = key_layer_8.transpose(-1, -2);  key_layer_8 = None
    attention_scores_24 = torch.matmul(query_layer_8, transpose_8);  query_layer_8 = transpose_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:349, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_25 = attention_scores_24 / 8.0;  attention_scores_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:352, code: attention_scores = attention_scores + attention_mask
    attention_scores_26 = attention_scores_25 + extended_attention_mask_3;  attention_scores_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_16 = torch.nn.functional.softmax(attention_scores_26, dim = -1);  attention_scores_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: attention_probs = self.dropout(attention_probs)
    attention_probs_17 = self.L__mod___bert_encoder_layer_8_attention_self_dropout(attention_probs_16);  attention_probs_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_24 = torch.matmul(attention_probs_17, value_layer_8);  attention_probs_17 = value_layer_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_35 = context_layer_24.permute(0, 2, 1, 3);  context_layer_24 = None
    context_layer_25 = permute_35.contiguous();  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    context_layer_26 = context_layer_25.view((1, 512, 768));  context_layer_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    hidden_states_72 = self.L__mod___bert_encoder_layer_8_attention_output_dense(context_layer_26);  context_layer_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:387, code: hidden_states = self.dropout(hidden_states)
    hidden_states_73 = self.L__mod___bert_encoder_layer_8_attention_output_dropout(hidden_states_72);  hidden_states_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_26 = hidden_states_73 + hidden_states_71;  hidden_states_73 = hidden_states_71 = None
    attention_output_16 = self.L__mod___bert_encoder_layer_8_attention_output_LayerNorm(add_26);  add_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    hidden_states_75 = self.L__mod___bert_encoder_layer_8_intermediate_dense(attention_output_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_8 = torch._C._nn.gelu(hidden_states_75);  hidden_states_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    hidden_states_77 = self.L__mod___bert_encoder_layer_8_output_dense(intermediate_output_8);  intermediate_output_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:465, code: hidden_states = self.dropout(hidden_states)
    hidden_states_78 = self.L__mod___bert_encoder_layer_8_output_dropout(hidden_states_77);  hidden_states_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_27 = hidden_states_78 + attention_output_16;  hidden_states_78 = attention_output_16 = None
    hidden_states_80 = self.L__mod___bert_encoder_layer_8_output_LayerNorm(add_27);  add_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    mixed_query_layer_9 = self.L__mod___bert_encoder_layer_9_attention_self_query(hidden_states_80)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    l__mod___bert_encoder_layer_9_attention_self_key = self.L__mod___bert_encoder_layer_9_attention_self_key(hidden_states_80)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    x_27 = l__mod___bert_encoder_layer_9_attention_self_key.view((1, 512, 12, 64));  l__mod___bert_encoder_layer_9_attention_self_key = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    key_layer_9 = x_27.permute(0, 2, 1, 3);  x_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    l__mod___bert_encoder_layer_9_attention_self_value = self.L__mod___bert_encoder_layer_9_attention_self_value(hidden_states_80)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    x_28 = l__mod___bert_encoder_layer_9_attention_self_value.view((1, 512, 12, 64));  l__mod___bert_encoder_layer_9_attention_self_value = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    value_layer_9 = x_28.permute(0, 2, 1, 3);  x_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    x_29 = mixed_query_layer_9.view((1, 512, 12, 64));  mixed_query_layer_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    query_layer_9 = x_29.permute(0, 2, 1, 3);  x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_9 = key_layer_9.transpose(-1, -2);  key_layer_9 = None
    attention_scores_27 = torch.matmul(query_layer_9, transpose_9);  query_layer_9 = transpose_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:349, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_28 = attention_scores_27 / 8.0;  attention_scores_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:352, code: attention_scores = attention_scores + attention_mask
    attention_scores_29 = attention_scores_28 + extended_attention_mask_3;  attention_scores_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_18 = torch.nn.functional.softmax(attention_scores_29, dim = -1);  attention_scores_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: attention_probs = self.dropout(attention_probs)
    attention_probs_19 = self.L__mod___bert_encoder_layer_9_attention_self_dropout(attention_probs_18);  attention_probs_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_27 = torch.matmul(attention_probs_19, value_layer_9);  attention_probs_19 = value_layer_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_39 = context_layer_27.permute(0, 2, 1, 3);  context_layer_27 = None
    context_layer_28 = permute_39.contiguous();  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    context_layer_29 = context_layer_28.view((1, 512, 768));  context_layer_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    hidden_states_81 = self.L__mod___bert_encoder_layer_9_attention_output_dense(context_layer_29);  context_layer_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:387, code: hidden_states = self.dropout(hidden_states)
    hidden_states_82 = self.L__mod___bert_encoder_layer_9_attention_output_dropout(hidden_states_81);  hidden_states_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_29 = hidden_states_82 + hidden_states_80;  hidden_states_82 = hidden_states_80 = None
    attention_output_18 = self.L__mod___bert_encoder_layer_9_attention_output_LayerNorm(add_29);  add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    hidden_states_84 = self.L__mod___bert_encoder_layer_9_intermediate_dense(attention_output_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_9 = torch._C._nn.gelu(hidden_states_84);  hidden_states_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    hidden_states_86 = self.L__mod___bert_encoder_layer_9_output_dense(intermediate_output_9);  intermediate_output_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:465, code: hidden_states = self.dropout(hidden_states)
    hidden_states_87 = self.L__mod___bert_encoder_layer_9_output_dropout(hidden_states_86);  hidden_states_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_30 = hidden_states_87 + attention_output_18;  hidden_states_87 = attention_output_18 = None
    hidden_states_89 = self.L__mod___bert_encoder_layer_9_output_LayerNorm(add_30);  add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    mixed_query_layer_10 = self.L__mod___bert_encoder_layer_10_attention_self_query(hidden_states_89)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    l__mod___bert_encoder_layer_10_attention_self_key = self.L__mod___bert_encoder_layer_10_attention_self_key(hidden_states_89)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    x_30 = l__mod___bert_encoder_layer_10_attention_self_key.view((1, 512, 12, 64));  l__mod___bert_encoder_layer_10_attention_self_key = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    key_layer_10 = x_30.permute(0, 2, 1, 3);  x_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    l__mod___bert_encoder_layer_10_attention_self_value = self.L__mod___bert_encoder_layer_10_attention_self_value(hidden_states_89)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    x_31 = l__mod___bert_encoder_layer_10_attention_self_value.view((1, 512, 12, 64));  l__mod___bert_encoder_layer_10_attention_self_value = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    value_layer_10 = x_31.permute(0, 2, 1, 3);  x_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    x_32 = mixed_query_layer_10.view((1, 512, 12, 64));  mixed_query_layer_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    query_layer_10 = x_32.permute(0, 2, 1, 3);  x_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_10 = key_layer_10.transpose(-1, -2);  key_layer_10 = None
    attention_scores_30 = torch.matmul(query_layer_10, transpose_10);  query_layer_10 = transpose_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:349, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_31 = attention_scores_30 / 8.0;  attention_scores_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:352, code: attention_scores = attention_scores + attention_mask
    attention_scores_32 = attention_scores_31 + extended_attention_mask_3;  attention_scores_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_20 = torch.nn.functional.softmax(attention_scores_32, dim = -1);  attention_scores_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: attention_probs = self.dropout(attention_probs)
    attention_probs_21 = self.L__mod___bert_encoder_layer_10_attention_self_dropout(attention_probs_20);  attention_probs_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_30 = torch.matmul(attention_probs_21, value_layer_10);  attention_probs_21 = value_layer_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_43 = context_layer_30.permute(0, 2, 1, 3);  context_layer_30 = None
    context_layer_31 = permute_43.contiguous();  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    context_layer_32 = context_layer_31.view((1, 512, 768));  context_layer_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    hidden_states_90 = self.L__mod___bert_encoder_layer_10_attention_output_dense(context_layer_32);  context_layer_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:387, code: hidden_states = self.dropout(hidden_states)
    hidden_states_91 = self.L__mod___bert_encoder_layer_10_attention_output_dropout(hidden_states_90);  hidden_states_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_32 = hidden_states_91 + hidden_states_89;  hidden_states_91 = hidden_states_89 = None
    attention_output_20 = self.L__mod___bert_encoder_layer_10_attention_output_LayerNorm(add_32);  add_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    hidden_states_93 = self.L__mod___bert_encoder_layer_10_intermediate_dense(attention_output_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_10 = torch._C._nn.gelu(hidden_states_93);  hidden_states_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    hidden_states_95 = self.L__mod___bert_encoder_layer_10_output_dense(intermediate_output_10);  intermediate_output_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:465, code: hidden_states = self.dropout(hidden_states)
    hidden_states_96 = self.L__mod___bert_encoder_layer_10_output_dropout(hidden_states_95);  hidden_states_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_33 = hidden_states_96 + attention_output_20;  hidden_states_96 = attention_output_20 = None
    hidden_states_98 = self.L__mod___bert_encoder_layer_10_output_LayerNorm(add_33);  add_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    mixed_query_layer_11 = self.L__mod___bert_encoder_layer_11_attention_self_query(hidden_states_98)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    l__mod___bert_encoder_layer_11_attention_self_key = self.L__mod___bert_encoder_layer_11_attention_self_key(hidden_states_98)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    x_33 = l__mod___bert_encoder_layer_11_attention_self_key.view((1, 512, 12, 64));  l__mod___bert_encoder_layer_11_attention_self_key = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    key_layer_11 = x_33.permute(0, 2, 1, 3);  x_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    l__mod___bert_encoder_layer_11_attention_self_value = self.L__mod___bert_encoder_layer_11_attention_self_value(hidden_states_98)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    x_34 = l__mod___bert_encoder_layer_11_attention_self_value.view((1, 512, 12, 64));  l__mod___bert_encoder_layer_11_attention_self_value = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    value_layer_11 = x_34.permute(0, 2, 1, 3);  x_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    x_35 = mixed_query_layer_11.view((1, 512, 12, 64));  mixed_query_layer_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    query_layer_11 = x_35.permute(0, 2, 1, 3);  x_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_11 = key_layer_11.transpose(-1, -2);  key_layer_11 = None
    attention_scores_33 = torch.matmul(query_layer_11, transpose_11);  query_layer_11 = transpose_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:349, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_34 = attention_scores_33 / 8.0;  attention_scores_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:352, code: attention_scores = attention_scores + attention_mask
    attention_scores_35 = attention_scores_34 + extended_attention_mask_3;  attention_scores_34 = extended_attention_mask_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_22 = torch.nn.functional.softmax(attention_scores_35, dim = -1);  attention_scores_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: attention_probs = self.dropout(attention_probs)
    attention_probs_23 = self.L__mod___bert_encoder_layer_11_attention_self_dropout(attention_probs_22);  attention_probs_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_33 = torch.matmul(attention_probs_23, value_layer_11);  attention_probs_23 = value_layer_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_47 = context_layer_33.permute(0, 2, 1, 3);  context_layer_33 = None
    context_layer_34 = permute_47.contiguous();  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    context_layer_35 = context_layer_34.view((1, 512, 768));  context_layer_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    hidden_states_99 = self.L__mod___bert_encoder_layer_11_attention_output_dense(context_layer_35);  context_layer_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:387, code: hidden_states = self.dropout(hidden_states)
    hidden_states_100 = self.L__mod___bert_encoder_layer_11_attention_output_dropout(hidden_states_99);  hidden_states_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_35 = hidden_states_100 + hidden_states_98;  hidden_states_100 = hidden_states_98 = None
    attention_output_22 = self.L__mod___bert_encoder_layer_11_attention_output_LayerNorm(add_35);  add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    hidden_states_102 = self.L__mod___bert_encoder_layer_11_intermediate_dense(attention_output_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_11 = torch._C._nn.gelu(hidden_states_102);  hidden_states_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    hidden_states_104 = self.L__mod___bert_encoder_layer_11_output_dense(intermediate_output_11);  intermediate_output_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:465, code: hidden_states = self.dropout(hidden_states)
    hidden_states_105 = self.L__mod___bert_encoder_layer_11_output_dropout(hidden_states_104);  hidden_states_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_36 = hidden_states_105 + attention_output_22;  hidden_states_105 = attention_output_22 = None
    sequence_output = self.L__mod___bert_encoder_layer_11_output_LayerNorm(add_36);  add_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:680, code: hidden_states = self.dense(hidden_states)
    hidden_states_108 = self.L__mod___cls_predictions_transform_dense(sequence_output);  sequence_output = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_109 = torch._C._nn.gelu(hidden_states_108);  hidden_states_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:682, code: hidden_states = self.LayerNorm(hidden_states)
    hidden_states_111 = self.L__mod___cls_predictions_transform_LayerNorm(hidden_states_109);  hidden_states_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:702, code: hidden_states = self.decoder(hidden_states)
    prediction_scores = self.L__mod___cls_predictions_decoder(hidden_states_111);  hidden_states_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:1378, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    view_48 = prediction_scores.view(-1, 30522)
    view_49 = l_cloned_inputs_labels_.view(-1);  l_cloned_inputs_labels_ = None
    loss = torch.nn.functional.cross_entropy(view_48, view_49, None, None, -100, None, 'mean', 0.0);  view_48 = view_49 = None
    return (loss, prediction_scores)
    