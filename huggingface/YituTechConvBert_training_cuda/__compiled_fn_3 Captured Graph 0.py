from __future__ import annotations



def forward(self, L_cloned_inputs_input_ids_ : torch.Tensor, L_cloned_inputs_labels_ : torch.Tensor):
    l_cloned_inputs_input_ids_ = L_cloned_inputs_input_ids_
    l_cloned_inputs_labels_ = L_cloned_inputs_labels_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:832, code: attention_mask = torch.ones(input_shape, device=device)
    attention_mask = torch.ones((1, 512), device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:835, code: buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
    l__mod___convbert_embeddings_token_type_ids = self.L__mod___convbert_embeddings_token_type_ids
    buffered_token_type_ids = l__mod___convbert_embeddings_token_type_ids[(slice(None, None, None), slice(None, 512, None))];  l__mod___convbert_embeddings_token_type_ids = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:836, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
    token_type_ids = buffered_token_type_ids.expand(1, 512);  buffered_token_type_ids = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:916, code: extended_attention_mask = attention_mask[:, None, None, :]
    extended_attention_mask = attention_mask[(slice(None, None, None), None, None, slice(None, None, None))];  attention_mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:927, code: extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask_1 = extended_attention_mask.to(dtype = torch.float32);  extended_attention_mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:928, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    sub = 1.0 - extended_attention_mask_1;  extended_attention_mask_1 = None
    extended_attention_mask_3 = sub * -3.4028234663852886e+38;  sub = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:216, code: position_ids = self.position_ids[:, :seq_length]
    l__mod___convbert_embeddings_position_ids = self.L__mod___convbert_embeddings_position_ids
    position_ids = l__mod___convbert_embeddings_position_ids[(slice(None, None, None), slice(None, 512, None))];  l__mod___convbert_embeddings_position_ids = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:230, code: inputs_embeds = self.word_embeddings(input_ids)
    inputs_embeds = self.L__mod___convbert_embeddings_word_embeddings(l_cloned_inputs_input_ids_);  l_cloned_inputs_input_ids_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:231, code: position_embeddings = self.position_embeddings(position_ids)
    position_embeddings = self.L__mod___convbert_embeddings_position_embeddings(position_ids);  position_ids = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:232, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    token_type_embeddings = self.L__mod___convbert_embeddings_token_type_embeddings(token_type_ids);  token_type_ids = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:234, code: embeddings = inputs_embeds + position_embeddings + token_type_embeddings
    add = inputs_embeds + position_embeddings;  inputs_embeds = position_embeddings = None
    embeddings = add + token_type_embeddings;  add = token_type_embeddings = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:235, code: embeddings = self.LayerNorm(embeddings)
    embeddings_1 = self.L__mod___convbert_embeddings_LayerNorm(embeddings);  embeddings = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:236, code: embeddings = self.dropout(embeddings)
    hidden_states = self.L__mod___convbert_embeddings_dropout(embeddings_1);  embeddings_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    mixed_query_layer = self.L__mod___convbert_encoder_layer_0_attention_self_query(hidden_states)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    mixed_key_layer = self.L__mod___convbert_encoder_layer_0_attention_self_key(hidden_states)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    mixed_value_layer = self.L__mod___convbert_encoder_layer_0_attention_self_value(hidden_states)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    transpose = hidden_states.transpose(1, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    x = self.L__mod___convbert_encoder_layer_0_attention_self_key_conv_attn_layer_depthwise(transpose);  transpose = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    x_1 = self.L__mod___convbert_encoder_layer_0_attention_self_key_conv_attn_layer_pointwise(x);  x = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    l__mod___convbert_encoder_layer_0_attention_self_key_conv_attn_layer_bias_1 = self.L__mod___convbert_encoder_layer_0_attention_self_key_conv_attn_layer_bias
    x_1 += l__mod___convbert_encoder_layer_0_attention_self_key_conv_attn_layer_bias_1;  mixed_key_conv_attn_layer = x_1;  x_1 = l__mod___convbert_encoder_layer_0_attention_self_key_conv_attn_layer_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:364, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
    mixed_key_conv_attn_layer_1 = mixed_key_conv_attn_layer.transpose(1, 2);  mixed_key_conv_attn_layer = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    x_3 = mixed_query_layer.view(1, 512, 6, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    query_layer = x_3.permute(0, 2, 1, 3);  x_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    x_4 = mixed_key_layer.view(1, 512, 6, 64);  mixed_key_layer = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    key_layer = x_4.permute(0, 2, 1, 3);  x_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    x_5 = mixed_value_layer.view(1, 512, 6, 64);  mixed_value_layer = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    value_layer = x_5.permute(0, 2, 1, 3);  x_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer_1, mixed_query_layer);  mixed_key_conv_attn_layer_1 = mixed_query_layer = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    conv_kernel_layer = self.L__mod___convbert_encoder_layer_0_attention_self_conv_kernel_layer(conv_attn_layer);  conv_attn_layer = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    conv_kernel_layer_1 = torch.reshape(conv_kernel_layer, [-1, 9, 1]);  conv_kernel_layer = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    conv_kernel_layer_2 = torch.softmax(conv_kernel_layer_1, dim = 1);  conv_kernel_layer_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    conv_out_layer = self.L__mod___convbert_encoder_layer_0_attention_self_conv_out_layer(hidden_states)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    conv_out_layer_1 = torch.reshape(conv_out_layer, [1, -1, 384]);  conv_out_layer = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    transpose_2 = conv_out_layer_1.transpose(1, 2);  conv_out_layer_1 = None
    contiguous = transpose_2.contiguous();  transpose_2 = None
    conv_out_layer_2 = contiguous.unsqueeze(-1);  contiguous = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    conv_out_layer_3 = torch.nn.functional.unfold(conv_out_layer_2, kernel_size = [9, 1], dilation = 1, padding = [4, 0], stride = 1);  conv_out_layer_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    transpose_3 = conv_out_layer_3.transpose(1, 2);  conv_out_layer_3 = None
    conv_out_layer_4 = transpose_3.reshape(1, -1, 384, 9);  transpose_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    conv_out_layer_5 = torch.reshape(conv_out_layer_4, [-1, 64, 9]);  conv_out_layer_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    conv_out_layer_6 = torch.matmul(conv_out_layer_5, conv_kernel_layer_2);  conv_out_layer_5 = conv_kernel_layer_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    conv_out_layer_7 = torch.reshape(conv_out_layer_6, [-1, 384]);  conv_out_layer_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_4 = key_layer.transpose(-1, -2);  key_layer = None
    attention_scores = torch.matmul(query_layer, transpose_4);  query_layer = transpose_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_1 = attention_scores / 8.0;  attention_scores = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:397, code: attention_scores = attention_scores + attention_mask
    attention_scores_2 = attention_scores_1 + extended_attention_mask_3;  attention_scores_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs = torch.nn.functional.softmax(attention_scores_2, dim = -1);  attention_scores_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    attention_probs_1 = self.L__mod___convbert_encoder_layer_0_attention_self_dropout(attention_probs);  attention_probs = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer = torch.matmul(attention_probs_1, value_layer);  attention_probs_1 = value_layer = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_3 = context_layer.permute(0, 2, 1, 3);  context_layer = None
    context_layer_1 = permute_3.contiguous();  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    conv_out = torch.reshape(conv_out_layer_7, [1, -1, 6, 64]);  conv_out_layer_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    context_layer_2 = torch.cat([context_layer_1, conv_out], 2);  context_layer_1 = conv_out = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    context_layer_3 = context_layer_2.view(1, 512, 768);  context_layer_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    hidden_states_1 = self.L__mod___convbert_encoder_layer_0_attention_output_dense(context_layer_3);  context_layer_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    hidden_states_2 = self.L__mod___convbert_encoder_layer_0_attention_output_dropout(hidden_states_1);  hidden_states_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_3 = hidden_states_2 + hidden_states;  hidden_states_2 = hidden_states = None
    attention_output = self.L__mod___convbert_encoder_layer_0_attention_output_LayerNorm(add_3);  add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    hidden_states_4 = self.L__mod___convbert_encoder_layer_0_intermediate_dense(attention_output)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output = torch._C._nn.gelu(hidden_states_4);  hidden_states_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    hidden_states_6 = self.L__mod___convbert_encoder_layer_0_output_dense(intermediate_output);  intermediate_output = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    hidden_states_7 = self.L__mod___convbert_encoder_layer_0_output_dropout(hidden_states_6);  hidden_states_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_4 = hidden_states_7 + attention_output;  hidden_states_7 = attention_output = None
    hidden_states_9 = self.L__mod___convbert_encoder_layer_0_output_LayerNorm(add_4);  add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    mixed_query_layer_1 = self.L__mod___convbert_encoder_layer_1_attention_self_query(hidden_states_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    mixed_key_layer_1 = self.L__mod___convbert_encoder_layer_1_attention_self_key(hidden_states_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    mixed_value_layer_1 = self.L__mod___convbert_encoder_layer_1_attention_self_value(hidden_states_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    transpose_5 = hidden_states_9.transpose(1, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    x_6 = self.L__mod___convbert_encoder_layer_1_attention_self_key_conv_attn_layer_depthwise(transpose_5);  transpose_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    x_7 = self.L__mod___convbert_encoder_layer_1_attention_self_key_conv_attn_layer_pointwise(x_6);  x_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    l__mod___convbert_encoder_layer_1_attention_self_key_conv_attn_layer_bias_1 = self.L__mod___convbert_encoder_layer_1_attention_self_key_conv_attn_layer_bias
    x_7 += l__mod___convbert_encoder_layer_1_attention_self_key_conv_attn_layer_bias_1;  mixed_key_conv_attn_layer_2 = x_7;  x_7 = l__mod___convbert_encoder_layer_1_attention_self_key_conv_attn_layer_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:364, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
    mixed_key_conv_attn_layer_3 = mixed_key_conv_attn_layer_2.transpose(1, 2);  mixed_key_conv_attn_layer_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    x_9 = mixed_query_layer_1.view(1, 512, 6, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    query_layer_1 = x_9.permute(0, 2, 1, 3);  x_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    x_10 = mixed_key_layer_1.view(1, 512, 6, 64);  mixed_key_layer_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    key_layer_1 = x_10.permute(0, 2, 1, 3);  x_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    x_11 = mixed_value_layer_1.view(1, 512, 6, 64);  mixed_value_layer_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    value_layer_1 = x_11.permute(0, 2, 1, 3);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    conv_attn_layer_1 = torch.multiply(mixed_key_conv_attn_layer_3, mixed_query_layer_1);  mixed_key_conv_attn_layer_3 = mixed_query_layer_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    conv_kernel_layer_3 = self.L__mod___convbert_encoder_layer_1_attention_self_conv_kernel_layer(conv_attn_layer_1);  conv_attn_layer_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    conv_kernel_layer_4 = torch.reshape(conv_kernel_layer_3, [-1, 9, 1]);  conv_kernel_layer_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    conv_kernel_layer_5 = torch.softmax(conv_kernel_layer_4, dim = 1);  conv_kernel_layer_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    conv_out_layer_8 = self.L__mod___convbert_encoder_layer_1_attention_self_conv_out_layer(hidden_states_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    conv_out_layer_9 = torch.reshape(conv_out_layer_8, [1, -1, 384]);  conv_out_layer_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    transpose_7 = conv_out_layer_9.transpose(1, 2);  conv_out_layer_9 = None
    contiguous_2 = transpose_7.contiguous();  transpose_7 = None
    conv_out_layer_10 = contiguous_2.unsqueeze(-1);  contiguous_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    conv_out_layer_11 = torch.nn.functional.unfold(conv_out_layer_10, kernel_size = [9, 1], dilation = 1, padding = [4, 0], stride = 1);  conv_out_layer_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    transpose_8 = conv_out_layer_11.transpose(1, 2);  conv_out_layer_11 = None
    conv_out_layer_12 = transpose_8.reshape(1, -1, 384, 9);  transpose_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    conv_out_layer_13 = torch.reshape(conv_out_layer_12, [-1, 64, 9]);  conv_out_layer_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    conv_out_layer_14 = torch.matmul(conv_out_layer_13, conv_kernel_layer_5);  conv_out_layer_13 = conv_kernel_layer_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    conv_out_layer_15 = torch.reshape(conv_out_layer_14, [-1, 384]);  conv_out_layer_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_9 = key_layer_1.transpose(-1, -2);  key_layer_1 = None
    attention_scores_3 = torch.matmul(query_layer_1, transpose_9);  query_layer_1 = transpose_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_4 = attention_scores_3 / 8.0;  attention_scores_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:397, code: attention_scores = attention_scores + attention_mask
    attention_scores_5 = attention_scores_4 + extended_attention_mask_3;  attention_scores_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_2 = torch.nn.functional.softmax(attention_scores_5, dim = -1);  attention_scores_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    attention_probs_3 = self.L__mod___convbert_encoder_layer_1_attention_self_dropout(attention_probs_2);  attention_probs_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_4 = torch.matmul(attention_probs_3, value_layer_1);  attention_probs_3 = value_layer_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_7 = context_layer_4.permute(0, 2, 1, 3);  context_layer_4 = None
    context_layer_5 = permute_7.contiguous();  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    conv_out_1 = torch.reshape(conv_out_layer_15, [1, -1, 6, 64]);  conv_out_layer_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    context_layer_6 = torch.cat([context_layer_5, conv_out_1], 2);  context_layer_5 = conv_out_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    context_layer_7 = context_layer_6.view(1, 512, 768);  context_layer_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    hidden_states_10 = self.L__mod___convbert_encoder_layer_1_attention_output_dense(context_layer_7);  context_layer_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    hidden_states_11 = self.L__mod___convbert_encoder_layer_1_attention_output_dropout(hidden_states_10);  hidden_states_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_6 = hidden_states_11 + hidden_states_9;  hidden_states_11 = hidden_states_9 = None
    attention_output_2 = self.L__mod___convbert_encoder_layer_1_attention_output_LayerNorm(add_6);  add_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    hidden_states_13 = self.L__mod___convbert_encoder_layer_1_intermediate_dense(attention_output_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_1 = torch._C._nn.gelu(hidden_states_13);  hidden_states_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    hidden_states_15 = self.L__mod___convbert_encoder_layer_1_output_dense(intermediate_output_1);  intermediate_output_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    hidden_states_16 = self.L__mod___convbert_encoder_layer_1_output_dropout(hidden_states_15);  hidden_states_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_7 = hidden_states_16 + attention_output_2;  hidden_states_16 = attention_output_2 = None
    hidden_states_18 = self.L__mod___convbert_encoder_layer_1_output_LayerNorm(add_7);  add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    mixed_query_layer_2 = self.L__mod___convbert_encoder_layer_2_attention_self_query(hidden_states_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    mixed_key_layer_2 = self.L__mod___convbert_encoder_layer_2_attention_self_key(hidden_states_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    mixed_value_layer_2 = self.L__mod___convbert_encoder_layer_2_attention_self_value(hidden_states_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    transpose_10 = hidden_states_18.transpose(1, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    x_12 = self.L__mod___convbert_encoder_layer_2_attention_self_key_conv_attn_layer_depthwise(transpose_10);  transpose_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    x_13 = self.L__mod___convbert_encoder_layer_2_attention_self_key_conv_attn_layer_pointwise(x_12);  x_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    l__mod___convbert_encoder_layer_2_attention_self_key_conv_attn_layer_bias_1 = self.L__mod___convbert_encoder_layer_2_attention_self_key_conv_attn_layer_bias
    x_13 += l__mod___convbert_encoder_layer_2_attention_self_key_conv_attn_layer_bias_1;  mixed_key_conv_attn_layer_4 = x_13;  x_13 = l__mod___convbert_encoder_layer_2_attention_self_key_conv_attn_layer_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:364, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
    mixed_key_conv_attn_layer_5 = mixed_key_conv_attn_layer_4.transpose(1, 2);  mixed_key_conv_attn_layer_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    x_15 = mixed_query_layer_2.view(1, 512, 6, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    query_layer_2 = x_15.permute(0, 2, 1, 3);  x_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    x_16 = mixed_key_layer_2.view(1, 512, 6, 64);  mixed_key_layer_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    key_layer_2 = x_16.permute(0, 2, 1, 3);  x_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    x_17 = mixed_value_layer_2.view(1, 512, 6, 64);  mixed_value_layer_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    value_layer_2 = x_17.permute(0, 2, 1, 3);  x_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    conv_attn_layer_2 = torch.multiply(mixed_key_conv_attn_layer_5, mixed_query_layer_2);  mixed_key_conv_attn_layer_5 = mixed_query_layer_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    conv_kernel_layer_6 = self.L__mod___convbert_encoder_layer_2_attention_self_conv_kernel_layer(conv_attn_layer_2);  conv_attn_layer_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    conv_kernel_layer_7 = torch.reshape(conv_kernel_layer_6, [-1, 9, 1]);  conv_kernel_layer_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    conv_kernel_layer_8 = torch.softmax(conv_kernel_layer_7, dim = 1);  conv_kernel_layer_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    conv_out_layer_16 = self.L__mod___convbert_encoder_layer_2_attention_self_conv_out_layer(hidden_states_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    conv_out_layer_17 = torch.reshape(conv_out_layer_16, [1, -1, 384]);  conv_out_layer_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    transpose_12 = conv_out_layer_17.transpose(1, 2);  conv_out_layer_17 = None
    contiguous_4 = transpose_12.contiguous();  transpose_12 = None
    conv_out_layer_18 = contiguous_4.unsqueeze(-1);  contiguous_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    conv_out_layer_19 = torch.nn.functional.unfold(conv_out_layer_18, kernel_size = [9, 1], dilation = 1, padding = [4, 0], stride = 1);  conv_out_layer_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    transpose_13 = conv_out_layer_19.transpose(1, 2);  conv_out_layer_19 = None
    conv_out_layer_20 = transpose_13.reshape(1, -1, 384, 9);  transpose_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    conv_out_layer_21 = torch.reshape(conv_out_layer_20, [-1, 64, 9]);  conv_out_layer_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    conv_out_layer_22 = torch.matmul(conv_out_layer_21, conv_kernel_layer_8);  conv_out_layer_21 = conv_kernel_layer_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    conv_out_layer_23 = torch.reshape(conv_out_layer_22, [-1, 384]);  conv_out_layer_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_14 = key_layer_2.transpose(-1, -2);  key_layer_2 = None
    attention_scores_6 = torch.matmul(query_layer_2, transpose_14);  query_layer_2 = transpose_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_7 = attention_scores_6 / 8.0;  attention_scores_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:397, code: attention_scores = attention_scores + attention_mask
    attention_scores_8 = attention_scores_7 + extended_attention_mask_3;  attention_scores_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_4 = torch.nn.functional.softmax(attention_scores_8, dim = -1);  attention_scores_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    attention_probs_5 = self.L__mod___convbert_encoder_layer_2_attention_self_dropout(attention_probs_4);  attention_probs_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_8 = torch.matmul(attention_probs_5, value_layer_2);  attention_probs_5 = value_layer_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_11 = context_layer_8.permute(0, 2, 1, 3);  context_layer_8 = None
    context_layer_9 = permute_11.contiguous();  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    conv_out_2 = torch.reshape(conv_out_layer_23, [1, -1, 6, 64]);  conv_out_layer_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    context_layer_10 = torch.cat([context_layer_9, conv_out_2], 2);  context_layer_9 = conv_out_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    context_layer_11 = context_layer_10.view(1, 512, 768);  context_layer_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    hidden_states_19 = self.L__mod___convbert_encoder_layer_2_attention_output_dense(context_layer_11);  context_layer_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    hidden_states_20 = self.L__mod___convbert_encoder_layer_2_attention_output_dropout(hidden_states_19);  hidden_states_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_9 = hidden_states_20 + hidden_states_18;  hidden_states_20 = hidden_states_18 = None
    attention_output_4 = self.L__mod___convbert_encoder_layer_2_attention_output_LayerNorm(add_9);  add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    hidden_states_22 = self.L__mod___convbert_encoder_layer_2_intermediate_dense(attention_output_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_2 = torch._C._nn.gelu(hidden_states_22);  hidden_states_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    hidden_states_24 = self.L__mod___convbert_encoder_layer_2_output_dense(intermediate_output_2);  intermediate_output_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    hidden_states_25 = self.L__mod___convbert_encoder_layer_2_output_dropout(hidden_states_24);  hidden_states_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_10 = hidden_states_25 + attention_output_4;  hidden_states_25 = attention_output_4 = None
    hidden_states_27 = self.L__mod___convbert_encoder_layer_2_output_LayerNorm(add_10);  add_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    mixed_query_layer_3 = self.L__mod___convbert_encoder_layer_3_attention_self_query(hidden_states_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    mixed_key_layer_3 = self.L__mod___convbert_encoder_layer_3_attention_self_key(hidden_states_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    mixed_value_layer_3 = self.L__mod___convbert_encoder_layer_3_attention_self_value(hidden_states_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    transpose_15 = hidden_states_27.transpose(1, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    x_18 = self.L__mod___convbert_encoder_layer_3_attention_self_key_conv_attn_layer_depthwise(transpose_15);  transpose_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    x_19 = self.L__mod___convbert_encoder_layer_3_attention_self_key_conv_attn_layer_pointwise(x_18);  x_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    l__mod___convbert_encoder_layer_3_attention_self_key_conv_attn_layer_bias_1 = self.L__mod___convbert_encoder_layer_3_attention_self_key_conv_attn_layer_bias
    x_19 += l__mod___convbert_encoder_layer_3_attention_self_key_conv_attn_layer_bias_1;  mixed_key_conv_attn_layer_6 = x_19;  x_19 = l__mod___convbert_encoder_layer_3_attention_self_key_conv_attn_layer_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:364, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
    mixed_key_conv_attn_layer_7 = mixed_key_conv_attn_layer_6.transpose(1, 2);  mixed_key_conv_attn_layer_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    x_21 = mixed_query_layer_3.view(1, 512, 6, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    query_layer_3 = x_21.permute(0, 2, 1, 3);  x_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    x_22 = mixed_key_layer_3.view(1, 512, 6, 64);  mixed_key_layer_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    key_layer_3 = x_22.permute(0, 2, 1, 3);  x_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    x_23 = mixed_value_layer_3.view(1, 512, 6, 64);  mixed_value_layer_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    value_layer_3 = x_23.permute(0, 2, 1, 3);  x_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    conv_attn_layer_3 = torch.multiply(mixed_key_conv_attn_layer_7, mixed_query_layer_3);  mixed_key_conv_attn_layer_7 = mixed_query_layer_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    conv_kernel_layer_9 = self.L__mod___convbert_encoder_layer_3_attention_self_conv_kernel_layer(conv_attn_layer_3);  conv_attn_layer_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    conv_kernel_layer_10 = torch.reshape(conv_kernel_layer_9, [-1, 9, 1]);  conv_kernel_layer_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    conv_kernel_layer_11 = torch.softmax(conv_kernel_layer_10, dim = 1);  conv_kernel_layer_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    conv_out_layer_24 = self.L__mod___convbert_encoder_layer_3_attention_self_conv_out_layer(hidden_states_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    conv_out_layer_25 = torch.reshape(conv_out_layer_24, [1, -1, 384]);  conv_out_layer_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    transpose_17 = conv_out_layer_25.transpose(1, 2);  conv_out_layer_25 = None
    contiguous_6 = transpose_17.contiguous();  transpose_17 = None
    conv_out_layer_26 = contiguous_6.unsqueeze(-1);  contiguous_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    conv_out_layer_27 = torch.nn.functional.unfold(conv_out_layer_26, kernel_size = [9, 1], dilation = 1, padding = [4, 0], stride = 1);  conv_out_layer_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    transpose_18 = conv_out_layer_27.transpose(1, 2);  conv_out_layer_27 = None
    conv_out_layer_28 = transpose_18.reshape(1, -1, 384, 9);  transpose_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    conv_out_layer_29 = torch.reshape(conv_out_layer_28, [-1, 64, 9]);  conv_out_layer_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    conv_out_layer_30 = torch.matmul(conv_out_layer_29, conv_kernel_layer_11);  conv_out_layer_29 = conv_kernel_layer_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    conv_out_layer_31 = torch.reshape(conv_out_layer_30, [-1, 384]);  conv_out_layer_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_19 = key_layer_3.transpose(-1, -2);  key_layer_3 = None
    attention_scores_9 = torch.matmul(query_layer_3, transpose_19);  query_layer_3 = transpose_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_10 = attention_scores_9 / 8.0;  attention_scores_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:397, code: attention_scores = attention_scores + attention_mask
    attention_scores_11 = attention_scores_10 + extended_attention_mask_3;  attention_scores_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_6 = torch.nn.functional.softmax(attention_scores_11, dim = -1);  attention_scores_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    attention_probs_7 = self.L__mod___convbert_encoder_layer_3_attention_self_dropout(attention_probs_6);  attention_probs_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_12 = torch.matmul(attention_probs_7, value_layer_3);  attention_probs_7 = value_layer_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_15 = context_layer_12.permute(0, 2, 1, 3);  context_layer_12 = None
    context_layer_13 = permute_15.contiguous();  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    conv_out_3 = torch.reshape(conv_out_layer_31, [1, -1, 6, 64]);  conv_out_layer_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    context_layer_14 = torch.cat([context_layer_13, conv_out_3], 2);  context_layer_13 = conv_out_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    context_layer_15 = context_layer_14.view(1, 512, 768);  context_layer_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    hidden_states_28 = self.L__mod___convbert_encoder_layer_3_attention_output_dense(context_layer_15);  context_layer_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    hidden_states_29 = self.L__mod___convbert_encoder_layer_3_attention_output_dropout(hidden_states_28);  hidden_states_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_12 = hidden_states_29 + hidden_states_27;  hidden_states_29 = hidden_states_27 = None
    attention_output_6 = self.L__mod___convbert_encoder_layer_3_attention_output_LayerNorm(add_12);  add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    hidden_states_31 = self.L__mod___convbert_encoder_layer_3_intermediate_dense(attention_output_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_3 = torch._C._nn.gelu(hidden_states_31);  hidden_states_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    hidden_states_33 = self.L__mod___convbert_encoder_layer_3_output_dense(intermediate_output_3);  intermediate_output_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    hidden_states_34 = self.L__mod___convbert_encoder_layer_3_output_dropout(hidden_states_33);  hidden_states_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_13 = hidden_states_34 + attention_output_6;  hidden_states_34 = attention_output_6 = None
    hidden_states_36 = self.L__mod___convbert_encoder_layer_3_output_LayerNorm(add_13);  add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    mixed_query_layer_4 = self.L__mod___convbert_encoder_layer_4_attention_self_query(hidden_states_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    mixed_key_layer_4 = self.L__mod___convbert_encoder_layer_4_attention_self_key(hidden_states_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    mixed_value_layer_4 = self.L__mod___convbert_encoder_layer_4_attention_self_value(hidden_states_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    transpose_20 = hidden_states_36.transpose(1, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    x_24 = self.L__mod___convbert_encoder_layer_4_attention_self_key_conv_attn_layer_depthwise(transpose_20);  transpose_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    x_25 = self.L__mod___convbert_encoder_layer_4_attention_self_key_conv_attn_layer_pointwise(x_24);  x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    l__mod___convbert_encoder_layer_4_attention_self_key_conv_attn_layer_bias_1 = self.L__mod___convbert_encoder_layer_4_attention_self_key_conv_attn_layer_bias
    x_25 += l__mod___convbert_encoder_layer_4_attention_self_key_conv_attn_layer_bias_1;  mixed_key_conv_attn_layer_8 = x_25;  x_25 = l__mod___convbert_encoder_layer_4_attention_self_key_conv_attn_layer_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:364, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
    mixed_key_conv_attn_layer_9 = mixed_key_conv_attn_layer_8.transpose(1, 2);  mixed_key_conv_attn_layer_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    x_27 = mixed_query_layer_4.view(1, 512, 6, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    query_layer_4 = x_27.permute(0, 2, 1, 3);  x_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    x_28 = mixed_key_layer_4.view(1, 512, 6, 64);  mixed_key_layer_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    key_layer_4 = x_28.permute(0, 2, 1, 3);  x_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    x_29 = mixed_value_layer_4.view(1, 512, 6, 64);  mixed_value_layer_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    value_layer_4 = x_29.permute(0, 2, 1, 3);  x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    conv_attn_layer_4 = torch.multiply(mixed_key_conv_attn_layer_9, mixed_query_layer_4);  mixed_key_conv_attn_layer_9 = mixed_query_layer_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    conv_kernel_layer_12 = self.L__mod___convbert_encoder_layer_4_attention_self_conv_kernel_layer(conv_attn_layer_4);  conv_attn_layer_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    conv_kernel_layer_13 = torch.reshape(conv_kernel_layer_12, [-1, 9, 1]);  conv_kernel_layer_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    conv_kernel_layer_14 = torch.softmax(conv_kernel_layer_13, dim = 1);  conv_kernel_layer_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    conv_out_layer_32 = self.L__mod___convbert_encoder_layer_4_attention_self_conv_out_layer(hidden_states_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    conv_out_layer_33 = torch.reshape(conv_out_layer_32, [1, -1, 384]);  conv_out_layer_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    transpose_22 = conv_out_layer_33.transpose(1, 2);  conv_out_layer_33 = None
    contiguous_8 = transpose_22.contiguous();  transpose_22 = None
    conv_out_layer_34 = contiguous_8.unsqueeze(-1);  contiguous_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    conv_out_layer_35 = torch.nn.functional.unfold(conv_out_layer_34, kernel_size = [9, 1], dilation = 1, padding = [4, 0], stride = 1);  conv_out_layer_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    transpose_23 = conv_out_layer_35.transpose(1, 2);  conv_out_layer_35 = None
    conv_out_layer_36 = transpose_23.reshape(1, -1, 384, 9);  transpose_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    conv_out_layer_37 = torch.reshape(conv_out_layer_36, [-1, 64, 9]);  conv_out_layer_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    conv_out_layer_38 = torch.matmul(conv_out_layer_37, conv_kernel_layer_14);  conv_out_layer_37 = conv_kernel_layer_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    conv_out_layer_39 = torch.reshape(conv_out_layer_38, [-1, 384]);  conv_out_layer_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_24 = key_layer_4.transpose(-1, -2);  key_layer_4 = None
    attention_scores_12 = torch.matmul(query_layer_4, transpose_24);  query_layer_4 = transpose_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_13 = attention_scores_12 / 8.0;  attention_scores_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:397, code: attention_scores = attention_scores + attention_mask
    attention_scores_14 = attention_scores_13 + extended_attention_mask_3;  attention_scores_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_8 = torch.nn.functional.softmax(attention_scores_14, dim = -1);  attention_scores_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    attention_probs_9 = self.L__mod___convbert_encoder_layer_4_attention_self_dropout(attention_probs_8);  attention_probs_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_16 = torch.matmul(attention_probs_9, value_layer_4);  attention_probs_9 = value_layer_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_19 = context_layer_16.permute(0, 2, 1, 3);  context_layer_16 = None
    context_layer_17 = permute_19.contiguous();  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    conv_out_4 = torch.reshape(conv_out_layer_39, [1, -1, 6, 64]);  conv_out_layer_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    context_layer_18 = torch.cat([context_layer_17, conv_out_4], 2);  context_layer_17 = conv_out_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    context_layer_19 = context_layer_18.view(1, 512, 768);  context_layer_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    hidden_states_37 = self.L__mod___convbert_encoder_layer_4_attention_output_dense(context_layer_19);  context_layer_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    hidden_states_38 = self.L__mod___convbert_encoder_layer_4_attention_output_dropout(hidden_states_37);  hidden_states_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_15 = hidden_states_38 + hidden_states_36;  hidden_states_38 = hidden_states_36 = None
    attention_output_8 = self.L__mod___convbert_encoder_layer_4_attention_output_LayerNorm(add_15);  add_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    hidden_states_40 = self.L__mod___convbert_encoder_layer_4_intermediate_dense(attention_output_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_4 = torch._C._nn.gelu(hidden_states_40);  hidden_states_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    hidden_states_42 = self.L__mod___convbert_encoder_layer_4_output_dense(intermediate_output_4);  intermediate_output_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    hidden_states_43 = self.L__mod___convbert_encoder_layer_4_output_dropout(hidden_states_42);  hidden_states_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_16 = hidden_states_43 + attention_output_8;  hidden_states_43 = attention_output_8 = None
    hidden_states_45 = self.L__mod___convbert_encoder_layer_4_output_LayerNorm(add_16);  add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    mixed_query_layer_5 = self.L__mod___convbert_encoder_layer_5_attention_self_query(hidden_states_45)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    mixed_key_layer_5 = self.L__mod___convbert_encoder_layer_5_attention_self_key(hidden_states_45)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    mixed_value_layer_5 = self.L__mod___convbert_encoder_layer_5_attention_self_value(hidden_states_45)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    transpose_25 = hidden_states_45.transpose(1, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    x_30 = self.L__mod___convbert_encoder_layer_5_attention_self_key_conv_attn_layer_depthwise(transpose_25);  transpose_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    x_31 = self.L__mod___convbert_encoder_layer_5_attention_self_key_conv_attn_layer_pointwise(x_30);  x_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    l__mod___convbert_encoder_layer_5_attention_self_key_conv_attn_layer_bias_1 = self.L__mod___convbert_encoder_layer_5_attention_self_key_conv_attn_layer_bias
    x_31 += l__mod___convbert_encoder_layer_5_attention_self_key_conv_attn_layer_bias_1;  mixed_key_conv_attn_layer_10 = x_31;  x_31 = l__mod___convbert_encoder_layer_5_attention_self_key_conv_attn_layer_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:364, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
    mixed_key_conv_attn_layer_11 = mixed_key_conv_attn_layer_10.transpose(1, 2);  mixed_key_conv_attn_layer_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    x_33 = mixed_query_layer_5.view(1, 512, 6, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    query_layer_5 = x_33.permute(0, 2, 1, 3);  x_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    x_34 = mixed_key_layer_5.view(1, 512, 6, 64);  mixed_key_layer_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    key_layer_5 = x_34.permute(0, 2, 1, 3);  x_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    x_35 = mixed_value_layer_5.view(1, 512, 6, 64);  mixed_value_layer_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    value_layer_5 = x_35.permute(0, 2, 1, 3);  x_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    conv_attn_layer_5 = torch.multiply(mixed_key_conv_attn_layer_11, mixed_query_layer_5);  mixed_key_conv_attn_layer_11 = mixed_query_layer_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    conv_kernel_layer_15 = self.L__mod___convbert_encoder_layer_5_attention_self_conv_kernel_layer(conv_attn_layer_5);  conv_attn_layer_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    conv_kernel_layer_16 = torch.reshape(conv_kernel_layer_15, [-1, 9, 1]);  conv_kernel_layer_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    conv_kernel_layer_17 = torch.softmax(conv_kernel_layer_16, dim = 1);  conv_kernel_layer_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    conv_out_layer_40 = self.L__mod___convbert_encoder_layer_5_attention_self_conv_out_layer(hidden_states_45)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    conv_out_layer_41 = torch.reshape(conv_out_layer_40, [1, -1, 384]);  conv_out_layer_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    transpose_27 = conv_out_layer_41.transpose(1, 2);  conv_out_layer_41 = None
    contiguous_10 = transpose_27.contiguous();  transpose_27 = None
    conv_out_layer_42 = contiguous_10.unsqueeze(-1);  contiguous_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    conv_out_layer_43 = torch.nn.functional.unfold(conv_out_layer_42, kernel_size = [9, 1], dilation = 1, padding = [4, 0], stride = 1);  conv_out_layer_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    transpose_28 = conv_out_layer_43.transpose(1, 2);  conv_out_layer_43 = None
    conv_out_layer_44 = transpose_28.reshape(1, -1, 384, 9);  transpose_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    conv_out_layer_45 = torch.reshape(conv_out_layer_44, [-1, 64, 9]);  conv_out_layer_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    conv_out_layer_46 = torch.matmul(conv_out_layer_45, conv_kernel_layer_17);  conv_out_layer_45 = conv_kernel_layer_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    conv_out_layer_47 = torch.reshape(conv_out_layer_46, [-1, 384]);  conv_out_layer_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_29 = key_layer_5.transpose(-1, -2);  key_layer_5 = None
    attention_scores_15 = torch.matmul(query_layer_5, transpose_29);  query_layer_5 = transpose_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_16 = attention_scores_15 / 8.0;  attention_scores_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:397, code: attention_scores = attention_scores + attention_mask
    attention_scores_17 = attention_scores_16 + extended_attention_mask_3;  attention_scores_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_10 = torch.nn.functional.softmax(attention_scores_17, dim = -1);  attention_scores_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    attention_probs_11 = self.L__mod___convbert_encoder_layer_5_attention_self_dropout(attention_probs_10);  attention_probs_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_20 = torch.matmul(attention_probs_11, value_layer_5);  attention_probs_11 = value_layer_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_23 = context_layer_20.permute(0, 2, 1, 3);  context_layer_20 = None
    context_layer_21 = permute_23.contiguous();  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    conv_out_5 = torch.reshape(conv_out_layer_47, [1, -1, 6, 64]);  conv_out_layer_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    context_layer_22 = torch.cat([context_layer_21, conv_out_5], 2);  context_layer_21 = conv_out_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    context_layer_23 = context_layer_22.view(1, 512, 768);  context_layer_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    hidden_states_46 = self.L__mod___convbert_encoder_layer_5_attention_output_dense(context_layer_23);  context_layer_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    hidden_states_47 = self.L__mod___convbert_encoder_layer_5_attention_output_dropout(hidden_states_46);  hidden_states_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_18 = hidden_states_47 + hidden_states_45;  hidden_states_47 = hidden_states_45 = None
    attention_output_10 = self.L__mod___convbert_encoder_layer_5_attention_output_LayerNorm(add_18);  add_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    hidden_states_49 = self.L__mod___convbert_encoder_layer_5_intermediate_dense(attention_output_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_5 = torch._C._nn.gelu(hidden_states_49);  hidden_states_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    hidden_states_51 = self.L__mod___convbert_encoder_layer_5_output_dense(intermediate_output_5);  intermediate_output_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    hidden_states_52 = self.L__mod___convbert_encoder_layer_5_output_dropout(hidden_states_51);  hidden_states_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_19 = hidden_states_52 + attention_output_10;  hidden_states_52 = attention_output_10 = None
    hidden_states_54 = self.L__mod___convbert_encoder_layer_5_output_LayerNorm(add_19);  add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    mixed_query_layer_6 = self.L__mod___convbert_encoder_layer_6_attention_self_query(hidden_states_54)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    mixed_key_layer_6 = self.L__mod___convbert_encoder_layer_6_attention_self_key(hidden_states_54)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    mixed_value_layer_6 = self.L__mod___convbert_encoder_layer_6_attention_self_value(hidden_states_54)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    transpose_30 = hidden_states_54.transpose(1, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    x_36 = self.L__mod___convbert_encoder_layer_6_attention_self_key_conv_attn_layer_depthwise(transpose_30);  transpose_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    x_37 = self.L__mod___convbert_encoder_layer_6_attention_self_key_conv_attn_layer_pointwise(x_36);  x_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    l__mod___convbert_encoder_layer_6_attention_self_key_conv_attn_layer_bias_1 = self.L__mod___convbert_encoder_layer_6_attention_self_key_conv_attn_layer_bias
    x_37 += l__mod___convbert_encoder_layer_6_attention_self_key_conv_attn_layer_bias_1;  mixed_key_conv_attn_layer_12 = x_37;  x_37 = l__mod___convbert_encoder_layer_6_attention_self_key_conv_attn_layer_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:364, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
    mixed_key_conv_attn_layer_13 = mixed_key_conv_attn_layer_12.transpose(1, 2);  mixed_key_conv_attn_layer_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    x_39 = mixed_query_layer_6.view(1, 512, 6, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    query_layer_6 = x_39.permute(0, 2, 1, 3);  x_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    x_40 = mixed_key_layer_6.view(1, 512, 6, 64);  mixed_key_layer_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    key_layer_6 = x_40.permute(0, 2, 1, 3);  x_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    x_41 = mixed_value_layer_6.view(1, 512, 6, 64);  mixed_value_layer_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    value_layer_6 = x_41.permute(0, 2, 1, 3);  x_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    conv_attn_layer_6 = torch.multiply(mixed_key_conv_attn_layer_13, mixed_query_layer_6);  mixed_key_conv_attn_layer_13 = mixed_query_layer_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    conv_kernel_layer_18 = self.L__mod___convbert_encoder_layer_6_attention_self_conv_kernel_layer(conv_attn_layer_6);  conv_attn_layer_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    conv_kernel_layer_19 = torch.reshape(conv_kernel_layer_18, [-1, 9, 1]);  conv_kernel_layer_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    conv_kernel_layer_20 = torch.softmax(conv_kernel_layer_19, dim = 1);  conv_kernel_layer_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    conv_out_layer_48 = self.L__mod___convbert_encoder_layer_6_attention_self_conv_out_layer(hidden_states_54)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    conv_out_layer_49 = torch.reshape(conv_out_layer_48, [1, -1, 384]);  conv_out_layer_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    transpose_32 = conv_out_layer_49.transpose(1, 2);  conv_out_layer_49 = None
    contiguous_12 = transpose_32.contiguous();  transpose_32 = None
    conv_out_layer_50 = contiguous_12.unsqueeze(-1);  contiguous_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    conv_out_layer_51 = torch.nn.functional.unfold(conv_out_layer_50, kernel_size = [9, 1], dilation = 1, padding = [4, 0], stride = 1);  conv_out_layer_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    transpose_33 = conv_out_layer_51.transpose(1, 2);  conv_out_layer_51 = None
    conv_out_layer_52 = transpose_33.reshape(1, -1, 384, 9);  transpose_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    conv_out_layer_53 = torch.reshape(conv_out_layer_52, [-1, 64, 9]);  conv_out_layer_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    conv_out_layer_54 = torch.matmul(conv_out_layer_53, conv_kernel_layer_20);  conv_out_layer_53 = conv_kernel_layer_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    conv_out_layer_55 = torch.reshape(conv_out_layer_54, [-1, 384]);  conv_out_layer_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_34 = key_layer_6.transpose(-1, -2);  key_layer_6 = None
    attention_scores_18 = torch.matmul(query_layer_6, transpose_34);  query_layer_6 = transpose_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_19 = attention_scores_18 / 8.0;  attention_scores_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:397, code: attention_scores = attention_scores + attention_mask
    attention_scores_20 = attention_scores_19 + extended_attention_mask_3;  attention_scores_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_12 = torch.nn.functional.softmax(attention_scores_20, dim = -1);  attention_scores_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    attention_probs_13 = self.L__mod___convbert_encoder_layer_6_attention_self_dropout(attention_probs_12);  attention_probs_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_24 = torch.matmul(attention_probs_13, value_layer_6);  attention_probs_13 = value_layer_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_27 = context_layer_24.permute(0, 2, 1, 3);  context_layer_24 = None
    context_layer_25 = permute_27.contiguous();  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    conv_out_6 = torch.reshape(conv_out_layer_55, [1, -1, 6, 64]);  conv_out_layer_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    context_layer_26 = torch.cat([context_layer_25, conv_out_6], 2);  context_layer_25 = conv_out_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    context_layer_27 = context_layer_26.view(1, 512, 768);  context_layer_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    hidden_states_55 = self.L__mod___convbert_encoder_layer_6_attention_output_dense(context_layer_27);  context_layer_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    hidden_states_56 = self.L__mod___convbert_encoder_layer_6_attention_output_dropout(hidden_states_55);  hidden_states_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_21 = hidden_states_56 + hidden_states_54;  hidden_states_56 = hidden_states_54 = None
    attention_output_12 = self.L__mod___convbert_encoder_layer_6_attention_output_LayerNorm(add_21);  add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    hidden_states_58 = self.L__mod___convbert_encoder_layer_6_intermediate_dense(attention_output_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_6 = torch._C._nn.gelu(hidden_states_58);  hidden_states_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    hidden_states_60 = self.L__mod___convbert_encoder_layer_6_output_dense(intermediate_output_6);  intermediate_output_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    hidden_states_61 = self.L__mod___convbert_encoder_layer_6_output_dropout(hidden_states_60);  hidden_states_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_22 = hidden_states_61 + attention_output_12;  hidden_states_61 = attention_output_12 = None
    hidden_states_63 = self.L__mod___convbert_encoder_layer_6_output_LayerNorm(add_22);  add_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    mixed_query_layer_7 = self.L__mod___convbert_encoder_layer_7_attention_self_query(hidden_states_63)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    mixed_key_layer_7 = self.L__mod___convbert_encoder_layer_7_attention_self_key(hidden_states_63)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    mixed_value_layer_7 = self.L__mod___convbert_encoder_layer_7_attention_self_value(hidden_states_63)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    transpose_35 = hidden_states_63.transpose(1, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    x_42 = self.L__mod___convbert_encoder_layer_7_attention_self_key_conv_attn_layer_depthwise(transpose_35);  transpose_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    x_43 = self.L__mod___convbert_encoder_layer_7_attention_self_key_conv_attn_layer_pointwise(x_42);  x_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    l__mod___convbert_encoder_layer_7_attention_self_key_conv_attn_layer_bias_1 = self.L__mod___convbert_encoder_layer_7_attention_self_key_conv_attn_layer_bias
    x_43 += l__mod___convbert_encoder_layer_7_attention_self_key_conv_attn_layer_bias_1;  mixed_key_conv_attn_layer_14 = x_43;  x_43 = l__mod___convbert_encoder_layer_7_attention_self_key_conv_attn_layer_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:364, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
    mixed_key_conv_attn_layer_15 = mixed_key_conv_attn_layer_14.transpose(1, 2);  mixed_key_conv_attn_layer_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    x_45 = mixed_query_layer_7.view(1, 512, 6, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    query_layer_7 = x_45.permute(0, 2, 1, 3);  x_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    x_46 = mixed_key_layer_7.view(1, 512, 6, 64);  mixed_key_layer_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    key_layer_7 = x_46.permute(0, 2, 1, 3);  x_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    x_47 = mixed_value_layer_7.view(1, 512, 6, 64);  mixed_value_layer_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    value_layer_7 = x_47.permute(0, 2, 1, 3);  x_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    conv_attn_layer_7 = torch.multiply(mixed_key_conv_attn_layer_15, mixed_query_layer_7);  mixed_key_conv_attn_layer_15 = mixed_query_layer_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    conv_kernel_layer_21 = self.L__mod___convbert_encoder_layer_7_attention_self_conv_kernel_layer(conv_attn_layer_7);  conv_attn_layer_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    conv_kernel_layer_22 = torch.reshape(conv_kernel_layer_21, [-1, 9, 1]);  conv_kernel_layer_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    conv_kernel_layer_23 = torch.softmax(conv_kernel_layer_22, dim = 1);  conv_kernel_layer_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    conv_out_layer_56 = self.L__mod___convbert_encoder_layer_7_attention_self_conv_out_layer(hidden_states_63)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    conv_out_layer_57 = torch.reshape(conv_out_layer_56, [1, -1, 384]);  conv_out_layer_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    transpose_37 = conv_out_layer_57.transpose(1, 2);  conv_out_layer_57 = None
    contiguous_14 = transpose_37.contiguous();  transpose_37 = None
    conv_out_layer_58 = contiguous_14.unsqueeze(-1);  contiguous_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    conv_out_layer_59 = torch.nn.functional.unfold(conv_out_layer_58, kernel_size = [9, 1], dilation = 1, padding = [4, 0], stride = 1);  conv_out_layer_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    transpose_38 = conv_out_layer_59.transpose(1, 2);  conv_out_layer_59 = None
    conv_out_layer_60 = transpose_38.reshape(1, -1, 384, 9);  transpose_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    conv_out_layer_61 = torch.reshape(conv_out_layer_60, [-1, 64, 9]);  conv_out_layer_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    conv_out_layer_62 = torch.matmul(conv_out_layer_61, conv_kernel_layer_23);  conv_out_layer_61 = conv_kernel_layer_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    conv_out_layer_63 = torch.reshape(conv_out_layer_62, [-1, 384]);  conv_out_layer_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_39 = key_layer_7.transpose(-1, -2);  key_layer_7 = None
    attention_scores_21 = torch.matmul(query_layer_7, transpose_39);  query_layer_7 = transpose_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_22 = attention_scores_21 / 8.0;  attention_scores_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:397, code: attention_scores = attention_scores + attention_mask
    attention_scores_23 = attention_scores_22 + extended_attention_mask_3;  attention_scores_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_14 = torch.nn.functional.softmax(attention_scores_23, dim = -1);  attention_scores_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    attention_probs_15 = self.L__mod___convbert_encoder_layer_7_attention_self_dropout(attention_probs_14);  attention_probs_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_28 = torch.matmul(attention_probs_15, value_layer_7);  attention_probs_15 = value_layer_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_31 = context_layer_28.permute(0, 2, 1, 3);  context_layer_28 = None
    context_layer_29 = permute_31.contiguous();  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    conv_out_7 = torch.reshape(conv_out_layer_63, [1, -1, 6, 64]);  conv_out_layer_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    context_layer_30 = torch.cat([context_layer_29, conv_out_7], 2);  context_layer_29 = conv_out_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    context_layer_31 = context_layer_30.view(1, 512, 768);  context_layer_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    hidden_states_64 = self.L__mod___convbert_encoder_layer_7_attention_output_dense(context_layer_31);  context_layer_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    hidden_states_65 = self.L__mod___convbert_encoder_layer_7_attention_output_dropout(hidden_states_64);  hidden_states_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_24 = hidden_states_65 + hidden_states_63;  hidden_states_65 = hidden_states_63 = None
    attention_output_14 = self.L__mod___convbert_encoder_layer_7_attention_output_LayerNorm(add_24);  add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    hidden_states_67 = self.L__mod___convbert_encoder_layer_7_intermediate_dense(attention_output_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_7 = torch._C._nn.gelu(hidden_states_67);  hidden_states_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    hidden_states_69 = self.L__mod___convbert_encoder_layer_7_output_dense(intermediate_output_7);  intermediate_output_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    hidden_states_70 = self.L__mod___convbert_encoder_layer_7_output_dropout(hidden_states_69);  hidden_states_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_25 = hidden_states_70 + attention_output_14;  hidden_states_70 = attention_output_14 = None
    hidden_states_72 = self.L__mod___convbert_encoder_layer_7_output_LayerNorm(add_25);  add_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    mixed_query_layer_8 = self.L__mod___convbert_encoder_layer_8_attention_self_query(hidden_states_72)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    mixed_key_layer_8 = self.L__mod___convbert_encoder_layer_8_attention_self_key(hidden_states_72)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    mixed_value_layer_8 = self.L__mod___convbert_encoder_layer_8_attention_self_value(hidden_states_72)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    transpose_40 = hidden_states_72.transpose(1, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    x_48 = self.L__mod___convbert_encoder_layer_8_attention_self_key_conv_attn_layer_depthwise(transpose_40);  transpose_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    x_49 = self.L__mod___convbert_encoder_layer_8_attention_self_key_conv_attn_layer_pointwise(x_48);  x_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    l__mod___convbert_encoder_layer_8_attention_self_key_conv_attn_layer_bias_1 = self.L__mod___convbert_encoder_layer_8_attention_self_key_conv_attn_layer_bias
    x_49 += l__mod___convbert_encoder_layer_8_attention_self_key_conv_attn_layer_bias_1;  mixed_key_conv_attn_layer_16 = x_49;  x_49 = l__mod___convbert_encoder_layer_8_attention_self_key_conv_attn_layer_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:364, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
    mixed_key_conv_attn_layer_17 = mixed_key_conv_attn_layer_16.transpose(1, 2);  mixed_key_conv_attn_layer_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    x_51 = mixed_query_layer_8.view(1, 512, 6, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    query_layer_8 = x_51.permute(0, 2, 1, 3);  x_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    x_52 = mixed_key_layer_8.view(1, 512, 6, 64);  mixed_key_layer_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    key_layer_8 = x_52.permute(0, 2, 1, 3);  x_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    x_53 = mixed_value_layer_8.view(1, 512, 6, 64);  mixed_value_layer_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    value_layer_8 = x_53.permute(0, 2, 1, 3);  x_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    conv_attn_layer_8 = torch.multiply(mixed_key_conv_attn_layer_17, mixed_query_layer_8);  mixed_key_conv_attn_layer_17 = mixed_query_layer_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    conv_kernel_layer_24 = self.L__mod___convbert_encoder_layer_8_attention_self_conv_kernel_layer(conv_attn_layer_8);  conv_attn_layer_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    conv_kernel_layer_25 = torch.reshape(conv_kernel_layer_24, [-1, 9, 1]);  conv_kernel_layer_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    conv_kernel_layer_26 = torch.softmax(conv_kernel_layer_25, dim = 1);  conv_kernel_layer_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    conv_out_layer_64 = self.L__mod___convbert_encoder_layer_8_attention_self_conv_out_layer(hidden_states_72)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    conv_out_layer_65 = torch.reshape(conv_out_layer_64, [1, -1, 384]);  conv_out_layer_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    transpose_42 = conv_out_layer_65.transpose(1, 2);  conv_out_layer_65 = None
    contiguous_16 = transpose_42.contiguous();  transpose_42 = None
    conv_out_layer_66 = contiguous_16.unsqueeze(-1);  contiguous_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    conv_out_layer_67 = torch.nn.functional.unfold(conv_out_layer_66, kernel_size = [9, 1], dilation = 1, padding = [4, 0], stride = 1);  conv_out_layer_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    transpose_43 = conv_out_layer_67.transpose(1, 2);  conv_out_layer_67 = None
    conv_out_layer_68 = transpose_43.reshape(1, -1, 384, 9);  transpose_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    conv_out_layer_69 = torch.reshape(conv_out_layer_68, [-1, 64, 9]);  conv_out_layer_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    conv_out_layer_70 = torch.matmul(conv_out_layer_69, conv_kernel_layer_26);  conv_out_layer_69 = conv_kernel_layer_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    conv_out_layer_71 = torch.reshape(conv_out_layer_70, [-1, 384]);  conv_out_layer_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_44 = key_layer_8.transpose(-1, -2);  key_layer_8 = None
    attention_scores_24 = torch.matmul(query_layer_8, transpose_44);  query_layer_8 = transpose_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_25 = attention_scores_24 / 8.0;  attention_scores_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:397, code: attention_scores = attention_scores + attention_mask
    attention_scores_26 = attention_scores_25 + extended_attention_mask_3;  attention_scores_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_16 = torch.nn.functional.softmax(attention_scores_26, dim = -1);  attention_scores_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    attention_probs_17 = self.L__mod___convbert_encoder_layer_8_attention_self_dropout(attention_probs_16);  attention_probs_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_32 = torch.matmul(attention_probs_17, value_layer_8);  attention_probs_17 = value_layer_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_35 = context_layer_32.permute(0, 2, 1, 3);  context_layer_32 = None
    context_layer_33 = permute_35.contiguous();  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    conv_out_8 = torch.reshape(conv_out_layer_71, [1, -1, 6, 64]);  conv_out_layer_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    context_layer_34 = torch.cat([context_layer_33, conv_out_8], 2);  context_layer_33 = conv_out_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    context_layer_35 = context_layer_34.view(1, 512, 768);  context_layer_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    hidden_states_73 = self.L__mod___convbert_encoder_layer_8_attention_output_dense(context_layer_35);  context_layer_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    hidden_states_74 = self.L__mod___convbert_encoder_layer_8_attention_output_dropout(hidden_states_73);  hidden_states_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_27 = hidden_states_74 + hidden_states_72;  hidden_states_74 = hidden_states_72 = None
    attention_output_16 = self.L__mod___convbert_encoder_layer_8_attention_output_LayerNorm(add_27);  add_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    hidden_states_76 = self.L__mod___convbert_encoder_layer_8_intermediate_dense(attention_output_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_8 = torch._C._nn.gelu(hidden_states_76);  hidden_states_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    hidden_states_78 = self.L__mod___convbert_encoder_layer_8_output_dense(intermediate_output_8);  intermediate_output_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    hidden_states_79 = self.L__mod___convbert_encoder_layer_8_output_dropout(hidden_states_78);  hidden_states_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_28 = hidden_states_79 + attention_output_16;  hidden_states_79 = attention_output_16 = None
    hidden_states_81 = self.L__mod___convbert_encoder_layer_8_output_LayerNorm(add_28);  add_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    mixed_query_layer_9 = self.L__mod___convbert_encoder_layer_9_attention_self_query(hidden_states_81)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    mixed_key_layer_9 = self.L__mod___convbert_encoder_layer_9_attention_self_key(hidden_states_81)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    mixed_value_layer_9 = self.L__mod___convbert_encoder_layer_9_attention_self_value(hidden_states_81)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    transpose_45 = hidden_states_81.transpose(1, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    x_54 = self.L__mod___convbert_encoder_layer_9_attention_self_key_conv_attn_layer_depthwise(transpose_45);  transpose_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    x_55 = self.L__mod___convbert_encoder_layer_9_attention_self_key_conv_attn_layer_pointwise(x_54);  x_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    l__mod___convbert_encoder_layer_9_attention_self_key_conv_attn_layer_bias_1 = self.L__mod___convbert_encoder_layer_9_attention_self_key_conv_attn_layer_bias
    x_55 += l__mod___convbert_encoder_layer_9_attention_self_key_conv_attn_layer_bias_1;  mixed_key_conv_attn_layer_18 = x_55;  x_55 = l__mod___convbert_encoder_layer_9_attention_self_key_conv_attn_layer_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:364, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
    mixed_key_conv_attn_layer_19 = mixed_key_conv_attn_layer_18.transpose(1, 2);  mixed_key_conv_attn_layer_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    x_57 = mixed_query_layer_9.view(1, 512, 6, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    query_layer_9 = x_57.permute(0, 2, 1, 3);  x_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    x_58 = mixed_key_layer_9.view(1, 512, 6, 64);  mixed_key_layer_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    key_layer_9 = x_58.permute(0, 2, 1, 3);  x_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    x_59 = mixed_value_layer_9.view(1, 512, 6, 64);  mixed_value_layer_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    value_layer_9 = x_59.permute(0, 2, 1, 3);  x_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    conv_attn_layer_9 = torch.multiply(mixed_key_conv_attn_layer_19, mixed_query_layer_9);  mixed_key_conv_attn_layer_19 = mixed_query_layer_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    conv_kernel_layer_27 = self.L__mod___convbert_encoder_layer_9_attention_self_conv_kernel_layer(conv_attn_layer_9);  conv_attn_layer_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    conv_kernel_layer_28 = torch.reshape(conv_kernel_layer_27, [-1, 9, 1]);  conv_kernel_layer_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    conv_kernel_layer_29 = torch.softmax(conv_kernel_layer_28, dim = 1);  conv_kernel_layer_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    conv_out_layer_72 = self.L__mod___convbert_encoder_layer_9_attention_self_conv_out_layer(hidden_states_81)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    conv_out_layer_73 = torch.reshape(conv_out_layer_72, [1, -1, 384]);  conv_out_layer_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    transpose_47 = conv_out_layer_73.transpose(1, 2);  conv_out_layer_73 = None
    contiguous_18 = transpose_47.contiguous();  transpose_47 = None
    conv_out_layer_74 = contiguous_18.unsqueeze(-1);  contiguous_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    conv_out_layer_75 = torch.nn.functional.unfold(conv_out_layer_74, kernel_size = [9, 1], dilation = 1, padding = [4, 0], stride = 1);  conv_out_layer_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    transpose_48 = conv_out_layer_75.transpose(1, 2);  conv_out_layer_75 = None
    conv_out_layer_76 = transpose_48.reshape(1, -1, 384, 9);  transpose_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    conv_out_layer_77 = torch.reshape(conv_out_layer_76, [-1, 64, 9]);  conv_out_layer_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    conv_out_layer_78 = torch.matmul(conv_out_layer_77, conv_kernel_layer_29);  conv_out_layer_77 = conv_kernel_layer_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    conv_out_layer_79 = torch.reshape(conv_out_layer_78, [-1, 384]);  conv_out_layer_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_49 = key_layer_9.transpose(-1, -2);  key_layer_9 = None
    attention_scores_27 = torch.matmul(query_layer_9, transpose_49);  query_layer_9 = transpose_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_28 = attention_scores_27 / 8.0;  attention_scores_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:397, code: attention_scores = attention_scores + attention_mask
    attention_scores_29 = attention_scores_28 + extended_attention_mask_3;  attention_scores_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_18 = torch.nn.functional.softmax(attention_scores_29, dim = -1);  attention_scores_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    attention_probs_19 = self.L__mod___convbert_encoder_layer_9_attention_self_dropout(attention_probs_18);  attention_probs_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_36 = torch.matmul(attention_probs_19, value_layer_9);  attention_probs_19 = value_layer_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_39 = context_layer_36.permute(0, 2, 1, 3);  context_layer_36 = None
    context_layer_37 = permute_39.contiguous();  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    conv_out_9 = torch.reshape(conv_out_layer_79, [1, -1, 6, 64]);  conv_out_layer_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    context_layer_38 = torch.cat([context_layer_37, conv_out_9], 2);  context_layer_37 = conv_out_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    context_layer_39 = context_layer_38.view(1, 512, 768);  context_layer_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    hidden_states_82 = self.L__mod___convbert_encoder_layer_9_attention_output_dense(context_layer_39);  context_layer_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    hidden_states_83 = self.L__mod___convbert_encoder_layer_9_attention_output_dropout(hidden_states_82);  hidden_states_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_30 = hidden_states_83 + hidden_states_81;  hidden_states_83 = hidden_states_81 = None
    attention_output_18 = self.L__mod___convbert_encoder_layer_9_attention_output_LayerNorm(add_30);  add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    hidden_states_85 = self.L__mod___convbert_encoder_layer_9_intermediate_dense(attention_output_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_9 = torch._C._nn.gelu(hidden_states_85);  hidden_states_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    hidden_states_87 = self.L__mod___convbert_encoder_layer_9_output_dense(intermediate_output_9);  intermediate_output_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    hidden_states_88 = self.L__mod___convbert_encoder_layer_9_output_dropout(hidden_states_87);  hidden_states_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_31 = hidden_states_88 + attention_output_18;  hidden_states_88 = attention_output_18 = None
    hidden_states_90 = self.L__mod___convbert_encoder_layer_9_output_LayerNorm(add_31);  add_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    mixed_query_layer_10 = self.L__mod___convbert_encoder_layer_10_attention_self_query(hidden_states_90)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    mixed_key_layer_10 = self.L__mod___convbert_encoder_layer_10_attention_self_key(hidden_states_90)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    mixed_value_layer_10 = self.L__mod___convbert_encoder_layer_10_attention_self_value(hidden_states_90)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    transpose_50 = hidden_states_90.transpose(1, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    x_60 = self.L__mod___convbert_encoder_layer_10_attention_self_key_conv_attn_layer_depthwise(transpose_50);  transpose_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    x_61 = self.L__mod___convbert_encoder_layer_10_attention_self_key_conv_attn_layer_pointwise(x_60);  x_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    l__mod___convbert_encoder_layer_10_attention_self_key_conv_attn_layer_bias_1 = self.L__mod___convbert_encoder_layer_10_attention_self_key_conv_attn_layer_bias
    x_61 += l__mod___convbert_encoder_layer_10_attention_self_key_conv_attn_layer_bias_1;  mixed_key_conv_attn_layer_20 = x_61;  x_61 = l__mod___convbert_encoder_layer_10_attention_self_key_conv_attn_layer_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:364, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
    mixed_key_conv_attn_layer_21 = mixed_key_conv_attn_layer_20.transpose(1, 2);  mixed_key_conv_attn_layer_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    x_63 = mixed_query_layer_10.view(1, 512, 6, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    query_layer_10 = x_63.permute(0, 2, 1, 3);  x_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    x_64 = mixed_key_layer_10.view(1, 512, 6, 64);  mixed_key_layer_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    key_layer_10 = x_64.permute(0, 2, 1, 3);  x_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    x_65 = mixed_value_layer_10.view(1, 512, 6, 64);  mixed_value_layer_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    value_layer_10 = x_65.permute(0, 2, 1, 3);  x_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    conv_attn_layer_10 = torch.multiply(mixed_key_conv_attn_layer_21, mixed_query_layer_10);  mixed_key_conv_attn_layer_21 = mixed_query_layer_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    conv_kernel_layer_30 = self.L__mod___convbert_encoder_layer_10_attention_self_conv_kernel_layer(conv_attn_layer_10);  conv_attn_layer_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    conv_kernel_layer_31 = torch.reshape(conv_kernel_layer_30, [-1, 9, 1]);  conv_kernel_layer_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    conv_kernel_layer_32 = torch.softmax(conv_kernel_layer_31, dim = 1);  conv_kernel_layer_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    conv_out_layer_80 = self.L__mod___convbert_encoder_layer_10_attention_self_conv_out_layer(hidden_states_90)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    conv_out_layer_81 = torch.reshape(conv_out_layer_80, [1, -1, 384]);  conv_out_layer_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    transpose_52 = conv_out_layer_81.transpose(1, 2);  conv_out_layer_81 = None
    contiguous_20 = transpose_52.contiguous();  transpose_52 = None
    conv_out_layer_82 = contiguous_20.unsqueeze(-1);  contiguous_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    conv_out_layer_83 = torch.nn.functional.unfold(conv_out_layer_82, kernel_size = [9, 1], dilation = 1, padding = [4, 0], stride = 1);  conv_out_layer_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    transpose_53 = conv_out_layer_83.transpose(1, 2);  conv_out_layer_83 = None
    conv_out_layer_84 = transpose_53.reshape(1, -1, 384, 9);  transpose_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    conv_out_layer_85 = torch.reshape(conv_out_layer_84, [-1, 64, 9]);  conv_out_layer_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    conv_out_layer_86 = torch.matmul(conv_out_layer_85, conv_kernel_layer_32);  conv_out_layer_85 = conv_kernel_layer_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    conv_out_layer_87 = torch.reshape(conv_out_layer_86, [-1, 384]);  conv_out_layer_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_54 = key_layer_10.transpose(-1, -2);  key_layer_10 = None
    attention_scores_30 = torch.matmul(query_layer_10, transpose_54);  query_layer_10 = transpose_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_31 = attention_scores_30 / 8.0;  attention_scores_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:397, code: attention_scores = attention_scores + attention_mask
    attention_scores_32 = attention_scores_31 + extended_attention_mask_3;  attention_scores_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_20 = torch.nn.functional.softmax(attention_scores_32, dim = -1);  attention_scores_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    attention_probs_21 = self.L__mod___convbert_encoder_layer_10_attention_self_dropout(attention_probs_20);  attention_probs_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_40 = torch.matmul(attention_probs_21, value_layer_10);  attention_probs_21 = value_layer_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_43 = context_layer_40.permute(0, 2, 1, 3);  context_layer_40 = None
    context_layer_41 = permute_43.contiguous();  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    conv_out_10 = torch.reshape(conv_out_layer_87, [1, -1, 6, 64]);  conv_out_layer_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    context_layer_42 = torch.cat([context_layer_41, conv_out_10], 2);  context_layer_41 = conv_out_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    context_layer_43 = context_layer_42.view(1, 512, 768);  context_layer_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    hidden_states_91 = self.L__mod___convbert_encoder_layer_10_attention_output_dense(context_layer_43);  context_layer_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    hidden_states_92 = self.L__mod___convbert_encoder_layer_10_attention_output_dropout(hidden_states_91);  hidden_states_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_33 = hidden_states_92 + hidden_states_90;  hidden_states_92 = hidden_states_90 = None
    attention_output_20 = self.L__mod___convbert_encoder_layer_10_attention_output_LayerNorm(add_33);  add_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    hidden_states_94 = self.L__mod___convbert_encoder_layer_10_intermediate_dense(attention_output_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_10 = torch._C._nn.gelu(hidden_states_94);  hidden_states_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    hidden_states_96 = self.L__mod___convbert_encoder_layer_10_output_dense(intermediate_output_10);  intermediate_output_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    hidden_states_97 = self.L__mod___convbert_encoder_layer_10_output_dropout(hidden_states_96);  hidden_states_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_34 = hidden_states_97 + attention_output_20;  hidden_states_97 = attention_output_20 = None
    hidden_states_99 = self.L__mod___convbert_encoder_layer_10_output_LayerNorm(add_34);  add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    mixed_query_layer_11 = self.L__mod___convbert_encoder_layer_11_attention_self_query(hidden_states_99)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    mixed_key_layer_11 = self.L__mod___convbert_encoder_layer_11_attention_self_key(hidden_states_99)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    mixed_value_layer_11 = self.L__mod___convbert_encoder_layer_11_attention_self_value(hidden_states_99)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    transpose_55 = hidden_states_99.transpose(1, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    x_66 = self.L__mod___convbert_encoder_layer_11_attention_self_key_conv_attn_layer_depthwise(transpose_55);  transpose_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    x_67 = self.L__mod___convbert_encoder_layer_11_attention_self_key_conv_attn_layer_pointwise(x_66);  x_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    l__mod___convbert_encoder_layer_11_attention_self_key_conv_attn_layer_bias_1 = self.L__mod___convbert_encoder_layer_11_attention_self_key_conv_attn_layer_bias
    x_67 += l__mod___convbert_encoder_layer_11_attention_self_key_conv_attn_layer_bias_1;  mixed_key_conv_attn_layer_22 = x_67;  x_67 = l__mod___convbert_encoder_layer_11_attention_self_key_conv_attn_layer_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:364, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
    mixed_key_conv_attn_layer_23 = mixed_key_conv_attn_layer_22.transpose(1, 2);  mixed_key_conv_attn_layer_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    x_69 = mixed_query_layer_11.view(1, 512, 6, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    query_layer_11 = x_69.permute(0, 2, 1, 3);  x_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    x_70 = mixed_key_layer_11.view(1, 512, 6, 64);  mixed_key_layer_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    key_layer_11 = x_70.permute(0, 2, 1, 3);  x_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    x_71 = mixed_value_layer_11.view(1, 512, 6, 64);  mixed_value_layer_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    value_layer_11 = x_71.permute(0, 2, 1, 3);  x_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    conv_attn_layer_11 = torch.multiply(mixed_key_conv_attn_layer_23, mixed_query_layer_11);  mixed_key_conv_attn_layer_23 = mixed_query_layer_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    conv_kernel_layer_33 = self.L__mod___convbert_encoder_layer_11_attention_self_conv_kernel_layer(conv_attn_layer_11);  conv_attn_layer_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    conv_kernel_layer_34 = torch.reshape(conv_kernel_layer_33, [-1, 9, 1]);  conv_kernel_layer_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    conv_kernel_layer_35 = torch.softmax(conv_kernel_layer_34, dim = 1);  conv_kernel_layer_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    conv_out_layer_88 = self.L__mod___convbert_encoder_layer_11_attention_self_conv_out_layer(hidden_states_99)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    conv_out_layer_89 = torch.reshape(conv_out_layer_88, [1, -1, 384]);  conv_out_layer_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    transpose_57 = conv_out_layer_89.transpose(1, 2);  conv_out_layer_89 = None
    contiguous_22 = transpose_57.contiguous();  transpose_57 = None
    conv_out_layer_90 = contiguous_22.unsqueeze(-1);  contiguous_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    conv_out_layer_91 = torch.nn.functional.unfold(conv_out_layer_90, kernel_size = [9, 1], dilation = 1, padding = [4, 0], stride = 1);  conv_out_layer_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    transpose_58 = conv_out_layer_91.transpose(1, 2);  conv_out_layer_91 = None
    conv_out_layer_92 = transpose_58.reshape(1, -1, 384, 9);  transpose_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    conv_out_layer_93 = torch.reshape(conv_out_layer_92, [-1, 64, 9]);  conv_out_layer_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    conv_out_layer_94 = torch.matmul(conv_out_layer_93, conv_kernel_layer_35);  conv_out_layer_93 = conv_kernel_layer_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    conv_out_layer_95 = torch.reshape(conv_out_layer_94, [-1, 384]);  conv_out_layer_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    transpose_59 = key_layer_11.transpose(-1, -2);  key_layer_11 = None
    attention_scores_33 = torch.matmul(query_layer_11, transpose_59);  query_layer_11 = transpose_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores_34 = attention_scores_33 / 8.0;  attention_scores_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:397, code: attention_scores = attention_scores + attention_mask
    attention_scores_35 = attention_scores_34 + extended_attention_mask_3;  attention_scores_34 = extended_attention_mask_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_22 = torch.nn.functional.softmax(attention_scores_35, dim = -1);  attention_scores_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    attention_probs_23 = self.L__mod___convbert_encoder_layer_11_attention_self_dropout(attention_probs_22);  attention_probs_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    context_layer_44 = torch.matmul(attention_probs_23, value_layer_11);  attention_probs_23 = value_layer_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_47 = context_layer_44.permute(0, 2, 1, 3);  context_layer_44 = None
    context_layer_45 = permute_47.contiguous();  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    conv_out_11 = torch.reshape(conv_out_layer_95, [1, -1, 6, 64]);  conv_out_layer_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    context_layer_46 = torch.cat([context_layer_45, conv_out_11], 2);  context_layer_45 = conv_out_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    context_layer_47 = context_layer_46.view(1, 512, 768);  context_layer_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    hidden_states_100 = self.L__mod___convbert_encoder_layer_11_attention_output_dense(context_layer_47);  context_layer_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    hidden_states_101 = self.L__mod___convbert_encoder_layer_11_attention_output_dropout(hidden_states_100);  hidden_states_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_36 = hidden_states_101 + hidden_states_99;  hidden_states_101 = hidden_states_99 = None
    attention_output_22 = self.L__mod___convbert_encoder_layer_11_attention_output_LayerNorm(add_36);  add_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    hidden_states_103 = self.L__mod___convbert_encoder_layer_11_intermediate_dense(attention_output_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    intermediate_output_11 = torch._C._nn.gelu(hidden_states_103);  hidden_states_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    hidden_states_105 = self.L__mod___convbert_encoder_layer_11_output_dense(intermediate_output_11);  intermediate_output_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    hidden_states_106 = self.L__mod___convbert_encoder_layer_11_output_dropout(hidden_states_105);  hidden_states_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_37 = hidden_states_106 + attention_output_22;  hidden_states_106 = attention_output_22 = None
    generator_sequence_output = self.L__mod___convbert_encoder_layer_11_output_LayerNorm(add_37);  add_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:873, code: hidden_states = self.dense(generator_hidden_states)
    hidden_states_109 = self.L__mod___generator_predictions_dense(generator_sequence_output);  generator_sequence_output = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    hidden_states_110 = torch._C._nn.gelu(hidden_states_109);  hidden_states_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:875, code: hidden_states = self.LayerNorm(hidden_states)
    prediction_scores = self.L__mod___generator_predictions_LayerNorm(hidden_states_110);  hidden_states_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:941, code: prediction_scores = self.generator_lm_head(prediction_scores)
    prediction_scores_1 = self.L__mod___generator_lm_head(prediction_scores);  prediction_scores = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:947, code: loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    view_48 = prediction_scores_1.view(-1, 30522)
    view_49 = l_cloned_inputs_labels_.view(-1);  l_cloned_inputs_labels_ = None
    loss = torch.nn.functional.cross_entropy(view_48, view_49, None, None, -100, None, 'mean', 0.0);  view_48 = view_49 = None
    return (loss, prediction_scores_1)
    