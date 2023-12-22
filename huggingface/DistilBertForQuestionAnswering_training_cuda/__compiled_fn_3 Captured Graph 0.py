from __future__ import annotations



def forward(self, L_cloned_inputs_input_ids_ : torch.Tensor, L_cloned_inputs_start_positions_ : torch.Tensor, L_cloned_inputs_end_positions_ : torch.Tensor):
    l_cloned_inputs_input_ids_ = L_cloned_inputs_input_ids_
    l_cloned_inputs_start_positions_ = L_cloned_inputs_start_positions_
    l_cloned_inputs_end_positions_ = L_cloned_inputs_end_positions_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:602, code: attention_mask = torch.ones(input_shape, device=device)  # (bs, seq_length)
    attention_mask = torch.ones((1, 128), device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:120, code: input_embeds = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)
    input_embeds = self.L__mod___distilbert_embeddings_word_embeddings(l_cloned_inputs_input_ids_);  l_cloned_inputs_input_ids_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:128, code: position_ids = self.position_ids[:, :seq_length]
    l__mod___distilbert_embeddings_position_ids = self.L__mod___distilbert_embeddings_position_ids
    position_ids = l__mod___distilbert_embeddings_position_ids[(slice(None, None, None), slice(None, 128, None))];  l__mod___distilbert_embeddings_position_ids = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:133, code: position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)
    position_embeddings = self.L__mod___distilbert_embeddings_position_embeddings(position_ids);  position_ids = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:135, code: embeddings = input_embeds + position_embeddings  # (bs, max_seq_length, dim)
    embeddings = input_embeds + position_embeddings;  input_embeds = position_embeddings = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:136, code: embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
    embeddings_1 = self.L__mod___distilbert_embeddings_LayerNorm(embeddings);  embeddings = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:137, code: embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
    hidden_state = self.L__mod___distilbert_embeddings_dropout(embeddings_1);  embeddings_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    l__mod___distilbert_transformer_layer_0_attention_q_lin = self.L__mod___distilbert_transformer_layer_0_attention_q_lin(hidden_state)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view = l__mod___distilbert_transformer_layer_0_attention_q_lin.view(1, -1, 12, 64);  l__mod___distilbert_transformer_layer_0_attention_q_lin = None
    q = view.transpose(1, 2);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    l__mod___distilbert_transformer_layer_0_attention_k_lin = self.L__mod___distilbert_transformer_layer_0_attention_k_lin(hidden_state)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_1 = l__mod___distilbert_transformer_layer_0_attention_k_lin.view(1, -1, 12, 64);  l__mod___distilbert_transformer_layer_0_attention_k_lin = None
    k = view_1.transpose(1, 2);  view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    l__mod___distilbert_transformer_layer_0_attention_v_lin = self.L__mod___distilbert_transformer_layer_0_attention_v_lin(hidden_state)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_2 = l__mod___distilbert_transformer_layer_0_attention_v_lin.view(1, -1, 12, 64);  l__mod___distilbert_transformer_layer_0_attention_v_lin = None
    v = view_2.transpose(1, 2);  view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:219, code: q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
    q_1 = q / 8.0;  q = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:220, code: scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    transpose_3 = k.transpose(2, 3);  k = None
    scores = torch.matmul(q_1, transpose_3);  q_1 = transpose_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:221, code: mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
    eq = attention_mask == 0
    view_3 = eq.view((1, 1, 1, 128));  eq = None
    mask = view_3.expand_as(scores);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:223, code: mask, torch.tensor(torch.finfo(scores.dtype).min)
    tensor = torch.tensor(-3.4028234663852886e+38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:222, code: scores = scores.masked_fill(
    scores_1 = scores.masked_fill(mask, tensor);  scores = mask = tensor = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:226, code: weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
    weights = torch.nn.functional.softmax(scores_1, dim = -1);  scores_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:227, code: weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)
    weights_1 = self.L__mod___distilbert_transformer_layer_0_attention_dropout(weights);  weights = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:233, code: context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    context = torch.matmul(weights_1, v);  weights_1 = v = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:213, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
    transpose_4 = context.transpose(1, 2);  context = None
    contiguous = transpose_4.contiguous();  transpose_4 = None
    context_1 = contiguous.view(1, -1, 768);  contiguous = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    sa_output = self.L__mod___distilbert_transformer_layer_0_attention_out_lin(context_1);  context_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    add_1 = sa_output + hidden_state;  sa_output = hidden_state = None
    sa_output_1 = self.L__mod___distilbert_transformer_layer_0_sa_layer_norm(add_1);  add_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    x = self.L__mod___distilbert_transformer_layer_0_ffn_lin1(sa_output_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    x_1 = torch._C._nn.gelu(x);  x = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    x_2 = self.L__mod___distilbert_transformer_layer_0_ffn_lin2(x_1);  x_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:260, code: x = self.dropout(x)
    ffn_output = self.L__mod___distilbert_transformer_layer_0_ffn_dropout(x_2);  x_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:314, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
    add_2 = ffn_output + sa_output_1;  ffn_output = sa_output_1 = None
    hidden_state_1 = self.L__mod___distilbert_transformer_layer_0_output_layer_norm(add_2);  add_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    l__mod___distilbert_transformer_layer_1_attention_q_lin = self.L__mod___distilbert_transformer_layer_1_attention_q_lin(hidden_state_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_5 = l__mod___distilbert_transformer_layer_1_attention_q_lin.view(1, -1, 12, 64);  l__mod___distilbert_transformer_layer_1_attention_q_lin = None
    q_2 = view_5.transpose(1, 2);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    l__mod___distilbert_transformer_layer_1_attention_k_lin = self.L__mod___distilbert_transformer_layer_1_attention_k_lin(hidden_state_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_6 = l__mod___distilbert_transformer_layer_1_attention_k_lin.view(1, -1, 12, 64);  l__mod___distilbert_transformer_layer_1_attention_k_lin = None
    k_1 = view_6.transpose(1, 2);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    l__mod___distilbert_transformer_layer_1_attention_v_lin = self.L__mod___distilbert_transformer_layer_1_attention_v_lin(hidden_state_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_7 = l__mod___distilbert_transformer_layer_1_attention_v_lin.view(1, -1, 12, 64);  l__mod___distilbert_transformer_layer_1_attention_v_lin = None
    v_1 = view_7.transpose(1, 2);  view_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:219, code: q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
    q_3 = q_2 / 8.0;  q_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:220, code: scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    transpose_8 = k_1.transpose(2, 3);  k_1 = None
    scores_2 = torch.matmul(q_3, transpose_8);  q_3 = transpose_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:221, code: mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
    eq_1 = attention_mask == 0
    view_8 = eq_1.view((1, 1, 1, 128));  eq_1 = None
    mask_1 = view_8.expand_as(scores_2);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:223, code: mask, torch.tensor(torch.finfo(scores.dtype).min)
    tensor_1 = torch.tensor(-3.4028234663852886e+38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:222, code: scores = scores.masked_fill(
    scores_3 = scores_2.masked_fill(mask_1, tensor_1);  scores_2 = mask_1 = tensor_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:226, code: weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
    weights_2 = torch.nn.functional.softmax(scores_3, dim = -1);  scores_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:227, code: weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)
    weights_3 = self.L__mod___distilbert_transformer_layer_1_attention_dropout(weights_2);  weights_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:233, code: context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    context_3 = torch.matmul(weights_3, v_1);  weights_3 = v_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:213, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
    transpose_9 = context_3.transpose(1, 2);  context_3 = None
    contiguous_1 = transpose_9.contiguous();  transpose_9 = None
    context_4 = contiguous_1.view(1, -1, 768);  contiguous_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    sa_output_2 = self.L__mod___distilbert_transformer_layer_1_attention_out_lin(context_4);  context_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    add_3 = sa_output_2 + hidden_state_1;  sa_output_2 = hidden_state_1 = None
    sa_output_3 = self.L__mod___distilbert_transformer_layer_1_sa_layer_norm(add_3);  add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    x_4 = self.L__mod___distilbert_transformer_layer_1_ffn_lin1(sa_output_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    x_5 = torch._C._nn.gelu(x_4);  x_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    x_6 = self.L__mod___distilbert_transformer_layer_1_ffn_lin2(x_5);  x_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:260, code: x = self.dropout(x)
    ffn_output_2 = self.L__mod___distilbert_transformer_layer_1_ffn_dropout(x_6);  x_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:314, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
    add_4 = ffn_output_2 + sa_output_3;  ffn_output_2 = sa_output_3 = None
    hidden_state_2 = self.L__mod___distilbert_transformer_layer_1_output_layer_norm(add_4);  add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    l__mod___distilbert_transformer_layer_2_attention_q_lin = self.L__mod___distilbert_transformer_layer_2_attention_q_lin(hidden_state_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_10 = l__mod___distilbert_transformer_layer_2_attention_q_lin.view(1, -1, 12, 64);  l__mod___distilbert_transformer_layer_2_attention_q_lin = None
    q_4 = view_10.transpose(1, 2);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    l__mod___distilbert_transformer_layer_2_attention_k_lin = self.L__mod___distilbert_transformer_layer_2_attention_k_lin(hidden_state_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_11 = l__mod___distilbert_transformer_layer_2_attention_k_lin.view(1, -1, 12, 64);  l__mod___distilbert_transformer_layer_2_attention_k_lin = None
    k_2 = view_11.transpose(1, 2);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    l__mod___distilbert_transformer_layer_2_attention_v_lin = self.L__mod___distilbert_transformer_layer_2_attention_v_lin(hidden_state_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_12 = l__mod___distilbert_transformer_layer_2_attention_v_lin.view(1, -1, 12, 64);  l__mod___distilbert_transformer_layer_2_attention_v_lin = None
    v_2 = view_12.transpose(1, 2);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:219, code: q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
    q_5 = q_4 / 8.0;  q_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:220, code: scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    transpose_13 = k_2.transpose(2, 3);  k_2 = None
    scores_4 = torch.matmul(q_5, transpose_13);  q_5 = transpose_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:221, code: mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
    eq_2 = attention_mask == 0
    view_13 = eq_2.view((1, 1, 1, 128));  eq_2 = None
    mask_2 = view_13.expand_as(scores_4);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:223, code: mask, torch.tensor(torch.finfo(scores.dtype).min)
    tensor_2 = torch.tensor(-3.4028234663852886e+38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:222, code: scores = scores.masked_fill(
    scores_5 = scores_4.masked_fill(mask_2, tensor_2);  scores_4 = mask_2 = tensor_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:226, code: weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
    weights_4 = torch.nn.functional.softmax(scores_5, dim = -1);  scores_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:227, code: weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)
    weights_5 = self.L__mod___distilbert_transformer_layer_2_attention_dropout(weights_4);  weights_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:233, code: context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    context_6 = torch.matmul(weights_5, v_2);  weights_5 = v_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:213, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
    transpose_14 = context_6.transpose(1, 2);  context_6 = None
    contiguous_2 = transpose_14.contiguous();  transpose_14 = None
    context_7 = contiguous_2.view(1, -1, 768);  contiguous_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    sa_output_4 = self.L__mod___distilbert_transformer_layer_2_attention_out_lin(context_7);  context_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    add_5 = sa_output_4 + hidden_state_2;  sa_output_4 = hidden_state_2 = None
    sa_output_5 = self.L__mod___distilbert_transformer_layer_2_sa_layer_norm(add_5);  add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    x_8 = self.L__mod___distilbert_transformer_layer_2_ffn_lin1(sa_output_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    x_9 = torch._C._nn.gelu(x_8);  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    x_10 = self.L__mod___distilbert_transformer_layer_2_ffn_lin2(x_9);  x_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:260, code: x = self.dropout(x)
    ffn_output_4 = self.L__mod___distilbert_transformer_layer_2_ffn_dropout(x_10);  x_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:314, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
    add_6 = ffn_output_4 + sa_output_5;  ffn_output_4 = sa_output_5 = None
    hidden_state_3 = self.L__mod___distilbert_transformer_layer_2_output_layer_norm(add_6);  add_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    l__mod___distilbert_transformer_layer_3_attention_q_lin = self.L__mod___distilbert_transformer_layer_3_attention_q_lin(hidden_state_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_15 = l__mod___distilbert_transformer_layer_3_attention_q_lin.view(1, -1, 12, 64);  l__mod___distilbert_transformer_layer_3_attention_q_lin = None
    q_6 = view_15.transpose(1, 2);  view_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    l__mod___distilbert_transformer_layer_3_attention_k_lin = self.L__mod___distilbert_transformer_layer_3_attention_k_lin(hidden_state_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_16 = l__mod___distilbert_transformer_layer_3_attention_k_lin.view(1, -1, 12, 64);  l__mod___distilbert_transformer_layer_3_attention_k_lin = None
    k_3 = view_16.transpose(1, 2);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    l__mod___distilbert_transformer_layer_3_attention_v_lin = self.L__mod___distilbert_transformer_layer_3_attention_v_lin(hidden_state_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_17 = l__mod___distilbert_transformer_layer_3_attention_v_lin.view(1, -1, 12, 64);  l__mod___distilbert_transformer_layer_3_attention_v_lin = None
    v_3 = view_17.transpose(1, 2);  view_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:219, code: q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
    q_7 = q_6 / 8.0;  q_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:220, code: scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    transpose_18 = k_3.transpose(2, 3);  k_3 = None
    scores_6 = torch.matmul(q_7, transpose_18);  q_7 = transpose_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:221, code: mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
    eq_3 = attention_mask == 0
    view_18 = eq_3.view((1, 1, 1, 128));  eq_3 = None
    mask_3 = view_18.expand_as(scores_6);  view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:223, code: mask, torch.tensor(torch.finfo(scores.dtype).min)
    tensor_3 = torch.tensor(-3.4028234663852886e+38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:222, code: scores = scores.masked_fill(
    scores_7 = scores_6.masked_fill(mask_3, tensor_3);  scores_6 = mask_3 = tensor_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:226, code: weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
    weights_6 = torch.nn.functional.softmax(scores_7, dim = -1);  scores_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:227, code: weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)
    weights_7 = self.L__mod___distilbert_transformer_layer_3_attention_dropout(weights_6);  weights_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:233, code: context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    context_9 = torch.matmul(weights_7, v_3);  weights_7 = v_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:213, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
    transpose_19 = context_9.transpose(1, 2);  context_9 = None
    contiguous_3 = transpose_19.contiguous();  transpose_19 = None
    context_10 = contiguous_3.view(1, -1, 768);  contiguous_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    sa_output_6 = self.L__mod___distilbert_transformer_layer_3_attention_out_lin(context_10);  context_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    add_7 = sa_output_6 + hidden_state_3;  sa_output_6 = hidden_state_3 = None
    sa_output_7 = self.L__mod___distilbert_transformer_layer_3_sa_layer_norm(add_7);  add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    x_12 = self.L__mod___distilbert_transformer_layer_3_ffn_lin1(sa_output_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    x_13 = torch._C._nn.gelu(x_12);  x_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    x_14 = self.L__mod___distilbert_transformer_layer_3_ffn_lin2(x_13);  x_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:260, code: x = self.dropout(x)
    ffn_output_6 = self.L__mod___distilbert_transformer_layer_3_ffn_dropout(x_14);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:314, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
    add_8 = ffn_output_6 + sa_output_7;  ffn_output_6 = sa_output_7 = None
    hidden_state_4 = self.L__mod___distilbert_transformer_layer_3_output_layer_norm(add_8);  add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    l__mod___distilbert_transformer_layer_4_attention_q_lin = self.L__mod___distilbert_transformer_layer_4_attention_q_lin(hidden_state_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_20 = l__mod___distilbert_transformer_layer_4_attention_q_lin.view(1, -1, 12, 64);  l__mod___distilbert_transformer_layer_4_attention_q_lin = None
    q_8 = view_20.transpose(1, 2);  view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    l__mod___distilbert_transformer_layer_4_attention_k_lin = self.L__mod___distilbert_transformer_layer_4_attention_k_lin(hidden_state_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_21 = l__mod___distilbert_transformer_layer_4_attention_k_lin.view(1, -1, 12, 64);  l__mod___distilbert_transformer_layer_4_attention_k_lin = None
    k_4 = view_21.transpose(1, 2);  view_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    l__mod___distilbert_transformer_layer_4_attention_v_lin = self.L__mod___distilbert_transformer_layer_4_attention_v_lin(hidden_state_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_22 = l__mod___distilbert_transformer_layer_4_attention_v_lin.view(1, -1, 12, 64);  l__mod___distilbert_transformer_layer_4_attention_v_lin = None
    v_4 = view_22.transpose(1, 2);  view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:219, code: q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
    q_9 = q_8 / 8.0;  q_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:220, code: scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    transpose_23 = k_4.transpose(2, 3);  k_4 = None
    scores_8 = torch.matmul(q_9, transpose_23);  q_9 = transpose_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:221, code: mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
    eq_4 = attention_mask == 0
    view_23 = eq_4.view((1, 1, 1, 128));  eq_4 = None
    mask_4 = view_23.expand_as(scores_8);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:223, code: mask, torch.tensor(torch.finfo(scores.dtype).min)
    tensor_4 = torch.tensor(-3.4028234663852886e+38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:222, code: scores = scores.masked_fill(
    scores_9 = scores_8.masked_fill(mask_4, tensor_4);  scores_8 = mask_4 = tensor_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:226, code: weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
    weights_8 = torch.nn.functional.softmax(scores_9, dim = -1);  scores_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:227, code: weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)
    weights_9 = self.L__mod___distilbert_transformer_layer_4_attention_dropout(weights_8);  weights_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:233, code: context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    context_12 = torch.matmul(weights_9, v_4);  weights_9 = v_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:213, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
    transpose_24 = context_12.transpose(1, 2);  context_12 = None
    contiguous_4 = transpose_24.contiguous();  transpose_24 = None
    context_13 = contiguous_4.view(1, -1, 768);  contiguous_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    sa_output_8 = self.L__mod___distilbert_transformer_layer_4_attention_out_lin(context_13);  context_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    add_9 = sa_output_8 + hidden_state_4;  sa_output_8 = hidden_state_4 = None
    sa_output_9 = self.L__mod___distilbert_transformer_layer_4_sa_layer_norm(add_9);  add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    x_16 = self.L__mod___distilbert_transformer_layer_4_ffn_lin1(sa_output_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    x_17 = torch._C._nn.gelu(x_16);  x_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    x_18 = self.L__mod___distilbert_transformer_layer_4_ffn_lin2(x_17);  x_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:260, code: x = self.dropout(x)
    ffn_output_8 = self.L__mod___distilbert_transformer_layer_4_ffn_dropout(x_18);  x_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:314, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
    add_10 = ffn_output_8 + sa_output_9;  ffn_output_8 = sa_output_9 = None
    hidden_state_5 = self.L__mod___distilbert_transformer_layer_4_output_layer_norm(add_10);  add_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    l__mod___distilbert_transformer_layer_5_attention_q_lin = self.L__mod___distilbert_transformer_layer_5_attention_q_lin(hidden_state_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_25 = l__mod___distilbert_transformer_layer_5_attention_q_lin.view(1, -1, 12, 64);  l__mod___distilbert_transformer_layer_5_attention_q_lin = None
    q_10 = view_25.transpose(1, 2);  view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    l__mod___distilbert_transformer_layer_5_attention_k_lin = self.L__mod___distilbert_transformer_layer_5_attention_k_lin(hidden_state_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_26 = l__mod___distilbert_transformer_layer_5_attention_k_lin.view(1, -1, 12, 64);  l__mod___distilbert_transformer_layer_5_attention_k_lin = None
    k_5 = view_26.transpose(1, 2);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    l__mod___distilbert_transformer_layer_5_attention_v_lin = self.L__mod___distilbert_transformer_layer_5_attention_v_lin(hidden_state_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_27 = l__mod___distilbert_transformer_layer_5_attention_v_lin.view(1, -1, 12, 64);  l__mod___distilbert_transformer_layer_5_attention_v_lin = None
    v_5 = view_27.transpose(1, 2);  view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:219, code: q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
    q_11 = q_10 / 8.0;  q_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:220, code: scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    transpose_28 = k_5.transpose(2, 3);  k_5 = None
    scores_10 = torch.matmul(q_11, transpose_28);  q_11 = transpose_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:221, code: mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
    eq_5 = attention_mask == 0;  attention_mask = None
    view_28 = eq_5.view((1, 1, 1, 128));  eq_5 = None
    mask_5 = view_28.expand_as(scores_10);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:223, code: mask, torch.tensor(torch.finfo(scores.dtype).min)
    tensor_5 = torch.tensor(-3.4028234663852886e+38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:222, code: scores = scores.masked_fill(
    scores_11 = scores_10.masked_fill(mask_5, tensor_5);  scores_10 = mask_5 = tensor_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:226, code: weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
    weights_10 = torch.nn.functional.softmax(scores_11, dim = -1);  scores_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:227, code: weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)
    weights_11 = self.L__mod___distilbert_transformer_layer_5_attention_dropout(weights_10);  weights_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:233, code: context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    context_15 = torch.matmul(weights_11, v_5);  weights_11 = v_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:213, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
    transpose_29 = context_15.transpose(1, 2);  context_15 = None
    contiguous_5 = transpose_29.contiguous();  transpose_29 = None
    context_16 = contiguous_5.view(1, -1, 768);  contiguous_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    sa_output_10 = self.L__mod___distilbert_transformer_layer_5_attention_out_lin(context_16);  context_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    add_11 = sa_output_10 + hidden_state_5;  sa_output_10 = hidden_state_5 = None
    sa_output_11 = self.L__mod___distilbert_transformer_layer_5_sa_layer_norm(add_11);  add_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    x_20 = self.L__mod___distilbert_transformer_layer_5_ffn_lin1(sa_output_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    x_21 = torch._C._nn.gelu(x_20);  x_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    x_22 = self.L__mod___distilbert_transformer_layer_5_ffn_lin2(x_21);  x_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:260, code: x = self.dropout(x)
    ffn_output_10 = self.L__mod___distilbert_transformer_layer_5_ffn_dropout(x_22);  x_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:314, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
    add_12 = ffn_output_10 + sa_output_11;  ffn_output_10 = sa_output_11 = None
    hidden_states = self.L__mod___distilbert_transformer_layer_5_output_layer_norm(add_12);  add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:922, code: hidden_states = self.dropout(hidden_states)  # (bs, max_query_len, dim)
    hidden_states_1 = self.L__mod___dropout(hidden_states);  hidden_states = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:923, code: logits = self.qa_outputs(hidden_states)  # (bs, max_query_len, 2)
    logits = self.L__mod___qa_outputs(hidden_states_1);  hidden_states_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:924, code: start_logits, end_logits = logits.split(1, dim=-1)
    split = logits.split(1, dim = -1);  logits = None
    start_logits = split[0]
    end_logits = split[1];  split = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:925, code: start_logits = start_logits.squeeze(-1).contiguous()  # (bs, max_query_len)
    squeeze = start_logits.squeeze(-1);  start_logits = None
    start_logits_1 = squeeze.contiguous();  squeeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:926, code: end_logits = end_logits.squeeze(-1).contiguous()  # (bs, max_query_len)
    squeeze_1 = end_logits.squeeze(-1);  end_logits = None
    end_logits_1 = squeeze_1.contiguous();  squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:937, code: start_positions = start_positions.clamp(0, ignored_index)
    start_positions = l_cloned_inputs_start_positions_.clamp(0, 128);  l_cloned_inputs_start_positions_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:938, code: end_positions = end_positions.clamp(0, ignored_index)
    end_positions = l_cloned_inputs_end_positions_.clamp(0, 128);  l_cloned_inputs_end_positions_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:941, code: start_loss = loss_fct(start_logits, start_positions)
    start_loss = torch.nn.functional.cross_entropy(start_logits_1, start_positions, None, None, 128, None, 'mean', 0.0);  start_positions = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:942, code: end_loss = loss_fct(end_logits, end_positions)
    end_loss = torch.nn.functional.cross_entropy(end_logits_1, end_positions, None, None, 128, None, 'mean', 0.0);  end_positions = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:943, code: total_loss = (start_loss + end_loss) / 2
    add_13 = start_loss + end_loss;  start_loss = end_loss = None
    loss = add_13 / 2;  add_13 = None
    return (loss, start_logits_1, end_logits_1)
    