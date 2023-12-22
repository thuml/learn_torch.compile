from __future__ import annotations



def forward(self, L_inputs_0_ : torch.Tensor, L_inputs_1_ : torch.Tensor):
    l_inputs_0_ = L_inputs_0_
    l_inputs_1_ = L_inputs_1_
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/bert.py:40, code: mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
    gt = l_inputs_0_ > 0
    unsqueeze = gt.unsqueeze(1);  gt = None
    repeat = unsqueeze.repeat(1, 128, 1);  unsqueeze = None
    mask = repeat.unsqueeze(1);  repeat = None
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/module.py:1520, code: return forward_call(*args, **kwargs)
    l__mod___embedding_token = self.L__mod___embedding_token
    forward = l__mod___embedding_token.forward(l_inputs_0_);  l__mod___embedding_token = l_inputs_0_ = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/position.py:26, code: return self.pe[:, :x.size(1)]
    l__mod___embedding_position_pe = self.L__mod___embedding_position_pe
    getitem = l__mod___embedding_position_pe[(slice(None, None, None), slice(None, 128, None))];  l__mod___embedding_position_pe = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/bert.py:32, code: x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
    add = forward + getitem;  forward = getitem = None
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/module.py:1520, code: return forward_call(*args, **kwargs)
    l__mod___embedding_segment = self.L__mod___embedding_segment
    forward_1 = l__mod___embedding_segment.forward(l_inputs_1_);  l__mod___embedding_segment = l_inputs_1_ = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/bert.py:32, code: x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
    x = add + forward_1;  add = forward_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/bert.py:33, code: return self.dropout(x)
    x_1 = self.L__mod___embedding_dropout(x);  x = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean = x_1.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    std = x_1.std(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    l__mod___transformer_blocks_0_input_sublayer_norm_a_2 = self.L__mod___transformer_blocks_0_input_sublayer_norm_a_2
    sub = x_1 - mean;  mean = None
    mul = l__mod___transformer_blocks_0_input_sublayer_norm_a_2 * sub;  l__mod___transformer_blocks_0_input_sublayer_norm_a_2 = sub = None
    add_2 = std + 1e-06;  std = None
    truediv = mul / add_2;  mul = add_2 = None
    l__mod___transformer_blocks_0_input_sublayer_norm_b_2 = self.L__mod___transformer_blocks_0_input_sublayer_norm_b_2
    x_4 = truediv + l__mod___transformer_blocks_0_input_sublayer_norm_b_2;  truediv = l__mod___transformer_blocks_0_input_sublayer_norm_b_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    l__mod___transformer_blocks_0_lambda_module_attention_linear_layers_0 = self.L__mod___transformer_blocks_0_lambda_module_attention_linear_layers_0(x_4)
    view = l__mod___transformer_blocks_0_lambda_module_attention_linear_layers_0.view(4, -1, 12, 64);  l__mod___transformer_blocks_0_lambda_module_attention_linear_layers_0 = None
    query = view.transpose(1, 2);  view = None
    l__mod___transformer_blocks_0_lambda_module_attention_linear_layers_1 = self.L__mod___transformer_blocks_0_lambda_module_attention_linear_layers_1(x_4)
    view_1 = l__mod___transformer_blocks_0_lambda_module_attention_linear_layers_1.view(4, -1, 12, 64);  l__mod___transformer_blocks_0_lambda_module_attention_linear_layers_1 = None
    key = view_1.transpose(1, 2);  view_1 = None
    l__mod___transformer_blocks_0_lambda_module_attention_linear_layers_2 = self.L__mod___transformer_blocks_0_lambda_module_attention_linear_layers_2(x_4);  x_4 = None
    view_2 = l__mod___transformer_blocks_0_lambda_module_attention_linear_layers_2.view(4, -1, 12, 64);  l__mod___transformer_blocks_0_lambda_module_attention_linear_layers_2 = None
    value = view_2.transpose(1, 2);  view_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:15, code: scores = torch.matmul(query, key.transpose(-2, -1)) \
    transpose_3 = key.transpose(-2, -1);  key = None
    matmul = torch.matmul(query, transpose_3);  query = transpose_3 = None
    scores = matmul / 8.0;  matmul = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    eq = mask == 0
    scores_1 = scores.masked_fill(eq, -1000000000.0);  scores = eq = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    p_attn = torch.nn.functional.softmax(scores_1, dim = -1);  scores_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:12, code: return self.dropout(x)
    attn = self.L__mod___transformer_blocks_0_lambda_module_attention_dropout_dropout(p_attn);  p_attn = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    x_5 = torch.matmul(attn, value);  attn = value = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    transpose_4 = x_5.transpose(1, 2);  x_5 = None
    contiguous = transpose_4.contiguous();  transpose_4 = None
    x_6 = contiguous.view(4, -1, 768);  contiguous = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    l__mod___transformer_blocks_0_lambda_module_attention_output_linear = self.L__mod___transformer_blocks_0_lambda_module_attention_output_linear(x_6);  x_6 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    l__mod___transformer_blocks_0_input_sublayer_dropout = self.L__mod___transformer_blocks_0_input_sublayer_dropout(l__mod___transformer_blocks_0_lambda_module_attention_output_linear);  l__mod___transformer_blocks_0_lambda_module_attention_output_linear = None
    x_7 = x_1 + l__mod___transformer_blocks_0_input_sublayer_dropout;  x_1 = l__mod___transformer_blocks_0_input_sublayer_dropout = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_2 = x_7.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    std_2 = x_7.std(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    l__mod___transformer_blocks_0_output_sublayer_norm_a_2 = self.L__mod___transformer_blocks_0_output_sublayer_norm_a_2
    sub_1 = x_7 - mean_2;  mean_2 = None
    mul_1 = l__mod___transformer_blocks_0_output_sublayer_norm_a_2 * sub_1;  l__mod___transformer_blocks_0_output_sublayer_norm_a_2 = sub_1 = None
    add_5 = std_2 + 1e-06;  std_2 = None
    truediv_2 = mul_1 / add_5;  mul_1 = add_5 = None
    l__mod___transformer_blocks_0_output_sublayer_norm_b_2 = self.L__mod___transformer_blocks_0_output_sublayer_norm_b_2
    add_6 = truediv_2 + l__mod___transformer_blocks_0_output_sublayer_norm_b_2;  truediv_2 = l__mod___transformer_blocks_0_output_sublayer_norm_b_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    l__mod___transformer_blocks_0_feed_forward_w_1 = self.L__mod___transformer_blocks_0_feed_forward_w_1(add_6);  add_6 = None
    l__mod___transformer_blocks_0_feed_forward_activation = self.L__mod___transformer_blocks_0_feed_forward_activation(l__mod___transformer_blocks_0_feed_forward_w_1);  l__mod___transformer_blocks_0_feed_forward_w_1 = None
    l__mod___transformer_blocks_0_feed_forward_dropout = self.L__mod___transformer_blocks_0_feed_forward_dropout(l__mod___transformer_blocks_0_feed_forward_activation);  l__mod___transformer_blocks_0_feed_forward_activation = None
    l__mod___transformer_blocks_0_feed_forward_w_2 = self.L__mod___transformer_blocks_0_feed_forward_w_2(l__mod___transformer_blocks_0_feed_forward_dropout);  l__mod___transformer_blocks_0_feed_forward_dropout = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    l__mod___transformer_blocks_0_output_sublayer_dropout = self.L__mod___transformer_blocks_0_output_sublayer_dropout(l__mod___transformer_blocks_0_feed_forward_w_2);  l__mod___transformer_blocks_0_feed_forward_w_2 = None
    x_8 = x_7 + l__mod___transformer_blocks_0_output_sublayer_dropout;  x_7 = l__mod___transformer_blocks_0_output_sublayer_dropout = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:49, code: return self.dropout(x)
    x_9 = self.L__mod___transformer_blocks_0_dropout(x_8);  x_8 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_4 = x_9.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    std_4 = x_9.std(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    l__mod___transformer_blocks_1_input_sublayer_norm_a_2 = self.L__mod___transformer_blocks_1_input_sublayer_norm_a_2
    sub_2 = x_9 - mean_4;  mean_4 = None
    mul_2 = l__mod___transformer_blocks_1_input_sublayer_norm_a_2 * sub_2;  l__mod___transformer_blocks_1_input_sublayer_norm_a_2 = sub_2 = None
    add_8 = std_4 + 1e-06;  std_4 = None
    truediv_3 = mul_2 / add_8;  mul_2 = add_8 = None
    l__mod___transformer_blocks_1_input_sublayer_norm_b_2 = self.L__mod___transformer_blocks_1_input_sublayer_norm_b_2
    x_12 = truediv_3 + l__mod___transformer_blocks_1_input_sublayer_norm_b_2;  truediv_3 = l__mod___transformer_blocks_1_input_sublayer_norm_b_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    l__mod___transformer_blocks_1_lambda_module_attention_linear_layers_0 = self.L__mod___transformer_blocks_1_lambda_module_attention_linear_layers_0(x_12)
    view_4 = l__mod___transformer_blocks_1_lambda_module_attention_linear_layers_0.view(4, -1, 12, 64);  l__mod___transformer_blocks_1_lambda_module_attention_linear_layers_0 = None
    query_1 = view_4.transpose(1, 2);  view_4 = None
    l__mod___transformer_blocks_1_lambda_module_attention_linear_layers_1 = self.L__mod___transformer_blocks_1_lambda_module_attention_linear_layers_1(x_12)
    view_5 = l__mod___transformer_blocks_1_lambda_module_attention_linear_layers_1.view(4, -1, 12, 64);  l__mod___transformer_blocks_1_lambda_module_attention_linear_layers_1 = None
    key_1 = view_5.transpose(1, 2);  view_5 = None
    l__mod___transformer_blocks_1_lambda_module_attention_linear_layers_2 = self.L__mod___transformer_blocks_1_lambda_module_attention_linear_layers_2(x_12);  x_12 = None
    view_6 = l__mod___transformer_blocks_1_lambda_module_attention_linear_layers_2.view(4, -1, 12, 64);  l__mod___transformer_blocks_1_lambda_module_attention_linear_layers_2 = None
    value_1 = view_6.transpose(1, 2);  view_6 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:15, code: scores = torch.matmul(query, key.transpose(-2, -1)) \
    transpose_8 = key_1.transpose(-2, -1);  key_1 = None
    matmul_2 = torch.matmul(query_1, transpose_8);  query_1 = transpose_8 = None
    scores_2 = matmul_2 / 8.0;  matmul_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    eq_1 = mask == 0
    scores_3 = scores_2.masked_fill(eq_1, -1000000000.0);  scores_2 = eq_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    p_attn_2 = torch.nn.functional.softmax(scores_3, dim = -1);  scores_3 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:12, code: return self.dropout(x)
    attn_1 = self.L__mod___transformer_blocks_1_lambda_module_attention_dropout_dropout(p_attn_2);  p_attn_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    x_13 = torch.matmul(attn_1, value_1);  attn_1 = value_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    transpose_9 = x_13.transpose(1, 2);  x_13 = None
    contiguous_1 = transpose_9.contiguous();  transpose_9 = None
    x_14 = contiguous_1.view(4, -1, 768);  contiguous_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    l__mod___transformer_blocks_1_lambda_module_attention_output_linear = self.L__mod___transformer_blocks_1_lambda_module_attention_output_linear(x_14);  x_14 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    l__mod___transformer_blocks_1_input_sublayer_dropout = self.L__mod___transformer_blocks_1_input_sublayer_dropout(l__mod___transformer_blocks_1_lambda_module_attention_output_linear);  l__mod___transformer_blocks_1_lambda_module_attention_output_linear = None
    x_15 = x_9 + l__mod___transformer_blocks_1_input_sublayer_dropout;  x_9 = l__mod___transformer_blocks_1_input_sublayer_dropout = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_6 = x_15.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    std_6 = x_15.std(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    l__mod___transformer_blocks_1_output_sublayer_norm_a_2 = self.L__mod___transformer_blocks_1_output_sublayer_norm_a_2
    sub_3 = x_15 - mean_6;  mean_6 = None
    mul_3 = l__mod___transformer_blocks_1_output_sublayer_norm_a_2 * sub_3;  l__mod___transformer_blocks_1_output_sublayer_norm_a_2 = sub_3 = None
    add_11 = std_6 + 1e-06;  std_6 = None
    truediv_5 = mul_3 / add_11;  mul_3 = add_11 = None
    l__mod___transformer_blocks_1_output_sublayer_norm_b_2 = self.L__mod___transformer_blocks_1_output_sublayer_norm_b_2
    add_12 = truediv_5 + l__mod___transformer_blocks_1_output_sublayer_norm_b_2;  truediv_5 = l__mod___transformer_blocks_1_output_sublayer_norm_b_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    l__mod___transformer_blocks_1_feed_forward_w_1 = self.L__mod___transformer_blocks_1_feed_forward_w_1(add_12);  add_12 = None
    l__mod___transformer_blocks_1_feed_forward_activation = self.L__mod___transformer_blocks_1_feed_forward_activation(l__mod___transformer_blocks_1_feed_forward_w_1);  l__mod___transformer_blocks_1_feed_forward_w_1 = None
    l__mod___transformer_blocks_1_feed_forward_dropout = self.L__mod___transformer_blocks_1_feed_forward_dropout(l__mod___transformer_blocks_1_feed_forward_activation);  l__mod___transformer_blocks_1_feed_forward_activation = None
    l__mod___transformer_blocks_1_feed_forward_w_2 = self.L__mod___transformer_blocks_1_feed_forward_w_2(l__mod___transformer_blocks_1_feed_forward_dropout);  l__mod___transformer_blocks_1_feed_forward_dropout = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    l__mod___transformer_blocks_1_output_sublayer_dropout = self.L__mod___transformer_blocks_1_output_sublayer_dropout(l__mod___transformer_blocks_1_feed_forward_w_2);  l__mod___transformer_blocks_1_feed_forward_w_2 = None
    x_16 = x_15 + l__mod___transformer_blocks_1_output_sublayer_dropout;  x_15 = l__mod___transformer_blocks_1_output_sublayer_dropout = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:49, code: return self.dropout(x)
    x_17 = self.L__mod___transformer_blocks_1_dropout(x_16);  x_16 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_8 = x_17.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    std_8 = x_17.std(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    l__mod___transformer_blocks_2_input_sublayer_norm_a_2 = self.L__mod___transformer_blocks_2_input_sublayer_norm_a_2
    sub_4 = x_17 - mean_8;  mean_8 = None
    mul_4 = l__mod___transformer_blocks_2_input_sublayer_norm_a_2 * sub_4;  l__mod___transformer_blocks_2_input_sublayer_norm_a_2 = sub_4 = None
    add_14 = std_8 + 1e-06;  std_8 = None
    truediv_6 = mul_4 / add_14;  mul_4 = add_14 = None
    l__mod___transformer_blocks_2_input_sublayer_norm_b_2 = self.L__mod___transformer_blocks_2_input_sublayer_norm_b_2
    x_20 = truediv_6 + l__mod___transformer_blocks_2_input_sublayer_norm_b_2;  truediv_6 = l__mod___transformer_blocks_2_input_sublayer_norm_b_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    l__mod___transformer_blocks_2_lambda_module_attention_linear_layers_0 = self.L__mod___transformer_blocks_2_lambda_module_attention_linear_layers_0(x_20)
    view_8 = l__mod___transformer_blocks_2_lambda_module_attention_linear_layers_0.view(4, -1, 12, 64);  l__mod___transformer_blocks_2_lambda_module_attention_linear_layers_0 = None
    query_2 = view_8.transpose(1, 2);  view_8 = None
    l__mod___transformer_blocks_2_lambda_module_attention_linear_layers_1 = self.L__mod___transformer_blocks_2_lambda_module_attention_linear_layers_1(x_20)
    view_9 = l__mod___transformer_blocks_2_lambda_module_attention_linear_layers_1.view(4, -1, 12, 64);  l__mod___transformer_blocks_2_lambda_module_attention_linear_layers_1 = None
    key_2 = view_9.transpose(1, 2);  view_9 = None
    l__mod___transformer_blocks_2_lambda_module_attention_linear_layers_2 = self.L__mod___transformer_blocks_2_lambda_module_attention_linear_layers_2(x_20);  x_20 = None
    view_10 = l__mod___transformer_blocks_2_lambda_module_attention_linear_layers_2.view(4, -1, 12, 64);  l__mod___transformer_blocks_2_lambda_module_attention_linear_layers_2 = None
    value_2 = view_10.transpose(1, 2);  view_10 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:15, code: scores = torch.matmul(query, key.transpose(-2, -1)) \
    transpose_13 = key_2.transpose(-2, -1);  key_2 = None
    matmul_4 = torch.matmul(query_2, transpose_13);  query_2 = transpose_13 = None
    scores_4 = matmul_4 / 8.0;  matmul_4 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    eq_2 = mask == 0
    scores_5 = scores_4.masked_fill(eq_2, -1000000000.0);  scores_4 = eq_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    p_attn_4 = torch.nn.functional.softmax(scores_5, dim = -1);  scores_5 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:12, code: return self.dropout(x)
    attn_2 = self.L__mod___transformer_blocks_2_lambda_module_attention_dropout_dropout(p_attn_4);  p_attn_4 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    x_21 = torch.matmul(attn_2, value_2);  attn_2 = value_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    transpose_14 = x_21.transpose(1, 2);  x_21 = None
    contiguous_2 = transpose_14.contiguous();  transpose_14 = None
    x_22 = contiguous_2.view(4, -1, 768);  contiguous_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    l__mod___transformer_blocks_2_lambda_module_attention_output_linear = self.L__mod___transformer_blocks_2_lambda_module_attention_output_linear(x_22);  x_22 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    l__mod___transformer_blocks_2_input_sublayer_dropout = self.L__mod___transformer_blocks_2_input_sublayer_dropout(l__mod___transformer_blocks_2_lambda_module_attention_output_linear);  l__mod___transformer_blocks_2_lambda_module_attention_output_linear = None
    x_23 = x_17 + l__mod___transformer_blocks_2_input_sublayer_dropout;  x_17 = l__mod___transformer_blocks_2_input_sublayer_dropout = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_10 = x_23.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    std_10 = x_23.std(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    l__mod___transformer_blocks_2_output_sublayer_norm_a_2 = self.L__mod___transformer_blocks_2_output_sublayer_norm_a_2
    sub_5 = x_23 - mean_10;  mean_10 = None
    mul_5 = l__mod___transformer_blocks_2_output_sublayer_norm_a_2 * sub_5;  l__mod___transformer_blocks_2_output_sublayer_norm_a_2 = sub_5 = None
    add_17 = std_10 + 1e-06;  std_10 = None
    truediv_8 = mul_5 / add_17;  mul_5 = add_17 = None
    l__mod___transformer_blocks_2_output_sublayer_norm_b_2 = self.L__mod___transformer_blocks_2_output_sublayer_norm_b_2
    add_18 = truediv_8 + l__mod___transformer_blocks_2_output_sublayer_norm_b_2;  truediv_8 = l__mod___transformer_blocks_2_output_sublayer_norm_b_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    l__mod___transformer_blocks_2_feed_forward_w_1 = self.L__mod___transformer_blocks_2_feed_forward_w_1(add_18);  add_18 = None
    l__mod___transformer_blocks_2_feed_forward_activation = self.L__mod___transformer_blocks_2_feed_forward_activation(l__mod___transformer_blocks_2_feed_forward_w_1);  l__mod___transformer_blocks_2_feed_forward_w_1 = None
    l__mod___transformer_blocks_2_feed_forward_dropout = self.L__mod___transformer_blocks_2_feed_forward_dropout(l__mod___transformer_blocks_2_feed_forward_activation);  l__mod___transformer_blocks_2_feed_forward_activation = None
    l__mod___transformer_blocks_2_feed_forward_w_2 = self.L__mod___transformer_blocks_2_feed_forward_w_2(l__mod___transformer_blocks_2_feed_forward_dropout);  l__mod___transformer_blocks_2_feed_forward_dropout = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    l__mod___transformer_blocks_2_output_sublayer_dropout = self.L__mod___transformer_blocks_2_output_sublayer_dropout(l__mod___transformer_blocks_2_feed_forward_w_2);  l__mod___transformer_blocks_2_feed_forward_w_2 = None
    x_24 = x_23 + l__mod___transformer_blocks_2_output_sublayer_dropout;  x_23 = l__mod___transformer_blocks_2_output_sublayer_dropout = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:49, code: return self.dropout(x)
    x_25 = self.L__mod___transformer_blocks_2_dropout(x_24);  x_24 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_12 = x_25.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    std_12 = x_25.std(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    l__mod___transformer_blocks_3_input_sublayer_norm_a_2 = self.L__mod___transformer_blocks_3_input_sublayer_norm_a_2
    sub_6 = x_25 - mean_12;  mean_12 = None
    mul_6 = l__mod___transformer_blocks_3_input_sublayer_norm_a_2 * sub_6;  l__mod___transformer_blocks_3_input_sublayer_norm_a_2 = sub_6 = None
    add_20 = std_12 + 1e-06;  std_12 = None
    truediv_9 = mul_6 / add_20;  mul_6 = add_20 = None
    l__mod___transformer_blocks_3_input_sublayer_norm_b_2 = self.L__mod___transformer_blocks_3_input_sublayer_norm_b_2
    x_28 = truediv_9 + l__mod___transformer_blocks_3_input_sublayer_norm_b_2;  truediv_9 = l__mod___transformer_blocks_3_input_sublayer_norm_b_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    l__mod___transformer_blocks_3_lambda_module_attention_linear_layers_0 = self.L__mod___transformer_blocks_3_lambda_module_attention_linear_layers_0(x_28)
    view_12 = l__mod___transformer_blocks_3_lambda_module_attention_linear_layers_0.view(4, -1, 12, 64);  l__mod___transformer_blocks_3_lambda_module_attention_linear_layers_0 = None
    query_3 = view_12.transpose(1, 2);  view_12 = None
    l__mod___transformer_blocks_3_lambda_module_attention_linear_layers_1 = self.L__mod___transformer_blocks_3_lambda_module_attention_linear_layers_1(x_28)
    view_13 = l__mod___transformer_blocks_3_lambda_module_attention_linear_layers_1.view(4, -1, 12, 64);  l__mod___transformer_blocks_3_lambda_module_attention_linear_layers_1 = None
    key_3 = view_13.transpose(1, 2);  view_13 = None
    l__mod___transformer_blocks_3_lambda_module_attention_linear_layers_2 = self.L__mod___transformer_blocks_3_lambda_module_attention_linear_layers_2(x_28);  x_28 = None
    view_14 = l__mod___transformer_blocks_3_lambda_module_attention_linear_layers_2.view(4, -1, 12, 64);  l__mod___transformer_blocks_3_lambda_module_attention_linear_layers_2 = None
    value_3 = view_14.transpose(1, 2);  view_14 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:15, code: scores = torch.matmul(query, key.transpose(-2, -1)) \
    transpose_18 = key_3.transpose(-2, -1);  key_3 = None
    matmul_6 = torch.matmul(query_3, transpose_18);  query_3 = transpose_18 = None
    scores_6 = matmul_6 / 8.0;  matmul_6 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    eq_3 = mask == 0
    scores_7 = scores_6.masked_fill(eq_3, -1000000000.0);  scores_6 = eq_3 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    p_attn_6 = torch.nn.functional.softmax(scores_7, dim = -1);  scores_7 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:12, code: return self.dropout(x)
    attn_3 = self.L__mod___transformer_blocks_3_lambda_module_attention_dropout_dropout(p_attn_6);  p_attn_6 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    x_29 = torch.matmul(attn_3, value_3);  attn_3 = value_3 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    transpose_19 = x_29.transpose(1, 2);  x_29 = None
    contiguous_3 = transpose_19.contiguous();  transpose_19 = None
    x_30 = contiguous_3.view(4, -1, 768);  contiguous_3 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    l__mod___transformer_blocks_3_lambda_module_attention_output_linear = self.L__mod___transformer_blocks_3_lambda_module_attention_output_linear(x_30);  x_30 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    l__mod___transformer_blocks_3_input_sublayer_dropout = self.L__mod___transformer_blocks_3_input_sublayer_dropout(l__mod___transformer_blocks_3_lambda_module_attention_output_linear);  l__mod___transformer_blocks_3_lambda_module_attention_output_linear = None
    x_31 = x_25 + l__mod___transformer_blocks_3_input_sublayer_dropout;  x_25 = l__mod___transformer_blocks_3_input_sublayer_dropout = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_14 = x_31.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    std_14 = x_31.std(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    l__mod___transformer_blocks_3_output_sublayer_norm_a_2 = self.L__mod___transformer_blocks_3_output_sublayer_norm_a_2
    sub_7 = x_31 - mean_14;  mean_14 = None
    mul_7 = l__mod___transformer_blocks_3_output_sublayer_norm_a_2 * sub_7;  l__mod___transformer_blocks_3_output_sublayer_norm_a_2 = sub_7 = None
    add_23 = std_14 + 1e-06;  std_14 = None
    truediv_11 = mul_7 / add_23;  mul_7 = add_23 = None
    l__mod___transformer_blocks_3_output_sublayer_norm_b_2 = self.L__mod___transformer_blocks_3_output_sublayer_norm_b_2
    add_24 = truediv_11 + l__mod___transformer_blocks_3_output_sublayer_norm_b_2;  truediv_11 = l__mod___transformer_blocks_3_output_sublayer_norm_b_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    l__mod___transformer_blocks_3_feed_forward_w_1 = self.L__mod___transformer_blocks_3_feed_forward_w_1(add_24);  add_24 = None
    l__mod___transformer_blocks_3_feed_forward_activation = self.L__mod___transformer_blocks_3_feed_forward_activation(l__mod___transformer_blocks_3_feed_forward_w_1);  l__mod___transformer_blocks_3_feed_forward_w_1 = None
    l__mod___transformer_blocks_3_feed_forward_dropout = self.L__mod___transformer_blocks_3_feed_forward_dropout(l__mod___transformer_blocks_3_feed_forward_activation);  l__mod___transformer_blocks_3_feed_forward_activation = None
    l__mod___transformer_blocks_3_feed_forward_w_2 = self.L__mod___transformer_blocks_3_feed_forward_w_2(l__mod___transformer_blocks_3_feed_forward_dropout);  l__mod___transformer_blocks_3_feed_forward_dropout = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    l__mod___transformer_blocks_3_output_sublayer_dropout = self.L__mod___transformer_blocks_3_output_sublayer_dropout(l__mod___transformer_blocks_3_feed_forward_w_2);  l__mod___transformer_blocks_3_feed_forward_w_2 = None
    x_32 = x_31 + l__mod___transformer_blocks_3_output_sublayer_dropout;  x_31 = l__mod___transformer_blocks_3_output_sublayer_dropout = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:49, code: return self.dropout(x)
    x_33 = self.L__mod___transformer_blocks_3_dropout(x_32);  x_32 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_16 = x_33.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    std_16 = x_33.std(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    l__mod___transformer_blocks_4_input_sublayer_norm_a_2 = self.L__mod___transformer_blocks_4_input_sublayer_norm_a_2
    sub_8 = x_33 - mean_16;  mean_16 = None
    mul_8 = l__mod___transformer_blocks_4_input_sublayer_norm_a_2 * sub_8;  l__mod___transformer_blocks_4_input_sublayer_norm_a_2 = sub_8 = None
    add_26 = std_16 + 1e-06;  std_16 = None
    truediv_12 = mul_8 / add_26;  mul_8 = add_26 = None
    l__mod___transformer_blocks_4_input_sublayer_norm_b_2 = self.L__mod___transformer_blocks_4_input_sublayer_norm_b_2
    x_36 = truediv_12 + l__mod___transformer_blocks_4_input_sublayer_norm_b_2;  truediv_12 = l__mod___transformer_blocks_4_input_sublayer_norm_b_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    l__mod___transformer_blocks_4_lambda_module_attention_linear_layers_0 = self.L__mod___transformer_blocks_4_lambda_module_attention_linear_layers_0(x_36)
    view_16 = l__mod___transformer_blocks_4_lambda_module_attention_linear_layers_0.view(4, -1, 12, 64);  l__mod___transformer_blocks_4_lambda_module_attention_linear_layers_0 = None
    query_4 = view_16.transpose(1, 2);  view_16 = None
    l__mod___transformer_blocks_4_lambda_module_attention_linear_layers_1 = self.L__mod___transformer_blocks_4_lambda_module_attention_linear_layers_1(x_36)
    view_17 = l__mod___transformer_blocks_4_lambda_module_attention_linear_layers_1.view(4, -1, 12, 64);  l__mod___transformer_blocks_4_lambda_module_attention_linear_layers_1 = None
    key_4 = view_17.transpose(1, 2);  view_17 = None
    l__mod___transformer_blocks_4_lambda_module_attention_linear_layers_2 = self.L__mod___transformer_blocks_4_lambda_module_attention_linear_layers_2(x_36);  x_36 = None
    view_18 = l__mod___transformer_blocks_4_lambda_module_attention_linear_layers_2.view(4, -1, 12, 64);  l__mod___transformer_blocks_4_lambda_module_attention_linear_layers_2 = None
    value_4 = view_18.transpose(1, 2);  view_18 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:15, code: scores = torch.matmul(query, key.transpose(-2, -1)) \
    transpose_23 = key_4.transpose(-2, -1);  key_4 = None
    matmul_8 = torch.matmul(query_4, transpose_23);  query_4 = transpose_23 = None
    scores_8 = matmul_8 / 8.0;  matmul_8 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    eq_4 = mask == 0
    scores_9 = scores_8.masked_fill(eq_4, -1000000000.0);  scores_8 = eq_4 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    p_attn_8 = torch.nn.functional.softmax(scores_9, dim = -1);  scores_9 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:12, code: return self.dropout(x)
    attn_4 = self.L__mod___transformer_blocks_4_lambda_module_attention_dropout_dropout(p_attn_8);  p_attn_8 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    x_37 = torch.matmul(attn_4, value_4);  attn_4 = value_4 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    transpose_24 = x_37.transpose(1, 2);  x_37 = None
    contiguous_4 = transpose_24.contiguous();  transpose_24 = None
    x_38 = contiguous_4.view(4, -1, 768);  contiguous_4 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    l__mod___transformer_blocks_4_lambda_module_attention_output_linear = self.L__mod___transformer_blocks_4_lambda_module_attention_output_linear(x_38);  x_38 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    l__mod___transformer_blocks_4_input_sublayer_dropout = self.L__mod___transformer_blocks_4_input_sublayer_dropout(l__mod___transformer_blocks_4_lambda_module_attention_output_linear);  l__mod___transformer_blocks_4_lambda_module_attention_output_linear = None
    x_39 = x_33 + l__mod___transformer_blocks_4_input_sublayer_dropout;  x_33 = l__mod___transformer_blocks_4_input_sublayer_dropout = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_18 = x_39.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    std_18 = x_39.std(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    l__mod___transformer_blocks_4_output_sublayer_norm_a_2 = self.L__mod___transformer_blocks_4_output_sublayer_norm_a_2
    sub_9 = x_39 - mean_18;  mean_18 = None
    mul_9 = l__mod___transformer_blocks_4_output_sublayer_norm_a_2 * sub_9;  l__mod___transformer_blocks_4_output_sublayer_norm_a_2 = sub_9 = None
    add_29 = std_18 + 1e-06;  std_18 = None
    truediv_14 = mul_9 / add_29;  mul_9 = add_29 = None
    l__mod___transformer_blocks_4_output_sublayer_norm_b_2 = self.L__mod___transformer_blocks_4_output_sublayer_norm_b_2
    add_30 = truediv_14 + l__mod___transformer_blocks_4_output_sublayer_norm_b_2;  truediv_14 = l__mod___transformer_blocks_4_output_sublayer_norm_b_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    l__mod___transformer_blocks_4_feed_forward_w_1 = self.L__mod___transformer_blocks_4_feed_forward_w_1(add_30);  add_30 = None
    l__mod___transformer_blocks_4_feed_forward_activation = self.L__mod___transformer_blocks_4_feed_forward_activation(l__mod___transformer_blocks_4_feed_forward_w_1);  l__mod___transformer_blocks_4_feed_forward_w_1 = None
    l__mod___transformer_blocks_4_feed_forward_dropout = self.L__mod___transformer_blocks_4_feed_forward_dropout(l__mod___transformer_blocks_4_feed_forward_activation);  l__mod___transformer_blocks_4_feed_forward_activation = None
    l__mod___transformer_blocks_4_feed_forward_w_2 = self.L__mod___transformer_blocks_4_feed_forward_w_2(l__mod___transformer_blocks_4_feed_forward_dropout);  l__mod___transformer_blocks_4_feed_forward_dropout = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    l__mod___transformer_blocks_4_output_sublayer_dropout = self.L__mod___transformer_blocks_4_output_sublayer_dropout(l__mod___transformer_blocks_4_feed_forward_w_2);  l__mod___transformer_blocks_4_feed_forward_w_2 = None
    x_40 = x_39 + l__mod___transformer_blocks_4_output_sublayer_dropout;  x_39 = l__mod___transformer_blocks_4_output_sublayer_dropout = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:49, code: return self.dropout(x)
    x_41 = self.L__mod___transformer_blocks_4_dropout(x_40);  x_40 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_20 = x_41.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    std_20 = x_41.std(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    l__mod___transformer_blocks_5_input_sublayer_norm_a_2 = self.L__mod___transformer_blocks_5_input_sublayer_norm_a_2
    sub_10 = x_41 - mean_20;  mean_20 = None
    mul_10 = l__mod___transformer_blocks_5_input_sublayer_norm_a_2 * sub_10;  l__mod___transformer_blocks_5_input_sublayer_norm_a_2 = sub_10 = None
    add_32 = std_20 + 1e-06;  std_20 = None
    truediv_15 = mul_10 / add_32;  mul_10 = add_32 = None
    l__mod___transformer_blocks_5_input_sublayer_norm_b_2 = self.L__mod___transformer_blocks_5_input_sublayer_norm_b_2
    x_44 = truediv_15 + l__mod___transformer_blocks_5_input_sublayer_norm_b_2;  truediv_15 = l__mod___transformer_blocks_5_input_sublayer_norm_b_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    l__mod___transformer_blocks_5_lambda_module_attention_linear_layers_0 = self.L__mod___transformer_blocks_5_lambda_module_attention_linear_layers_0(x_44)
    view_20 = l__mod___transformer_blocks_5_lambda_module_attention_linear_layers_0.view(4, -1, 12, 64);  l__mod___transformer_blocks_5_lambda_module_attention_linear_layers_0 = None
    query_5 = view_20.transpose(1, 2);  view_20 = None
    l__mod___transformer_blocks_5_lambda_module_attention_linear_layers_1 = self.L__mod___transformer_blocks_5_lambda_module_attention_linear_layers_1(x_44)
    view_21 = l__mod___transformer_blocks_5_lambda_module_attention_linear_layers_1.view(4, -1, 12, 64);  l__mod___transformer_blocks_5_lambda_module_attention_linear_layers_1 = None
    key_5 = view_21.transpose(1, 2);  view_21 = None
    l__mod___transformer_blocks_5_lambda_module_attention_linear_layers_2 = self.L__mod___transformer_blocks_5_lambda_module_attention_linear_layers_2(x_44);  x_44 = None
    view_22 = l__mod___transformer_blocks_5_lambda_module_attention_linear_layers_2.view(4, -1, 12, 64);  l__mod___transformer_blocks_5_lambda_module_attention_linear_layers_2 = None
    value_5 = view_22.transpose(1, 2);  view_22 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:15, code: scores = torch.matmul(query, key.transpose(-2, -1)) \
    transpose_28 = key_5.transpose(-2, -1);  key_5 = None
    matmul_10 = torch.matmul(query_5, transpose_28);  query_5 = transpose_28 = None
    scores_10 = matmul_10 / 8.0;  matmul_10 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    eq_5 = mask == 0
    scores_11 = scores_10.masked_fill(eq_5, -1000000000.0);  scores_10 = eq_5 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    p_attn_10 = torch.nn.functional.softmax(scores_11, dim = -1);  scores_11 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:12, code: return self.dropout(x)
    attn_5 = self.L__mod___transformer_blocks_5_lambda_module_attention_dropout_dropout(p_attn_10);  p_attn_10 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    x_45 = torch.matmul(attn_5, value_5);  attn_5 = value_5 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    transpose_29 = x_45.transpose(1, 2);  x_45 = None
    contiguous_5 = transpose_29.contiguous();  transpose_29 = None
    x_46 = contiguous_5.view(4, -1, 768);  contiguous_5 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    l__mod___transformer_blocks_5_lambda_module_attention_output_linear = self.L__mod___transformer_blocks_5_lambda_module_attention_output_linear(x_46);  x_46 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    l__mod___transformer_blocks_5_input_sublayer_dropout = self.L__mod___transformer_blocks_5_input_sublayer_dropout(l__mod___transformer_blocks_5_lambda_module_attention_output_linear);  l__mod___transformer_blocks_5_lambda_module_attention_output_linear = None
    x_47 = x_41 + l__mod___transformer_blocks_5_input_sublayer_dropout;  x_41 = l__mod___transformer_blocks_5_input_sublayer_dropout = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_22 = x_47.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    std_22 = x_47.std(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    l__mod___transformer_blocks_5_output_sublayer_norm_a_2 = self.L__mod___transformer_blocks_5_output_sublayer_norm_a_2
    sub_11 = x_47 - mean_22;  mean_22 = None
    mul_11 = l__mod___transformer_blocks_5_output_sublayer_norm_a_2 * sub_11;  l__mod___transformer_blocks_5_output_sublayer_norm_a_2 = sub_11 = None
    add_35 = std_22 + 1e-06;  std_22 = None
    truediv_17 = mul_11 / add_35;  mul_11 = add_35 = None
    l__mod___transformer_blocks_5_output_sublayer_norm_b_2 = self.L__mod___transformer_blocks_5_output_sublayer_norm_b_2
    add_36 = truediv_17 + l__mod___transformer_blocks_5_output_sublayer_norm_b_2;  truediv_17 = l__mod___transformer_blocks_5_output_sublayer_norm_b_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    l__mod___transformer_blocks_5_feed_forward_w_1 = self.L__mod___transformer_blocks_5_feed_forward_w_1(add_36);  add_36 = None
    l__mod___transformer_blocks_5_feed_forward_activation = self.L__mod___transformer_blocks_5_feed_forward_activation(l__mod___transformer_blocks_5_feed_forward_w_1);  l__mod___transformer_blocks_5_feed_forward_w_1 = None
    l__mod___transformer_blocks_5_feed_forward_dropout = self.L__mod___transformer_blocks_5_feed_forward_dropout(l__mod___transformer_blocks_5_feed_forward_activation);  l__mod___transformer_blocks_5_feed_forward_activation = None
    l__mod___transformer_blocks_5_feed_forward_w_2 = self.L__mod___transformer_blocks_5_feed_forward_w_2(l__mod___transformer_blocks_5_feed_forward_dropout);  l__mod___transformer_blocks_5_feed_forward_dropout = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    l__mod___transformer_blocks_5_output_sublayer_dropout = self.L__mod___transformer_blocks_5_output_sublayer_dropout(l__mod___transformer_blocks_5_feed_forward_w_2);  l__mod___transformer_blocks_5_feed_forward_w_2 = None
    x_48 = x_47 + l__mod___transformer_blocks_5_output_sublayer_dropout;  x_47 = l__mod___transformer_blocks_5_output_sublayer_dropout = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:49, code: return self.dropout(x)
    x_49 = self.L__mod___transformer_blocks_5_dropout(x_48);  x_48 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_24 = x_49.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    std_24 = x_49.std(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    l__mod___transformer_blocks_6_input_sublayer_norm_a_2 = self.L__mod___transformer_blocks_6_input_sublayer_norm_a_2
    sub_12 = x_49 - mean_24;  mean_24 = None
    mul_12 = l__mod___transformer_blocks_6_input_sublayer_norm_a_2 * sub_12;  l__mod___transformer_blocks_6_input_sublayer_norm_a_2 = sub_12 = None
    add_38 = std_24 + 1e-06;  std_24 = None
    truediv_18 = mul_12 / add_38;  mul_12 = add_38 = None
    l__mod___transformer_blocks_6_input_sublayer_norm_b_2 = self.L__mod___transformer_blocks_6_input_sublayer_norm_b_2
    x_52 = truediv_18 + l__mod___transformer_blocks_6_input_sublayer_norm_b_2;  truediv_18 = l__mod___transformer_blocks_6_input_sublayer_norm_b_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    l__mod___transformer_blocks_6_lambda_module_attention_linear_layers_0 = self.L__mod___transformer_blocks_6_lambda_module_attention_linear_layers_0(x_52)
    view_24 = l__mod___transformer_blocks_6_lambda_module_attention_linear_layers_0.view(4, -1, 12, 64);  l__mod___transformer_blocks_6_lambda_module_attention_linear_layers_0 = None
    query_6 = view_24.transpose(1, 2);  view_24 = None
    l__mod___transformer_blocks_6_lambda_module_attention_linear_layers_1 = self.L__mod___transformer_blocks_6_lambda_module_attention_linear_layers_1(x_52)
    view_25 = l__mod___transformer_blocks_6_lambda_module_attention_linear_layers_1.view(4, -1, 12, 64);  l__mod___transformer_blocks_6_lambda_module_attention_linear_layers_1 = None
    key_6 = view_25.transpose(1, 2);  view_25 = None
    l__mod___transformer_blocks_6_lambda_module_attention_linear_layers_2 = self.L__mod___transformer_blocks_6_lambda_module_attention_linear_layers_2(x_52);  x_52 = None
    view_26 = l__mod___transformer_blocks_6_lambda_module_attention_linear_layers_2.view(4, -1, 12, 64);  l__mod___transformer_blocks_6_lambda_module_attention_linear_layers_2 = None
    value_6 = view_26.transpose(1, 2);  view_26 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:15, code: scores = torch.matmul(query, key.transpose(-2, -1)) \
    transpose_33 = key_6.transpose(-2, -1);  key_6 = None
    matmul_12 = torch.matmul(query_6, transpose_33);  query_6 = transpose_33 = None
    scores_12 = matmul_12 / 8.0;  matmul_12 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    eq_6 = mask == 0
    scores_13 = scores_12.masked_fill(eq_6, -1000000000.0);  scores_12 = eq_6 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    p_attn_12 = torch.nn.functional.softmax(scores_13, dim = -1);  scores_13 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:12, code: return self.dropout(x)
    attn_6 = self.L__mod___transformer_blocks_6_lambda_module_attention_dropout_dropout(p_attn_12);  p_attn_12 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    x_53 = torch.matmul(attn_6, value_6);  attn_6 = value_6 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    transpose_34 = x_53.transpose(1, 2);  x_53 = None
    contiguous_6 = transpose_34.contiguous();  transpose_34 = None
    x_54 = contiguous_6.view(4, -1, 768);  contiguous_6 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    l__mod___transformer_blocks_6_lambda_module_attention_output_linear = self.L__mod___transformer_blocks_6_lambda_module_attention_output_linear(x_54);  x_54 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    l__mod___transformer_blocks_6_input_sublayer_dropout = self.L__mod___transformer_blocks_6_input_sublayer_dropout(l__mod___transformer_blocks_6_lambda_module_attention_output_linear);  l__mod___transformer_blocks_6_lambda_module_attention_output_linear = None
    x_55 = x_49 + l__mod___transformer_blocks_6_input_sublayer_dropout;  x_49 = l__mod___transformer_blocks_6_input_sublayer_dropout = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_26 = x_55.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    std_26 = x_55.std(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    l__mod___transformer_blocks_6_output_sublayer_norm_a_2 = self.L__mod___transformer_blocks_6_output_sublayer_norm_a_2
    sub_13 = x_55 - mean_26;  mean_26 = None
    mul_13 = l__mod___transformer_blocks_6_output_sublayer_norm_a_2 * sub_13;  l__mod___transformer_blocks_6_output_sublayer_norm_a_2 = sub_13 = None
    add_41 = std_26 + 1e-06;  std_26 = None
    truediv_20 = mul_13 / add_41;  mul_13 = add_41 = None
    l__mod___transformer_blocks_6_output_sublayer_norm_b_2 = self.L__mod___transformer_blocks_6_output_sublayer_norm_b_2
    add_42 = truediv_20 + l__mod___transformer_blocks_6_output_sublayer_norm_b_2;  truediv_20 = l__mod___transformer_blocks_6_output_sublayer_norm_b_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    l__mod___transformer_blocks_6_feed_forward_w_1 = self.L__mod___transformer_blocks_6_feed_forward_w_1(add_42);  add_42 = None
    l__mod___transformer_blocks_6_feed_forward_activation = self.L__mod___transformer_blocks_6_feed_forward_activation(l__mod___transformer_blocks_6_feed_forward_w_1);  l__mod___transformer_blocks_6_feed_forward_w_1 = None
    l__mod___transformer_blocks_6_feed_forward_dropout = self.L__mod___transformer_blocks_6_feed_forward_dropout(l__mod___transformer_blocks_6_feed_forward_activation);  l__mod___transformer_blocks_6_feed_forward_activation = None
    l__mod___transformer_blocks_6_feed_forward_w_2 = self.L__mod___transformer_blocks_6_feed_forward_w_2(l__mod___transformer_blocks_6_feed_forward_dropout);  l__mod___transformer_blocks_6_feed_forward_dropout = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    l__mod___transformer_blocks_6_output_sublayer_dropout = self.L__mod___transformer_blocks_6_output_sublayer_dropout(l__mod___transformer_blocks_6_feed_forward_w_2);  l__mod___transformer_blocks_6_feed_forward_w_2 = None
    x_56 = x_55 + l__mod___transformer_blocks_6_output_sublayer_dropout;  x_55 = l__mod___transformer_blocks_6_output_sublayer_dropout = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:49, code: return self.dropout(x)
    x_57 = self.L__mod___transformer_blocks_6_dropout(x_56);  x_56 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_28 = x_57.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    std_28 = x_57.std(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    l__mod___transformer_blocks_7_input_sublayer_norm_a_2 = self.L__mod___transformer_blocks_7_input_sublayer_norm_a_2
    sub_14 = x_57 - mean_28;  mean_28 = None
    mul_14 = l__mod___transformer_blocks_7_input_sublayer_norm_a_2 * sub_14;  l__mod___transformer_blocks_7_input_sublayer_norm_a_2 = sub_14 = None
    add_44 = std_28 + 1e-06;  std_28 = None
    truediv_21 = mul_14 / add_44;  mul_14 = add_44 = None
    l__mod___transformer_blocks_7_input_sublayer_norm_b_2 = self.L__mod___transformer_blocks_7_input_sublayer_norm_b_2
    x_60 = truediv_21 + l__mod___transformer_blocks_7_input_sublayer_norm_b_2;  truediv_21 = l__mod___transformer_blocks_7_input_sublayer_norm_b_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    l__mod___transformer_blocks_7_lambda_module_attention_linear_layers_0 = self.L__mod___transformer_blocks_7_lambda_module_attention_linear_layers_0(x_60)
    view_28 = l__mod___transformer_blocks_7_lambda_module_attention_linear_layers_0.view(4, -1, 12, 64);  l__mod___transformer_blocks_7_lambda_module_attention_linear_layers_0 = None
    query_7 = view_28.transpose(1, 2);  view_28 = None
    l__mod___transformer_blocks_7_lambda_module_attention_linear_layers_1 = self.L__mod___transformer_blocks_7_lambda_module_attention_linear_layers_1(x_60)
    view_29 = l__mod___transformer_blocks_7_lambda_module_attention_linear_layers_1.view(4, -1, 12, 64);  l__mod___transformer_blocks_7_lambda_module_attention_linear_layers_1 = None
    key_7 = view_29.transpose(1, 2);  view_29 = None
    l__mod___transformer_blocks_7_lambda_module_attention_linear_layers_2 = self.L__mod___transformer_blocks_7_lambda_module_attention_linear_layers_2(x_60);  x_60 = None
    view_30 = l__mod___transformer_blocks_7_lambda_module_attention_linear_layers_2.view(4, -1, 12, 64);  l__mod___transformer_blocks_7_lambda_module_attention_linear_layers_2 = None
    value_7 = view_30.transpose(1, 2);  view_30 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:15, code: scores = torch.matmul(query, key.transpose(-2, -1)) \
    transpose_38 = key_7.transpose(-2, -1);  key_7 = None
    matmul_14 = torch.matmul(query_7, transpose_38);  query_7 = transpose_38 = None
    scores_14 = matmul_14 / 8.0;  matmul_14 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    eq_7 = mask == 0
    scores_15 = scores_14.masked_fill(eq_7, -1000000000.0);  scores_14 = eq_7 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    p_attn_14 = torch.nn.functional.softmax(scores_15, dim = -1);  scores_15 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:12, code: return self.dropout(x)
    attn_7 = self.L__mod___transformer_blocks_7_lambda_module_attention_dropout_dropout(p_attn_14);  p_attn_14 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    x_61 = torch.matmul(attn_7, value_7);  attn_7 = value_7 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    transpose_39 = x_61.transpose(1, 2);  x_61 = None
    contiguous_7 = transpose_39.contiguous();  transpose_39 = None
    x_62 = contiguous_7.view(4, -1, 768);  contiguous_7 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    l__mod___transformer_blocks_7_lambda_module_attention_output_linear = self.L__mod___transformer_blocks_7_lambda_module_attention_output_linear(x_62);  x_62 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    l__mod___transformer_blocks_7_input_sublayer_dropout = self.L__mod___transformer_blocks_7_input_sublayer_dropout(l__mod___transformer_blocks_7_lambda_module_attention_output_linear);  l__mod___transformer_blocks_7_lambda_module_attention_output_linear = None
    x_63 = x_57 + l__mod___transformer_blocks_7_input_sublayer_dropout;  x_57 = l__mod___transformer_blocks_7_input_sublayer_dropout = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_30 = x_63.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    std_30 = x_63.std(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    l__mod___transformer_blocks_7_output_sublayer_norm_a_2 = self.L__mod___transformer_blocks_7_output_sublayer_norm_a_2
    sub_15 = x_63 - mean_30;  mean_30 = None
    mul_15 = l__mod___transformer_blocks_7_output_sublayer_norm_a_2 * sub_15;  l__mod___transformer_blocks_7_output_sublayer_norm_a_2 = sub_15 = None
    add_47 = std_30 + 1e-06;  std_30 = None
    truediv_23 = mul_15 / add_47;  mul_15 = add_47 = None
    l__mod___transformer_blocks_7_output_sublayer_norm_b_2 = self.L__mod___transformer_blocks_7_output_sublayer_norm_b_2
    add_48 = truediv_23 + l__mod___transformer_blocks_7_output_sublayer_norm_b_2;  truediv_23 = l__mod___transformer_blocks_7_output_sublayer_norm_b_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    l__mod___transformer_blocks_7_feed_forward_w_1 = self.L__mod___transformer_blocks_7_feed_forward_w_1(add_48);  add_48 = None
    l__mod___transformer_blocks_7_feed_forward_activation = self.L__mod___transformer_blocks_7_feed_forward_activation(l__mod___transformer_blocks_7_feed_forward_w_1);  l__mod___transformer_blocks_7_feed_forward_w_1 = None
    l__mod___transformer_blocks_7_feed_forward_dropout = self.L__mod___transformer_blocks_7_feed_forward_dropout(l__mod___transformer_blocks_7_feed_forward_activation);  l__mod___transformer_blocks_7_feed_forward_activation = None
    l__mod___transformer_blocks_7_feed_forward_w_2 = self.L__mod___transformer_blocks_7_feed_forward_w_2(l__mod___transformer_blocks_7_feed_forward_dropout);  l__mod___transformer_blocks_7_feed_forward_dropout = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    l__mod___transformer_blocks_7_output_sublayer_dropout = self.L__mod___transformer_blocks_7_output_sublayer_dropout(l__mod___transformer_blocks_7_feed_forward_w_2);  l__mod___transformer_blocks_7_feed_forward_w_2 = None
    x_64 = x_63 + l__mod___transformer_blocks_7_output_sublayer_dropout;  x_63 = l__mod___transformer_blocks_7_output_sublayer_dropout = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:49, code: return self.dropout(x)
    x_65 = self.L__mod___transformer_blocks_7_dropout(x_64);  x_64 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_32 = x_65.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    std_32 = x_65.std(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    l__mod___transformer_blocks_8_input_sublayer_norm_a_2 = self.L__mod___transformer_blocks_8_input_sublayer_norm_a_2
    sub_16 = x_65 - mean_32;  mean_32 = None
    mul_16 = l__mod___transformer_blocks_8_input_sublayer_norm_a_2 * sub_16;  l__mod___transformer_blocks_8_input_sublayer_norm_a_2 = sub_16 = None
    add_50 = std_32 + 1e-06;  std_32 = None
    truediv_24 = mul_16 / add_50;  mul_16 = add_50 = None
    l__mod___transformer_blocks_8_input_sublayer_norm_b_2 = self.L__mod___transformer_blocks_8_input_sublayer_norm_b_2
    x_68 = truediv_24 + l__mod___transformer_blocks_8_input_sublayer_norm_b_2;  truediv_24 = l__mod___transformer_blocks_8_input_sublayer_norm_b_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    l__mod___transformer_blocks_8_lambda_module_attention_linear_layers_0 = self.L__mod___transformer_blocks_8_lambda_module_attention_linear_layers_0(x_68)
    view_32 = l__mod___transformer_blocks_8_lambda_module_attention_linear_layers_0.view(4, -1, 12, 64);  l__mod___transformer_blocks_8_lambda_module_attention_linear_layers_0 = None
    query_8 = view_32.transpose(1, 2);  view_32 = None
    l__mod___transformer_blocks_8_lambda_module_attention_linear_layers_1 = self.L__mod___transformer_blocks_8_lambda_module_attention_linear_layers_1(x_68)
    view_33 = l__mod___transformer_blocks_8_lambda_module_attention_linear_layers_1.view(4, -1, 12, 64);  l__mod___transformer_blocks_8_lambda_module_attention_linear_layers_1 = None
    key_8 = view_33.transpose(1, 2);  view_33 = None
    l__mod___transformer_blocks_8_lambda_module_attention_linear_layers_2 = self.L__mod___transformer_blocks_8_lambda_module_attention_linear_layers_2(x_68);  x_68 = None
    view_34 = l__mod___transformer_blocks_8_lambda_module_attention_linear_layers_2.view(4, -1, 12, 64);  l__mod___transformer_blocks_8_lambda_module_attention_linear_layers_2 = None
    value_8 = view_34.transpose(1, 2);  view_34 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:15, code: scores = torch.matmul(query, key.transpose(-2, -1)) \
    transpose_43 = key_8.transpose(-2, -1);  key_8 = None
    matmul_16 = torch.matmul(query_8, transpose_43);  query_8 = transpose_43 = None
    scores_16 = matmul_16 / 8.0;  matmul_16 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    eq_8 = mask == 0
    scores_17 = scores_16.masked_fill(eq_8, -1000000000.0);  scores_16 = eq_8 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    p_attn_16 = torch.nn.functional.softmax(scores_17, dim = -1);  scores_17 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:12, code: return self.dropout(x)
    attn_8 = self.L__mod___transformer_blocks_8_lambda_module_attention_dropout_dropout(p_attn_16);  p_attn_16 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    x_69 = torch.matmul(attn_8, value_8);  attn_8 = value_8 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    transpose_44 = x_69.transpose(1, 2);  x_69 = None
    contiguous_8 = transpose_44.contiguous();  transpose_44 = None
    x_70 = contiguous_8.view(4, -1, 768);  contiguous_8 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    l__mod___transformer_blocks_8_lambda_module_attention_output_linear = self.L__mod___transformer_blocks_8_lambda_module_attention_output_linear(x_70);  x_70 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    l__mod___transformer_blocks_8_input_sublayer_dropout = self.L__mod___transformer_blocks_8_input_sublayer_dropout(l__mod___transformer_blocks_8_lambda_module_attention_output_linear);  l__mod___transformer_blocks_8_lambda_module_attention_output_linear = None
    x_71 = x_65 + l__mod___transformer_blocks_8_input_sublayer_dropout;  x_65 = l__mod___transformer_blocks_8_input_sublayer_dropout = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_34 = x_71.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    std_34 = x_71.std(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    l__mod___transformer_blocks_8_output_sublayer_norm_a_2 = self.L__mod___transformer_blocks_8_output_sublayer_norm_a_2
    sub_17 = x_71 - mean_34;  mean_34 = None
    mul_17 = l__mod___transformer_blocks_8_output_sublayer_norm_a_2 * sub_17;  l__mod___transformer_blocks_8_output_sublayer_norm_a_2 = sub_17 = None
    add_53 = std_34 + 1e-06;  std_34 = None
    truediv_26 = mul_17 / add_53;  mul_17 = add_53 = None
    l__mod___transformer_blocks_8_output_sublayer_norm_b_2 = self.L__mod___transformer_blocks_8_output_sublayer_norm_b_2
    add_54 = truediv_26 + l__mod___transformer_blocks_8_output_sublayer_norm_b_2;  truediv_26 = l__mod___transformer_blocks_8_output_sublayer_norm_b_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    l__mod___transformer_blocks_8_feed_forward_w_1 = self.L__mod___transformer_blocks_8_feed_forward_w_1(add_54);  add_54 = None
    l__mod___transformer_blocks_8_feed_forward_activation = self.L__mod___transformer_blocks_8_feed_forward_activation(l__mod___transformer_blocks_8_feed_forward_w_1);  l__mod___transformer_blocks_8_feed_forward_w_1 = None
    l__mod___transformer_blocks_8_feed_forward_dropout = self.L__mod___transformer_blocks_8_feed_forward_dropout(l__mod___transformer_blocks_8_feed_forward_activation);  l__mod___transformer_blocks_8_feed_forward_activation = None
    l__mod___transformer_blocks_8_feed_forward_w_2 = self.L__mod___transformer_blocks_8_feed_forward_w_2(l__mod___transformer_blocks_8_feed_forward_dropout);  l__mod___transformer_blocks_8_feed_forward_dropout = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    l__mod___transformer_blocks_8_output_sublayer_dropout = self.L__mod___transformer_blocks_8_output_sublayer_dropout(l__mod___transformer_blocks_8_feed_forward_w_2);  l__mod___transformer_blocks_8_feed_forward_w_2 = None
    x_72 = x_71 + l__mod___transformer_blocks_8_output_sublayer_dropout;  x_71 = l__mod___transformer_blocks_8_output_sublayer_dropout = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:49, code: return self.dropout(x)
    x_73 = self.L__mod___transformer_blocks_8_dropout(x_72);  x_72 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_36 = x_73.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    std_36 = x_73.std(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    l__mod___transformer_blocks_9_input_sublayer_norm_a_2 = self.L__mod___transformer_blocks_9_input_sublayer_norm_a_2
    sub_18 = x_73 - mean_36;  mean_36 = None
    mul_18 = l__mod___transformer_blocks_9_input_sublayer_norm_a_2 * sub_18;  l__mod___transformer_blocks_9_input_sublayer_norm_a_2 = sub_18 = None
    add_56 = std_36 + 1e-06;  std_36 = None
    truediv_27 = mul_18 / add_56;  mul_18 = add_56 = None
    l__mod___transformer_blocks_9_input_sublayer_norm_b_2 = self.L__mod___transformer_blocks_9_input_sublayer_norm_b_2
    x_76 = truediv_27 + l__mod___transformer_blocks_9_input_sublayer_norm_b_2;  truediv_27 = l__mod___transformer_blocks_9_input_sublayer_norm_b_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    l__mod___transformer_blocks_9_lambda_module_attention_linear_layers_0 = self.L__mod___transformer_blocks_9_lambda_module_attention_linear_layers_0(x_76)
    view_36 = l__mod___transformer_blocks_9_lambda_module_attention_linear_layers_0.view(4, -1, 12, 64);  l__mod___transformer_blocks_9_lambda_module_attention_linear_layers_0 = None
    query_9 = view_36.transpose(1, 2);  view_36 = None
    l__mod___transformer_blocks_9_lambda_module_attention_linear_layers_1 = self.L__mod___transformer_blocks_9_lambda_module_attention_linear_layers_1(x_76)
    view_37 = l__mod___transformer_blocks_9_lambda_module_attention_linear_layers_1.view(4, -1, 12, 64);  l__mod___transformer_blocks_9_lambda_module_attention_linear_layers_1 = None
    key_9 = view_37.transpose(1, 2);  view_37 = None
    l__mod___transformer_blocks_9_lambda_module_attention_linear_layers_2 = self.L__mod___transformer_blocks_9_lambda_module_attention_linear_layers_2(x_76);  x_76 = None
    view_38 = l__mod___transformer_blocks_9_lambda_module_attention_linear_layers_2.view(4, -1, 12, 64);  l__mod___transformer_blocks_9_lambda_module_attention_linear_layers_2 = None
    value_9 = view_38.transpose(1, 2);  view_38 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:15, code: scores = torch.matmul(query, key.transpose(-2, -1)) \
    transpose_48 = key_9.transpose(-2, -1);  key_9 = None
    matmul_18 = torch.matmul(query_9, transpose_48);  query_9 = transpose_48 = None
    scores_18 = matmul_18 / 8.0;  matmul_18 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    eq_9 = mask == 0
    scores_19 = scores_18.masked_fill(eq_9, -1000000000.0);  scores_18 = eq_9 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    p_attn_18 = torch.nn.functional.softmax(scores_19, dim = -1);  scores_19 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:12, code: return self.dropout(x)
    attn_9 = self.L__mod___transformer_blocks_9_lambda_module_attention_dropout_dropout(p_attn_18);  p_attn_18 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    x_77 = torch.matmul(attn_9, value_9);  attn_9 = value_9 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    transpose_49 = x_77.transpose(1, 2);  x_77 = None
    contiguous_9 = transpose_49.contiguous();  transpose_49 = None
    x_78 = contiguous_9.view(4, -1, 768);  contiguous_9 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    l__mod___transformer_blocks_9_lambda_module_attention_output_linear = self.L__mod___transformer_blocks_9_lambda_module_attention_output_linear(x_78);  x_78 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    l__mod___transformer_blocks_9_input_sublayer_dropout = self.L__mod___transformer_blocks_9_input_sublayer_dropout(l__mod___transformer_blocks_9_lambda_module_attention_output_linear);  l__mod___transformer_blocks_9_lambda_module_attention_output_linear = None
    x_79 = x_73 + l__mod___transformer_blocks_9_input_sublayer_dropout;  x_73 = l__mod___transformer_blocks_9_input_sublayer_dropout = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_38 = x_79.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    std_38 = x_79.std(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    l__mod___transformer_blocks_9_output_sublayer_norm_a_2 = self.L__mod___transformer_blocks_9_output_sublayer_norm_a_2
    sub_19 = x_79 - mean_38;  mean_38 = None
    mul_19 = l__mod___transformer_blocks_9_output_sublayer_norm_a_2 * sub_19;  l__mod___transformer_blocks_9_output_sublayer_norm_a_2 = sub_19 = None
    add_59 = std_38 + 1e-06;  std_38 = None
    truediv_29 = mul_19 / add_59;  mul_19 = add_59 = None
    l__mod___transformer_blocks_9_output_sublayer_norm_b_2 = self.L__mod___transformer_blocks_9_output_sublayer_norm_b_2
    add_60 = truediv_29 + l__mod___transformer_blocks_9_output_sublayer_norm_b_2;  truediv_29 = l__mod___transformer_blocks_9_output_sublayer_norm_b_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    l__mod___transformer_blocks_9_feed_forward_w_1 = self.L__mod___transformer_blocks_9_feed_forward_w_1(add_60);  add_60 = None
    l__mod___transformer_blocks_9_feed_forward_activation = self.L__mod___transformer_blocks_9_feed_forward_activation(l__mod___transformer_blocks_9_feed_forward_w_1);  l__mod___transformer_blocks_9_feed_forward_w_1 = None
    l__mod___transformer_blocks_9_feed_forward_dropout = self.L__mod___transformer_blocks_9_feed_forward_dropout(l__mod___transformer_blocks_9_feed_forward_activation);  l__mod___transformer_blocks_9_feed_forward_activation = None
    l__mod___transformer_blocks_9_feed_forward_w_2 = self.L__mod___transformer_blocks_9_feed_forward_w_2(l__mod___transformer_blocks_9_feed_forward_dropout);  l__mod___transformer_blocks_9_feed_forward_dropout = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    l__mod___transformer_blocks_9_output_sublayer_dropout = self.L__mod___transformer_blocks_9_output_sublayer_dropout(l__mod___transformer_blocks_9_feed_forward_w_2);  l__mod___transformer_blocks_9_feed_forward_w_2 = None
    x_80 = x_79 + l__mod___transformer_blocks_9_output_sublayer_dropout;  x_79 = l__mod___transformer_blocks_9_output_sublayer_dropout = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:49, code: return self.dropout(x)
    x_81 = self.L__mod___transformer_blocks_9_dropout(x_80);  x_80 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_40 = x_81.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    std_40 = x_81.std(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    l__mod___transformer_blocks_10_input_sublayer_norm_a_2 = self.L__mod___transformer_blocks_10_input_sublayer_norm_a_2
    sub_20 = x_81 - mean_40;  mean_40 = None
    mul_20 = l__mod___transformer_blocks_10_input_sublayer_norm_a_2 * sub_20;  l__mod___transformer_blocks_10_input_sublayer_norm_a_2 = sub_20 = None
    add_62 = std_40 + 1e-06;  std_40 = None
    truediv_30 = mul_20 / add_62;  mul_20 = add_62 = None
    l__mod___transformer_blocks_10_input_sublayer_norm_b_2 = self.L__mod___transformer_blocks_10_input_sublayer_norm_b_2
    x_84 = truediv_30 + l__mod___transformer_blocks_10_input_sublayer_norm_b_2;  truediv_30 = l__mod___transformer_blocks_10_input_sublayer_norm_b_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    l__mod___transformer_blocks_10_lambda_module_attention_linear_layers_0 = self.L__mod___transformer_blocks_10_lambda_module_attention_linear_layers_0(x_84)
    view_40 = l__mod___transformer_blocks_10_lambda_module_attention_linear_layers_0.view(4, -1, 12, 64);  l__mod___transformer_blocks_10_lambda_module_attention_linear_layers_0 = None
    query_10 = view_40.transpose(1, 2);  view_40 = None
    l__mod___transformer_blocks_10_lambda_module_attention_linear_layers_1 = self.L__mod___transformer_blocks_10_lambda_module_attention_linear_layers_1(x_84)
    view_41 = l__mod___transformer_blocks_10_lambda_module_attention_linear_layers_1.view(4, -1, 12, 64);  l__mod___transformer_blocks_10_lambda_module_attention_linear_layers_1 = None
    key_10 = view_41.transpose(1, 2);  view_41 = None
    l__mod___transformer_blocks_10_lambda_module_attention_linear_layers_2 = self.L__mod___transformer_blocks_10_lambda_module_attention_linear_layers_2(x_84);  x_84 = None
    view_42 = l__mod___transformer_blocks_10_lambda_module_attention_linear_layers_2.view(4, -1, 12, 64);  l__mod___transformer_blocks_10_lambda_module_attention_linear_layers_2 = None
    value_10 = view_42.transpose(1, 2);  view_42 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:15, code: scores = torch.matmul(query, key.transpose(-2, -1)) \
    transpose_53 = key_10.transpose(-2, -1);  key_10 = None
    matmul_20 = torch.matmul(query_10, transpose_53);  query_10 = transpose_53 = None
    scores_20 = matmul_20 / 8.0;  matmul_20 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    eq_10 = mask == 0
    scores_21 = scores_20.masked_fill(eq_10, -1000000000.0);  scores_20 = eq_10 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    p_attn_20 = torch.nn.functional.softmax(scores_21, dim = -1);  scores_21 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:12, code: return self.dropout(x)
    attn_10 = self.L__mod___transformer_blocks_10_lambda_module_attention_dropout_dropout(p_attn_20);  p_attn_20 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    x_85 = torch.matmul(attn_10, value_10);  attn_10 = value_10 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    transpose_54 = x_85.transpose(1, 2);  x_85 = None
    contiguous_10 = transpose_54.contiguous();  transpose_54 = None
    x_86 = contiguous_10.view(4, -1, 768);  contiguous_10 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    l__mod___transformer_blocks_10_lambda_module_attention_output_linear = self.L__mod___transformer_blocks_10_lambda_module_attention_output_linear(x_86);  x_86 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    l__mod___transformer_blocks_10_input_sublayer_dropout = self.L__mod___transformer_blocks_10_input_sublayer_dropout(l__mod___transformer_blocks_10_lambda_module_attention_output_linear);  l__mod___transformer_blocks_10_lambda_module_attention_output_linear = None
    x_87 = x_81 + l__mod___transformer_blocks_10_input_sublayer_dropout;  x_81 = l__mod___transformer_blocks_10_input_sublayer_dropout = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_42 = x_87.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    std_42 = x_87.std(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    l__mod___transformer_blocks_10_output_sublayer_norm_a_2 = self.L__mod___transformer_blocks_10_output_sublayer_norm_a_2
    sub_21 = x_87 - mean_42;  mean_42 = None
    mul_21 = l__mod___transformer_blocks_10_output_sublayer_norm_a_2 * sub_21;  l__mod___transformer_blocks_10_output_sublayer_norm_a_2 = sub_21 = None
    add_65 = std_42 + 1e-06;  std_42 = None
    truediv_32 = mul_21 / add_65;  mul_21 = add_65 = None
    l__mod___transformer_blocks_10_output_sublayer_norm_b_2 = self.L__mod___transformer_blocks_10_output_sublayer_norm_b_2
    add_66 = truediv_32 + l__mod___transformer_blocks_10_output_sublayer_norm_b_2;  truediv_32 = l__mod___transformer_blocks_10_output_sublayer_norm_b_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    l__mod___transformer_blocks_10_feed_forward_w_1 = self.L__mod___transformer_blocks_10_feed_forward_w_1(add_66);  add_66 = None
    l__mod___transformer_blocks_10_feed_forward_activation = self.L__mod___transformer_blocks_10_feed_forward_activation(l__mod___transformer_blocks_10_feed_forward_w_1);  l__mod___transformer_blocks_10_feed_forward_w_1 = None
    l__mod___transformer_blocks_10_feed_forward_dropout = self.L__mod___transformer_blocks_10_feed_forward_dropout(l__mod___transformer_blocks_10_feed_forward_activation);  l__mod___transformer_blocks_10_feed_forward_activation = None
    l__mod___transformer_blocks_10_feed_forward_w_2 = self.L__mod___transformer_blocks_10_feed_forward_w_2(l__mod___transformer_blocks_10_feed_forward_dropout);  l__mod___transformer_blocks_10_feed_forward_dropout = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    l__mod___transformer_blocks_10_output_sublayer_dropout = self.L__mod___transformer_blocks_10_output_sublayer_dropout(l__mod___transformer_blocks_10_feed_forward_w_2);  l__mod___transformer_blocks_10_feed_forward_w_2 = None
    x_88 = x_87 + l__mod___transformer_blocks_10_output_sublayer_dropout;  x_87 = l__mod___transformer_blocks_10_output_sublayer_dropout = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:49, code: return self.dropout(x)
    x_89 = self.L__mod___transformer_blocks_10_dropout(x_88);  x_88 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_44 = x_89.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    std_44 = x_89.std(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    l__mod___transformer_blocks_11_input_sublayer_norm_a_2 = self.L__mod___transformer_blocks_11_input_sublayer_norm_a_2
    sub_22 = x_89 - mean_44;  mean_44 = None
    mul_22 = l__mod___transformer_blocks_11_input_sublayer_norm_a_2 * sub_22;  l__mod___transformer_blocks_11_input_sublayer_norm_a_2 = sub_22 = None
    add_68 = std_44 + 1e-06;  std_44 = None
    truediv_33 = mul_22 / add_68;  mul_22 = add_68 = None
    l__mod___transformer_blocks_11_input_sublayer_norm_b_2 = self.L__mod___transformer_blocks_11_input_sublayer_norm_b_2
    x_92 = truediv_33 + l__mod___transformer_blocks_11_input_sublayer_norm_b_2;  truediv_33 = l__mod___transformer_blocks_11_input_sublayer_norm_b_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    l__mod___transformer_blocks_11_lambda_module_attention_linear_layers_0 = self.L__mod___transformer_blocks_11_lambda_module_attention_linear_layers_0(x_92)
    view_44 = l__mod___transformer_blocks_11_lambda_module_attention_linear_layers_0.view(4, -1, 12, 64);  l__mod___transformer_blocks_11_lambda_module_attention_linear_layers_0 = None
    query_11 = view_44.transpose(1, 2);  view_44 = None
    l__mod___transformer_blocks_11_lambda_module_attention_linear_layers_1 = self.L__mod___transformer_blocks_11_lambda_module_attention_linear_layers_1(x_92)
    view_45 = l__mod___transformer_blocks_11_lambda_module_attention_linear_layers_1.view(4, -1, 12, 64);  l__mod___transformer_blocks_11_lambda_module_attention_linear_layers_1 = None
    key_11 = view_45.transpose(1, 2);  view_45 = None
    l__mod___transformer_blocks_11_lambda_module_attention_linear_layers_2 = self.L__mod___transformer_blocks_11_lambda_module_attention_linear_layers_2(x_92);  x_92 = None
    view_46 = l__mod___transformer_blocks_11_lambda_module_attention_linear_layers_2.view(4, -1, 12, 64);  l__mod___transformer_blocks_11_lambda_module_attention_linear_layers_2 = None
    value_11 = view_46.transpose(1, 2);  view_46 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:15, code: scores = torch.matmul(query, key.transpose(-2, -1)) \
    transpose_58 = key_11.transpose(-2, -1);  key_11 = None
    matmul_22 = torch.matmul(query_11, transpose_58);  query_11 = transpose_58 = None
    scores_22 = matmul_22 / 8.0;  matmul_22 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    eq_11 = mask == 0
    scores_23 = scores_22.masked_fill(eq_11, -1000000000.0);  scores_22 = eq_11 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    p_attn_22 = torch.nn.functional.softmax(scores_23, dim = -1);  scores_23 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:12, code: return self.dropout(x)
    attn_11 = self.L__mod___transformer_blocks_11_lambda_module_attention_dropout_dropout(p_attn_22);  p_attn_22 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    x_93 = torch.matmul(attn_11, value_11);  attn_11 = value_11 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    transpose_59 = x_93.transpose(1, 2);  x_93 = None
    contiguous_11 = transpose_59.contiguous();  transpose_59 = None
    x_94 = contiguous_11.view(4, -1, 768);  contiguous_11 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    l__mod___transformer_blocks_11_lambda_module_attention_output_linear = self.L__mod___transformer_blocks_11_lambda_module_attention_output_linear(x_94);  x_94 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    l__mod___transformer_blocks_11_input_sublayer_dropout = self.L__mod___transformer_blocks_11_input_sublayer_dropout(l__mod___transformer_blocks_11_lambda_module_attention_output_linear);  l__mod___transformer_blocks_11_lambda_module_attention_output_linear = None
    x_95 = x_89 + l__mod___transformer_blocks_11_input_sublayer_dropout;  x_89 = l__mod___transformer_blocks_11_input_sublayer_dropout = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_46 = x_95.mean(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    std_46 = x_95.std(-1, keepdim = True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    l__mod___transformer_blocks_11_output_sublayer_norm_a_2 = self.L__mod___transformer_blocks_11_output_sublayer_norm_a_2
    sub_23 = x_95 - mean_46;  mean_46 = None
    mul_23 = l__mod___transformer_blocks_11_output_sublayer_norm_a_2 * sub_23;  l__mod___transformer_blocks_11_output_sublayer_norm_a_2 = sub_23 = None
    add_71 = std_46 + 1e-06;  std_46 = None
    truediv_35 = mul_23 / add_71;  mul_23 = add_71 = None
    l__mod___transformer_blocks_11_output_sublayer_norm_b_2 = self.L__mod___transformer_blocks_11_output_sublayer_norm_b_2
    add_72 = truediv_35 + l__mod___transformer_blocks_11_output_sublayer_norm_b_2;  truediv_35 = l__mod___transformer_blocks_11_output_sublayer_norm_b_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    l__mod___transformer_blocks_11_feed_forward_w_1 = self.L__mod___transformer_blocks_11_feed_forward_w_1(add_72);  add_72 = None
    l__mod___transformer_blocks_11_feed_forward_activation = self.L__mod___transformer_blocks_11_feed_forward_activation(l__mod___transformer_blocks_11_feed_forward_w_1);  l__mod___transformer_blocks_11_feed_forward_w_1 = None
    l__mod___transformer_blocks_11_feed_forward_dropout = self.L__mod___transformer_blocks_11_feed_forward_dropout(l__mod___transformer_blocks_11_feed_forward_activation);  l__mod___transformer_blocks_11_feed_forward_activation = None
    l__mod___transformer_blocks_11_feed_forward_w_2 = self.L__mod___transformer_blocks_11_feed_forward_w_2(l__mod___transformer_blocks_11_feed_forward_dropout);  l__mod___transformer_blocks_11_feed_forward_dropout = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    l__mod___transformer_blocks_11_output_sublayer_dropout = self.L__mod___transformer_blocks_11_output_sublayer_dropout(l__mod___transformer_blocks_11_feed_forward_w_2);  l__mod___transformer_blocks_11_feed_forward_w_2 = None
    x_96 = x_95 + l__mod___transformer_blocks_11_output_sublayer_dropout;  x_95 = l__mod___transformer_blocks_11_output_sublayer_dropout = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:49, code: return self.dropout(x)
    x_97 = self.L__mod___transformer_blocks_11_dropout(x_96);  x_96 = None
    return (x_97, mask)
    