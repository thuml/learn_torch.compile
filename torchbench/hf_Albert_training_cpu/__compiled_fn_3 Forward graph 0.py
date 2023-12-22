from __future__ import annotations



def forward(self, primals_1: "f32[30000, 128]", primals_2: "f32[2, 128]", primals_3: "f32[512, 128]", primals_4: "f32[128]", primals_5: "f32[128]", primals_6: "f32[768, 128]", primals_7: "f32[768]", primals_8: "f32[768, 768]", primals_9: "f32[768]", primals_10: "f32[768, 768]", primals_11: "f32[768]", primals_12: "f32[768, 768]", primals_13: "f32[768]", primals_14: "f32[768, 768]", primals_15: "f32[768]", primals_16: "f32[768]", primals_17: "f32[768]", primals_18: "f32[3072, 768]", primals_19: "f32[3072]", primals_20: "f32[768, 3072]", primals_21: "f32[768]", primals_22: "f32[768]", primals_23: "f32[768]", primals_24: "f32[128, 768]", primals_25: "f32[128]", primals_26: "f32[128]", primals_27: "f32[128]", primals_28: "f32[30000, 128]", primals_29: "f32[30000]", primals_30: "i64[1, 512]", primals_31: "i64[1, 512]", primals_32: "i64[4, 512]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:718, code: buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
    slice_1: "i64[1, 512]" = torch.ops.aten.slice.Tensor(primals_30, 0, 0, 9223372036854775807);  primals_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:719, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
    expand: "i64[4, 512]" = torch.ops.aten.expand.default(slice_1, [4, 512]);  slice_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:236, code: position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
    slice_2: "i64[1, 512]" = torch.ops.aten.slice.Tensor(primals_31, 0, 0, 9223372036854775807);  primals_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:250, code: inputs_embeds = self.word_embeddings(input_ids)
    embedding: "f32[4, 512, 128]" = torch.ops.aten.embedding.default(primals_1, primals_32, 0);  primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:251, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    embedding_1: "f32[4, 512, 128]" = torch.ops.aten.embedding.default(primals_2, expand);  primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:253, code: embeddings = inputs_embeds + token_type_embeddings
    add: "f32[4, 512, 128]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:255, code: position_embeddings = self.position_embeddings(position_ids)
    embedding_2: "f32[1, 512, 128]" = torch.ops.aten.embedding.default(primals_3, slice_2);  primals_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:256, code: embeddings += position_embeddings
    add_1: "f32[4, 512, 128]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:257, code: embeddings = self.LayerNorm(embeddings)
    var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
    getitem: "f32[4, 512, 1]" = var_mean[0]
    getitem_1: "f32[4, 512, 1]" = var_mean[1];  var_mean = None
    add_2: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
    rsqrt: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
    sub_1: "f32[4, 512, 128]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
    mul_1: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = None
    mul_2: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul_1, primals_4)
    add_3: "f32[4, 512, 128]" = torch.ops.aten.add.Tensor(mul_2, primals_5);  mul_2 = primals_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:258, code: embeddings = self.dropout(embeddings)
    clone: "f32[4, 512, 128]" = torch.ops.aten.clone.default(add_3);  add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:467, code: hidden_states = self.embedding_hidden_mapping_in(hidden_states)
    view: "f32[2048, 128]" = torch.ops.aten.view.default(clone, [2048, 128]);  clone = None
    permute: "f32[128, 768]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
    addmm: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_7, view, permute);  primals_7 = None
    view_1: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm, [4, 512, 768]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_2: "f32[2048, 768]" = torch.ops.aten.view.default(view_1, [2048, 768])
    permute_1: "f32[768, 768]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
    addmm_1: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_9, view_2, permute_1)
    view_3: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_1, [4, 512, 768]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
    addmm_2: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_11, view_2, permute_2)
    view_5: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_2, [4, 512, 768]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    permute_3: "f32[768, 768]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
    addmm_3: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_13, view_2, permute_3)
    view_7: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_3, [4, 512, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_8: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_3, [4, 512, 12, 64]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_4: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_9: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_5, [4, 512, 12, 64]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_5: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_10: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_7, [4, 512, 12, 64]);  view_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_6: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_7: "f32[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_5, [0, 1, 3, 2]);  permute_5 = None
    expand_1: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_4, [4, 12, 512, 64]);  permute_4 = None
    clone_1: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_11: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_1, [48, 512, 64]);  clone_1 = None
    expand_2: "f32[4, 12, 64, 512]" = torch.ops.aten.expand.default(permute_7, [4, 12, 64, 512]);  permute_7 = None
    clone_2: "f32[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
    view_12: "f32[48, 64, 512]" = torch.ops.aten.view.default(clone_2, [48, 64, 512]);  clone_2 = None
    bmm: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_11, view_12)
    view_13: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm, [4, 12, 512, 512]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:336, code: attention_scores = attention_scores + attention_mask
    div: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_13, 8.0);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(div, [-1], True)
    sub_2: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(div, amax);  div = amax = None
    exp: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
    sum_1: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_1: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(div_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    clone_3: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_1);  div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_3: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_3, [4, 12, 512, 512]);  clone_3 = None
    view_14: "f32[48, 512, 512]" = torch.ops.aten.view.default(expand_3, [48, 512, 512]);  expand_3 = None
    expand_4: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_6, [4, 12, 512, 64]);  permute_6 = None
    clone_4: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
    view_15: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_4, [48, 512, 64]);  clone_4 = None
    bmm_1: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_14, view_15)
    view_16: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_1, [4, 12, 512, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    permute_8: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
    clone_5: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
    view_17: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_5, [4, 512, 768]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_18: "f32[2048, 768]" = torch.ops.aten.view.default(view_17, [2048, 768]);  view_17 = None
    permute_9: "f32[768, 768]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
    addmm_4: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_15, view_18, permute_9)
    view_19: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_4, [4, 512, 768]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:369, code: projected_context_layer_dropout = self.output_dropout(projected_context_layer)
    clone_6: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_19);  view_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_5: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_1, clone_6);  view_1 = clone_6 = None
    var_mean_1 = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
    getitem_2: "f32[4, 512, 1]" = var_mean_1[0]
    getitem_3: "f32[4, 512, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
    rsqrt_1: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_3: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_5, getitem_3);  add_5 = getitem_3 = None
    mul_3: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = None
    mul_4: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_3, primals_16)
    add_7: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_4, primals_17);  mul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_20: "f32[2048, 768]" = torch.ops.aten.view.default(add_7, [2048, 768])
    permute_10: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
    addmm_5: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_19, view_20, permute_10)
    view_21: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_5, [4, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_5: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_21, 0.5)
    pow_1: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_21, 3.0)
    mul_6: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
    add_8: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(view_21, mul_6);  view_21 = mul_6 = None
    mul_7: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_8, 0.7978845608028654);  add_8 = None
    tanh: "f32[4, 512, 3072]" = torch.ops.aten.tanh.default(mul_7);  mul_7 = None
    add_9: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0)
    mul_8: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_5, add_9);  mul_5 = add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_22: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_8, [2048, 3072]);  mul_8 = None
    permute_11: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
    addmm_6: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_21, view_22, permute_11)
    view_23: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_6, [4, 512, 768]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_10: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_23, add_7);  view_23 = add_7 = None
    var_mean_2 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
    getitem_4: "f32[4, 512, 1]" = var_mean_2[0]
    getitem_5: "f32[4, 512, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
    rsqrt_2: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_4: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_10, getitem_5);  add_10 = getitem_5 = None
    mul_9: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = None
    mul_10: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_9, primals_22)
    add_12: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_10, primals_23);  mul_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_24: "f32[2048, 768]" = torch.ops.aten.view.default(add_12, [2048, 768])
    addmm_7: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_9, view_24, permute_1)
    view_25: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_7, [4, 512, 768]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    addmm_8: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_11, view_24, permute_2)
    view_27: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_8, [4, 512, 768]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    addmm_9: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_13, view_24, permute_3)
    view_29: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_9, [4, 512, 768]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_30: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_25, [4, 512, 12, 64]);  view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_15: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_31: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_27, [4, 512, 12, 64]);  view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_16: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_32: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_29, [4, 512, 12, 64]);  view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_17: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_18: "f32[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_16, [0, 1, 3, 2]);  permute_16 = None
    expand_5: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_15, [4, 12, 512, 64]);  permute_15 = None
    clone_7: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
    view_33: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_7, [48, 512, 64]);  clone_7 = None
    expand_6: "f32[4, 12, 64, 512]" = torch.ops.aten.expand.default(permute_18, [4, 12, 64, 512]);  permute_18 = None
    clone_8: "f32[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
    view_34: "f32[48, 64, 512]" = torch.ops.aten.view.default(clone_8, [48, 64, 512]);  clone_8 = None
    bmm_2: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_33, view_34)
    view_35: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_2, [4, 12, 512, 512]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:336, code: attention_scores = attention_scores + attention_mask
    div_2: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_35, 8.0);  view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_1: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(div_2, [-1], True)
    sub_5: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(div_2, amax_1);  div_2 = amax_1 = None
    exp_1: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
    sum_2: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_3: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_2: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(div_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    clone_9: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_3);  div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_7: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_9, [4, 12, 512, 512]);  clone_9 = None
    view_36: "f32[48, 512, 512]" = torch.ops.aten.view.default(expand_7, [48, 512, 512]);  expand_7 = None
    expand_8: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_17, [4, 12, 512, 64]);  permute_17 = None
    clone_10: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
    view_37: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_10, [48, 512, 64]);  clone_10 = None
    bmm_3: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_36, view_37)
    view_38: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_3, [4, 12, 512, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    permute_19: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
    clone_11: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
    view_39: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_11, [4, 512, 768]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_40: "f32[2048, 768]" = torch.ops.aten.view.default(view_39, [2048, 768]);  view_39 = None
    addmm_10: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_15, view_40, permute_9)
    view_41: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_10, [4, 512, 768]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:369, code: projected_context_layer_dropout = self.output_dropout(projected_context_layer)
    clone_12: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_41);  view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_14: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_12, clone_12);  add_12 = clone_12 = None
    var_mean_3 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
    getitem_6: "f32[4, 512, 1]" = var_mean_3[0]
    getitem_7: "f32[4, 512, 1]" = var_mean_3[1];  var_mean_3 = None
    add_15: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
    rsqrt_3: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
    sub_6: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_14, getitem_7);  add_14 = getitem_7 = None
    mul_11: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_3);  sub_6 = None
    mul_12: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_11, primals_16)
    add_16: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_12, primals_17);  mul_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_42: "f32[2048, 768]" = torch.ops.aten.view.default(add_16, [2048, 768])
    addmm_11: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_19, view_42, permute_10)
    view_43: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_11, [4, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_13: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_43, 0.5)
    pow_2: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_43, 3.0)
    mul_14: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
    add_17: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(view_43, mul_14);  view_43 = mul_14 = None
    mul_15: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_17, 0.7978845608028654);  add_17 = None
    tanh_1: "f32[4, 512, 3072]" = torch.ops.aten.tanh.default(mul_15);  mul_15 = None
    add_18: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_1, 1.0)
    mul_16: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_13, add_18);  mul_13 = add_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_44: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_16, [2048, 3072]);  mul_16 = None
    addmm_12: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_21, view_44, permute_11)
    view_45: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_12, [4, 512, 768]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_19: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_45, add_16);  view_45 = add_16 = None
    var_mean_4 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
    getitem_8: "f32[4, 512, 1]" = var_mean_4[0]
    getitem_9: "f32[4, 512, 1]" = var_mean_4[1];  var_mean_4 = None
    add_20: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
    rsqrt_4: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
    sub_7: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_9);  add_19 = getitem_9 = None
    mul_17: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_4);  sub_7 = None
    mul_18: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_17, primals_22)
    add_21: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_18, primals_23);  mul_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_46: "f32[2048, 768]" = torch.ops.aten.view.default(add_21, [2048, 768])
    addmm_13: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_9, view_46, permute_1)
    view_47: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_13, [4, 512, 768]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    addmm_14: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_11, view_46, permute_2)
    view_49: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_14, [4, 512, 768]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    addmm_15: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_13, view_46, permute_3)
    view_51: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_15, [4, 512, 768]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_52: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_47, [4, 512, 12, 64]);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_26: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_53: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_49, [4, 512, 12, 64]);  view_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_27: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_54: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_51, [4, 512, 12, 64]);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_28: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_29: "f32[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_27, [0, 1, 3, 2]);  permute_27 = None
    expand_9: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_26, [4, 12, 512, 64]);  permute_26 = None
    clone_13: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    view_55: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_13, [48, 512, 64]);  clone_13 = None
    expand_10: "f32[4, 12, 64, 512]" = torch.ops.aten.expand.default(permute_29, [4, 12, 64, 512]);  permute_29 = None
    clone_14: "f32[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
    view_56: "f32[48, 64, 512]" = torch.ops.aten.view.default(clone_14, [48, 64, 512]);  clone_14 = None
    bmm_4: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_55, view_56)
    view_57: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_4, [4, 12, 512, 512]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:336, code: attention_scores = attention_scores + attention_mask
    div_4: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_57, 8.0);  view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_2: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(div_4, [-1], True)
    sub_8: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(div_4, amax_2);  div_4 = amax_2 = None
    exp_2: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
    sum_3: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_5: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_4: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(div_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    clone_15: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_5);  div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_11: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_15, [4, 12, 512, 512]);  clone_15 = None
    view_58: "f32[48, 512, 512]" = torch.ops.aten.view.default(expand_11, [48, 512, 512]);  expand_11 = None
    expand_12: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_28, [4, 12, 512, 64]);  permute_28 = None
    clone_16: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
    view_59: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_16, [48, 512, 64]);  clone_16 = None
    bmm_5: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_58, view_59)
    view_60: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_5, [4, 12, 512, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    permute_30: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
    clone_17: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
    view_61: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_17, [4, 512, 768]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_62: "f32[2048, 768]" = torch.ops.aten.view.default(view_61, [2048, 768]);  view_61 = None
    addmm_16: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_15, view_62, permute_9)
    view_63: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_16, [4, 512, 768]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:369, code: projected_context_layer_dropout = self.output_dropout(projected_context_layer)
    clone_18: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_63);  view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_23: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_21, clone_18);  add_21 = clone_18 = None
    var_mean_5 = torch.ops.aten.var_mean.correction(add_23, [2], correction = 0, keepdim = True)
    getitem_10: "f32[4, 512, 1]" = var_mean_5[0]
    getitem_11: "f32[4, 512, 1]" = var_mean_5[1];  var_mean_5 = None
    add_24: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
    rsqrt_5: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
    sub_9: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_23, getitem_11);  add_23 = getitem_11 = None
    mul_19: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_5);  sub_9 = None
    mul_20: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_19, primals_16)
    add_25: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_20, primals_17);  mul_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_64: "f32[2048, 768]" = torch.ops.aten.view.default(add_25, [2048, 768])
    addmm_17: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_19, view_64, permute_10)
    view_65: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_17, [4, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_21: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_65, 0.5)
    pow_3: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_65, 3.0)
    mul_22: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
    add_26: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(view_65, mul_22);  view_65 = mul_22 = None
    mul_23: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_26, 0.7978845608028654);  add_26 = None
    tanh_2: "f32[4, 512, 3072]" = torch.ops.aten.tanh.default(mul_23);  mul_23 = None
    add_27: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_2, 1.0)
    mul_24: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_21, add_27);  mul_21 = add_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_66: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_24, [2048, 3072]);  mul_24 = None
    addmm_18: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_21, view_66, permute_11)
    view_67: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_18, [4, 512, 768]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_28: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_67, add_25);  view_67 = add_25 = None
    var_mean_6 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
    getitem_12: "f32[4, 512, 1]" = var_mean_6[0]
    getitem_13: "f32[4, 512, 1]" = var_mean_6[1];  var_mean_6 = None
    add_29: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
    rsqrt_6: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
    sub_10: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_28, getitem_13);  add_28 = getitem_13 = None
    mul_25: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_6);  sub_10 = None
    mul_26: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_25, primals_22)
    add_30: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_26, primals_23);  mul_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_68: "f32[2048, 768]" = torch.ops.aten.view.default(add_30, [2048, 768])
    addmm_19: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_9, view_68, permute_1)
    view_69: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_19, [4, 512, 768]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    addmm_20: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_11, view_68, permute_2)
    view_71: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_20, [4, 512, 768]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    addmm_21: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_13, view_68, permute_3)
    view_73: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_21, [4, 512, 768]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_74: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_69, [4, 512, 12, 64]);  view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_37: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_75: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_71, [4, 512, 12, 64]);  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_38: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_76: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_73, [4, 512, 12, 64]);  view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_39: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_40: "f32[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_38, [0, 1, 3, 2]);  permute_38 = None
    expand_13: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_37, [4, 12, 512, 64]);  permute_37 = None
    clone_19: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
    view_77: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_19, [48, 512, 64]);  clone_19 = None
    expand_14: "f32[4, 12, 64, 512]" = torch.ops.aten.expand.default(permute_40, [4, 12, 64, 512]);  permute_40 = None
    clone_20: "f32[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
    view_78: "f32[48, 64, 512]" = torch.ops.aten.view.default(clone_20, [48, 64, 512]);  clone_20 = None
    bmm_6: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_77, view_78)
    view_79: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_6, [4, 12, 512, 512]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:336, code: attention_scores = attention_scores + attention_mask
    div_6: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_79, 8.0);  view_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_3: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(div_6, [-1], True)
    sub_11: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(div_6, amax_3);  div_6 = amax_3 = None
    exp_3: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
    sum_4: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_7: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    alias_6: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(div_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    clone_21: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_7);  div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_15: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_21, [4, 12, 512, 512]);  clone_21 = None
    view_80: "f32[48, 512, 512]" = torch.ops.aten.view.default(expand_15, [48, 512, 512]);  expand_15 = None
    expand_16: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_39, [4, 12, 512, 64]);  permute_39 = None
    clone_22: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
    view_81: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_22, [48, 512, 64]);  clone_22 = None
    bmm_7: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_80, view_81)
    view_82: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_7, [4, 12, 512, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    permute_41: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
    clone_23: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None
    view_83: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_23, [4, 512, 768]);  clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_84: "f32[2048, 768]" = torch.ops.aten.view.default(view_83, [2048, 768]);  view_83 = None
    addmm_22: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_15, view_84, permute_9)
    view_85: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_22, [4, 512, 768]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:369, code: projected_context_layer_dropout = self.output_dropout(projected_context_layer)
    clone_24: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_85);  view_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_32: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_30, clone_24);  add_30 = clone_24 = None
    var_mean_7 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
    getitem_14: "f32[4, 512, 1]" = var_mean_7[0]
    getitem_15: "f32[4, 512, 1]" = var_mean_7[1];  var_mean_7 = None
    add_33: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
    rsqrt_7: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
    sub_12: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_32, getitem_15);  add_32 = getitem_15 = None
    mul_27: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_7);  sub_12 = None
    mul_28: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_27, primals_16)
    add_34: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_28, primals_17);  mul_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_86: "f32[2048, 768]" = torch.ops.aten.view.default(add_34, [2048, 768])
    addmm_23: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_19, view_86, permute_10)
    view_87: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_23, [4, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_29: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_87, 0.5)
    pow_4: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_87, 3.0)
    mul_30: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
    add_35: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(view_87, mul_30);  view_87 = mul_30 = None
    mul_31: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_35, 0.7978845608028654);  add_35 = None
    tanh_3: "f32[4, 512, 3072]" = torch.ops.aten.tanh.default(mul_31);  mul_31 = None
    add_36: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_3, 1.0)
    mul_32: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_29, add_36);  mul_29 = add_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_88: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_32, [2048, 3072]);  mul_32 = None
    addmm_24: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_21, view_88, permute_11)
    view_89: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_24, [4, 512, 768]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_37: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_89, add_34);  view_89 = add_34 = None
    var_mean_8 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
    getitem_16: "f32[4, 512, 1]" = var_mean_8[0]
    getitem_17: "f32[4, 512, 1]" = var_mean_8[1];  var_mean_8 = None
    add_38: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
    rsqrt_8: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    sub_13: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_37, getitem_17);  add_37 = getitem_17 = None
    mul_33: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_8);  sub_13 = None
    mul_34: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_33, primals_22)
    add_39: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_34, primals_23);  mul_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_90: "f32[2048, 768]" = torch.ops.aten.view.default(add_39, [2048, 768])
    addmm_25: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_9, view_90, permute_1)
    view_91: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_25, [4, 512, 768]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    addmm_26: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_11, view_90, permute_2)
    view_93: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_26, [4, 512, 768]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    addmm_27: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_13, view_90, permute_3)
    view_95: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_27, [4, 512, 768]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_96: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_91, [4, 512, 12, 64]);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_48: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_97: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_93, [4, 512, 12, 64]);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_49: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_97, [0, 2, 1, 3]);  view_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_98: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_95, [4, 512, 12, 64]);  view_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_50: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_51: "f32[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_49, [0, 1, 3, 2]);  permute_49 = None
    expand_17: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_48, [4, 12, 512, 64]);  permute_48 = None
    clone_25: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
    view_99: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_25, [48, 512, 64]);  clone_25 = None
    expand_18: "f32[4, 12, 64, 512]" = torch.ops.aten.expand.default(permute_51, [4, 12, 64, 512]);  permute_51 = None
    clone_26: "f32[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
    view_100: "f32[48, 64, 512]" = torch.ops.aten.view.default(clone_26, [48, 64, 512]);  clone_26 = None
    bmm_8: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_99, view_100)
    view_101: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_8, [4, 12, 512, 512]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:336, code: attention_scores = attention_scores + attention_mask
    div_8: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_101, 8.0);  view_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_4: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(div_8, [-1], True)
    sub_14: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(div_8, amax_4);  div_8 = amax_4 = None
    exp_4: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
    sum_5: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_9: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    alias_8: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(div_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    clone_27: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_9);  div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_19: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_27, [4, 12, 512, 512]);  clone_27 = None
    view_102: "f32[48, 512, 512]" = torch.ops.aten.view.default(expand_19, [48, 512, 512]);  expand_19 = None
    expand_20: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_50, [4, 12, 512, 64]);  permute_50 = None
    clone_28: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
    view_103: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_28, [48, 512, 64]);  clone_28 = None
    bmm_9: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_102, view_103)
    view_104: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_9, [4, 12, 512, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    permute_52: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
    clone_29: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None
    view_105: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_29, [4, 512, 768]);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_106: "f32[2048, 768]" = torch.ops.aten.view.default(view_105, [2048, 768]);  view_105 = None
    addmm_28: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_15, view_106, permute_9)
    view_107: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_28, [4, 512, 768]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:369, code: projected_context_layer_dropout = self.output_dropout(projected_context_layer)
    clone_30: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_107);  view_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_41: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_39, clone_30);  add_39 = clone_30 = None
    var_mean_9 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
    getitem_18: "f32[4, 512, 1]" = var_mean_9[0]
    getitem_19: "f32[4, 512, 1]" = var_mean_9[1];  var_mean_9 = None
    add_42: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
    rsqrt_9: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_15: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_41, getitem_19);  add_41 = getitem_19 = None
    mul_35: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_9);  sub_15 = None
    mul_36: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_35, primals_16)
    add_43: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_36, primals_17);  mul_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_108: "f32[2048, 768]" = torch.ops.aten.view.default(add_43, [2048, 768])
    addmm_29: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_19, view_108, permute_10)
    view_109: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_29, [4, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_37: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_109, 0.5)
    pow_5: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_109, 3.0)
    mul_38: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
    add_44: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(view_109, mul_38);  view_109 = mul_38 = None
    mul_39: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_44, 0.7978845608028654);  add_44 = None
    tanh_4: "f32[4, 512, 3072]" = torch.ops.aten.tanh.default(mul_39);  mul_39 = None
    add_45: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_4, 1.0)
    mul_40: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_37, add_45);  mul_37 = add_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_110: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_40, [2048, 3072]);  mul_40 = None
    addmm_30: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_21, view_110, permute_11)
    view_111: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_30, [4, 512, 768]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_46: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_111, add_43);  view_111 = add_43 = None
    var_mean_10 = torch.ops.aten.var_mean.correction(add_46, [2], correction = 0, keepdim = True)
    getitem_20: "f32[4, 512, 1]" = var_mean_10[0]
    getitem_21: "f32[4, 512, 1]" = var_mean_10[1];  var_mean_10 = None
    add_47: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
    rsqrt_10: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    sub_16: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_46, getitem_21);  add_46 = getitem_21 = None
    mul_41: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = None
    mul_42: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_41, primals_22)
    add_48: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_42, primals_23);  mul_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_112: "f32[2048, 768]" = torch.ops.aten.view.default(add_48, [2048, 768])
    addmm_31: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_9, view_112, permute_1)
    view_113: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_31, [4, 512, 768]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    addmm_32: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_11, view_112, permute_2)
    view_115: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_32, [4, 512, 768]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    addmm_33: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_13, view_112, permute_3)
    view_117: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_33, [4, 512, 768]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_118: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_113, [4, 512, 12, 64]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_59: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_119: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_115, [4, 512, 12, 64]);  view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_60: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_120: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_117, [4, 512, 12, 64]);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_61: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_62: "f32[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_60, [0, 1, 3, 2]);  permute_60 = None
    expand_21: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_59, [4, 12, 512, 64]);  permute_59 = None
    clone_31: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    view_121: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_31, [48, 512, 64]);  clone_31 = None
    expand_22: "f32[4, 12, 64, 512]" = torch.ops.aten.expand.default(permute_62, [4, 12, 64, 512]);  permute_62 = None
    clone_32: "f32[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
    view_122: "f32[48, 64, 512]" = torch.ops.aten.view.default(clone_32, [48, 64, 512]);  clone_32 = None
    bmm_10: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_121, view_122)
    view_123: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_10, [4, 12, 512, 512]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:336, code: attention_scores = attention_scores + attention_mask
    div_10: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_123, 8.0);  view_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_5: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(div_10, [-1], True)
    sub_17: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(div_10, amax_5);  div_10 = amax_5 = None
    exp_5: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
    sum_6: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_11: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    alias_10: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(div_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    clone_33: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_11);  div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_23: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_33, [4, 12, 512, 512]);  clone_33 = None
    view_124: "f32[48, 512, 512]" = torch.ops.aten.view.default(expand_23, [48, 512, 512]);  expand_23 = None
    expand_24: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_61, [4, 12, 512, 64]);  permute_61 = None
    clone_34: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
    view_125: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_34, [48, 512, 64]);  clone_34 = None
    bmm_11: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_124, view_125)
    view_126: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_11, [4, 12, 512, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    permute_63: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
    clone_35: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None
    view_127: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_35, [4, 512, 768]);  clone_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_128: "f32[2048, 768]" = torch.ops.aten.view.default(view_127, [2048, 768]);  view_127 = None
    addmm_34: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_15, view_128, permute_9)
    view_129: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_34, [4, 512, 768]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:369, code: projected_context_layer_dropout = self.output_dropout(projected_context_layer)
    clone_36: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_129);  view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_50: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_48, clone_36);  add_48 = clone_36 = None
    var_mean_11 = torch.ops.aten.var_mean.correction(add_50, [2], correction = 0, keepdim = True)
    getitem_22: "f32[4, 512, 1]" = var_mean_11[0]
    getitem_23: "f32[4, 512, 1]" = var_mean_11[1];  var_mean_11 = None
    add_51: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
    rsqrt_11: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
    sub_18: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_50, getitem_23);  add_50 = getitem_23 = None
    mul_43: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_11);  sub_18 = None
    mul_44: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_43, primals_16)
    add_52: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_44, primals_17);  mul_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_130: "f32[2048, 768]" = torch.ops.aten.view.default(add_52, [2048, 768])
    addmm_35: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_19, view_130, permute_10)
    view_131: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_35, [4, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_45: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_131, 0.5)
    pow_6: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_131, 3.0)
    mul_46: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
    add_53: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(view_131, mul_46);  view_131 = mul_46 = None
    mul_47: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_53, 0.7978845608028654);  add_53 = None
    tanh_5: "f32[4, 512, 3072]" = torch.ops.aten.tanh.default(mul_47);  mul_47 = None
    add_54: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_5, 1.0)
    mul_48: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_45, add_54);  mul_45 = add_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_132: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_48, [2048, 3072]);  mul_48 = None
    addmm_36: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_21, view_132, permute_11)
    view_133: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_36, [4, 512, 768]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_55: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_133, add_52);  view_133 = add_52 = None
    var_mean_12 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
    getitem_24: "f32[4, 512, 1]" = var_mean_12[0]
    getitem_25: "f32[4, 512, 1]" = var_mean_12[1];  var_mean_12 = None
    add_56: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
    rsqrt_12: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_19: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_55, getitem_25);  add_55 = getitem_25 = None
    mul_49: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_12);  sub_19 = None
    mul_50: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_49, primals_22)
    add_57: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_50, primals_23);  mul_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_134: "f32[2048, 768]" = torch.ops.aten.view.default(add_57, [2048, 768])
    addmm_37: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_9, view_134, permute_1)
    view_135: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_37, [4, 512, 768]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    addmm_38: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_11, view_134, permute_2)
    view_137: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_38, [4, 512, 768]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    addmm_39: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_13, view_134, permute_3)
    view_139: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_39, [4, 512, 768]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_140: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_135, [4, 512, 12, 64]);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_70: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_141: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_137, [4, 512, 12, 64]);  view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_71: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_141, [0, 2, 1, 3]);  view_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_142: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_139, [4, 512, 12, 64]);  view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_72: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_73: "f32[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_71, [0, 1, 3, 2]);  permute_71 = None
    expand_25: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_70, [4, 12, 512, 64]);  permute_70 = None
    clone_37: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
    view_143: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_37, [48, 512, 64]);  clone_37 = None
    expand_26: "f32[4, 12, 64, 512]" = torch.ops.aten.expand.default(permute_73, [4, 12, 64, 512]);  permute_73 = None
    clone_38: "f32[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
    view_144: "f32[48, 64, 512]" = torch.ops.aten.view.default(clone_38, [48, 64, 512]);  clone_38 = None
    bmm_12: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_143, view_144)
    view_145: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_12, [4, 12, 512, 512]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:336, code: attention_scores = attention_scores + attention_mask
    div_12: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_145, 8.0);  view_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_6: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(div_12, [-1], True)
    sub_20: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(div_12, amax_6);  div_12 = amax_6 = None
    exp_6: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
    sum_7: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_13: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    alias_12: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(div_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    clone_39: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_13);  div_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_27: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_39, [4, 12, 512, 512]);  clone_39 = None
    view_146: "f32[48, 512, 512]" = torch.ops.aten.view.default(expand_27, [48, 512, 512]);  expand_27 = None
    expand_28: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_72, [4, 12, 512, 64]);  permute_72 = None
    clone_40: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
    view_147: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_40, [48, 512, 64]);  clone_40 = None
    bmm_13: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_146, view_147)
    view_148: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_13, [4, 12, 512, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    permute_74: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
    clone_41: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_74, memory_format = torch.contiguous_format);  permute_74 = None
    view_149: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_41, [4, 512, 768]);  clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_150: "f32[2048, 768]" = torch.ops.aten.view.default(view_149, [2048, 768]);  view_149 = None
    addmm_40: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_15, view_150, permute_9)
    view_151: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_40, [4, 512, 768]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:369, code: projected_context_layer_dropout = self.output_dropout(projected_context_layer)
    clone_42: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_151);  view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_59: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_57, clone_42);  add_57 = clone_42 = None
    var_mean_13 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
    getitem_26: "f32[4, 512, 1]" = var_mean_13[0]
    getitem_27: "f32[4, 512, 1]" = var_mean_13[1];  var_mean_13 = None
    add_60: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
    rsqrt_13: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_21: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_27);  add_59 = getitem_27 = None
    mul_51: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_13);  sub_21 = None
    mul_52: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_51, primals_16)
    add_61: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_52, primals_17);  mul_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_152: "f32[2048, 768]" = torch.ops.aten.view.default(add_61, [2048, 768])
    addmm_41: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_19, view_152, permute_10)
    view_153: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_41, [4, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_53: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_153, 0.5)
    pow_7: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_153, 3.0)
    mul_54: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
    add_62: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(view_153, mul_54);  view_153 = mul_54 = None
    mul_55: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_62, 0.7978845608028654);  add_62 = None
    tanh_6: "f32[4, 512, 3072]" = torch.ops.aten.tanh.default(mul_55);  mul_55 = None
    add_63: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_6, 1.0)
    mul_56: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_53, add_63);  mul_53 = add_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_154: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_56, [2048, 3072]);  mul_56 = None
    addmm_42: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_21, view_154, permute_11)
    view_155: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_42, [4, 512, 768]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_64: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_155, add_61);  view_155 = add_61 = None
    var_mean_14 = torch.ops.aten.var_mean.correction(add_64, [2], correction = 0, keepdim = True)
    getitem_28: "f32[4, 512, 1]" = var_mean_14[0]
    getitem_29: "f32[4, 512, 1]" = var_mean_14[1];  var_mean_14 = None
    add_65: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
    rsqrt_14: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
    sub_22: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_64, getitem_29);  add_64 = getitem_29 = None
    mul_57: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_14);  sub_22 = None
    mul_58: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_57, primals_22)
    add_66: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_58, primals_23);  mul_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_156: "f32[2048, 768]" = torch.ops.aten.view.default(add_66, [2048, 768])
    addmm_43: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_9, view_156, permute_1)
    view_157: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_43, [4, 512, 768]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    addmm_44: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_11, view_156, permute_2)
    view_159: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_44, [4, 512, 768]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    addmm_45: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_13, view_156, permute_3)
    view_161: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_45, [4, 512, 768]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_162: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_157, [4, 512, 12, 64]);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_81: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_163: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_159, [4, 512, 12, 64]);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_82: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_163, [0, 2, 1, 3]);  view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_164: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_161, [4, 512, 12, 64]);  view_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_83: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_84: "f32[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_82, [0, 1, 3, 2]);  permute_82 = None
    expand_29: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_81, [4, 12, 512, 64]);  permute_81 = None
    clone_43: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
    view_165: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_43, [48, 512, 64]);  clone_43 = None
    expand_30: "f32[4, 12, 64, 512]" = torch.ops.aten.expand.default(permute_84, [4, 12, 64, 512]);  permute_84 = None
    clone_44: "f32[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
    view_166: "f32[48, 64, 512]" = torch.ops.aten.view.default(clone_44, [48, 64, 512]);  clone_44 = None
    bmm_14: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_165, view_166)
    view_167: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_14, [4, 12, 512, 512]);  bmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:336, code: attention_scores = attention_scores + attention_mask
    div_14: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_167, 8.0);  view_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_7: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(div_14, [-1], True)
    sub_23: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(div_14, amax_7);  div_14 = amax_7 = None
    exp_7: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub_23);  sub_23 = None
    sum_8: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_15: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    alias_14: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(div_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    clone_45: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_15);  div_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_31: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_45, [4, 12, 512, 512]);  clone_45 = None
    view_168: "f32[48, 512, 512]" = torch.ops.aten.view.default(expand_31, [48, 512, 512]);  expand_31 = None
    expand_32: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_83, [4, 12, 512, 64]);  permute_83 = None
    clone_46: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
    view_169: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_46, [48, 512, 64]);  clone_46 = None
    bmm_15: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_168, view_169)
    view_170: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_15, [4, 12, 512, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    permute_85: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
    clone_47: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None
    view_171: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_47, [4, 512, 768]);  clone_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_172: "f32[2048, 768]" = torch.ops.aten.view.default(view_171, [2048, 768]);  view_171 = None
    addmm_46: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_15, view_172, permute_9)
    view_173: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_46, [4, 512, 768]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:369, code: projected_context_layer_dropout = self.output_dropout(projected_context_layer)
    clone_48: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_173);  view_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_68: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_66, clone_48);  add_66 = clone_48 = None
    var_mean_15 = torch.ops.aten.var_mean.correction(add_68, [2], correction = 0, keepdim = True)
    getitem_30: "f32[4, 512, 1]" = var_mean_15[0]
    getitem_31: "f32[4, 512, 1]" = var_mean_15[1];  var_mean_15 = None
    add_69: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
    rsqrt_15: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
    sub_24: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_68, getitem_31);  add_68 = getitem_31 = None
    mul_59: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_15);  sub_24 = None
    mul_60: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_59, primals_16)
    add_70: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_60, primals_17);  mul_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_174: "f32[2048, 768]" = torch.ops.aten.view.default(add_70, [2048, 768])
    addmm_47: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_19, view_174, permute_10)
    view_175: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_47, [4, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_61: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_175, 0.5)
    pow_8: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_175, 3.0)
    mul_62: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
    add_71: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(view_175, mul_62);  view_175 = mul_62 = None
    mul_63: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_71, 0.7978845608028654);  add_71 = None
    tanh_7: "f32[4, 512, 3072]" = torch.ops.aten.tanh.default(mul_63);  mul_63 = None
    add_72: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_7, 1.0)
    mul_64: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_61, add_72);  mul_61 = add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_176: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_64, [2048, 3072]);  mul_64 = None
    addmm_48: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_21, view_176, permute_11)
    view_177: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_48, [4, 512, 768]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_73: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_177, add_70);  view_177 = add_70 = None
    var_mean_16 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
    getitem_32: "f32[4, 512, 1]" = var_mean_16[0]
    getitem_33: "f32[4, 512, 1]" = var_mean_16[1];  var_mean_16 = None
    add_74: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
    rsqrt_16: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_25: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_73, getitem_33);  add_73 = getitem_33 = None
    mul_65: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_16);  sub_25 = None
    mul_66: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_65, primals_22)
    add_75: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_66, primals_23);  mul_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_178: "f32[2048, 768]" = torch.ops.aten.view.default(add_75, [2048, 768])
    addmm_49: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_9, view_178, permute_1)
    view_179: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_49, [4, 512, 768]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    addmm_50: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_11, view_178, permute_2)
    view_181: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_50, [4, 512, 768]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    addmm_51: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_13, view_178, permute_3)
    view_183: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_51, [4, 512, 768]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_184: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_179, [4, 512, 12, 64]);  view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_92: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_185: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_181, [4, 512, 12, 64]);  view_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_93: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_186: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_183, [4, 512, 12, 64]);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_94: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_95: "f32[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_93, [0, 1, 3, 2]);  permute_93 = None
    expand_33: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_92, [4, 12, 512, 64]);  permute_92 = None
    clone_49: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
    view_187: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_49, [48, 512, 64]);  clone_49 = None
    expand_34: "f32[4, 12, 64, 512]" = torch.ops.aten.expand.default(permute_95, [4, 12, 64, 512]);  permute_95 = None
    clone_50: "f32[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
    view_188: "f32[48, 64, 512]" = torch.ops.aten.view.default(clone_50, [48, 64, 512]);  clone_50 = None
    bmm_16: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_187, view_188)
    view_189: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_16, [4, 12, 512, 512]);  bmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:336, code: attention_scores = attention_scores + attention_mask
    div_16: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_189, 8.0);  view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_8: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(div_16, [-1], True)
    sub_26: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(div_16, amax_8);  div_16 = amax_8 = None
    exp_8: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub_26);  sub_26 = None
    sum_9: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_17: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    alias_16: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(div_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    clone_51: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_17);  div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_35: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_51, [4, 12, 512, 512]);  clone_51 = None
    view_190: "f32[48, 512, 512]" = torch.ops.aten.view.default(expand_35, [48, 512, 512]);  expand_35 = None
    expand_36: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_94, [4, 12, 512, 64]);  permute_94 = None
    clone_52: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
    view_191: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_52, [48, 512, 64]);  clone_52 = None
    bmm_17: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_190, view_191)
    view_192: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_17, [4, 12, 512, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    permute_96: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
    clone_53: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_96, memory_format = torch.contiguous_format);  permute_96 = None
    view_193: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_53, [4, 512, 768]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_194: "f32[2048, 768]" = torch.ops.aten.view.default(view_193, [2048, 768]);  view_193 = None
    addmm_52: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_15, view_194, permute_9)
    view_195: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_52, [4, 512, 768]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:369, code: projected_context_layer_dropout = self.output_dropout(projected_context_layer)
    clone_54: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_195);  view_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_77: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_75, clone_54);  add_75 = clone_54 = None
    var_mean_17 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
    getitem_34: "f32[4, 512, 1]" = var_mean_17[0]
    getitem_35: "f32[4, 512, 1]" = var_mean_17[1];  var_mean_17 = None
    add_78: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
    rsqrt_17: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    sub_27: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_77, getitem_35);  add_77 = getitem_35 = None
    mul_67: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_17);  sub_27 = None
    mul_68: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_67, primals_16)
    add_79: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_68, primals_17);  mul_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_196: "f32[2048, 768]" = torch.ops.aten.view.default(add_79, [2048, 768])
    addmm_53: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_19, view_196, permute_10)
    view_197: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_53, [4, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_69: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_197, 0.5)
    pow_9: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_197, 3.0)
    mul_70: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
    add_80: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(view_197, mul_70);  view_197 = mul_70 = None
    mul_71: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_80, 0.7978845608028654);  add_80 = None
    tanh_8: "f32[4, 512, 3072]" = torch.ops.aten.tanh.default(mul_71);  mul_71 = None
    add_81: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_8, 1.0)
    mul_72: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_69, add_81);  mul_69 = add_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_198: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_72, [2048, 3072]);  mul_72 = None
    addmm_54: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_21, view_198, permute_11)
    view_199: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_54, [4, 512, 768]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_82: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_199, add_79);  view_199 = add_79 = None
    var_mean_18 = torch.ops.aten.var_mean.correction(add_82, [2], correction = 0, keepdim = True)
    getitem_36: "f32[4, 512, 1]" = var_mean_18[0]
    getitem_37: "f32[4, 512, 1]" = var_mean_18[1];  var_mean_18 = None
    add_83: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
    rsqrt_18: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
    sub_28: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_82, getitem_37);  add_82 = getitem_37 = None
    mul_73: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_18);  sub_28 = None
    mul_74: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_73, primals_22)
    add_84: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_74, primals_23);  mul_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_200: "f32[2048, 768]" = torch.ops.aten.view.default(add_84, [2048, 768])
    addmm_55: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_9, view_200, permute_1)
    view_201: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_55, [4, 512, 768]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    addmm_56: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_11, view_200, permute_2)
    view_203: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_56, [4, 512, 768]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    addmm_57: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_13, view_200, permute_3)
    view_205: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_57, [4, 512, 768]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_206: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_201, [4, 512, 12, 64]);  view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_103: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_207: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_203, [4, 512, 12, 64]);  view_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_104: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_208: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_205, [4, 512, 12, 64]);  view_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_105: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_106: "f32[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_104, [0, 1, 3, 2]);  permute_104 = None
    expand_37: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_103, [4, 12, 512, 64]);  permute_103 = None
    clone_55: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
    view_209: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_55, [48, 512, 64]);  clone_55 = None
    expand_38: "f32[4, 12, 64, 512]" = torch.ops.aten.expand.default(permute_106, [4, 12, 64, 512]);  permute_106 = None
    clone_56: "f32[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
    view_210: "f32[48, 64, 512]" = torch.ops.aten.view.default(clone_56, [48, 64, 512]);  clone_56 = None
    bmm_18: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_209, view_210)
    view_211: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_18, [4, 12, 512, 512]);  bmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:336, code: attention_scores = attention_scores + attention_mask
    div_18: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_211, 8.0);  view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_9: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(div_18, [-1], True)
    sub_29: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(div_18, amax_9);  div_18 = amax_9 = None
    exp_9: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub_29);  sub_29 = None
    sum_10: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_19: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    alias_18: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(div_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    clone_57: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_19);  div_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_39: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_57, [4, 12, 512, 512]);  clone_57 = None
    view_212: "f32[48, 512, 512]" = torch.ops.aten.view.default(expand_39, [48, 512, 512]);  expand_39 = None
    expand_40: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_105, [4, 12, 512, 64]);  permute_105 = None
    clone_58: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
    view_213: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_58, [48, 512, 64]);  clone_58 = None
    bmm_19: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_212, view_213)
    view_214: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_19, [4, 12, 512, 64]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    permute_107: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_214, [0, 2, 1, 3]);  view_214 = None
    clone_59: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_107, memory_format = torch.contiguous_format);  permute_107 = None
    view_215: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_59, [4, 512, 768]);  clone_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_216: "f32[2048, 768]" = torch.ops.aten.view.default(view_215, [2048, 768]);  view_215 = None
    addmm_58: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_15, view_216, permute_9)
    view_217: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_58, [4, 512, 768]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:369, code: projected_context_layer_dropout = self.output_dropout(projected_context_layer)
    clone_60: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_217);  view_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_86: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_84, clone_60);  add_84 = clone_60 = None
    var_mean_19 = torch.ops.aten.var_mean.correction(add_86, [2], correction = 0, keepdim = True)
    getitem_38: "f32[4, 512, 1]" = var_mean_19[0]
    getitem_39: "f32[4, 512, 1]" = var_mean_19[1];  var_mean_19 = None
    add_87: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
    rsqrt_19: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
    sub_30: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_86, getitem_39);  add_86 = getitem_39 = None
    mul_75: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_19);  sub_30 = None
    mul_76: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_75, primals_16)
    add_88: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_76, primals_17);  mul_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_218: "f32[2048, 768]" = torch.ops.aten.view.default(add_88, [2048, 768])
    addmm_59: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_19, view_218, permute_10)
    view_219: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_59, [4, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_77: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_219, 0.5)
    pow_10: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_219, 3.0)
    mul_78: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
    add_89: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(view_219, mul_78);  view_219 = mul_78 = None
    mul_79: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_89, 0.7978845608028654);  add_89 = None
    tanh_9: "f32[4, 512, 3072]" = torch.ops.aten.tanh.default(mul_79);  mul_79 = None
    add_90: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_9, 1.0)
    mul_80: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_77, add_90);  mul_77 = add_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_220: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_80, [2048, 3072]);  mul_80 = None
    addmm_60: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_21, view_220, permute_11)
    view_221: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_60, [4, 512, 768]);  addmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_91: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_221, add_88);  view_221 = add_88 = None
    var_mean_20 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
    getitem_40: "f32[4, 512, 1]" = var_mean_20[0]
    getitem_41: "f32[4, 512, 1]" = var_mean_20[1];  var_mean_20 = None
    add_92: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
    rsqrt_20: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
    sub_31: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_91, getitem_41);  add_91 = getitem_41 = None
    mul_81: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_20);  sub_31 = None
    mul_82: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_81, primals_22)
    add_93: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_82, primals_23);  mul_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_222: "f32[2048, 768]" = torch.ops.aten.view.default(add_93, [2048, 768])
    addmm_61: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_9, view_222, permute_1)
    view_223: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_61, [4, 512, 768]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    addmm_62: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_11, view_222, permute_2)
    view_225: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_62, [4, 512, 768]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    addmm_63: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_13, view_222, permute_3)
    view_227: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_63, [4, 512, 768]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_228: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_223, [4, 512, 12, 64]);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_114: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_229: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_225, [4, 512, 12, 64]);  view_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_115: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_229, [0, 2, 1, 3]);  view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_230: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_227, [4, 512, 12, 64]);  view_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_116: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_230, [0, 2, 1, 3]);  view_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_117: "f32[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_115, [0, 1, 3, 2]);  permute_115 = None
    expand_41: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_114, [4, 12, 512, 64]);  permute_114 = None
    clone_61: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
    view_231: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_61, [48, 512, 64]);  clone_61 = None
    expand_42: "f32[4, 12, 64, 512]" = torch.ops.aten.expand.default(permute_117, [4, 12, 64, 512]);  permute_117 = None
    clone_62: "f32[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
    view_232: "f32[48, 64, 512]" = torch.ops.aten.view.default(clone_62, [48, 64, 512]);  clone_62 = None
    bmm_20: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_231, view_232)
    view_233: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_20, [4, 12, 512, 512]);  bmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:336, code: attention_scores = attention_scores + attention_mask
    div_20: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_233, 8.0);  view_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_10: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(div_20, [-1], True)
    sub_32: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(div_20, amax_10);  div_20 = amax_10 = None
    exp_10: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub_32);  sub_32 = None
    sum_11: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_21: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    alias_20: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(div_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    clone_63: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_21);  div_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_43: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_63, [4, 12, 512, 512]);  clone_63 = None
    view_234: "f32[48, 512, 512]" = torch.ops.aten.view.default(expand_43, [48, 512, 512]);  expand_43 = None
    expand_44: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_116, [4, 12, 512, 64]);  permute_116 = None
    clone_64: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
    view_235: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_64, [48, 512, 64]);  clone_64 = None
    bmm_21: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_234, view_235)
    view_236: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_21, [4, 12, 512, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    permute_118: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
    clone_65: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_118, memory_format = torch.contiguous_format);  permute_118 = None
    view_237: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_65, [4, 512, 768]);  clone_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_238: "f32[2048, 768]" = torch.ops.aten.view.default(view_237, [2048, 768]);  view_237 = None
    addmm_64: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_15, view_238, permute_9)
    view_239: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_64, [4, 512, 768]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:369, code: projected_context_layer_dropout = self.output_dropout(projected_context_layer)
    clone_66: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_239);  view_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_95: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_93, clone_66);  add_93 = clone_66 = None
    var_mean_21 = torch.ops.aten.var_mean.correction(add_95, [2], correction = 0, keepdim = True)
    getitem_42: "f32[4, 512, 1]" = var_mean_21[0]
    getitem_43: "f32[4, 512, 1]" = var_mean_21[1];  var_mean_21 = None
    add_96: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
    rsqrt_21: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
    sub_33: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_95, getitem_43);  add_95 = getitem_43 = None
    mul_83: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_21);  sub_33 = None
    mul_84: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_83, primals_16)
    add_97: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_84, primals_17);  mul_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_240: "f32[2048, 768]" = torch.ops.aten.view.default(add_97, [2048, 768])
    addmm_65: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_19, view_240, permute_10)
    view_241: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_65, [4, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_85: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_241, 0.5)
    pow_11: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_241, 3.0)
    mul_86: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
    add_98: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(view_241, mul_86);  view_241 = mul_86 = None
    mul_87: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_98, 0.7978845608028654);  add_98 = None
    tanh_10: "f32[4, 512, 3072]" = torch.ops.aten.tanh.default(mul_87);  mul_87 = None
    add_99: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_10, 1.0)
    mul_88: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_85, add_99);  mul_85 = add_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_242: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_88, [2048, 3072]);  mul_88 = None
    addmm_66: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_21, view_242, permute_11)
    view_243: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_66, [4, 512, 768]);  addmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_100: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_243, add_97);  view_243 = add_97 = None
    var_mean_22 = torch.ops.aten.var_mean.correction(add_100, [2], correction = 0, keepdim = True)
    getitem_44: "f32[4, 512, 1]" = var_mean_22[0]
    getitem_45: "f32[4, 512, 1]" = var_mean_22[1];  var_mean_22 = None
    add_101: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
    rsqrt_22: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
    sub_34: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_100, getitem_45);  add_100 = getitem_45 = None
    mul_89: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_22);  sub_34 = None
    mul_90: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_89, primals_22)
    add_102: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_90, primals_23);  mul_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_244: "f32[2048, 768]" = torch.ops.aten.view.default(add_102, [2048, 768])
    addmm_67: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_9, view_244, permute_1);  primals_9 = None
    view_245: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_67, [4, 512, 768]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    addmm_68: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_11, view_244, permute_2);  primals_11 = None
    view_247: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_68, [4, 512, 768]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    addmm_69: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_13, view_244, permute_3);  primals_13 = None
    view_249: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_69, [4, 512, 768]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_250: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_245, [4, 512, 12, 64]);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_125: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_251: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_247, [4, 512, 12, 64]);  view_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_126: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_251, [0, 2, 1, 3]);  view_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_252: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_249, [4, 512, 12, 64]);  view_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_127: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_128: "f32[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_126, [0, 1, 3, 2]);  permute_126 = None
    expand_45: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_125, [4, 12, 512, 64]);  permute_125 = None
    clone_67: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
    view_253: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_67, [48, 512, 64]);  clone_67 = None
    expand_46: "f32[4, 12, 64, 512]" = torch.ops.aten.expand.default(permute_128, [4, 12, 64, 512]);  permute_128 = None
    clone_68: "f32[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
    view_254: "f32[48, 64, 512]" = torch.ops.aten.view.default(clone_68, [48, 64, 512]);  clone_68 = None
    bmm_22: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_253, view_254)
    view_255: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_22, [4, 12, 512, 512]);  bmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:336, code: attention_scores = attention_scores + attention_mask
    div_22: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_255, 8.0);  view_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_11: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(div_22, [-1], True)
    sub_35: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(div_22, amax_11);  div_22 = amax_11 = None
    exp_11: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub_35);  sub_35 = None
    sum_12: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_23: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    alias_22: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(div_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    clone_69: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_23);  div_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_47: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_69, [4, 12, 512, 512]);  clone_69 = None
    view_256: "f32[48, 512, 512]" = torch.ops.aten.view.default(expand_47, [48, 512, 512]);  expand_47 = None
    expand_48: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_127, [4, 12, 512, 64]);  permute_127 = None
    clone_70: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
    view_257: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_70, [48, 512, 64]);  clone_70 = None
    bmm_23: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_256, view_257)
    view_258: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_23, [4, 12, 512, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    permute_129: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None
    clone_71: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None
    view_259: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_71, [4, 512, 768]);  clone_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_260: "f32[2048, 768]" = torch.ops.aten.view.default(view_259, [2048, 768]);  view_259 = None
    addmm_70: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_15, view_260, permute_9);  primals_15 = None
    view_261: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_70, [4, 512, 768]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:369, code: projected_context_layer_dropout = self.output_dropout(projected_context_layer)
    clone_72: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_261);  view_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_104: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_102, clone_72);  add_102 = clone_72 = None
    var_mean_23 = torch.ops.aten.var_mean.correction(add_104, [2], correction = 0, keepdim = True)
    getitem_46: "f32[4, 512, 1]" = var_mean_23[0]
    getitem_47: "f32[4, 512, 1]" = var_mean_23[1];  var_mean_23 = None
    add_105: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
    rsqrt_23: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
    sub_36: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_104, getitem_47);  add_104 = getitem_47 = None
    mul_91: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_23);  sub_36 = None
    mul_92: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_91, primals_16)
    add_106: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_92, primals_17);  mul_92 = primals_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_262: "f32[2048, 768]" = torch.ops.aten.view.default(add_106, [2048, 768])
    addmm_71: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_19, view_262, permute_10);  primals_19 = None
    view_263: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_71, [4, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_93: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_263, 0.5)
    pow_12: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_263, 3.0)
    mul_94: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
    add_107: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(view_263, mul_94);  view_263 = mul_94 = None
    mul_95: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_107, 0.7978845608028654);  add_107 = None
    tanh_11: "f32[4, 512, 3072]" = torch.ops.aten.tanh.default(mul_95);  mul_95 = None
    add_108: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_11, 1.0)
    mul_96: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_93, add_108);  mul_93 = add_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_264: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_96, [2048, 3072]);  mul_96 = None
    addmm_72: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_21, view_264, permute_11);  primals_21 = None
    view_265: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_72, [4, 512, 768]);  addmm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_109: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_265, add_106);  view_265 = add_106 = None
    var_mean_24 = torch.ops.aten.var_mean.correction(add_109, [2], correction = 0, keepdim = True)
    getitem_48: "f32[4, 512, 1]" = var_mean_24[0]
    getitem_49: "f32[4, 512, 1]" = var_mean_24[1];  var_mean_24 = None
    add_110: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
    rsqrt_24: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
    sub_37: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_109, getitem_49);  add_109 = getitem_49 = None
    mul_97: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_24);  sub_37 = None
    mul_98: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_97, primals_22)
    add_111: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_98, primals_23);  mul_98 = primals_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:880, code: hidden_states = self.dense(hidden_states)
    view_266: "f32[2048, 768]" = torch.ops.aten.view.default(add_111, [2048, 768]);  add_111 = None
    permute_133: "f32[768, 128]" = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
    addmm_73: "f32[2048, 128]" = torch.ops.aten.addmm.default(primals_25, view_266, permute_133);  primals_25 = None
    view_267: "f32[4, 512, 128]" = torch.ops.aten.view.default(addmm_73, [4, 512, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_99: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(view_267, 0.5)
    pow_13: "f32[4, 512, 128]" = torch.ops.aten.pow.Tensor_Scalar(view_267, 3.0)
    mul_100: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
    add_112: "f32[4, 512, 128]" = torch.ops.aten.add.Tensor(view_267, mul_100);  view_267 = mul_100 = None
    mul_101: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(add_112, 0.7978845608028654);  add_112 = None
    tanh_12: "f32[4, 512, 128]" = torch.ops.aten.tanh.default(mul_101);  mul_101 = None
    add_113: "f32[4, 512, 128]" = torch.ops.aten.add.Tensor(tanh_12, 1.0)
    mul_102: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul_99, add_113);  mul_99 = add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:882, code: hidden_states = self.LayerNorm(hidden_states)
    var_mean_25 = torch.ops.aten.var_mean.correction(mul_102, [2], correction = 0, keepdim = True)
    getitem_50: "f32[4, 512, 1]" = var_mean_25[0]
    getitem_51: "f32[4, 512, 1]" = var_mean_25[1];  var_mean_25 = None
    add_114: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
    rsqrt_25: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
    sub_38: "f32[4, 512, 128]" = torch.ops.aten.sub.Tensor(mul_102, getitem_51);  mul_102 = None
    mul_103: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_25);  sub_38 = None
    mul_104: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul_103, primals_26);  mul_103 = None
    add_115: "f32[4, 512, 128]" = torch.ops.aten.add.Tensor(mul_104, primals_27);  mul_104 = primals_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:883, code: hidden_states = self.decoder(hidden_states)
    view_268: "f32[2048, 128]" = torch.ops.aten.view.default(add_115, [2048, 128]);  add_115 = None
    permute_134: "f32[128, 30000]" = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
    addmm_74: "f32[2048, 30000]" = torch.ops.aten.addmm.default(primals_29, view_268, permute_134);  primals_29 = None
    view_269: "f32[4, 512, 30000]" = torch.ops.aten.view.default(addmm_74, [4, 512, 30000]);  addmm_74 = None
    permute_135: "f32[30000, 128]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:880, code: hidden_states = self.dense(hidden_states)
    permute_139: "f32[128, 768]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    div_25: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    permute_143: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    permute_147: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    div_26: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    permute_151: "f32[768, 768]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    permute_156: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_256, [0, 2, 1]);  view_256 = None
    permute_157: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_257, [0, 2, 1]);  view_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_27: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(alias_22);  alias_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_158: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_253, [0, 2, 1]);  view_253 = None
    permute_159: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    permute_164: "f32[768, 768]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    permute_168: "f32[768, 768]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    permute_172: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    div_28: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    div_29: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    permute_189: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_234, [0, 2, 1]);  view_234 = None
    permute_190: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_235, [0, 2, 1]);  view_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_29: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(alias_20);  alias_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_191: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_231, [0, 2, 1]);  view_231 = None
    permute_192: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    div_31: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    div_32: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    permute_222: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_212, [0, 2, 1]);  view_212 = None
    permute_223: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_213, [0, 2, 1]);  view_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_31: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(alias_18);  alias_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_224: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_209, [0, 2, 1]);  view_209 = None
    permute_225: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    div_34: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    div_35: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    permute_255: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_190, [0, 2, 1]);  view_190 = None
    permute_256: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_191, [0, 2, 1]);  view_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_33: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(alias_16);  alias_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_257: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_187, [0, 2, 1]);  view_187 = None
    permute_258: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    div_37: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    div_38: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    permute_288: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_168, [0, 2, 1]);  view_168 = None
    permute_289: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_169, [0, 2, 1]);  view_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_35: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_290: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_165, [0, 2, 1]);  view_165 = None
    permute_291: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    div_40: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    div_41: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    permute_321: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_146, [0, 2, 1]);  view_146 = None
    permute_322: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_147, [0, 2, 1]);  view_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_37: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_323: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_143, [0, 2, 1]);  view_143 = None
    permute_324: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    div_43: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    div_44: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    permute_354: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_124, [0, 2, 1]);  view_124 = None
    permute_355: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_125, [0, 2, 1]);  view_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_39: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_356: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_121, [0, 2, 1]);  view_121 = None
    permute_357: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    div_46: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    div_47: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    permute_387: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_102, [0, 2, 1]);  view_102 = None
    permute_388: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_103, [0, 2, 1]);  view_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_41: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_389: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_99, [0, 2, 1]);  view_99 = None
    permute_390: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    div_49: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    div_50: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    permute_420: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_80, [0, 2, 1]);  view_80 = None
    permute_421: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_81, [0, 2, 1]);  view_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_43: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_422: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_77, [0, 2, 1]);  view_77 = None
    permute_423: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    div_52: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    div_53: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    permute_453: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_58, [0, 2, 1]);  view_58 = None
    permute_454: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_59, [0, 2, 1]);  view_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_45: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_455: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_55, [0, 2, 1]);  view_55 = None
    permute_456: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    div_55: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    div_56: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    permute_486: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_36, [0, 2, 1]);  view_36 = None
    permute_487: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_37, [0, 2, 1]);  view_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_47: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_488: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_33, [0, 2, 1]);  view_33 = None
    permute_489: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    div_58: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    div_59: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    permute_519: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_14, [0, 2, 1]);  view_14 = None
    permute_520: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_15, [0, 2, 1]);  view_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_49: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(alias);  alias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_521: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
    permute_522: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:467, code: hidden_states = self.embedding_hidden_mapping_in(hidden_states)
    permute_539: "f32[768, 128]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:257, code: embeddings = self.LayerNorm(embeddings)
    div_61: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 128);  rsqrt = None
    return [view_269, primals_4, primals_16, primals_22, primals_26, primals_32, expand, slice_2, mul_1, view, view_2, view_18, mul_3, view_20, addmm_5, tanh, view_22, mul_9, view_24, view_40, mul_11, view_42, addmm_11, tanh_1, view_44, mul_17, view_46, view_62, mul_19, view_64, addmm_17, tanh_2, view_66, mul_25, view_68, view_84, mul_27, view_86, addmm_23, tanh_3, view_88, mul_33, view_90, view_106, mul_35, view_108, addmm_29, tanh_4, view_110, mul_41, view_112, view_128, mul_43, view_130, addmm_35, tanh_5, view_132, mul_49, view_134, view_150, mul_51, view_152, addmm_41, tanh_6, view_154, mul_57, view_156, view_172, mul_59, view_174, addmm_47, tanh_7, view_176, mul_65, view_178, view_194, mul_67, view_196, addmm_53, tanh_8, view_198, mul_73, view_200, view_216, mul_75, view_218, addmm_59, tanh_9, view_220, mul_81, view_222, view_238, mul_83, view_240, addmm_65, tanh_10, view_242, mul_89, view_244, view_260, mul_91, view_262, addmm_71, tanh_11, view_264, mul_97, view_266, addmm_73, tanh_12, getitem_51, rsqrt_25, view_268, permute_135, permute_139, div_25, permute_143, permute_147, div_26, permute_151, permute_156, permute_157, alias_27, permute_158, permute_159, permute_164, permute_168, permute_172, div_28, div_29, permute_189, permute_190, alias_29, permute_191, permute_192, div_31, div_32, permute_222, permute_223, alias_31, permute_224, permute_225, div_34, div_35, permute_255, permute_256, alias_33, permute_257, permute_258, div_37, div_38, permute_288, permute_289, alias_35, permute_290, permute_291, div_40, div_41, permute_321, permute_322, alias_37, permute_323, permute_324, div_43, div_44, permute_354, permute_355, alias_39, permute_356, permute_357, div_46, div_47, permute_387, permute_388, alias_41, permute_389, permute_390, div_49, div_50, permute_420, permute_421, alias_43, permute_422, permute_423, div_52, div_53, permute_453, permute_454, alias_45, permute_455, permute_456, div_55, div_56, permute_486, permute_487, alias_47, permute_488, permute_489, div_58, div_59, permute_519, permute_520, alias_49, permute_521, permute_522, permute_539, div_61]
    