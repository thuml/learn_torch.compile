from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[768, 768]"; primals_2: "f32[768]"; primals_3: "f32[768]"; primals_4: "f32[768]"; primals_5: "f32[50265, 768]"; primals_6: "f32[50265]"; primals_7: "f32[1, 1024, 768]"; primals_8: "i64[1, 1024]"; tangents_1: "f32[]"; tangents_2: "f32[1, 1024, 50265]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, tangents_1, tangents_2, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1397, code: x = self.dense(features)
    view: "f32[1024, 768]" = torch.ops.aten.view.default(primals_7, [1024, 768]);  primals_7 = None
    permute: "f32[768, 768]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
    addmm: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_2, view, permute);  primals_2 = None
    view_1: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm, [1, 1024, 768]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1, 0.5)
    mul_1: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1, 0.7071067811865476)
    erf: "f32[1, 1024, 768]" = torch.ops.aten.erf.default(mul_1);  mul_1 = None
    add: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_2: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul, add);  mul = add = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1399, code: x = self.layer_norm(x)
    var_mean = torch.ops.aten.var_mean.correction(mul_2, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 1024, 1]" = var_mean[0]
    getitem_1: "f32[1, 1024, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_2, getitem_1)
    mul_3: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_4: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_3, primals_3);  mul_3 = None
    add_2: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_4, primals_4);  mul_4 = primals_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1402, code: x = self.decoder(x)
    view_2: "f32[1024, 768]" = torch.ops.aten.view.default(add_2, [1024, 768]);  add_2 = None
    permute_1: "f32[768, 50265]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
    addmm_1: "f32[1024, 50265]" = torch.ops.aten.addmm.default(primals_6, view_2, permute_1);  primals_6 = None
    view_3: "f32[1, 1024, 50265]" = torch.ops.aten.view.default(addmm_1, [1, 1024, 50265]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1864, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    view_4: "f32[1024, 50265]" = torch.ops.aten.view.default(view_3, [-1, 50265])
    view_5: "i64[1024]" = torch.ops.aten.view.default(primals_8, [-1]);  primals_8 = None
    amax: "f32[1024, 1]" = torch.ops.aten.amax.default(view_4, [1], True)
    sub_1: "f32[1024, 50265]" = torch.ops.aten.sub.Tensor(view_4, amax);  view_4 = amax = None
    exp: "f32[1024, 50265]" = torch.ops.aten.exp.default(sub_1)
    sum_1: "f32[1024, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
    log: "f32[1024, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
    sub_2: "f32[1024, 50265]" = torch.ops.aten.sub.Tensor(sub_1, log);  sub_1 = log = None
    alias: "f32[1024, 50265]" = torch.ops.aten.alias.default(sub_2)
    ne: "b8[1024]" = torch.ops.aten.ne.Scalar(view_5, -100)
    scalar_tensor: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'))
    where: "i64[1024]" = torch.ops.aten.where.self(ne, view_5, scalar_tensor);  ne = scalar_tensor = None
    unsqueeze: "i64[1024, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
    gather: "f32[1024, 1]" = torch.ops.aten.gather.default(sub_2, 1, unsqueeze);  sub_2 = unsqueeze = None
    squeeze: "f32[1024]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[1024]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    ne_1: "b8[1024]" = torch.ops.aten.ne.Scalar(view_5, -100)
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_1: "f32[1024]" = torch.ops.aten.where.self(ne_1, neg, scalar_tensor_1);  ne_1 = neg = scalar_tensor_1 = None
    ne_2: "b8[1024]" = torch.ops.aten.ne.Scalar(view_5, -100)
    sum_2: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
    sum_3: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
    div: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type);  sum_3 = None
    div_1: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type);  tangents_1 = convert_element_type = None
    unsqueeze_1: "i64[1024, 1]" = torch.ops.aten.unsqueeze.default(view_5, 1);  view_5 = None
    ne_3: "b8[1024, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_1, -100)
    scalar_tensor_2: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'))
    where_2: "i64[1024, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_1, scalar_tensor_2);  ne_3 = scalar_tensor_2 = None
    full: "f32[1024, 50265]" = torch.ops.aten.full.default([1024, 50265], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    scatter: "f32[1024, 50265]" = torch.ops.aten.scatter.value(full, 1, where_2, -1.0);  full = where_2 = None
    ne_4: "b8[1024, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_1, -100);  unsqueeze_1 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_3: "f32[1024, 1]" = torch.ops.aten.where.self(ne_4, div_1, scalar_tensor_3);  ne_4 = div_1 = scalar_tensor_3 = None
    mul_5: "f32[1024, 50265]" = torch.ops.aten.mul.Tensor(scatter, where_3);  scatter = where_3 = None
    alias_1: "f32[1024, 50265]" = torch.ops.aten.alias.default(alias);  alias = None
    exp_1: "f32[1024, 50265]" = torch.ops.aten.exp.default(alias_1);  alias_1 = None
    sum_4: "f32[1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_5, [1], True)
    mul_6: "f32[1024, 50265]" = torch.ops.aten.mul.Tensor(exp_1, sum_4);  exp_1 = sum_4 = None
    sub_3: "f32[1024, 50265]" = torch.ops.aten.sub.Tensor(mul_5, mul_6);  mul_5 = mul_6 = None
    view_6: "f32[1, 1024, 50265]" = torch.ops.aten.view.default(sub_3, [1, 1024, 50265]);  sub_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1864, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    add_3: "f32[1, 1024, 50265]" = torch.ops.aten.add.Tensor(tangents_2, view_6);  tangents_2 = view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1402, code: x = self.decoder(x)
    view_7: "f32[1024, 50265]" = torch.ops.aten.view.default(add_3, [1024, 50265]);  add_3 = None
    permute_2: "f32[50265, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    mm: "f32[1024, 768]" = torch.ops.aten.mm.default(view_7, permute_2);  permute_2 = None
    permute_3: "f32[50265, 1024]" = torch.ops.aten.permute.default(view_7, [1, 0])
    mm_1: "f32[50265, 768]" = torch.ops.aten.mm.default(permute_3, view_2);  permute_3 = view_2 = None
    permute_4: "f32[768, 50265]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_5: "f32[1, 50265]" = torch.ops.aten.sum.dim_IntList(view_7, [0], True);  view_7 = None
    view_8: "f32[50265]" = torch.ops.aten.view.default(sum_5, [50265]);  sum_5 = None
    permute_5: "f32[50265, 768]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
    view_9: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm, [1, 1024, 768]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1399, code: x = self.layer_norm(x)
    sub_4: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_2, getitem_1);  mul_2 = getitem_1 = None
    mul_7: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt);  sub_4 = None
    mul_8: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_9, primals_3);  primals_3 = None
    mul_9: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_8, 768)
    sum_6: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_8, [2], True)
    mul_10: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_8, mul_7);  mul_8 = None
    sum_7: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_10, [2], True);  mul_10 = None
    mul_11: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_7, sum_7);  sum_7 = None
    sub_5: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_9, sum_6);  mul_9 = sum_6 = None
    sub_6: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_5, mul_11);  sub_5 = mul_11 = None
    div_2: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
    mul_12: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_2, sub_6);  div_2 = sub_6 = None
    mul_13: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_9, mul_7);  mul_7 = None
    sum_8: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_13, [0, 1]);  mul_13 = None
    sum_9: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_9, [0, 1]);  view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_14: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1, 0.7071067811865476)
    erf_1: "f32[1, 1024, 768]" = torch.ops.aten.erf.default(mul_14);  mul_14 = None
    add_4: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_15: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_4, 0.5);  add_4 = None
    mul_16: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1, view_1)
    mul_17: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_16, -0.5);  mul_16 = None
    exp_2: "f32[1, 1024, 768]" = torch.ops.aten.exp.default(mul_17);  mul_17 = None
    mul_18: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
    mul_19: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1, mul_18);  view_1 = mul_18 = None
    add_5: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_15, mul_19);  mul_15 = mul_19 = None
    mul_20: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_12, add_5);  mul_12 = add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1397, code: x = self.dense(features)
    view_10: "f32[1024, 768]" = torch.ops.aten.view.default(mul_20, [1024, 768]);  mul_20 = None
    permute_6: "f32[768, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    mm_2: "f32[1024, 768]" = torch.ops.aten.mm.default(view_10, permute_6);  permute_6 = None
    permute_7: "f32[768, 1024]" = torch.ops.aten.permute.default(view_10, [1, 0])
    mm_3: "f32[768, 768]" = torch.ops.aten.mm.default(permute_7, view);  permute_7 = view = None
    permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_10: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_10, [0], True);  view_10 = None
    view_11: "f32[768]" = torch.ops.aten.view.default(sum_10, [768]);  sum_10 = None
    permute_9: "f32[768, 768]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    view_12: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_2, [1, 1024, 768]);  mm_2 = None
    return pytree.tree_unflatten([div, view_3, permute_9, view_11, sum_8, sum_9, permute_5, view_8, view_12, None], self._out_spec)
    