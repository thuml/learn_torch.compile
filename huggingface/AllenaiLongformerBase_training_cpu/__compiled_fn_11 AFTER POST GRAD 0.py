from __future__ import annotations



def forward(self, primals_1: "f32[768, 768]", primals_2: "f32[768]", primals_3: "f32[768]", primals_4: "f32[768]", primals_5: "f32[50265, 768]", primals_6: "f32[50265]", primals_7: "f32[1, 1024, 768]", primals_8: "i64[1, 1024]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1397, code: x = self.dense(features)
    view: "f32[1024, 768]" = torch.ops.aten.reshape.default(primals_7, [1024, 768]);  primals_7 = None
    permute: "f32[768, 768]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
    addmm: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_2, view, permute);  primals_2 = None
    view_1: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(addmm, [1, 1024, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1, 0.5)
    mul_1: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1, 0.7071067811865476);  view_1 = None
    erf: "f32[1, 1024, 768]" = torch.ops.aten.erf.default(mul_1);  mul_1 = None
    add: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_2: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul, add);  mul = add = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1399, code: x = self.layer_norm(x)
    var_mean = torch.ops.aten.var_mean.correction(mul_2, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 1024, 1]" = var_mean[0]
    getitem_1: "f32[1, 1024, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_2, getitem_1);  mul_2 = getitem_1 = None
    mul_3: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_4: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_3, primals_3)
    add_2: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_4, primals_4);  mul_4 = primals_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1402, code: x = self.decoder(x)
    view_2: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_2, [1024, 768]);  add_2 = None
    permute_1: "f32[768, 50265]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
    addmm_1: "f32[1024, 50265]" = torch.ops.aten.addmm.default(primals_6, view_2, permute_1);  primals_6 = None
    view_3: "f32[1, 1024, 50265]" = torch.ops.aten.reshape.default(addmm_1, [1, 1024, 50265]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1864, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    view_4: "f32[1024, 50265]" = torch.ops.aten.reshape.default(view_3, [-1, 50265])
    view_5: "i64[1024]" = torch.ops.aten.reshape.default(primals_8, [-1])
    amax: "f32[1024, 1]" = torch.ops.aten.amax.default(view_4, [1], True)
    sub_1: "f32[1024, 50265]" = torch.ops.aten.sub.Tensor(view_4, amax);  view_4 = amax = None
    exp: "f32[1024, 50265]" = torch.ops.aten.exp.default(sub_1)
    sum_1: "f32[1024, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
    log: "f32[1024, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
    sub_2: "f32[1024, 50265]" = torch.ops.aten.sub.Tensor(sub_1, log);  sub_1 = log = None
    ne: "b8[1024]" = torch.ops.aten.ne.Scalar(view_5, -100)
    full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where: "i64[1024]" = torch.ops.aten.where.self(ne, view_5, full_default);  view_5 = full_default = None
    unsqueeze: "i64[1024, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
    gather: "f32[1024, 1]" = torch.ops.aten.gather.default(sub_2, 1, unsqueeze);  unsqueeze = None
    squeeze: "f32[1024]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[1024]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_1: "f32[1024]" = torch.ops.aten.where.self(ne, neg, full_default_1);  neg = full_default_1 = None
    sum_2: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
    sum_3: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
    div: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type);  sum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1402, code: x = self.decoder(x)
    permute_2: "f32[50265, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1399, code: x = self.layer_norm(x)
    div_2: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1397, code: x = self.dense(features)
    permute_6: "f32[768, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    return [div, view_3, primals_3, primals_8, view, addmm, mul_3, view_2, sub_2, convert_element_type, permute_2, div_2, permute_6]
    