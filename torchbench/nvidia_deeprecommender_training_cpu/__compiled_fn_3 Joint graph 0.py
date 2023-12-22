from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[512, 197951]"; primals_2: "f32[512, 512]"; primals_3: "f32[1024, 512]"; primals_4: "f32[512]"; primals_5: "f32[512]"; primals_6: "f32[1024]"; primals_7: "f32[512, 1024]"; primals_8: "f32[512, 512]"; primals_9: "f32[197951, 512]"; primals_10: "f32[512]"; primals_11: "f32[512]"; primals_12: "f32[197951]"; primals_13: "f32[4, 197951]"; tangents_1: "f32[4, 197951]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:101, code: x = activation(input=F.linear(input=x, weight=w, bias=self.encode_b[ind]), kind=self._nl_type)
    permute: "f32[197951, 512]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
    addmm: "f32[4, 512]" = torch.ops.aten.addmm.default(primals_4, primals_13, permute);  primals_4 = permute = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:11, code: return F.selu(input)
    gt: "b8[4, 512]" = torch.ops.aten.gt.Scalar(addmm, 0)
    mul: "f32[4, 512]" = torch.ops.aten.mul.Tensor(addmm, 1.0507009873554805)
    mul_1: "f32[4, 512]" = torch.ops.aten.mul.Tensor(addmm, 1.0)
    expm1: "f32[4, 512]" = torch.ops.aten.expm1.default(mul_1);  mul_1 = None
    mul_2: "f32[4, 512]" = torch.ops.aten.mul.Tensor(expm1, 1.7580993408473766);  expm1 = None
    where: "f32[4, 512]" = torch.ops.aten.where.self(gt, mul, mul_2);  gt = mul = mul_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:101, code: x = activation(input=F.linear(input=x, weight=w, bias=self.encode_b[ind]), kind=self._nl_type)
    permute_1: "f32[512, 512]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
    addmm_1: "f32[4, 512]" = torch.ops.aten.addmm.default(primals_5, where, permute_1);  primals_5 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:11, code: return F.selu(input)
    gt_1: "b8[4, 512]" = torch.ops.aten.gt.Scalar(addmm_1, 0)
    mul_3: "f32[4, 512]" = torch.ops.aten.mul.Tensor(addmm_1, 1.0507009873554805)
    mul_4: "f32[4, 512]" = torch.ops.aten.mul.Tensor(addmm_1, 1.0)
    expm1_1: "f32[4, 512]" = torch.ops.aten.expm1.default(mul_4);  mul_4 = None
    mul_5: "f32[4, 512]" = torch.ops.aten.mul.Tensor(expm1_1, 1.7580993408473766);  expm1_1 = None
    where_1: "f32[4, 512]" = torch.ops.aten.where.self(gt_1, mul_3, mul_5);  gt_1 = mul_3 = mul_5 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:101, code: x = activation(input=F.linear(input=x, weight=w, bias=self.encode_b[ind]), kind=self._nl_type)
    permute_2: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
    addmm_2: "f32[4, 1024]" = torch.ops.aten.addmm.default(primals_6, where_1, permute_2);  primals_6 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:11, code: return F.selu(input)
    gt_2: "b8[4, 1024]" = torch.ops.aten.gt.Scalar(addmm_2, 0)
    mul_6: "f32[4, 1024]" = torch.ops.aten.mul.Tensor(addmm_2, 1.0507009873554805)
    mul_7: "f32[4, 1024]" = torch.ops.aten.mul.Tensor(addmm_2, 1.0)
    expm1_2: "f32[4, 1024]" = torch.ops.aten.expm1.default(mul_7);  mul_7 = None
    mul_8: "f32[4, 1024]" = torch.ops.aten.mul.Tensor(expm1_2, 1.7580993408473766);  expm1_2 = None
    where_2: "f32[4, 1024]" = torch.ops.aten.where.self(gt_2, mul_6, mul_8);  gt_2 = mul_6 = mul_8 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:103, code: x = self.drop(x)
    clone: "f32[4, 1024]" = torch.ops.aten.clone.default(where_2);  where_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:116, code: z = activation(input=F.linear(input=z, weight=w, bias=self.decode_b[ind]),
    permute_3: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
    addmm_3: "f32[4, 512]" = torch.ops.aten.addmm.default(primals_10, clone, permute_3);  primals_10 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:11, code: return F.selu(input)
    gt_3: "b8[4, 512]" = torch.ops.aten.gt.Scalar(addmm_3, 0)
    mul_9: "f32[4, 512]" = torch.ops.aten.mul.Tensor(addmm_3, 1.0507009873554805)
    mul_10: "f32[4, 512]" = torch.ops.aten.mul.Tensor(addmm_3, 1.0)
    expm1_3: "f32[4, 512]" = torch.ops.aten.expm1.default(mul_10);  mul_10 = None
    mul_11: "f32[4, 512]" = torch.ops.aten.mul.Tensor(expm1_3, 1.7580993408473766);  expm1_3 = None
    where_3: "f32[4, 512]" = torch.ops.aten.where.self(gt_3, mul_9, mul_11);  gt_3 = mul_9 = mul_11 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:116, code: z = activation(input=F.linear(input=z, weight=w, bias=self.decode_b[ind]),
    permute_4: "f32[512, 512]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
    addmm_4: "f32[4, 512]" = torch.ops.aten.addmm.default(primals_11, where_3, permute_4);  primals_11 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:11, code: return F.selu(input)
    gt_4: "b8[4, 512]" = torch.ops.aten.gt.Scalar(addmm_4, 0)
    mul_12: "f32[4, 512]" = torch.ops.aten.mul.Tensor(addmm_4, 1.0507009873554805)
    mul_13: "f32[4, 512]" = torch.ops.aten.mul.Tensor(addmm_4, 1.0)
    expm1_4: "f32[4, 512]" = torch.ops.aten.expm1.default(mul_13);  mul_13 = None
    mul_14: "f32[4, 512]" = torch.ops.aten.mul.Tensor(expm1_4, 1.7580993408473766);  expm1_4 = None
    where_4: "f32[4, 512]" = torch.ops.aten.where.self(gt_4, mul_12, mul_14);  gt_4 = mul_12 = mul_14 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:116, code: z = activation(input=F.linear(input=z, weight=w, bias=self.decode_b[ind]),
    permute_5: "f32[512, 197951]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
    addmm_5: "f32[4, 197951]" = torch.ops.aten.addmm.default(primals_12, where_4, permute_5);  primals_12 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:11, code: return F.selu(input)
    gt_5: "b8[4, 197951]" = torch.ops.aten.gt.Scalar(addmm_5, 0)
    mul_15: "f32[4, 197951]" = torch.ops.aten.mul.Tensor(addmm_5, 1.0507009873554805)
    mul_16: "f32[4, 197951]" = torch.ops.aten.mul.Tensor(addmm_5, 1.0)
    expm1_5: "f32[4, 197951]" = torch.ops.aten.expm1.default(mul_16);  mul_16 = None
    mul_17: "f32[4, 197951]" = torch.ops.aten.mul.Tensor(expm1_5, 1.7580993408473766);  expm1_5 = None
    where_5: "f32[4, 197951]" = torch.ops.aten.where.self(gt_5, mul_15, mul_17);  gt_5 = mul_15 = mul_17 = None
    le: "b8[4, 197951]" = torch.ops.aten.le.Scalar(addmm_5, 0)
    mul_18: "f32[4, 197951]" = torch.ops.aten.mul.Tensor(tangents_1, 1)
    mul_19: "f32[4, 197951]" = torch.ops.aten.mul.Tensor(mul_18, 1.7580993408473766);  mul_18 = None
    mul_20: "f32[4, 197951]" = torch.ops.aten.mul.Tensor(addmm_5, 1);  addmm_5 = None
    exp: "f32[4, 197951]" = torch.ops.aten.exp.default(mul_20);  mul_20 = None
    mul_21: "f32[4, 197951]" = torch.ops.aten.mul.Tensor(mul_19, exp);  mul_19 = exp = None
    mul_22: "f32[4, 197951]" = torch.ops.aten.mul.Tensor(tangents_1, 1.0507009873554805);  tangents_1 = None
    where_6: "f32[4, 197951]" = torch.ops.aten.where.self(le, mul_21, mul_22);  le = mul_21 = mul_22 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:116, code: z = activation(input=F.linear(input=z, weight=w, bias=self.decode_b[ind]),
    permute_6: "f32[197951, 512]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
    mm: "f32[4, 512]" = torch.ops.aten.mm.default(where_6, permute_6);  permute_6 = None
    permute_7: "f32[197951, 4]" = torch.ops.aten.permute.default(where_6, [1, 0])
    mm_1: "f32[197951, 512]" = torch.ops.aten.mm.default(permute_7, where_4);  permute_7 = where_4 = None
    permute_8: "f32[512, 197951]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 197951]" = torch.ops.aten.sum.dim_IntList(where_6, [0], True);  where_6 = None
    view: "f32[197951]" = torch.ops.aten.view.default(sum_1, [197951]);  sum_1 = None
    permute_9: "f32[197951, 512]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:11, code: return F.selu(input)
    le_1: "b8[4, 512]" = torch.ops.aten.le.Scalar(addmm_4, 0)
    mul_23: "f32[4, 512]" = torch.ops.aten.mul.Tensor(mm, 1)
    mul_24: "f32[4, 512]" = torch.ops.aten.mul.Tensor(mul_23, 1.7580993408473766);  mul_23 = None
    mul_25: "f32[4, 512]" = torch.ops.aten.mul.Tensor(addmm_4, 1);  addmm_4 = None
    exp_1: "f32[4, 512]" = torch.ops.aten.exp.default(mul_25);  mul_25 = None
    mul_26: "f32[4, 512]" = torch.ops.aten.mul.Tensor(mul_24, exp_1);  mul_24 = exp_1 = None
    mul_27: "f32[4, 512]" = torch.ops.aten.mul.Tensor(mm, 1.0507009873554805);  mm = None
    where_7: "f32[4, 512]" = torch.ops.aten.where.self(le_1, mul_26, mul_27);  le_1 = mul_26 = mul_27 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:116, code: z = activation(input=F.linear(input=z, weight=w, bias=self.decode_b[ind]),
    permute_10: "f32[512, 512]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
    mm_2: "f32[4, 512]" = torch.ops.aten.mm.default(where_7, permute_10);  permute_10 = None
    permute_11: "f32[512, 4]" = torch.ops.aten.permute.default(where_7, [1, 0])
    mm_3: "f32[512, 512]" = torch.ops.aten.mm.default(permute_11, where_3);  permute_11 = where_3 = None
    permute_12: "f32[512, 512]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_2: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(where_7, [0], True);  where_7 = None
    view_1: "f32[512]" = torch.ops.aten.view.default(sum_2, [512]);  sum_2 = None
    permute_13: "f32[512, 512]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:11, code: return F.selu(input)
    le_2: "b8[4, 512]" = torch.ops.aten.le.Scalar(addmm_3, 0)
    mul_28: "f32[4, 512]" = torch.ops.aten.mul.Tensor(mm_2, 1)
    mul_29: "f32[4, 512]" = torch.ops.aten.mul.Tensor(mul_28, 1.7580993408473766);  mul_28 = None
    mul_30: "f32[4, 512]" = torch.ops.aten.mul.Tensor(addmm_3, 1);  addmm_3 = None
    exp_2: "f32[4, 512]" = torch.ops.aten.exp.default(mul_30);  mul_30 = None
    mul_31: "f32[4, 512]" = torch.ops.aten.mul.Tensor(mul_29, exp_2);  mul_29 = exp_2 = None
    mul_32: "f32[4, 512]" = torch.ops.aten.mul.Tensor(mm_2, 1.0507009873554805);  mm_2 = None
    where_8: "f32[4, 512]" = torch.ops.aten.where.self(le_2, mul_31, mul_32);  le_2 = mul_31 = mul_32 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:116, code: z = activation(input=F.linear(input=z, weight=w, bias=self.decode_b[ind]),
    permute_14: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    mm_4: "f32[4, 1024]" = torch.ops.aten.mm.default(where_8, permute_14);  permute_14 = None
    permute_15: "f32[512, 4]" = torch.ops.aten.permute.default(where_8, [1, 0])
    mm_5: "f32[512, 1024]" = torch.ops.aten.mm.default(permute_15, clone);  permute_15 = clone = None
    permute_16: "f32[1024, 512]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_3: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(where_8, [0], True);  where_8 = None
    view_2: "f32[512]" = torch.ops.aten.view.default(sum_3, [512]);  sum_3 = None
    permute_17: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:11, code: return F.selu(input)
    le_3: "b8[4, 1024]" = torch.ops.aten.le.Scalar(addmm_2, 0)
    mul_33: "f32[4, 1024]" = torch.ops.aten.mul.Tensor(mm_4, 1)
    mul_34: "f32[4, 1024]" = torch.ops.aten.mul.Tensor(mul_33, 1.7580993408473766);  mul_33 = None
    mul_35: "f32[4, 1024]" = torch.ops.aten.mul.Tensor(addmm_2, 1);  addmm_2 = None
    exp_3: "f32[4, 1024]" = torch.ops.aten.exp.default(mul_35);  mul_35 = None
    mul_36: "f32[4, 1024]" = torch.ops.aten.mul.Tensor(mul_34, exp_3);  mul_34 = exp_3 = None
    mul_37: "f32[4, 1024]" = torch.ops.aten.mul.Tensor(mm_4, 1.0507009873554805);  mm_4 = None
    where_9: "f32[4, 1024]" = torch.ops.aten.where.self(le_3, mul_36, mul_37);  le_3 = mul_36 = mul_37 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:101, code: x = activation(input=F.linear(input=x, weight=w, bias=self.encode_b[ind]), kind=self._nl_type)
    permute_18: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    mm_6: "f32[4, 512]" = torch.ops.aten.mm.default(where_9, permute_18);  permute_18 = None
    permute_19: "f32[1024, 4]" = torch.ops.aten.permute.default(where_9, [1, 0])
    mm_7: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_19, where_1);  permute_19 = where_1 = None
    permute_20: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_4: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(where_9, [0], True);  where_9 = None
    view_3: "f32[1024]" = torch.ops.aten.view.default(sum_4, [1024]);  sum_4 = None
    permute_21: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:11, code: return F.selu(input)
    le_4: "b8[4, 512]" = torch.ops.aten.le.Scalar(addmm_1, 0)
    mul_38: "f32[4, 512]" = torch.ops.aten.mul.Tensor(mm_6, 1)
    mul_39: "f32[4, 512]" = torch.ops.aten.mul.Tensor(mul_38, 1.7580993408473766);  mul_38 = None
    mul_40: "f32[4, 512]" = torch.ops.aten.mul.Tensor(addmm_1, 1);  addmm_1 = None
    exp_4: "f32[4, 512]" = torch.ops.aten.exp.default(mul_40);  mul_40 = None
    mul_41: "f32[4, 512]" = torch.ops.aten.mul.Tensor(mul_39, exp_4);  mul_39 = exp_4 = None
    mul_42: "f32[4, 512]" = torch.ops.aten.mul.Tensor(mm_6, 1.0507009873554805);  mm_6 = None
    where_10: "f32[4, 512]" = torch.ops.aten.where.self(le_4, mul_41, mul_42);  le_4 = mul_41 = mul_42 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:101, code: x = activation(input=F.linear(input=x, weight=w, bias=self.encode_b[ind]), kind=self._nl_type)
    permute_22: "f32[512, 512]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    mm_8: "f32[4, 512]" = torch.ops.aten.mm.default(where_10, permute_22);  permute_22 = None
    permute_23: "f32[512, 4]" = torch.ops.aten.permute.default(where_10, [1, 0])
    mm_9: "f32[512, 512]" = torch.ops.aten.mm.default(permute_23, where);  permute_23 = where = None
    permute_24: "f32[512, 512]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_5: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(where_10, [0], True);  where_10 = None
    view_4: "f32[512]" = torch.ops.aten.view.default(sum_5, [512]);  sum_5 = None
    permute_25: "f32[512, 512]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:11, code: return F.selu(input)
    le_5: "b8[4, 512]" = torch.ops.aten.le.Scalar(addmm, 0)
    mul_43: "f32[4, 512]" = torch.ops.aten.mul.Tensor(mm_8, 1)
    mul_44: "f32[4, 512]" = torch.ops.aten.mul.Tensor(mul_43, 1.7580993408473766);  mul_43 = None
    mul_45: "f32[4, 512]" = torch.ops.aten.mul.Tensor(addmm, 1);  addmm = None
    exp_5: "f32[4, 512]" = torch.ops.aten.exp.default(mul_45);  mul_45 = None
    mul_46: "f32[4, 512]" = torch.ops.aten.mul.Tensor(mul_44, exp_5);  mul_44 = exp_5 = None
    mul_47: "f32[4, 512]" = torch.ops.aten.mul.Tensor(mm_8, 1.0507009873554805);  mm_8 = None
    where_11: "f32[4, 512]" = torch.ops.aten.where.self(le_5, mul_46, mul_47);  le_5 = mul_46 = mul_47 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:101, code: x = activation(input=F.linear(input=x, weight=w, bias=self.encode_b[ind]), kind=self._nl_type)
    permute_26: "f32[512, 4]" = torch.ops.aten.permute.default(where_11, [1, 0])
    mm_10: "f32[512, 197951]" = torch.ops.aten.mm.default(permute_26, primals_13);  permute_26 = primals_13 = None
    permute_27: "f32[197951, 512]" = torch.ops.aten.permute.default(mm_10, [1, 0]);  mm_10 = None
    sum_6: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(where_11, [0], True);  where_11 = None
    view_5: "f32[512]" = torch.ops.aten.view.default(sum_6, [512]);  sum_6 = None
    permute_28: "f32[512, 197951]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
    return pytree.tree_unflatten([where_5, permute_28, permute_25, permute_21, view_5, view_4, view_3, permute_17, permute_13, permute_9, view_2, view_1, view, None], self._out_spec)
    