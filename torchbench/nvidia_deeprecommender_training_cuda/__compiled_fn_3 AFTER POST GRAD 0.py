from __future__ import annotations



def forward(self, primals_1: "f32[512, 197951]", primals_2: "f32[512, 512]", primals_3: "f32[1024, 512]", primals_4: "f32[512]", primals_5: "f32[512]", primals_6: "f32[1024]", primals_7: "f32[512, 1024]", primals_8: "f32[512, 512]", primals_9: "f32[197951, 512]", primals_10: "f32[512]", primals_11: "f32[512]", primals_12: "f32[197951]", primals_13: "f32[4, 197951]"):
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
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:116, code: z = activation(input=F.linear(input=z, weight=w, bias=self.decode_b[ind]),
    permute_3: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
    addmm_3: "f32[4, 512]" = torch.ops.aten.addmm.default(primals_10, where_2, permute_3);  primals_10 = None
    
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
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:116, code: z = activation(input=F.linear(input=z, weight=w, bias=self.decode_b[ind]),
    permute_6: "f32[197951, 512]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
    permute_10: "f32[512, 512]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
    permute_14: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:101, code: x = activation(input=F.linear(input=x, weight=w, bias=self.encode_b[ind]), kind=self._nl_type)
    permute_18: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    permute_22: "f32[512, 512]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    return [where_5, primals_13, addmm, where, addmm_1, where_1, addmm_2, where_2, addmm_3, where_3, addmm_4, where_4, addmm_5, permute_6, permute_10, permute_14, permute_18, permute_22]
    