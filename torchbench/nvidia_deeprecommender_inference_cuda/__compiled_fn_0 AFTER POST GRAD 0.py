from __future__ import annotations



def forward(self, arg0_1: "f32[512, 197951]", arg1_1: "f32[512, 512]", arg2_1: "f32[1024, 512]", arg3_1: "f32[512]", arg4_1: "f32[512]", arg5_1: "f32[1024]", arg6_1: "f32[512, 1024]", arg7_1: "f32[512, 512]", arg8_1: "f32[197951, 512]", arg9_1: "f32[512]", arg10_1: "f32[512]", arg11_1: "f32[197951]", arg12_1: "f32[4, 197951]"):
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:101, code: x = activation(input=F.linear(input=x, weight=w, bias=self.encode_b[ind]), kind=self._nl_type)
    permute: "f32[197951, 512]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
    
    # No stacktrace found for following nodes
    mm_default_5: "f32[4, 512]" = torch.ops.aten.mm.default(arg12_1, permute);  arg12_1 = permute = None
    add_tensor_5: "f32[4, 512]" = torch.ops.aten.add.Tensor(mm_default_5, arg3_1);  mm_default_5 = arg3_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:11, code: return F.selu(input)
    gt: "b8[4, 512]" = torch.ops.aten.gt.Scalar(add_tensor_5, 0)
    mul: "f32[4, 512]" = torch.ops.aten.mul.Tensor(add_tensor_5, 1.0507009873554805)
    mul_1: "f32[4, 512]" = torch.ops.aten.mul.Tensor(add_tensor_5, 1.0);  add_tensor_5 = None
    expm1: "f32[4, 512]" = torch.ops.aten.expm1.default(mul_1);  mul_1 = None
    mul_2: "f32[4, 512]" = torch.ops.aten.mul.Tensor(expm1, 1.7580993408473766);  expm1 = None
    where: "f32[4, 512]" = torch.ops.aten.where.self(gt, mul, mul_2);  gt = mul = mul_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:101, code: x = activation(input=F.linear(input=x, weight=w, bias=self.encode_b[ind]), kind=self._nl_type)
    permute_1: "f32[512, 512]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
    
    # No stacktrace found for following nodes
    mm_default_4: "f32[4, 512]" = torch.ops.aten.mm.default(where, permute_1);  where = permute_1 = None
    add_tensor_4: "f32[4, 512]" = torch.ops.aten.add.Tensor(mm_default_4, arg4_1);  mm_default_4 = arg4_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:11, code: return F.selu(input)
    gt_1: "b8[4, 512]" = torch.ops.aten.gt.Scalar(add_tensor_4, 0)
    mul_3: "f32[4, 512]" = torch.ops.aten.mul.Tensor(add_tensor_4, 1.0507009873554805)
    mul_4: "f32[4, 512]" = torch.ops.aten.mul.Tensor(add_tensor_4, 1.0);  add_tensor_4 = None
    expm1_1: "f32[4, 512]" = torch.ops.aten.expm1.default(mul_4);  mul_4 = None
    mul_5: "f32[4, 512]" = torch.ops.aten.mul.Tensor(expm1_1, 1.7580993408473766);  expm1_1 = None
    where_1: "f32[4, 512]" = torch.ops.aten.where.self(gt_1, mul_3, mul_5);  gt_1 = mul_3 = mul_5 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:101, code: x = activation(input=F.linear(input=x, weight=w, bias=self.encode_b[ind]), kind=self._nl_type)
    permute_2: "f32[512, 1024]" = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
    
    # No stacktrace found for following nodes
    mm_default_3: "f32[4, 1024]" = torch.ops.aten.mm.default(where_1, permute_2);  where_1 = permute_2 = None
    add_tensor_3: "f32[4, 1024]" = torch.ops.aten.add.Tensor(mm_default_3, arg5_1);  mm_default_3 = arg5_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:11, code: return F.selu(input)
    gt_2: "b8[4, 1024]" = torch.ops.aten.gt.Scalar(add_tensor_3, 0)
    mul_6: "f32[4, 1024]" = torch.ops.aten.mul.Tensor(add_tensor_3, 1.0507009873554805)
    mul_7: "f32[4, 1024]" = torch.ops.aten.mul.Tensor(add_tensor_3, 1.0);  add_tensor_3 = None
    expm1_2: "f32[4, 1024]" = torch.ops.aten.expm1.default(mul_7);  mul_7 = None
    mul_8: "f32[4, 1024]" = torch.ops.aten.mul.Tensor(expm1_2, 1.7580993408473766);  expm1_2 = None
    where_2: "f32[4, 1024]" = torch.ops.aten.where.self(gt_2, mul_6, mul_8);  gt_2 = mul_6 = mul_8 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:116, code: z = activation(input=F.linear(input=z, weight=w, bias=self.decode_b[ind]),
    permute_3: "f32[1024, 512]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
    
    # No stacktrace found for following nodes
    mm_default_2: "f32[4, 512]" = torch.ops.aten.mm.default(where_2, permute_3);  where_2 = permute_3 = None
    add_tensor_2: "f32[4, 512]" = torch.ops.aten.add.Tensor(mm_default_2, arg9_1);  mm_default_2 = arg9_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:11, code: return F.selu(input)
    gt_3: "b8[4, 512]" = torch.ops.aten.gt.Scalar(add_tensor_2, 0)
    mul_9: "f32[4, 512]" = torch.ops.aten.mul.Tensor(add_tensor_2, 1.0507009873554805)
    mul_10: "f32[4, 512]" = torch.ops.aten.mul.Tensor(add_tensor_2, 1.0);  add_tensor_2 = None
    expm1_3: "f32[4, 512]" = torch.ops.aten.expm1.default(mul_10);  mul_10 = None
    mul_11: "f32[4, 512]" = torch.ops.aten.mul.Tensor(expm1_3, 1.7580993408473766);  expm1_3 = None
    where_3: "f32[4, 512]" = torch.ops.aten.where.self(gt_3, mul_9, mul_11);  gt_3 = mul_9 = mul_11 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:116, code: z = activation(input=F.linear(input=z, weight=w, bias=self.decode_b[ind]),
    permute_4: "f32[512, 512]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
    
    # No stacktrace found for following nodes
    mm_default_1: "f32[4, 512]" = torch.ops.aten.mm.default(where_3, permute_4);  where_3 = permute_4 = None
    add_tensor_1: "f32[4, 512]" = torch.ops.aten.add.Tensor(mm_default_1, arg10_1);  mm_default_1 = arg10_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:11, code: return F.selu(input)
    gt_4: "b8[4, 512]" = torch.ops.aten.gt.Scalar(add_tensor_1, 0)
    mul_12: "f32[4, 512]" = torch.ops.aten.mul.Tensor(add_tensor_1, 1.0507009873554805)
    mul_13: "f32[4, 512]" = torch.ops.aten.mul.Tensor(add_tensor_1, 1.0);  add_tensor_1 = None
    expm1_4: "f32[4, 512]" = torch.ops.aten.expm1.default(mul_13);  mul_13 = None
    mul_14: "f32[4, 512]" = torch.ops.aten.mul.Tensor(expm1_4, 1.7580993408473766);  expm1_4 = None
    where_4: "f32[4, 512]" = torch.ops.aten.where.self(gt_4, mul_12, mul_14);  gt_4 = mul_12 = mul_14 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:116, code: z = activation(input=F.linear(input=z, weight=w, bias=self.decode_b[ind]),
    permute_5: "f32[512, 197951]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
    
    # No stacktrace found for following nodes
    mm_default: "f32[4, 197951]" = torch.ops.aten.mm.default(where_4, permute_5);  where_4 = permute_5 = None
    add_tensor: "f32[4, 197951]" = torch.ops.aten.add.Tensor(mm_default, arg11_1);  mm_default = arg11_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:11, code: return F.selu(input)
    gt_5: "b8[4, 197951]" = torch.ops.aten.gt.Scalar(add_tensor, 0)
    mul_15: "f32[4, 197951]" = torch.ops.aten.mul.Tensor(add_tensor, 1.0507009873554805)
    mul_16: "f32[4, 197951]" = torch.ops.aten.mul.Tensor(add_tensor, 1.0);  add_tensor = None
    expm1_5: "f32[4, 197951]" = torch.ops.aten.expm1.default(mul_16);  mul_16 = None
    mul_17: "f32[4, 197951]" = torch.ops.aten.mul.Tensor(expm1_5, 1.7580993408473766);  expm1_5 = None
    where_5: "f32[4, 197951]" = torch.ops.aten.where.self(gt_5, mul_15, mul_17);  gt_5 = mul_15 = mul_17 = None
    return (where_5,)
    