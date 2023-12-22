from __future__ import annotations



def forward(self, primals_1: "f32[64, 3, 4, 4]", primals_2: "f32[128, 64, 4, 4]", primals_3: "f32[128]", primals_4: "f32[128]", primals_5: "f32[256, 128, 4, 4]", primals_6: "f32[256]", primals_7: "f32[256]", primals_8: "f32[512, 256, 4, 4]", primals_9: "f32[512]", primals_10: "f32[512]", primals_11: "f32[1, 512, 4, 4]", primals_12: "f32[128]", primals_13: "f32[128]", primals_14: "i64[]", primals_15: "f32[256]", primals_16: "f32[256]", primals_17: "i64[]", primals_18: "f32[512]", primals_19: "f32[512]", primals_20: "i64[]", primals_21: "f32[4, 3, 64, 64]"):
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/dcgan/__init__.py:134, code: return self.main(input)
    convolution: "f32[4, 64, 32, 32]" = torch.ops.aten.convolution.default(primals_21, primals_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    gt: "b8[4, 64, 32, 32]" = torch.ops.aten.gt.Scalar(convolution, 0)
    mul: "f32[4, 64, 32, 32]" = torch.ops.aten.mul.Tensor(convolution, 0.2)
    where: "f32[4, 64, 32, 32]" = torch.ops.aten.where.self(gt, convolution, mul);  gt = convolution = mul = None
    convolution_1: "f32[4, 128, 16, 16]" = torch.ops.aten.convolution.default(where, primals_2, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    add: "f32[128]" = torch.ops.aten.add.Tensor(primals_13, 1e-05)
    sqrt: "f32[128]" = torch.ops.aten.sqrt.default(add);  add = None
    reciprocal: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
    mul_1: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
    unsqueeze: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1)
    unsqueeze_1: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    unsqueeze_2: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_1, -1);  mul_1 = None
    unsqueeze_3: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    sub: "f32[4, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_1);  unsqueeze_1 = None
    mul_2: "f32[4, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
    unsqueeze_4: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1)
    unsqueeze_5: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_3: "f32[4, 128, 16, 16]" = torch.ops.aten.mul.Tensor(mul_2, unsqueeze_5);  mul_2 = unsqueeze_5 = None
    unsqueeze_6: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_4, -1);  primals_4 = None
    unsqueeze_7: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_1: "f32[4, 128, 16, 16]" = torch.ops.aten.add.Tensor(mul_3, unsqueeze_7);  mul_3 = unsqueeze_7 = None
    gt_1: "b8[4, 128, 16, 16]" = torch.ops.aten.gt.Scalar(add_1, 0)
    mul_4: "f32[4, 128, 16, 16]" = torch.ops.aten.mul.Tensor(add_1, 0.2)
    where_1: "f32[4, 128, 16, 16]" = torch.ops.aten.where.self(gt_1, add_1, mul_4);  gt_1 = add_1 = mul_4 = None
    convolution_2: "f32[4, 256, 8, 8]" = torch.ops.aten.convolution.default(where_1, primals_5, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    add_2: "f32[256]" = torch.ops.aten.add.Tensor(primals_16, 1e-05)
    sqrt_1: "f32[256]" = torch.ops.aten.sqrt.default(add_2);  add_2 = None
    reciprocal_1: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
    mul_5: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
    unsqueeze_8: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1)
    unsqueeze_9: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    unsqueeze_10: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_5, -1);  mul_5 = None
    unsqueeze_11: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    sub_1: "f32[4, 256, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_9);  unsqueeze_9 = None
    mul_6: "f32[4, 256, 8, 8]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
    unsqueeze_12: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1)
    unsqueeze_13: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_7: "f32[4, 256, 8, 8]" = torch.ops.aten.mul.Tensor(mul_6, unsqueeze_13);  mul_6 = unsqueeze_13 = None
    unsqueeze_14: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1);  primals_7 = None
    unsqueeze_15: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_3: "f32[4, 256, 8, 8]" = torch.ops.aten.add.Tensor(mul_7, unsqueeze_15);  mul_7 = unsqueeze_15 = None
    gt_2: "b8[4, 256, 8, 8]" = torch.ops.aten.gt.Scalar(add_3, 0)
    mul_8: "f32[4, 256, 8, 8]" = torch.ops.aten.mul.Tensor(add_3, 0.2)
    where_2: "f32[4, 256, 8, 8]" = torch.ops.aten.where.self(gt_2, add_3, mul_8);  gt_2 = add_3 = mul_8 = None
    convolution_3: "f32[4, 512, 4, 4]" = torch.ops.aten.convolution.default(where_2, primals_8, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    add_4: "f32[512]" = torch.ops.aten.add.Tensor(primals_19, 1e-05)
    sqrt_2: "f32[512]" = torch.ops.aten.sqrt.default(add_4);  add_4 = None
    reciprocal_2: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
    mul_9: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
    unsqueeze_16: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1)
    unsqueeze_17: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    unsqueeze_18: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_9, -1);  mul_9 = None
    unsqueeze_19: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    sub_2: "f32[4, 512, 4, 4]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_17);  unsqueeze_17 = None
    mul_10: "f32[4, 512, 4, 4]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
    unsqueeze_20: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1)
    unsqueeze_21: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_11: "f32[4, 512, 4, 4]" = torch.ops.aten.mul.Tensor(mul_10, unsqueeze_21);  mul_10 = unsqueeze_21 = None
    unsqueeze_22: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_10, -1);  primals_10 = None
    unsqueeze_23: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_5: "f32[4, 512, 4, 4]" = torch.ops.aten.add.Tensor(mul_11, unsqueeze_23);  mul_11 = unsqueeze_23 = None
    gt_3: "b8[4, 512, 4, 4]" = torch.ops.aten.gt.Scalar(add_5, 0)
    mul_12: "f32[4, 512, 4, 4]" = torch.ops.aten.mul.Tensor(add_5, 0.2)
    where_3: "f32[4, 512, 4, 4]" = torch.ops.aten.where.self(gt_3, add_5, mul_12);  gt_3 = add_5 = mul_12 = None
    convolution_4: "f32[4, 1, 1, 1]" = torch.ops.aten.convolution.default(where_3, primals_11, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    sigmoid: "f32[4, 1, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_4);  convolution_4 = None
    return [sigmoid, primals_1, primals_2, primals_3, primals_5, primals_6, primals_8, primals_9, primals_11, primals_12, primals_13, primals_15, primals_16, primals_18, primals_19, primals_21, where, convolution_1, where_1, convolution_2, where_2, convolution_3, where_3, sigmoid]
    