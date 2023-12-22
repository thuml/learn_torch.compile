from __future__ import annotations



def forward(self, primals_1: "f32[64, 3, 4, 4]", primals_2: "f32[128, 64, 4, 4]", primals_3: "f32[128]", primals_5: "f32[256, 128, 4, 4]", primals_6: "f32[256]", primals_8: "f32[512, 256, 4, 4]", primals_9: "f32[512]", primals_11: "f32[1, 512, 4, 4]", primals_12: "f32[128]", primals_13: "f32[128]", primals_15: "f32[256]", primals_16: "f32[256]", primals_18: "f32[512]", primals_19: "f32[512]", primals_21: "f32[4, 3, 64, 64]", where: "f32[4, 64, 32, 32]", convolution_1: "f32[4, 128, 16, 16]", where_1: "f32[4, 128, 16, 16]", convolution_2: "f32[4, 256, 8, 8]", where_2: "f32[4, 256, 8, 8]", convolution_3: "f32[4, 512, 4, 4]", where_3: "f32[4, 512, 4, 4]", sigmoid: "f32[4, 1, 1, 1]", tangents_1: "f32[4, 1, 1, 1]"):
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/dcgan/__init__.py:134, code: return self.main(input)
    sub_3: "f32[4, 1, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid)
    mul_13: "f32[4, 1, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid, sub_3);  sigmoid = sub_3 = None
    mul_14: "f32[4, 1, 1, 1]" = torch.ops.aten.mul.Tensor(tangents_1, mul_13);  tangents_1 = mul_13 = None
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_14, where_3, primals_11, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_14 = primals_11 = None
    getitem: "f32[4, 512, 4, 4]" = convolution_backward[0]
    getitem_1: "f32[1, 512, 4, 4]" = convolution_backward[1];  convolution_backward = None
    gt_4: "b8[4, 512, 4, 4]" = torch.ops.aten.gt.Scalar(where_3, 0);  where_3 = None
    mul_15: "f32[4, 512, 4, 4]" = torch.ops.aten.mul.Tensor(getitem, 0.2)
    where_4: "f32[4, 512, 4, 4]" = torch.ops.aten.where.self(gt_4, getitem, mul_15);  gt_4 = getitem = mul_15 = None
    add_6: "f32[512]" = torch.ops.aten.add.Tensor(primals_19, 1e-05);  primals_19 = None
    rsqrt: "f32[512]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    unsqueeze_24: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_18, 0);  primals_18 = None
    unsqueeze_25: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, 2);  unsqueeze_24 = None
    unsqueeze_26: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_25, 3);  unsqueeze_25 = None
    sum_1: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_4: "f32[4, 512, 4, 4]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_26);  convolution_3 = unsqueeze_26 = None
    mul_16: "f32[4, 512, 4, 4]" = torch.ops.aten.mul.Tensor(where_4, sub_4);  sub_4 = None
    sum_2: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_16, [0, 2, 3]);  mul_16 = None
    mul_21: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt, primals_9);  primals_9 = None
    unsqueeze_33: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_21, 0);  mul_21 = None
    unsqueeze_34: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_33, 2);  unsqueeze_33 = None
    unsqueeze_35: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, 3);  unsqueeze_34 = None
    mul_22: "f32[4, 512, 4, 4]" = torch.ops.aten.mul.Tensor(where_4, unsqueeze_35);  where_4 = unsqueeze_35 = None
    mul_23: "f32[512]" = torch.ops.aten.mul.Tensor(sum_2, rsqrt);  sum_2 = rsqrt = None
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_22, where_2, primals_8, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_22 = primals_8 = None
    getitem_3: "f32[4, 256, 8, 8]" = convolution_backward_1[0]
    getitem_4: "f32[512, 256, 4, 4]" = convolution_backward_1[1];  convolution_backward_1 = None
    gt_5: "b8[4, 256, 8, 8]" = torch.ops.aten.gt.Scalar(where_2, 0);  where_2 = None
    mul_24: "f32[4, 256, 8, 8]" = torch.ops.aten.mul.Tensor(getitem_3, 0.2)
    where_5: "f32[4, 256, 8, 8]" = torch.ops.aten.where.self(gt_5, getitem_3, mul_24);  gt_5 = getitem_3 = mul_24 = None
    add_7: "f32[256]" = torch.ops.aten.add.Tensor(primals_16, 1e-05);  primals_16 = None
    rsqrt_1: "f32[256]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    unsqueeze_36: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_15, 0);  primals_15 = None
    unsqueeze_37: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, 2);  unsqueeze_36 = None
    unsqueeze_38: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_37, 3);  unsqueeze_37 = None
    sum_3: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_5: "f32[4, 256, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_38);  convolution_2 = unsqueeze_38 = None
    mul_25: "f32[4, 256, 8, 8]" = torch.ops.aten.mul.Tensor(where_5, sub_5);  sub_5 = None
    sum_4: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_25, [0, 2, 3]);  mul_25 = None
    mul_30: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_1, primals_6);  primals_6 = None
    unsqueeze_45: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_30, 0);  mul_30 = None
    unsqueeze_46: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_45, 2);  unsqueeze_45 = None
    unsqueeze_47: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, 3);  unsqueeze_46 = None
    mul_31: "f32[4, 256, 8, 8]" = torch.ops.aten.mul.Tensor(where_5, unsqueeze_47);  where_5 = unsqueeze_47 = None
    mul_32: "f32[256]" = torch.ops.aten.mul.Tensor(sum_4, rsqrt_1);  sum_4 = rsqrt_1 = None
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_31, where_1, primals_5, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_31 = primals_5 = None
    getitem_6: "f32[4, 128, 16, 16]" = convolution_backward_2[0]
    getitem_7: "f32[256, 128, 4, 4]" = convolution_backward_2[1];  convolution_backward_2 = None
    gt_6: "b8[4, 128, 16, 16]" = torch.ops.aten.gt.Scalar(where_1, 0);  where_1 = None
    mul_33: "f32[4, 128, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_6, 0.2)
    where_6: "f32[4, 128, 16, 16]" = torch.ops.aten.where.self(gt_6, getitem_6, mul_33);  gt_6 = getitem_6 = mul_33 = None
    add_8: "f32[128]" = torch.ops.aten.add.Tensor(primals_13, 1e-05);  primals_13 = None
    rsqrt_2: "f32[128]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
    unsqueeze_48: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_12, 0);  primals_12 = None
    unsqueeze_49: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, 2);  unsqueeze_48 = None
    unsqueeze_50: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_49, 3);  unsqueeze_49 = None
    sum_5: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_6: "f32[4, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_50);  convolution_1 = unsqueeze_50 = None
    mul_34: "f32[4, 128, 16, 16]" = torch.ops.aten.mul.Tensor(where_6, sub_6);  sub_6 = None
    sum_6: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_34, [0, 2, 3]);  mul_34 = None
    mul_39: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_2, primals_3);  primals_3 = None
    unsqueeze_57: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_39, 0);  mul_39 = None
    unsqueeze_58: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_57, 2);  unsqueeze_57 = None
    unsqueeze_59: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, 3);  unsqueeze_58 = None
    mul_40: "f32[4, 128, 16, 16]" = torch.ops.aten.mul.Tensor(where_6, unsqueeze_59);  where_6 = unsqueeze_59 = None
    mul_41: "f32[128]" = torch.ops.aten.mul.Tensor(sum_6, rsqrt_2);  sum_6 = rsqrt_2 = None
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_40, where, primals_2, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_40 = primals_2 = None
    getitem_9: "f32[4, 64, 32, 32]" = convolution_backward_3[0]
    getitem_10: "f32[128, 64, 4, 4]" = convolution_backward_3[1];  convolution_backward_3 = None
    gt_7: "b8[4, 64, 32, 32]" = torch.ops.aten.gt.Scalar(where, 0);  where = None
    mul_42: "f32[4, 64, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_9, 0.2)
    where_7: "f32[4, 64, 32, 32]" = torch.ops.aten.where.self(gt_7, getitem_9, mul_42);  gt_7 = getitem_9 = mul_42 = None
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(where_7, primals_21, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  where_7 = primals_21 = primals_1 = None
    getitem_13: "f32[64, 3, 4, 4]" = convolution_backward_4[1];  convolution_backward_4 = None
    return [getitem_13, getitem_10, mul_41, sum_5, getitem_7, mul_32, sum_3, getitem_4, mul_23, sum_1, getitem_1, None, None, None, None, None, None, None, None, None, None]
    