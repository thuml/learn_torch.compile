from __future__ import annotations



def forward(self, primals_1: "f32[64, 3, 3, 3]", primals_3: "f32[64, 64, 3, 3]", primals_5: "f32[128, 64, 3, 3]", primals_7: "f32[128, 128, 3, 3]", primals_9: "f32[256, 128, 3, 3]", primals_11: "f32[256, 256, 3, 3]", primals_13: "f32[256, 256, 3, 3]", primals_15: "f32[512, 256, 3, 3]", primals_17: "f32[512, 512, 3, 3]", primals_19: "f32[512, 512, 3, 3]", primals_21: "f32[512, 512, 3, 3]", primals_23: "f32[512, 512, 3, 3]", primals_25: "f32[512, 512, 3, 3]", primals_33: "f32[4, 3, 224, 224]", relu: "f32[4, 64, 224, 224]", relu_1: "f32[4, 64, 224, 224]", getitem: "f32[4, 64, 112, 112]", getitem_1: "i64[4, 64, 112, 112]", relu_2: "f32[4, 128, 112, 112]", relu_3: "f32[4, 128, 112, 112]", getitem_2: "f32[4, 128, 56, 56]", getitem_3: "i64[4, 128, 56, 56]", relu_4: "f32[4, 256, 56, 56]", relu_5: "f32[4, 256, 56, 56]", relu_6: "f32[4, 256, 56, 56]", getitem_4: "f32[4, 256, 28, 28]", getitem_5: "i64[4, 256, 28, 28]", relu_7: "f32[4, 512, 28, 28]", relu_8: "f32[4, 512, 28, 28]", relu_9: "f32[4, 512, 28, 28]", getitem_6: "f32[4, 512, 14, 14]", getitem_7: "i64[4, 512, 14, 14]", relu_10: "f32[4, 512, 14, 14]", relu_11: "f32[4, 512, 14, 14]", relu_12: "f32[4, 512, 14, 14]", getitem_8: "f32[4, 512, 7, 7]", getitem_9: "i64[4, 512, 7, 7]", view: "f32[4, 25088]", clone: "f32[4, 4096]", clone_1: "f32[4, 4096]", permute_3: "f32[1000, 4096]", le: "b8[4, 4096]", permute_7: "f32[4096, 4096]", le_1: "b8[4, 4096]", permute_11: "f32[4096, 25088]", tangents_1: "f32[4, 1000]"):
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/vgg.py:69, code: x = self.classifier(x)
    mm: "f32[4, 4096]" = torch.ops.aten.mm.default(tangents_1, permute_3);  permute_3 = None
    permute_4: "f32[1000, 4]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 4096]" = torch.ops.aten.mm.default(permute_4, clone_1);  permute_4 = clone_1 = None
    permute_5: "f32[4096, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_1: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_6: "f32[1000, 4096]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where: "f32[4, 4096]" = torch.ops.aten.where.self(le, full_default, mm);  le = mm = None
    mm_2: "f32[4, 4096]" = torch.ops.aten.mm.default(where, permute_7);  permute_7 = None
    permute_8: "f32[4096, 4]" = torch.ops.aten.permute.default(where, [1, 0])
    mm_3: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_8, clone);  permute_8 = clone = None
    permute_9: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_2: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(where, [0], True);  where = None
    view_2: "f32[4096]" = torch.ops.aten.view.default(sum_2, [4096]);  sum_2 = None
    permute_10: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    where_1: "f32[4, 4096]" = torch.ops.aten.where.self(le_1, full_default, mm_2);  le_1 = mm_2 = None
    mm_4: "f32[4, 25088]" = torch.ops.aten.mm.default(where_1, permute_11);  permute_11 = None
    permute_12: "f32[4096, 4]" = torch.ops.aten.permute.default(where_1, [1, 0])
    mm_5: "f32[4096, 25088]" = torch.ops.aten.mm.default(permute_12, view);  permute_12 = view = None
    permute_13: "f32[25088, 4096]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_3: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(where_1, [0], True);  where_1 = None
    view_3: "f32[4096]" = torch.ops.aten.view.default(sum_3, [4096]);  sum_3 = None
    permute_14: "f32[4096, 25088]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/vgg.py:68, code: x = torch.flatten(x, 1)
    view_4: "f32[4, 512, 7, 7]" = torch.ops.aten.view.default(mm_4, [4, 512, 7, 7]);  mm_4 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/vgg.py:67, code: x = self.avgpool(x)
    _adaptive_avg_pool2d_backward: "f32[4, 512, 7, 7]" = torch.ops.aten._adaptive_avg_pool2d_backward.default(view_4, getitem_8);  view_4 = getitem_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/vgg.py:66, code: x = self.features(x)
    max_pool2d_with_indices_backward: "f32[4, 512, 14, 14]" = torch.ops.aten.max_pool2d_with_indices_backward.default(_adaptive_avg_pool2d_backward, relu_12, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_9);  _adaptive_avg_pool2d_backward = getitem_9 = None
    alias_22: "f32[4, 512, 14, 14]" = torch.ops.aten.alias.default(relu_12);  relu_12 = None
    alias_23: "f32[4, 512, 14, 14]" = torch.ops.aten.alias.default(alias_22);  alias_22 = None
    le_2: "b8[4, 512, 14, 14]" = torch.ops.aten.le.Scalar(alias_23, 0);  alias_23 = None
    where_2: "f32[4, 512, 14, 14]" = torch.ops.aten.where.self(le_2, full_default, max_pool2d_with_indices_backward);  le_2 = max_pool2d_with_indices_backward = None
    sum_4: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    convolution_backward = torch.ops.aten.convolution_backward.default(where_2, relu_11, primals_25, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_2 = primals_25 = None
    getitem_10: "f32[4, 512, 14, 14]" = convolution_backward[0]
    getitem_11: "f32[512, 512, 3, 3]" = convolution_backward[1];  convolution_backward = None
    alias_25: "f32[4, 512, 14, 14]" = torch.ops.aten.alias.default(relu_11);  relu_11 = None
    alias_26: "f32[4, 512, 14, 14]" = torch.ops.aten.alias.default(alias_25);  alias_25 = None
    le_3: "b8[4, 512, 14, 14]" = torch.ops.aten.le.Scalar(alias_26, 0);  alias_26 = None
    where_3: "f32[4, 512, 14, 14]" = torch.ops.aten.where.self(le_3, full_default, getitem_10);  le_3 = getitem_10 = None
    sum_5: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(where_3, relu_10, primals_23, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_3 = primals_23 = None
    getitem_13: "f32[4, 512, 14, 14]" = convolution_backward_1[0]
    getitem_14: "f32[512, 512, 3, 3]" = convolution_backward_1[1];  convolution_backward_1 = None
    alias_28: "f32[4, 512, 14, 14]" = torch.ops.aten.alias.default(relu_10);  relu_10 = None
    alias_29: "f32[4, 512, 14, 14]" = torch.ops.aten.alias.default(alias_28);  alias_28 = None
    le_4: "b8[4, 512, 14, 14]" = torch.ops.aten.le.Scalar(alias_29, 0);  alias_29 = None
    where_4: "f32[4, 512, 14, 14]" = torch.ops.aten.where.self(le_4, full_default, getitem_13);  le_4 = getitem_13 = None
    sum_6: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(where_4, getitem_6, primals_21, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_4 = getitem_6 = primals_21 = None
    getitem_16: "f32[4, 512, 14, 14]" = convolution_backward_2[0]
    getitem_17: "f32[512, 512, 3, 3]" = convolution_backward_2[1];  convolution_backward_2 = None
    max_pool2d_with_indices_backward_1: "f32[4, 512, 28, 28]" = torch.ops.aten.max_pool2d_with_indices_backward.default(getitem_16, relu_9, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_7);  getitem_16 = getitem_7 = None
    alias_31: "f32[4, 512, 28, 28]" = torch.ops.aten.alias.default(relu_9);  relu_9 = None
    alias_32: "f32[4, 512, 28, 28]" = torch.ops.aten.alias.default(alias_31);  alias_31 = None
    le_5: "b8[4, 512, 28, 28]" = torch.ops.aten.le.Scalar(alias_32, 0);  alias_32 = None
    where_5: "f32[4, 512, 28, 28]" = torch.ops.aten.where.self(le_5, full_default, max_pool2d_with_indices_backward_1);  le_5 = max_pool2d_with_indices_backward_1 = None
    sum_7: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(where_5, relu_8, primals_19, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_5 = primals_19 = None
    getitem_19: "f32[4, 512, 28, 28]" = convolution_backward_3[0]
    getitem_20: "f32[512, 512, 3, 3]" = convolution_backward_3[1];  convolution_backward_3 = None
    alias_34: "f32[4, 512, 28, 28]" = torch.ops.aten.alias.default(relu_8);  relu_8 = None
    alias_35: "f32[4, 512, 28, 28]" = torch.ops.aten.alias.default(alias_34);  alias_34 = None
    le_6: "b8[4, 512, 28, 28]" = torch.ops.aten.le.Scalar(alias_35, 0);  alias_35 = None
    where_6: "f32[4, 512, 28, 28]" = torch.ops.aten.where.self(le_6, full_default, getitem_19);  le_6 = getitem_19 = None
    sum_8: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(where_6, relu_7, primals_17, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_6 = primals_17 = None
    getitem_22: "f32[4, 512, 28, 28]" = convolution_backward_4[0]
    getitem_23: "f32[512, 512, 3, 3]" = convolution_backward_4[1];  convolution_backward_4 = None
    alias_37: "f32[4, 512, 28, 28]" = torch.ops.aten.alias.default(relu_7);  relu_7 = None
    alias_38: "f32[4, 512, 28, 28]" = torch.ops.aten.alias.default(alias_37);  alias_37 = None
    le_7: "b8[4, 512, 28, 28]" = torch.ops.aten.le.Scalar(alias_38, 0);  alias_38 = None
    where_7: "f32[4, 512, 28, 28]" = torch.ops.aten.where.self(le_7, full_default, getitem_22);  le_7 = getitem_22 = None
    sum_9: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(where_7, getitem_4, primals_15, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_7 = getitem_4 = primals_15 = None
    getitem_25: "f32[4, 256, 28, 28]" = convolution_backward_5[0]
    getitem_26: "f32[512, 256, 3, 3]" = convolution_backward_5[1];  convolution_backward_5 = None
    max_pool2d_with_indices_backward_2: "f32[4, 256, 56, 56]" = torch.ops.aten.max_pool2d_with_indices_backward.default(getitem_25, relu_6, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_5);  getitem_25 = getitem_5 = None
    alias_40: "f32[4, 256, 56, 56]" = torch.ops.aten.alias.default(relu_6);  relu_6 = None
    alias_41: "f32[4, 256, 56, 56]" = torch.ops.aten.alias.default(alias_40);  alias_40 = None
    le_8: "b8[4, 256, 56, 56]" = torch.ops.aten.le.Scalar(alias_41, 0);  alias_41 = None
    where_8: "f32[4, 256, 56, 56]" = torch.ops.aten.where.self(le_8, full_default, max_pool2d_with_indices_backward_2);  le_8 = max_pool2d_with_indices_backward_2 = None
    sum_10: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(where_8, relu_5, primals_13, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_8 = primals_13 = None
    getitem_28: "f32[4, 256, 56, 56]" = convolution_backward_6[0]
    getitem_29: "f32[256, 256, 3, 3]" = convolution_backward_6[1];  convolution_backward_6 = None
    alias_43: "f32[4, 256, 56, 56]" = torch.ops.aten.alias.default(relu_5);  relu_5 = None
    alias_44: "f32[4, 256, 56, 56]" = torch.ops.aten.alias.default(alias_43);  alias_43 = None
    le_9: "b8[4, 256, 56, 56]" = torch.ops.aten.le.Scalar(alias_44, 0);  alias_44 = None
    where_9: "f32[4, 256, 56, 56]" = torch.ops.aten.where.self(le_9, full_default, getitem_28);  le_9 = getitem_28 = None
    sum_11: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(where_9, relu_4, primals_11, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_9 = primals_11 = None
    getitem_31: "f32[4, 256, 56, 56]" = convolution_backward_7[0]
    getitem_32: "f32[256, 256, 3, 3]" = convolution_backward_7[1];  convolution_backward_7 = None
    alias_46: "f32[4, 256, 56, 56]" = torch.ops.aten.alias.default(relu_4);  relu_4 = None
    alias_47: "f32[4, 256, 56, 56]" = torch.ops.aten.alias.default(alias_46);  alias_46 = None
    le_10: "b8[4, 256, 56, 56]" = torch.ops.aten.le.Scalar(alias_47, 0);  alias_47 = None
    where_10: "f32[4, 256, 56, 56]" = torch.ops.aten.where.self(le_10, full_default, getitem_31);  le_10 = getitem_31 = None
    sum_12: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(where_10, getitem_2, primals_9, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_10 = getitem_2 = primals_9 = None
    getitem_34: "f32[4, 128, 56, 56]" = convolution_backward_8[0]
    getitem_35: "f32[256, 128, 3, 3]" = convolution_backward_8[1];  convolution_backward_8 = None
    max_pool2d_with_indices_backward_3: "f32[4, 128, 112, 112]" = torch.ops.aten.max_pool2d_with_indices_backward.default(getitem_34, relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_3);  getitem_34 = getitem_3 = None
    alias_49: "f32[4, 128, 112, 112]" = torch.ops.aten.alias.default(relu_3);  relu_3 = None
    alias_50: "f32[4, 128, 112, 112]" = torch.ops.aten.alias.default(alias_49);  alias_49 = None
    le_11: "b8[4, 128, 112, 112]" = torch.ops.aten.le.Scalar(alias_50, 0);  alias_50 = None
    where_11: "f32[4, 128, 112, 112]" = torch.ops.aten.where.self(le_11, full_default, max_pool2d_with_indices_backward_3);  le_11 = max_pool2d_with_indices_backward_3 = None
    sum_13: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(where_11, relu_2, primals_7, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_11 = primals_7 = None
    getitem_37: "f32[4, 128, 112, 112]" = convolution_backward_9[0]
    getitem_38: "f32[128, 128, 3, 3]" = convolution_backward_9[1];  convolution_backward_9 = None
    alias_52: "f32[4, 128, 112, 112]" = torch.ops.aten.alias.default(relu_2);  relu_2 = None
    alias_53: "f32[4, 128, 112, 112]" = torch.ops.aten.alias.default(alias_52);  alias_52 = None
    le_12: "b8[4, 128, 112, 112]" = torch.ops.aten.le.Scalar(alias_53, 0);  alias_53 = None
    where_12: "f32[4, 128, 112, 112]" = torch.ops.aten.where.self(le_12, full_default, getitem_37);  le_12 = getitem_37 = None
    sum_14: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(where_12, getitem, primals_5, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_12 = getitem = primals_5 = None
    getitem_40: "f32[4, 64, 112, 112]" = convolution_backward_10[0]
    getitem_41: "f32[128, 64, 3, 3]" = convolution_backward_10[1];  convolution_backward_10 = None
    max_pool2d_with_indices_backward_4: "f32[4, 64, 224, 224]" = torch.ops.aten.max_pool2d_with_indices_backward.default(getitem_40, relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_1);  getitem_40 = getitem_1 = None
    alias_55: "f32[4, 64, 224, 224]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_56: "f32[4, 64, 224, 224]" = torch.ops.aten.alias.default(alias_55);  alias_55 = None
    le_13: "b8[4, 64, 224, 224]" = torch.ops.aten.le.Scalar(alias_56, 0);  alias_56 = None
    where_13: "f32[4, 64, 224, 224]" = torch.ops.aten.where.self(le_13, full_default, max_pool2d_with_indices_backward_4);  le_13 = max_pool2d_with_indices_backward_4 = None
    sum_15: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(where_13, relu, primals_3, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_13 = primals_3 = None
    getitem_43: "f32[4, 64, 224, 224]" = convolution_backward_11[0]
    getitem_44: "f32[64, 64, 3, 3]" = convolution_backward_11[1];  convolution_backward_11 = None
    alias_58: "f32[4, 64, 224, 224]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_59: "f32[4, 64, 224, 224]" = torch.ops.aten.alias.default(alias_58);  alias_58 = None
    le_14: "b8[4, 64, 224, 224]" = torch.ops.aten.le.Scalar(alias_59, 0);  alias_59 = None
    where_14: "f32[4, 64, 224, 224]" = torch.ops.aten.where.self(le_14, full_default, getitem_43);  le_14 = full_default = getitem_43 = None
    sum_16: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(where_14, primals_33, primals_1, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  where_14 = primals_33 = primals_1 = None
    getitem_47: "f32[64, 3, 3, 3]" = convolution_backward_12[1];  convolution_backward_12 = None
    return [getitem_47, sum_16, getitem_44, sum_15, getitem_41, sum_14, getitem_38, sum_13, getitem_35, sum_12, getitem_32, sum_11, getitem_29, sum_10, getitem_26, sum_9, getitem_23, sum_8, getitem_20, sum_7, getitem_17, sum_6, getitem_14, sum_5, getitem_11, sum_4, permute_14, view_3, permute_10, view_2, permute_6, view_1, None]
    