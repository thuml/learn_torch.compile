from __future__ import annotations



def forward(self, primals_1: "f32[64, 3, 11, 11]", primals_3: "f32[192, 64, 5, 5]", primals_5: "f32[384, 192, 3, 3]", primals_7: "f32[256, 384, 3, 3]", primals_9: "f32[256, 256, 3, 3]", primals_17: "f32[4, 3, 224, 224]", relu: "f32[4, 64, 55, 55]", getitem: "f32[4, 64, 27, 27]", getitem_1: "i64[4, 64, 27, 27]", relu_1: "f32[4, 192, 27, 27]", getitem_2: "f32[4, 192, 13, 13]", getitem_3: "i64[4, 192, 13, 13]", relu_2: "f32[4, 384, 13, 13]", relu_3: "f32[4, 256, 13, 13]", relu_4: "f32[4, 256, 13, 13]", getitem_4: "f32[4, 256, 6, 6]", getitem_5: "i64[4, 256, 6, 6]", clone: "f32[4, 9216]", clone_1: "f32[4, 4096]", relu_6: "f32[4, 4096]", permute_3: "f32[1000, 4096]", permute_7: "f32[4096, 4096]", le_1: "b8[4, 4096]", permute_11: "f32[4096, 9216]", tangents_1: "f32[4, 1000]"):
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/alexnet.py:51, code: x = self.classifier(x)
    mm: "f32[4, 4096]" = torch.ops.aten.mm.default(tangents_1, permute_3);  permute_3 = None
    permute_4: "f32[1000, 4]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 4096]" = torch.ops.aten.mm.default(permute_4, relu_6);  permute_4 = None
    permute_5: "f32[4096, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_1: "f32[1000]" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
    permute_6: "f32[1000, 4096]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
    le: "b8[4, 4096]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where: "f32[4, 4096]" = torch.ops.aten.where.self(le, full_default, mm);  le = mm = None
    mm_2: "f32[4, 4096]" = torch.ops.aten.mm.default(where, permute_7);  permute_7 = None
    permute_8: "f32[4096, 4]" = torch.ops.aten.permute.default(where, [1, 0])
    mm_3: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_8, clone_1);  permute_8 = clone_1 = None
    permute_9: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_2: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(where, [0], True);  where = None
    view_2: "f32[4096]" = torch.ops.aten.reshape.default(sum_2, [4096]);  sum_2 = None
    permute_10: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    where_1: "f32[4, 4096]" = torch.ops.aten.where.self(le_1, full_default, mm_2);  le_1 = mm_2 = None
    mm_4: "f32[4, 9216]" = torch.ops.aten.mm.default(where_1, permute_11);  permute_11 = None
    permute_12: "f32[4096, 4]" = torch.ops.aten.permute.default(where_1, [1, 0])
    mm_5: "f32[4096, 9216]" = torch.ops.aten.mm.default(permute_12, clone);  permute_12 = clone = None
    permute_13: "f32[9216, 4096]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_3: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(where_1, [0], True);  where_1 = None
    view_3: "f32[4096]" = torch.ops.aten.reshape.default(sum_3, [4096]);  sum_3 = None
    permute_14: "f32[4096, 9216]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/alexnet.py:50, code: x = torch.flatten(x, 1)
    view_4: "f32[4, 256, 6, 6]" = torch.ops.aten.reshape.default(mm_4, [4, 256, 6, 6]);  mm_4 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/alexnet.py:49, code: x = self.avgpool(x)
    _adaptive_avg_pool2d_backward: "f32[4, 256, 6, 6]" = torch.ops.aten._adaptive_avg_pool2d_backward.default(view_4, getitem_4);  view_4 = getitem_4 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/alexnet.py:48, code: x = self.features(x)
    max_pool2d_with_indices_backward: "f32[4, 256, 13, 13]" = torch.ops.aten.max_pool2d_with_indices_backward.default(_adaptive_avg_pool2d_backward, relu_4, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_5);  _adaptive_avg_pool2d_backward = getitem_5 = None
    le_2: "b8[4, 256, 13, 13]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
    where_2: "f32[4, 256, 13, 13]" = torch.ops.aten.where.self(le_2, full_default, max_pool2d_with_indices_backward);  le_2 = max_pool2d_with_indices_backward = None
    convolution_backward = torch.ops.aten.convolution_backward.default(where_2, relu_3, primals_9, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]);  where_2 = primals_9 = None
    getitem_6: "f32[4, 256, 13, 13]" = convolution_backward[0]
    getitem_7: "f32[256, 256, 3, 3]" = convolution_backward[1]
    getitem_8: "f32[256]" = convolution_backward[2];  convolution_backward = None
    le_3: "b8[4, 256, 13, 13]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
    where_3: "f32[4, 256, 13, 13]" = torch.ops.aten.where.self(le_3, full_default, getitem_6);  le_3 = getitem_6 = None
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(where_3, relu_2, primals_7, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]);  where_3 = primals_7 = None
    getitem_9: "f32[4, 384, 13, 13]" = convolution_backward_1[0]
    getitem_10: "f32[256, 384, 3, 3]" = convolution_backward_1[1]
    getitem_11: "f32[256]" = convolution_backward_1[2];  convolution_backward_1 = None
    le_4: "b8[4, 384, 13, 13]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
    where_4: "f32[4, 384, 13, 13]" = torch.ops.aten.where.self(le_4, full_default, getitem_9);  le_4 = getitem_9 = None
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(where_4, getitem_2, primals_5, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]);  where_4 = getitem_2 = primals_5 = None
    getitem_12: "f32[4, 192, 13, 13]" = convolution_backward_2[0]
    getitem_13: "f32[384, 192, 3, 3]" = convolution_backward_2[1]
    getitem_14: "f32[384]" = convolution_backward_2[2];  convolution_backward_2 = None
    max_pool2d_with_indices_backward_1: "f32[4, 192, 27, 27]" = torch.ops.aten.max_pool2d_with_indices_backward.default(getitem_12, relu_1, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_3);  getitem_12 = getitem_3 = None
    le_5: "b8[4, 192, 27, 27]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
    where_5: "f32[4, 192, 27, 27]" = torch.ops.aten.where.self(le_5, full_default, max_pool2d_with_indices_backward_1);  le_5 = max_pool2d_with_indices_backward_1 = None
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(where_5, getitem, primals_3, [192], [1, 1], [2, 2], [1, 1], False, [0, 0], 1, [True, True, True]);  where_5 = getitem = primals_3 = None
    getitem_15: "f32[4, 64, 27, 27]" = convolution_backward_3[0]
    getitem_16: "f32[192, 64, 5, 5]" = convolution_backward_3[1]
    getitem_17: "f32[192]" = convolution_backward_3[2];  convolution_backward_3 = None
    max_pool2d_with_indices_backward_2: "f32[4, 64, 55, 55]" = torch.ops.aten.max_pool2d_with_indices_backward.default(getitem_15, relu, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_1);  getitem_15 = getitem_1 = None
    le_6: "b8[4, 64, 55, 55]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
    where_6: "f32[4, 64, 55, 55]" = torch.ops.aten.where.self(le_6, full_default, max_pool2d_with_indices_backward_2);  le_6 = full_default = max_pool2d_with_indices_backward_2 = None
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(where_6, primals_17, primals_1, [64], [4, 4], [2, 2], [1, 1], False, [0, 0], 1, [False, True, True]);  where_6 = primals_17 = primals_1 = None
    getitem_19: "f32[64, 3, 11, 11]" = convolution_backward_4[1]
    getitem_20: "f32[64]" = convolution_backward_4[2];  convolution_backward_4 = None
    return [getitem_19, getitem_20, getitem_16, getitem_17, getitem_13, getitem_14, getitem_10, getitem_11, getitem_7, getitem_8, permute_14, view_3, permute_10, view_2, permute_6, view_1, None]
    