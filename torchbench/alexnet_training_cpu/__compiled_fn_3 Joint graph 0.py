from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[64, 3, 11, 11]"; primals_2: "f32[64]"; primals_3: "f32[192, 64, 5, 5]"; primals_4: "f32[192]"; primals_5: "f32[384, 192, 3, 3]"; primals_6: "f32[384]"; primals_7: "f32[256, 384, 3, 3]"; primals_8: "f32[256]"; primals_9: "f32[256, 256, 3, 3]"; primals_10: "f32[256]"; primals_11: "f32[4096, 9216]"; primals_12: "f32[4096]"; primals_13: "f32[4096, 4096]"; primals_14: "f32[4096]"; primals_15: "f32[1000, 4096]"; primals_16: "f32[1000]"; primals_17: "f32[4, 3, 224, 224]"; tangents_1: "f32[4, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/alexnet.py:48, code: x = self.features(x)
    convolution: "f32[4, 64, 55, 55]" = torch.ops.aten.convolution.default(primals_17, primals_1, primals_2, [4, 4], [2, 2], [1, 1], False, [0, 0], 1);  primals_2 = None
    relu: "f32[4, 64, 55, 55]" = torch.ops.aten.relu.default(convolution);  convolution = None
    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(relu, [3, 3], [2, 2])
    getitem: "f32[4, 64, 27, 27]" = max_pool2d_with_indices[0]
    getitem_1: "i64[4, 64, 27, 27]" = max_pool2d_with_indices[1];  max_pool2d_with_indices = None
    convolution_1: "f32[4, 192, 27, 27]" = torch.ops.aten.convolution.default(getitem, primals_3, primals_4, [1, 1], [2, 2], [1, 1], False, [0, 0], 1);  primals_4 = None
    relu_1: "f32[4, 192, 27, 27]" = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None
    max_pool2d_with_indices_1 = torch.ops.aten.max_pool2d_with_indices.default(relu_1, [3, 3], [2, 2])
    getitem_2: "f32[4, 192, 13, 13]" = max_pool2d_with_indices_1[0]
    getitem_3: "i64[4, 192, 13, 13]" = max_pool2d_with_indices_1[1];  max_pool2d_with_indices_1 = None
    convolution_2: "f32[4, 384, 13, 13]" = torch.ops.aten.convolution.default(getitem_2, primals_5, primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_6 = None
    relu_2: "f32[4, 384, 13, 13]" = torch.ops.aten.relu.default(convolution_2);  convolution_2 = None
    convolution_3: "f32[4, 256, 13, 13]" = torch.ops.aten.convolution.default(relu_2, primals_7, primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_8 = None
    relu_3: "f32[4, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None
    convolution_4: "f32[4, 256, 13, 13]" = torch.ops.aten.convolution.default(relu_3, primals_9, primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_10 = None
    relu_4: "f32[4, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_4);  convolution_4 = None
    max_pool2d_with_indices_2 = torch.ops.aten.max_pool2d_with_indices.default(relu_4, [3, 3], [2, 2])
    getitem_4: "f32[4, 256, 6, 6]" = max_pool2d_with_indices_2[0]
    getitem_5: "i64[4, 256, 6, 6]" = max_pool2d_with_indices_2[1];  max_pool2d_with_indices_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/alexnet.py:49, code: x = self.avgpool(x)
    _adaptive_avg_pool2d: "f32[4, 256, 6, 6]" = torch.ops.aten._adaptive_avg_pool2d.default(getitem_4, [6, 6])
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/alexnet.py:50, code: x = torch.flatten(x, 1)
    view: "f32[4, 9216]" = torch.ops.aten.view.default(_adaptive_avg_pool2d, [4, 9216]);  _adaptive_avg_pool2d = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/alexnet.py:51, code: x = self.classifier(x)
    clone: "f32[4, 9216]" = torch.ops.aten.clone.default(view);  view = None
    permute: "f32[9216, 4096]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
    addmm: "f32[4, 4096]" = torch.ops.aten.addmm.default(primals_12, clone, permute);  primals_12 = None
    relu_5: "f32[4, 4096]" = torch.ops.aten.relu.default(addmm);  addmm = None
    clone_1: "f32[4, 4096]" = torch.ops.aten.clone.default(relu_5)
    permute_1: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
    addmm_1: "f32[4, 4096]" = torch.ops.aten.addmm.default(primals_14, clone_1, permute_1);  primals_14 = None
    relu_6: "f32[4, 4096]" = torch.ops.aten.relu.default(addmm_1);  addmm_1 = None
    permute_2: "f32[4096, 1000]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
    addmm_2: "f32[4, 1000]" = torch.ops.aten.addmm.default(primals_16, relu_6, permute_2);  primals_16 = None
    permute_3: "f32[1000, 4096]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    mm: "f32[4, 4096]" = torch.ops.aten.mm.default(tangents_1, permute_3);  permute_3 = None
    permute_4: "f32[1000, 4]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 4096]" = torch.ops.aten.mm.default(permute_4, relu_6);  permute_4 = None
    permute_5: "f32[4096, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_1: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_6: "f32[1000, 4096]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
    alias_8: "f32[4, 4096]" = torch.ops.aten.alias.default(relu_6);  relu_6 = None
    alias_9: "f32[4, 4096]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    le: "b8[4, 4096]" = torch.ops.aten.le.Scalar(alias_9, 0);  alias_9 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where: "f32[4, 4096]" = torch.ops.aten.where.self(le, scalar_tensor, mm);  le = scalar_tensor = mm = None
    permute_7: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    mm_2: "f32[4, 4096]" = torch.ops.aten.mm.default(where, permute_7);  permute_7 = None
    permute_8: "f32[4096, 4]" = torch.ops.aten.permute.default(where, [1, 0])
    mm_3: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_8, clone_1);  permute_8 = clone_1 = None
    permute_9: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_2: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(where, [0], True);  where = None
    view_2: "f32[4096]" = torch.ops.aten.view.default(sum_2, [4096]);  sum_2 = None
    permute_10: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    alias_11: "f32[4, 4096]" = torch.ops.aten.alias.default(relu_5);  relu_5 = None
    alias_12: "f32[4, 4096]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    le_1: "b8[4, 4096]" = torch.ops.aten.le.Scalar(alias_12, 0);  alias_12 = None
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_1: "f32[4, 4096]" = torch.ops.aten.where.self(le_1, scalar_tensor_1, mm_2);  le_1 = scalar_tensor_1 = mm_2 = None
    permute_11: "f32[4096, 9216]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    mm_4: "f32[4, 9216]" = torch.ops.aten.mm.default(where_1, permute_11);  permute_11 = None
    permute_12: "f32[4096, 4]" = torch.ops.aten.permute.default(where_1, [1, 0])
    mm_5: "f32[4096, 9216]" = torch.ops.aten.mm.default(permute_12, clone);  permute_12 = clone = None
    permute_13: "f32[9216, 4096]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_3: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(where_1, [0], True);  where_1 = None
    view_3: "f32[4096]" = torch.ops.aten.view.default(sum_3, [4096]);  sum_3 = None
    permute_14: "f32[4096, 9216]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/alexnet.py:50, code: x = torch.flatten(x, 1)
    view_4: "f32[4, 256, 6, 6]" = torch.ops.aten.view.default(mm_4, [4, 256, 6, 6]);  mm_4 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/alexnet.py:49, code: x = self.avgpool(x)
    _adaptive_avg_pool2d_backward: "f32[4, 256, 6, 6]" = torch.ops.aten._adaptive_avg_pool2d_backward.default(view_4, getitem_4);  view_4 = getitem_4 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/alexnet.py:48, code: x = self.features(x)
    max_pool2d_with_indices_backward: "f32[4, 256, 13, 13]" = torch.ops.aten.max_pool2d_with_indices_backward.default(_adaptive_avg_pool2d_backward, relu_4, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_5);  _adaptive_avg_pool2d_backward = getitem_5 = None
    alias_14: "f32[4, 256, 13, 13]" = torch.ops.aten.alias.default(relu_4);  relu_4 = None
    alias_15: "f32[4, 256, 13, 13]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    le_2: "b8[4, 256, 13, 13]" = torch.ops.aten.le.Scalar(alias_15, 0);  alias_15 = None
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_2: "f32[4, 256, 13, 13]" = torch.ops.aten.where.self(le_2, scalar_tensor_2, max_pool2d_with_indices_backward);  le_2 = scalar_tensor_2 = max_pool2d_with_indices_backward = None
    convolution_backward = torch.ops.aten.convolution_backward.default(where_2, relu_3, primals_9, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]);  where_2 = primals_9 = None
    getitem_6: "f32[4, 256, 13, 13]" = convolution_backward[0]
    getitem_7: "f32[256, 256, 3, 3]" = convolution_backward[1]
    getitem_8: "f32[256]" = convolution_backward[2];  convolution_backward = None
    alias_17: "f32[4, 256, 13, 13]" = torch.ops.aten.alias.default(relu_3);  relu_3 = None
    alias_18: "f32[4, 256, 13, 13]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
    le_3: "b8[4, 256, 13, 13]" = torch.ops.aten.le.Scalar(alias_18, 0);  alias_18 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_3: "f32[4, 256, 13, 13]" = torch.ops.aten.where.self(le_3, scalar_tensor_3, getitem_6);  le_3 = scalar_tensor_3 = getitem_6 = None
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(where_3, relu_2, primals_7, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]);  where_3 = primals_7 = None
    getitem_9: "f32[4, 384, 13, 13]" = convolution_backward_1[0]
    getitem_10: "f32[256, 384, 3, 3]" = convolution_backward_1[1]
    getitem_11: "f32[256]" = convolution_backward_1[2];  convolution_backward_1 = None
    alias_20: "f32[4, 384, 13, 13]" = torch.ops.aten.alias.default(relu_2);  relu_2 = None
    alias_21: "f32[4, 384, 13, 13]" = torch.ops.aten.alias.default(alias_20);  alias_20 = None
    le_4: "b8[4, 384, 13, 13]" = torch.ops.aten.le.Scalar(alias_21, 0);  alias_21 = None
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_4: "f32[4, 384, 13, 13]" = torch.ops.aten.where.self(le_4, scalar_tensor_4, getitem_9);  le_4 = scalar_tensor_4 = getitem_9 = None
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(where_4, getitem_2, primals_5, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]);  where_4 = getitem_2 = primals_5 = None
    getitem_12: "f32[4, 192, 13, 13]" = convolution_backward_2[0]
    getitem_13: "f32[384, 192, 3, 3]" = convolution_backward_2[1]
    getitem_14: "f32[384]" = convolution_backward_2[2];  convolution_backward_2 = None
    max_pool2d_with_indices_backward_1: "f32[4, 192, 27, 27]" = torch.ops.aten.max_pool2d_with_indices_backward.default(getitem_12, relu_1, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_3);  getitem_12 = getitem_3 = None
    alias_23: "f32[4, 192, 27, 27]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_24: "f32[4, 192, 27, 27]" = torch.ops.aten.alias.default(alias_23);  alias_23 = None
    le_5: "b8[4, 192, 27, 27]" = torch.ops.aten.le.Scalar(alias_24, 0);  alias_24 = None
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_5: "f32[4, 192, 27, 27]" = torch.ops.aten.where.self(le_5, scalar_tensor_5, max_pool2d_with_indices_backward_1);  le_5 = scalar_tensor_5 = max_pool2d_with_indices_backward_1 = None
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(where_5, getitem, primals_3, [192], [1, 1], [2, 2], [1, 1], False, [0, 0], 1, [True, True, True]);  where_5 = getitem = primals_3 = None
    getitem_15: "f32[4, 64, 27, 27]" = convolution_backward_3[0]
    getitem_16: "f32[192, 64, 5, 5]" = convolution_backward_3[1]
    getitem_17: "f32[192]" = convolution_backward_3[2];  convolution_backward_3 = None
    max_pool2d_with_indices_backward_2: "f32[4, 64, 55, 55]" = torch.ops.aten.max_pool2d_with_indices_backward.default(getitem_15, relu, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_1);  getitem_15 = getitem_1 = None
    alias_26: "f32[4, 64, 55, 55]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_27: "f32[4, 64, 55, 55]" = torch.ops.aten.alias.default(alias_26);  alias_26 = None
    le_6: "b8[4, 64, 55, 55]" = torch.ops.aten.le.Scalar(alias_27, 0);  alias_27 = None
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_6: "f32[4, 64, 55, 55]" = torch.ops.aten.where.self(le_6, scalar_tensor_6, max_pool2d_with_indices_backward_2);  le_6 = scalar_tensor_6 = max_pool2d_with_indices_backward_2 = None
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(where_6, primals_17, primals_1, [64], [4, 4], [2, 2], [1, 1], False, [0, 0], 1, [False, True, True]);  where_6 = primals_17 = primals_1 = None
    getitem_19: "f32[64, 3, 11, 11]" = convolution_backward_4[1]
    getitem_20: "f32[64]" = convolution_backward_4[2];  convolution_backward_4 = None
    return pytree.tree_unflatten([addmm_2, getitem_19, getitem_20, getitem_16, getitem_17, getitem_13, getitem_14, getitem_10, getitem_11, getitem_7, getitem_8, permute_14, view_3, permute_10, view_2, permute_6, view_1, None], self._out_spec)
    