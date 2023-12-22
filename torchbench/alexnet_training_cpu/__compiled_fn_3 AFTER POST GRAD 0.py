from __future__ import annotations



def forward(self, primals_1: "f32[64, 3, 11, 11]", primals_2: "f32[64]", primals_3: "f32[192, 64, 5, 5]", primals_4: "f32[192]", primals_5: "f32[384, 192, 3, 3]", primals_6: "f32[384]", primals_7: "f32[256, 384, 3, 3]", primals_8: "f32[256]", primals_9: "f32[256, 256, 3, 3]", primals_10: "f32[256]", primals_11: "f32[4096, 9216]", primals_12: "f32[4096]", primals_13: "f32[4096, 4096]", primals_14: "f32[4096]", primals_15: "f32[1000, 4096]", primals_16: "f32[1000]", primals_17: "f32[4, 3, 224, 224]"):
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
    view: "f32[4, 9216]" = torch.ops.aten.reshape.default(_adaptive_avg_pool2d, [4, 9216]);  _adaptive_avg_pool2d = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/alexnet.py:51, code: x = self.classifier(x)
    permute: "f32[9216, 4096]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
    addmm: "f32[4, 4096]" = torch.ops.aten.addmm.default(primals_12, view, permute);  primals_12 = None
    relu_5: "f32[4, 4096]" = torch.ops.aten.relu.default(addmm);  addmm = None
    permute_1: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
    addmm_1: "f32[4, 4096]" = torch.ops.aten.addmm.default(primals_14, relu_5, permute_1);  primals_14 = None
    relu_6: "f32[4, 4096]" = torch.ops.aten.relu.default(addmm_1);  addmm_1 = None
    permute_2: "f32[4096, 1000]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
    addmm_2: "f32[4, 1000]" = torch.ops.aten.addmm.default(primals_16, relu_6, permute_2);  primals_16 = None
    permute_3: "f32[1000, 4096]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    permute_7: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    le_1: "b8[4, 4096]" = torch.ops.aten.le.Scalar(relu_5, 0)
    permute_11: "f32[4096, 9216]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    return [addmm_2, primals_1, primals_3, primals_5, primals_7, primals_9, primals_17, relu, getitem, getitem_1, relu_1, getitem_2, getitem_3, relu_2, relu_3, relu_4, getitem_4, getitem_5, view, relu_5, relu_6, permute_3, permute_7, le_1, permute_11]
    