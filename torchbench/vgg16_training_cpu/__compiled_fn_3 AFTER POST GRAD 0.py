from __future__ import annotations



def forward(self, primals_1: "f32[64, 3, 3, 3]", primals_2: "f32[64]", primals_3: "f32[64, 64, 3, 3]", primals_4: "f32[64]", primals_5: "f32[128, 64, 3, 3]", primals_6: "f32[128]", primals_7: "f32[128, 128, 3, 3]", primals_8: "f32[128]", primals_9: "f32[256, 128, 3, 3]", primals_10: "f32[256]", primals_11: "f32[256, 256, 3, 3]", primals_12: "f32[256]", primals_13: "f32[256, 256, 3, 3]", primals_14: "f32[256]", primals_15: "f32[512, 256, 3, 3]", primals_16: "f32[512]", primals_17: "f32[512, 512, 3, 3]", primals_18: "f32[512]", primals_19: "f32[512, 512, 3, 3]", primals_20: "f32[512]", primals_21: "f32[512, 512, 3, 3]", primals_22: "f32[512]", primals_23: "f32[512, 512, 3, 3]", primals_24: "f32[512]", primals_25: "f32[512, 512, 3, 3]", primals_26: "f32[512]", primals_27: "f32[4096, 25088]", primals_28: "f32[4096]", primals_29: "f32[4096, 4096]", primals_30: "f32[4096]", primals_31: "f32[1000, 4096]", primals_32: "f32[1000]", primals_33: "f32[4, 3, 224, 224]"):
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/vgg.py:66, code: x = self.features(x)
    convolution: "f32[4, 64, 224, 224]" = torch.ops.aten.convolution.default(primals_33, primals_1, primals_2, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_2 = None
    relu: "f32[4, 64, 224, 224]" = torch.ops.aten.relu.default(convolution);  convolution = None
    convolution_1: "f32[4, 64, 224, 224]" = torch.ops.aten.convolution.default(relu, primals_3, primals_4, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_4 = None
    relu_1: "f32[4, 64, 224, 224]" = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None
    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(relu_1, [2, 2], [2, 2])
    getitem: "f32[4, 64, 112, 112]" = max_pool2d_with_indices[0]
    getitem_1: "i64[4, 64, 112, 112]" = max_pool2d_with_indices[1];  max_pool2d_with_indices = None
    convolution_2: "f32[4, 128, 112, 112]" = torch.ops.aten.convolution.default(getitem, primals_5, primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_6 = None
    relu_2: "f32[4, 128, 112, 112]" = torch.ops.aten.relu.default(convolution_2);  convolution_2 = None
    convolution_3: "f32[4, 128, 112, 112]" = torch.ops.aten.convolution.default(relu_2, primals_7, primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_8 = None
    relu_3: "f32[4, 128, 112, 112]" = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None
    max_pool2d_with_indices_1 = torch.ops.aten.max_pool2d_with_indices.default(relu_3, [2, 2], [2, 2])
    getitem_2: "f32[4, 128, 56, 56]" = max_pool2d_with_indices_1[0]
    getitem_3: "i64[4, 128, 56, 56]" = max_pool2d_with_indices_1[1];  max_pool2d_with_indices_1 = None
    convolution_4: "f32[4, 256, 56, 56]" = torch.ops.aten.convolution.default(getitem_2, primals_9, primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_10 = None
    relu_4: "f32[4, 256, 56, 56]" = torch.ops.aten.relu.default(convolution_4);  convolution_4 = None
    convolution_5: "f32[4, 256, 56, 56]" = torch.ops.aten.convolution.default(relu_4, primals_11, primals_12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_12 = None
    relu_5: "f32[4, 256, 56, 56]" = torch.ops.aten.relu.default(convolution_5);  convolution_5 = None
    convolution_6: "f32[4, 256, 56, 56]" = torch.ops.aten.convolution.default(relu_5, primals_13, primals_14, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_14 = None
    relu_6: "f32[4, 256, 56, 56]" = torch.ops.aten.relu.default(convolution_6);  convolution_6 = None
    max_pool2d_with_indices_2 = torch.ops.aten.max_pool2d_with_indices.default(relu_6, [2, 2], [2, 2])
    getitem_4: "f32[4, 256, 28, 28]" = max_pool2d_with_indices_2[0]
    getitem_5: "i64[4, 256, 28, 28]" = max_pool2d_with_indices_2[1];  max_pool2d_with_indices_2 = None
    convolution_7: "f32[4, 512, 28, 28]" = torch.ops.aten.convolution.default(getitem_4, primals_15, primals_16, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_16 = None
    relu_7: "f32[4, 512, 28, 28]" = torch.ops.aten.relu.default(convolution_7);  convolution_7 = None
    convolution_8: "f32[4, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_7, primals_17, primals_18, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_18 = None
    relu_8: "f32[4, 512, 28, 28]" = torch.ops.aten.relu.default(convolution_8);  convolution_8 = None
    convolution_9: "f32[4, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_8, primals_19, primals_20, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_20 = None
    relu_9: "f32[4, 512, 28, 28]" = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None
    max_pool2d_with_indices_3 = torch.ops.aten.max_pool2d_with_indices.default(relu_9, [2, 2], [2, 2])
    getitem_6: "f32[4, 512, 14, 14]" = max_pool2d_with_indices_3[0]
    getitem_7: "i64[4, 512, 14, 14]" = max_pool2d_with_indices_3[1];  max_pool2d_with_indices_3 = None
    convolution_10: "f32[4, 512, 14, 14]" = torch.ops.aten.convolution.default(getitem_6, primals_21, primals_22, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_22 = None
    relu_10: "f32[4, 512, 14, 14]" = torch.ops.aten.relu.default(convolution_10);  convolution_10 = None
    convolution_11: "f32[4, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_10, primals_23, primals_24, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_24 = None
    relu_11: "f32[4, 512, 14, 14]" = torch.ops.aten.relu.default(convolution_11);  convolution_11 = None
    convolution_12: "f32[4, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_11, primals_25, primals_26, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_26 = None
    relu_12: "f32[4, 512, 14, 14]" = torch.ops.aten.relu.default(convolution_12);  convolution_12 = None
    max_pool2d_with_indices_4 = torch.ops.aten.max_pool2d_with_indices.default(relu_12, [2, 2], [2, 2])
    getitem_8: "f32[4, 512, 7, 7]" = max_pool2d_with_indices_4[0]
    getitem_9: "i64[4, 512, 7, 7]" = max_pool2d_with_indices_4[1];  max_pool2d_with_indices_4 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/vgg.py:67, code: x = self.avgpool(x)
    _adaptive_avg_pool2d: "f32[4, 512, 7, 7]" = torch.ops.aten._adaptive_avg_pool2d.default(getitem_8, [7, 7])
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/vgg.py:68, code: x = torch.flatten(x, 1)
    view: "f32[4, 25088]" = torch.ops.aten.reshape.default(_adaptive_avg_pool2d, [4, 25088]);  _adaptive_avg_pool2d = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/vgg.py:69, code: x = self.classifier(x)
    permute: "f32[25088, 4096]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
    addmm: "f32[4, 4096]" = torch.ops.aten.addmm.default(primals_28, view, permute);  primals_28 = None
    relu_13: "f32[4, 4096]" = torch.ops.aten.relu.default(addmm);  addmm = None
    permute_1: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_29, [1, 0]);  primals_29 = None
    addmm_1: "f32[4, 4096]" = torch.ops.aten.addmm.default(primals_30, relu_13, permute_1);  primals_30 = None
    relu_14: "f32[4, 4096]" = torch.ops.aten.relu.default(addmm_1);  addmm_1 = None
    permute_2: "f32[4096, 1000]" = torch.ops.aten.permute.default(primals_31, [1, 0]);  primals_31 = None
    addmm_2: "f32[4, 1000]" = torch.ops.aten.addmm.default(primals_32, relu_14, permute_2);  primals_32 = None
    permute_3: "f32[1000, 4096]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    le: "b8[4, 4096]" = torch.ops.aten.le.Scalar(relu_14, 0)
    permute_7: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    le_1: "b8[4, 4096]" = torch.ops.aten.le.Scalar(relu_13, 0)
    permute_11: "f32[4096, 25088]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    return [addmm_2, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_33, relu, relu_1, getitem, getitem_1, relu_2, relu_3, getitem_2, getitem_3, relu_4, relu_5, relu_6, getitem_4, getitem_5, relu_7, relu_8, relu_9, getitem_6, getitem_7, relu_10, relu_11, relu_12, getitem_8, getitem_9, view, relu_13, relu_14, permute_3, le, permute_7, le_1, permute_11]
    