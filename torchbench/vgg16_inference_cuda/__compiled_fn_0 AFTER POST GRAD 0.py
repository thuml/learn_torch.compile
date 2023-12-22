from __future__ import annotations



def forward(self, arg0_1: "f32[64, 3, 3, 3]", arg1_1: "f32[64]", arg2_1: "f32[64, 64, 3, 3]", arg3_1: "f32[64]", arg4_1: "f32[128, 64, 3, 3]", arg5_1: "f32[128]", arg6_1: "f32[128, 128, 3, 3]", arg7_1: "f32[128]", arg8_1: "f32[256, 128, 3, 3]", arg9_1: "f32[256]", arg10_1: "f32[256, 256, 3, 3]", arg11_1: "f32[256]", arg12_1: "f32[256, 256, 3, 3]", arg13_1: "f32[256]", arg14_1: "f32[512, 256, 3, 3]", arg15_1: "f32[512]", arg16_1: "f32[512, 512, 3, 3]", arg17_1: "f32[512]", arg18_1: "f32[512, 512, 3, 3]", arg19_1: "f32[512]", arg20_1: "f32[512, 512, 3, 3]", arg21_1: "f32[512]", arg22_1: "f32[512, 512, 3, 3]", arg23_1: "f32[512]", arg24_1: "f32[512, 512, 3, 3]", arg25_1: "f32[512]", arg26_1: "f32[4096, 25088]", arg27_1: "f32[4096]", arg28_1: "f32[4096, 4096]", arg29_1: "f32[4096]", arg30_1: "f32[1000, 4096]", arg31_1: "f32[1000]", arg32_1: "f32[4, 3, 224, 224]"):
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/vgg.py:66, code: x = self.features(x)
    convolution: "f32[4, 64, 224, 224]" = torch.ops.aten.convolution.default(arg32_1, arg0_1, arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg32_1 = arg0_1 = arg1_1 = None
    relu: "f32[4, 64, 224, 224]" = torch.ops.aten.relu.default(convolution);  convolution = None
    convolution_1: "f32[4, 64, 224, 224]" = torch.ops.aten.convolution.default(relu, arg2_1, arg3_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu = arg2_1 = arg3_1 = None
    relu_1: "f32[4, 64, 224, 224]" = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None
    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(relu_1, [2, 2], [2, 2]);  relu_1 = None
    getitem: "f32[4, 64, 112, 112]" = max_pool2d_with_indices[0];  max_pool2d_with_indices = None
    convolution_2: "f32[4, 128, 112, 112]" = torch.ops.aten.convolution.default(getitem, arg4_1, arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem = arg4_1 = arg5_1 = None
    relu_2: "f32[4, 128, 112, 112]" = torch.ops.aten.relu.default(convolution_2);  convolution_2 = None
    convolution_3: "f32[4, 128, 112, 112]" = torch.ops.aten.convolution.default(relu_2, arg6_1, arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_2 = arg6_1 = arg7_1 = None
    relu_3: "f32[4, 128, 112, 112]" = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None
    max_pool2d_with_indices_1 = torch.ops.aten.max_pool2d_with_indices.default(relu_3, [2, 2], [2, 2]);  relu_3 = None
    getitem_2: "f32[4, 128, 56, 56]" = max_pool2d_with_indices_1[0];  max_pool2d_with_indices_1 = None
    convolution_4: "f32[4, 256, 56, 56]" = torch.ops.aten.convolution.default(getitem_2, arg8_1, arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_2 = arg8_1 = arg9_1 = None
    relu_4: "f32[4, 256, 56, 56]" = torch.ops.aten.relu.default(convolution_4);  convolution_4 = None
    convolution_5: "f32[4, 256, 56, 56]" = torch.ops.aten.convolution.default(relu_4, arg10_1, arg11_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_4 = arg10_1 = arg11_1 = None
    relu_5: "f32[4, 256, 56, 56]" = torch.ops.aten.relu.default(convolution_5);  convolution_5 = None
    convolution_6: "f32[4, 256, 56, 56]" = torch.ops.aten.convolution.default(relu_5, arg12_1, arg13_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_5 = arg12_1 = arg13_1 = None
    relu_6: "f32[4, 256, 56, 56]" = torch.ops.aten.relu.default(convolution_6);  convolution_6 = None
    max_pool2d_with_indices_2 = torch.ops.aten.max_pool2d_with_indices.default(relu_6, [2, 2], [2, 2]);  relu_6 = None
    getitem_4: "f32[4, 256, 28, 28]" = max_pool2d_with_indices_2[0];  max_pool2d_with_indices_2 = None
    convolution_7: "f32[4, 512, 28, 28]" = torch.ops.aten.convolution.default(getitem_4, arg14_1, arg15_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_4 = arg14_1 = arg15_1 = None
    relu_7: "f32[4, 512, 28, 28]" = torch.ops.aten.relu.default(convolution_7);  convolution_7 = None
    convolution_8: "f32[4, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_7, arg16_1, arg17_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_7 = arg16_1 = arg17_1 = None
    relu_8: "f32[4, 512, 28, 28]" = torch.ops.aten.relu.default(convolution_8);  convolution_8 = None
    convolution_9: "f32[4, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_8, arg18_1, arg19_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_8 = arg18_1 = arg19_1 = None
    relu_9: "f32[4, 512, 28, 28]" = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None
    max_pool2d_with_indices_3 = torch.ops.aten.max_pool2d_with_indices.default(relu_9, [2, 2], [2, 2]);  relu_9 = None
    getitem_6: "f32[4, 512, 14, 14]" = max_pool2d_with_indices_3[0];  max_pool2d_with_indices_3 = None
    convolution_10: "f32[4, 512, 14, 14]" = torch.ops.aten.convolution.default(getitem_6, arg20_1, arg21_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_6 = arg20_1 = arg21_1 = None
    relu_10: "f32[4, 512, 14, 14]" = torch.ops.aten.relu.default(convolution_10);  convolution_10 = None
    convolution_11: "f32[4, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_10, arg22_1, arg23_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_10 = arg22_1 = arg23_1 = None
    relu_11: "f32[4, 512, 14, 14]" = torch.ops.aten.relu.default(convolution_11);  convolution_11 = None
    convolution_12: "f32[4, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_11, arg24_1, arg25_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_11 = arg24_1 = arg25_1 = None
    relu_12: "f32[4, 512, 14, 14]" = torch.ops.aten.relu.default(convolution_12);  convolution_12 = None
    max_pool2d_with_indices_4 = torch.ops.aten.max_pool2d_with_indices.default(relu_12, [2, 2], [2, 2]);  relu_12 = None
    getitem_8: "f32[4, 512, 7, 7]" = max_pool2d_with_indices_4[0];  max_pool2d_with_indices_4 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/vgg.py:67, code: x = self.avgpool(x)
    _adaptive_avg_pool2d: "f32[4, 512, 7, 7]" = torch.ops.aten._adaptive_avg_pool2d.default(getitem_8, [7, 7]);  getitem_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/vgg.py:68, code: x = torch.flatten(x, 1)
    view: "f32[4, 25088]" = torch.ops.aten.reshape.default(_adaptive_avg_pool2d, [4, 25088]);  _adaptive_avg_pool2d = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/vgg.py:69, code: x = self.classifier(x)
    permute: "f32[25088, 4096]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
    
    # No stacktrace found for following nodes
    mm_default_1: "f32[4, 4096]" = torch.ops.aten.mm.default(view, permute);  view = permute = None
    add_tensor_1: "f32[4, 4096]" = torch.ops.aten.add.Tensor(mm_default_1, arg27_1);  mm_default_1 = arg27_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/vgg.py:69, code: x = self.classifier(x)
    relu_13: "f32[4, 4096]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None
    permute_1: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
    
    # No stacktrace found for following nodes
    mm_default: "f32[4, 4096]" = torch.ops.aten.mm.default(relu_13, permute_1);  relu_13 = permute_1 = None
    add_tensor: "f32[4, 4096]" = torch.ops.aten.add.Tensor(mm_default, arg29_1);  mm_default = arg29_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/vgg.py:69, code: x = self.classifier(x)
    relu_14: "f32[4, 4096]" = torch.ops.aten.relu.default(add_tensor);  add_tensor = None
    permute_2: "f32[4096, 1000]" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
    addmm_2: "f32[4, 1000]" = torch.ops.aten.addmm.default(arg31_1, relu_14, permute_2);  arg31_1 = relu_14 = permute_2 = None
    return (addmm_2,)
    