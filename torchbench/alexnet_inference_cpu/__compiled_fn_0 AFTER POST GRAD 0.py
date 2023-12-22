from __future__ import annotations



def forward(self, arg0_1: "f32[64, 3, 11, 11]", arg1_1: "f32[64]", arg2_1: "f32[192, 64, 5, 5]", arg3_1: "f32[192]", arg4_1: "f32[384, 192, 3, 3]", arg5_1: "f32[384]", arg6_1: "f32[256, 384, 3, 3]", arg7_1: "f32[256]", arg8_1: "f32[256, 256, 3, 3]", arg9_1: "f32[256]", arg10_1: "f32[4096, 9216]", arg11_1: "f32[4096]", arg12_1: "f32[4096, 4096]", arg13_1: "f32[4096]", arg14_1: "f32[1000, 4096]", arg15_1: "f32[1000]", arg16_1: "f32[4, 3, 224, 224]"):
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/alexnet.py:48, code: x = self.features(x)
    convolution: "f32[4, 64, 55, 55]" = torch.ops.aten.convolution.default(arg16_1, arg0_1, arg1_1, [4, 4], [2, 2], [1, 1], False, [0, 0], 1);  arg16_1 = arg0_1 = arg1_1 = None
    relu: "f32[4, 64, 55, 55]" = torch.ops.aten.relu.default(convolution);  convolution = None
    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(relu, [3, 3], [2, 2]);  relu = None
    getitem: "f32[4, 64, 27, 27]" = max_pool2d_with_indices[0];  max_pool2d_with_indices = None
    convolution_1: "f32[4, 192, 27, 27]" = torch.ops.aten.convolution.default(getitem, arg2_1, arg3_1, [1, 1], [2, 2], [1, 1], False, [0, 0], 1);  getitem = arg2_1 = arg3_1 = None
    relu_1: "f32[4, 192, 27, 27]" = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None
    max_pool2d_with_indices_1 = torch.ops.aten.max_pool2d_with_indices.default(relu_1, [3, 3], [2, 2]);  relu_1 = None
    getitem_2: "f32[4, 192, 13, 13]" = max_pool2d_with_indices_1[0];  max_pool2d_with_indices_1 = None
    convolution_2: "f32[4, 384, 13, 13]" = torch.ops.aten.convolution.default(getitem_2, arg4_1, arg5_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_2 = arg4_1 = arg5_1 = None
    relu_2: "f32[4, 384, 13, 13]" = torch.ops.aten.relu.default(convolution_2);  convolution_2 = None
    convolution_3: "f32[4, 256, 13, 13]" = torch.ops.aten.convolution.default(relu_2, arg6_1, arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_2 = arg6_1 = arg7_1 = None
    relu_3: "f32[4, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None
    convolution_4: "f32[4, 256, 13, 13]" = torch.ops.aten.convolution.default(relu_3, arg8_1, arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_3 = arg8_1 = arg9_1 = None
    relu_4: "f32[4, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_4);  convolution_4 = None
    max_pool2d_with_indices_2 = torch.ops.aten.max_pool2d_with_indices.default(relu_4, [3, 3], [2, 2]);  relu_4 = None
    getitem_4: "f32[4, 256, 6, 6]" = max_pool2d_with_indices_2[0];  max_pool2d_with_indices_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/alexnet.py:49, code: x = self.avgpool(x)
    _adaptive_avg_pool2d: "f32[4, 256, 6, 6]" = torch.ops.aten._adaptive_avg_pool2d.default(getitem_4, [6, 6]);  getitem_4 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/alexnet.py:50, code: x = torch.flatten(x, 1)
    view: "f32[4, 9216]" = torch.ops.aten.reshape.default(_adaptive_avg_pool2d, [4, 9216]);  _adaptive_avg_pool2d = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/alexnet.py:51, code: x = self.classifier(x)
    permute: "f32[9216, 4096]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
    addmm: "f32[4, 4096]" = torch.ops.aten.addmm.default(arg11_1, view, permute);  arg11_1 = view = permute = None
    relu_5: "f32[4, 4096]" = torch.ops.aten.relu.default(addmm);  addmm = None
    permute_1: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
    addmm_1: "f32[4, 4096]" = torch.ops.aten.addmm.default(arg13_1, relu_5, permute_1);  arg13_1 = relu_5 = permute_1 = None
    relu_6: "f32[4, 4096]" = torch.ops.aten.relu.default(addmm_1);  addmm_1 = None
    permute_2: "f32[4096, 1000]" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
    addmm_2: "f32[4, 1000]" = torch.ops.aten.addmm.default(arg15_1, relu_6, permute_2);  arg15_1 = relu_6 = permute_2 = None
    return (addmm_2,)
    