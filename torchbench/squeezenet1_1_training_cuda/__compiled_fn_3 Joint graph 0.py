from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[64, 3, 3, 3]"; primals_2: "f32[64]"; primals_3: "f32[16, 64, 1, 1]"; primals_4: "f32[16]"; primals_5: "f32[64, 16, 1, 1]"; primals_6: "f32[64]"; primals_7: "f32[64, 16, 3, 3]"; primals_8: "f32[64]"; primals_9: "f32[16, 128, 1, 1]"; primals_10: "f32[16]"; primals_11: "f32[64, 16, 1, 1]"; primals_12: "f32[64]"; primals_13: "f32[64, 16, 3, 3]"; primals_14: "f32[64]"; primals_15: "f32[32, 128, 1, 1]"; primals_16: "f32[32]"; primals_17: "f32[128, 32, 1, 1]"; primals_18: "f32[128]"; primals_19: "f32[128, 32, 3, 3]"; primals_20: "f32[128]"; primals_21: "f32[32, 256, 1, 1]"; primals_22: "f32[32]"; primals_23: "f32[128, 32, 1, 1]"; primals_24: "f32[128]"; primals_25: "f32[128, 32, 3, 3]"; primals_26: "f32[128]"; primals_27: "f32[48, 256, 1, 1]"; primals_28: "f32[48]"; primals_29: "f32[192, 48, 1, 1]"; primals_30: "f32[192]"; primals_31: "f32[192, 48, 3, 3]"; primals_32: "f32[192]"; primals_33: "f32[48, 384, 1, 1]"; primals_34: "f32[48]"; primals_35: "f32[192, 48, 1, 1]"; primals_36: "f32[192]"; primals_37: "f32[192, 48, 3, 3]"; primals_38: "f32[192]"; primals_39: "f32[64, 384, 1, 1]"; primals_40: "f32[64]"; primals_41: "f32[256, 64, 1, 1]"; primals_42: "f32[256]"; primals_43: "f32[256, 64, 3, 3]"; primals_44: "f32[256]"; primals_45: "f32[64, 512, 1, 1]"; primals_46: "f32[64]"; primals_47: "f32[256, 64, 1, 1]"; primals_48: "f32[256]"; primals_49: "f32[256, 64, 3, 3]"; primals_50: "f32[256]"; primals_51: "f32[1000, 512, 1, 1]"; primals_52: "f32[1000]"; primals_53: "f32[4, 3, 224, 224]"; tangents_1: "f32[4, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:95, code: x = self.features(x)
    convolution: "f32[4, 64, 111, 111]" = torch.ops.aten.convolution.default(primals_53, primals_1, primals_2, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_2 = None
    relu: "f32[4, 64, 111, 111]" = torch.ops.aten.relu.default(convolution);  convolution = None
    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(relu, [3, 3], [2, 2], [0, 0], [1, 1], True)
    getitem: "f32[4, 64, 55, 55]" = max_pool2d_with_indices[0]
    getitem_1: "i64[4, 64, 55, 55]" = max_pool2d_with_indices[1];  max_pool2d_with_indices = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    convolution_1: "f32[4, 16, 55, 55]" = torch.ops.aten.convolution.default(getitem, primals_3, primals_4, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_4 = None
    relu_1: "f32[4, 16, 55, 55]" = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    convolution_2: "f32[4, 64, 55, 55]" = torch.ops.aten.convolution.default(relu_1, primals_5, primals_6, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_6 = None
    relu_2: "f32[4, 64, 55, 55]" = torch.ops.aten.relu.default(convolution_2);  convolution_2 = None
    convolution_3: "f32[4, 64, 55, 55]" = torch.ops.aten.convolution.default(relu_1, primals_7, primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_8 = None
    relu_3: "f32[4, 64, 55, 55]" = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    cat: "f32[4, 128, 55, 55]" = torch.ops.aten.cat.default([relu_2, relu_3], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    convolution_4: "f32[4, 16, 55, 55]" = torch.ops.aten.convolution.default(cat, primals_9, primals_10, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_10 = None
    relu_4: "f32[4, 16, 55, 55]" = torch.ops.aten.relu.default(convolution_4);  convolution_4 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    convolution_5: "f32[4, 64, 55, 55]" = torch.ops.aten.convolution.default(relu_4, primals_11, primals_12, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_12 = None
    relu_5: "f32[4, 64, 55, 55]" = torch.ops.aten.relu.default(convolution_5);  convolution_5 = None
    convolution_6: "f32[4, 64, 55, 55]" = torch.ops.aten.convolution.default(relu_4, primals_13, primals_14, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_14 = None
    relu_6: "f32[4, 64, 55, 55]" = torch.ops.aten.relu.default(convolution_6);  convolution_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    cat_1: "f32[4, 128, 55, 55]" = torch.ops.aten.cat.default([relu_5, relu_6], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:95, code: x = self.features(x)
    max_pool2d_with_indices_1 = torch.ops.aten.max_pool2d_with_indices.default(cat_1, [3, 3], [2, 2], [0, 0], [1, 1], True)
    getitem_2: "f32[4, 128, 27, 27]" = max_pool2d_with_indices_1[0]
    getitem_3: "i64[4, 128, 27, 27]" = max_pool2d_with_indices_1[1];  max_pool2d_with_indices_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    convolution_7: "f32[4, 32, 27, 27]" = torch.ops.aten.convolution.default(getitem_2, primals_15, primals_16, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_16 = None
    relu_7: "f32[4, 32, 27, 27]" = torch.ops.aten.relu.default(convolution_7);  convolution_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    convolution_8: "f32[4, 128, 27, 27]" = torch.ops.aten.convolution.default(relu_7, primals_17, primals_18, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_18 = None
    relu_8: "f32[4, 128, 27, 27]" = torch.ops.aten.relu.default(convolution_8);  convolution_8 = None
    convolution_9: "f32[4, 128, 27, 27]" = torch.ops.aten.convolution.default(relu_7, primals_19, primals_20, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_20 = None
    relu_9: "f32[4, 128, 27, 27]" = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    cat_2: "f32[4, 256, 27, 27]" = torch.ops.aten.cat.default([relu_8, relu_9], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    convolution_10: "f32[4, 32, 27, 27]" = torch.ops.aten.convolution.default(cat_2, primals_21, primals_22, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_22 = None
    relu_10: "f32[4, 32, 27, 27]" = torch.ops.aten.relu.default(convolution_10);  convolution_10 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    convolution_11: "f32[4, 128, 27, 27]" = torch.ops.aten.convolution.default(relu_10, primals_23, primals_24, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_24 = None
    relu_11: "f32[4, 128, 27, 27]" = torch.ops.aten.relu.default(convolution_11);  convolution_11 = None
    convolution_12: "f32[4, 128, 27, 27]" = torch.ops.aten.convolution.default(relu_10, primals_25, primals_26, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_26 = None
    relu_12: "f32[4, 128, 27, 27]" = torch.ops.aten.relu.default(convolution_12);  convolution_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    cat_3: "f32[4, 256, 27, 27]" = torch.ops.aten.cat.default([relu_11, relu_12], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:95, code: x = self.features(x)
    max_pool2d_with_indices_2 = torch.ops.aten.max_pool2d_with_indices.default(cat_3, [3, 3], [2, 2], [0, 0], [1, 1], True)
    getitem_4: "f32[4, 256, 13, 13]" = max_pool2d_with_indices_2[0]
    getitem_5: "i64[4, 256, 13, 13]" = max_pool2d_with_indices_2[1];  max_pool2d_with_indices_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    convolution_13: "f32[4, 48, 13, 13]" = torch.ops.aten.convolution.default(getitem_4, primals_27, primals_28, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_28 = None
    relu_13: "f32[4, 48, 13, 13]" = torch.ops.aten.relu.default(convolution_13);  convolution_13 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    convolution_14: "f32[4, 192, 13, 13]" = torch.ops.aten.convolution.default(relu_13, primals_29, primals_30, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_30 = None
    relu_14: "f32[4, 192, 13, 13]" = torch.ops.aten.relu.default(convolution_14);  convolution_14 = None
    convolution_15: "f32[4, 192, 13, 13]" = torch.ops.aten.convolution.default(relu_13, primals_31, primals_32, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_32 = None
    relu_15: "f32[4, 192, 13, 13]" = torch.ops.aten.relu.default(convolution_15);  convolution_15 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    cat_4: "f32[4, 384, 13, 13]" = torch.ops.aten.cat.default([relu_14, relu_15], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    convolution_16: "f32[4, 48, 13, 13]" = torch.ops.aten.convolution.default(cat_4, primals_33, primals_34, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_34 = None
    relu_16: "f32[4, 48, 13, 13]" = torch.ops.aten.relu.default(convolution_16);  convolution_16 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    convolution_17: "f32[4, 192, 13, 13]" = torch.ops.aten.convolution.default(relu_16, primals_35, primals_36, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_36 = None
    relu_17: "f32[4, 192, 13, 13]" = torch.ops.aten.relu.default(convolution_17);  convolution_17 = None
    convolution_18: "f32[4, 192, 13, 13]" = torch.ops.aten.convolution.default(relu_16, primals_37, primals_38, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_38 = None
    relu_18: "f32[4, 192, 13, 13]" = torch.ops.aten.relu.default(convolution_18);  convolution_18 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    cat_5: "f32[4, 384, 13, 13]" = torch.ops.aten.cat.default([relu_17, relu_18], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    convolution_19: "f32[4, 64, 13, 13]" = torch.ops.aten.convolution.default(cat_5, primals_39, primals_40, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_40 = None
    relu_19: "f32[4, 64, 13, 13]" = torch.ops.aten.relu.default(convolution_19);  convolution_19 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    convolution_20: "f32[4, 256, 13, 13]" = torch.ops.aten.convolution.default(relu_19, primals_41, primals_42, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_42 = None
    relu_20: "f32[4, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_20);  convolution_20 = None
    convolution_21: "f32[4, 256, 13, 13]" = torch.ops.aten.convolution.default(relu_19, primals_43, primals_44, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_44 = None
    relu_21: "f32[4, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_21);  convolution_21 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    cat_6: "f32[4, 512, 13, 13]" = torch.ops.aten.cat.default([relu_20, relu_21], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    convolution_22: "f32[4, 64, 13, 13]" = torch.ops.aten.convolution.default(cat_6, primals_45, primals_46, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_46 = None
    relu_22: "f32[4, 64, 13, 13]" = torch.ops.aten.relu.default(convolution_22);  convolution_22 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    convolution_23: "f32[4, 256, 13, 13]" = torch.ops.aten.convolution.default(relu_22, primals_47, primals_48, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_48 = None
    relu_23: "f32[4, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_23);  convolution_23 = None
    convolution_24: "f32[4, 256, 13, 13]" = torch.ops.aten.convolution.default(relu_22, primals_49, primals_50, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_50 = None
    relu_24: "f32[4, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_24);  convolution_24 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    cat_7: "f32[4, 512, 13, 13]" = torch.ops.aten.cat.default([relu_23, relu_24], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:96, code: x = self.classifier(x)
    clone: "f32[4, 512, 13, 13]" = torch.ops.aten.clone.default(cat_7);  cat_7 = None
    convolution_25: "f32[4, 1000, 13, 13]" = torch.ops.aten.convolution.default(clone, primals_51, primals_52, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_52 = None
    relu_25: "f32[4, 1000, 13, 13]" = torch.ops.aten.relu.default(convolution_25);  convolution_25 = None
    mean: "f32[4, 1000, 1, 1]" = torch.ops.aten.mean.dim(relu_25, [-1, -2], True)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:97, code: return torch.flatten(x, 1)
    view: "f32[4, 1000]" = torch.ops.aten.view.default(mean, [4, 1000]);  mean = None
    view_1: "f32[4, 1000, 1, 1]" = torch.ops.aten.view.default(tangents_1, [4, 1000, 1, 1]);  tangents_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:96, code: x = self.classifier(x)
    expand: "f32[4, 1000, 13, 13]" = torch.ops.aten.expand.default(view_1, [4, 1000, 13, 13]);  view_1 = None
    div: "f32[4, 1000, 13, 13]" = torch.ops.aten.div.Scalar(expand, 169);  expand = None
    alias_27: "f32[4, 1000, 13, 13]" = torch.ops.aten.alias.default(relu_25);  relu_25 = None
    alias_28: "f32[4, 1000, 13, 13]" = torch.ops.aten.alias.default(alias_27);  alias_27 = None
    le: "b8[4, 1000, 13, 13]" = torch.ops.aten.le.Scalar(alias_28, 0);  alias_28 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where: "f32[4, 1000, 13, 13]" = torch.ops.aten.where.self(le, scalar_tensor, div);  le = scalar_tensor = div = None
    sum_1: "f32[1000]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    convolution_backward = torch.ops.aten.convolution_backward.default(where, clone, primals_51, [1000], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where = clone = primals_51 = None
    getitem_6: "f32[4, 512, 13, 13]" = convolution_backward[0]
    getitem_7: "f32[1000, 512, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    slice_1: "f32[4, 256, 13, 13]" = torch.ops.aten.slice.Tensor(getitem_6, 1, 0, 256)
    slice_2: "f32[4, 256, 13, 13]" = torch.ops.aten.slice.Tensor(getitem_6, 1, 256, 512);  getitem_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    alias_30: "f32[4, 256, 13, 13]" = torch.ops.aten.alias.default(relu_24);  relu_24 = None
    alias_31: "f32[4, 256, 13, 13]" = torch.ops.aten.alias.default(alias_30);  alias_30 = None
    le_1: "b8[4, 256, 13, 13]" = torch.ops.aten.le.Scalar(alias_31, 0);  alias_31 = None
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_1: "f32[4, 256, 13, 13]" = torch.ops.aten.where.self(le_1, scalar_tensor_1, slice_2);  le_1 = scalar_tensor_1 = slice_2 = None
    sum_2: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(where_1, relu_22, primals_49, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_1 = primals_49 = None
    getitem_9: "f32[4, 64, 13, 13]" = convolution_backward_1[0]
    getitem_10: "f32[256, 64, 3, 3]" = convolution_backward_1[1];  convolution_backward_1 = None
    alias_33: "f32[4, 256, 13, 13]" = torch.ops.aten.alias.default(relu_23);  relu_23 = None
    alias_34: "f32[4, 256, 13, 13]" = torch.ops.aten.alias.default(alias_33);  alias_33 = None
    le_2: "b8[4, 256, 13, 13]" = torch.ops.aten.le.Scalar(alias_34, 0);  alias_34 = None
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_2: "f32[4, 256, 13, 13]" = torch.ops.aten.where.self(le_2, scalar_tensor_2, slice_1);  le_2 = scalar_tensor_2 = slice_1 = None
    sum_3: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(where_2, relu_22, primals_47, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_2 = primals_47 = None
    getitem_12: "f32[4, 64, 13, 13]" = convolution_backward_2[0]
    getitem_13: "f32[256, 64, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    add: "f32[4, 64, 13, 13]" = torch.ops.aten.add.Tensor(getitem_9, getitem_12);  getitem_9 = getitem_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    alias_36: "f32[4, 64, 13, 13]" = torch.ops.aten.alias.default(relu_22);  relu_22 = None
    alias_37: "f32[4, 64, 13, 13]" = torch.ops.aten.alias.default(alias_36);  alias_36 = None
    le_3: "b8[4, 64, 13, 13]" = torch.ops.aten.le.Scalar(alias_37, 0);  alias_37 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_3: "f32[4, 64, 13, 13]" = torch.ops.aten.where.self(le_3, scalar_tensor_3, add);  le_3 = scalar_tensor_3 = add = None
    sum_4: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(where_3, cat_6, primals_45, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_3 = cat_6 = primals_45 = None
    getitem_15: "f32[4, 512, 13, 13]" = convolution_backward_3[0]
    getitem_16: "f32[64, 512, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    slice_3: "f32[4, 256, 13, 13]" = torch.ops.aten.slice.Tensor(getitem_15, 1, 0, 256)
    slice_4: "f32[4, 256, 13, 13]" = torch.ops.aten.slice.Tensor(getitem_15, 1, 256, 512);  getitem_15 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    alias_39: "f32[4, 256, 13, 13]" = torch.ops.aten.alias.default(relu_21);  relu_21 = None
    alias_40: "f32[4, 256, 13, 13]" = torch.ops.aten.alias.default(alias_39);  alias_39 = None
    le_4: "b8[4, 256, 13, 13]" = torch.ops.aten.le.Scalar(alias_40, 0);  alias_40 = None
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_4: "f32[4, 256, 13, 13]" = torch.ops.aten.where.self(le_4, scalar_tensor_4, slice_4);  le_4 = scalar_tensor_4 = slice_4 = None
    sum_5: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(where_4, relu_19, primals_43, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_4 = primals_43 = None
    getitem_18: "f32[4, 64, 13, 13]" = convolution_backward_4[0]
    getitem_19: "f32[256, 64, 3, 3]" = convolution_backward_4[1];  convolution_backward_4 = None
    alias_42: "f32[4, 256, 13, 13]" = torch.ops.aten.alias.default(relu_20);  relu_20 = None
    alias_43: "f32[4, 256, 13, 13]" = torch.ops.aten.alias.default(alias_42);  alias_42 = None
    le_5: "b8[4, 256, 13, 13]" = torch.ops.aten.le.Scalar(alias_43, 0);  alias_43 = None
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_5: "f32[4, 256, 13, 13]" = torch.ops.aten.where.self(le_5, scalar_tensor_5, slice_3);  le_5 = scalar_tensor_5 = slice_3 = None
    sum_6: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(where_5, relu_19, primals_41, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_5 = primals_41 = None
    getitem_21: "f32[4, 64, 13, 13]" = convolution_backward_5[0]
    getitem_22: "f32[256, 64, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    add_1: "f32[4, 64, 13, 13]" = torch.ops.aten.add.Tensor(getitem_18, getitem_21);  getitem_18 = getitem_21 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    alias_45: "f32[4, 64, 13, 13]" = torch.ops.aten.alias.default(relu_19);  relu_19 = None
    alias_46: "f32[4, 64, 13, 13]" = torch.ops.aten.alias.default(alias_45);  alias_45 = None
    le_6: "b8[4, 64, 13, 13]" = torch.ops.aten.le.Scalar(alias_46, 0);  alias_46 = None
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_6: "f32[4, 64, 13, 13]" = torch.ops.aten.where.self(le_6, scalar_tensor_6, add_1);  le_6 = scalar_tensor_6 = add_1 = None
    sum_7: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(where_6, cat_5, primals_39, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_6 = cat_5 = primals_39 = None
    getitem_24: "f32[4, 384, 13, 13]" = convolution_backward_6[0]
    getitem_25: "f32[64, 384, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    slice_5: "f32[4, 192, 13, 13]" = torch.ops.aten.slice.Tensor(getitem_24, 1, 0, 192)
    slice_6: "f32[4, 192, 13, 13]" = torch.ops.aten.slice.Tensor(getitem_24, 1, 192, 384);  getitem_24 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    alias_48: "f32[4, 192, 13, 13]" = torch.ops.aten.alias.default(relu_18);  relu_18 = None
    alias_49: "f32[4, 192, 13, 13]" = torch.ops.aten.alias.default(alias_48);  alias_48 = None
    le_7: "b8[4, 192, 13, 13]" = torch.ops.aten.le.Scalar(alias_49, 0);  alias_49 = None
    scalar_tensor_7: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_7: "f32[4, 192, 13, 13]" = torch.ops.aten.where.self(le_7, scalar_tensor_7, slice_6);  le_7 = scalar_tensor_7 = slice_6 = None
    sum_8: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(where_7, relu_16, primals_37, [192], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_7 = primals_37 = None
    getitem_27: "f32[4, 48, 13, 13]" = convolution_backward_7[0]
    getitem_28: "f32[192, 48, 3, 3]" = convolution_backward_7[1];  convolution_backward_7 = None
    alias_51: "f32[4, 192, 13, 13]" = torch.ops.aten.alias.default(relu_17);  relu_17 = None
    alias_52: "f32[4, 192, 13, 13]" = torch.ops.aten.alias.default(alias_51);  alias_51 = None
    le_8: "b8[4, 192, 13, 13]" = torch.ops.aten.le.Scalar(alias_52, 0);  alias_52 = None
    scalar_tensor_8: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_8: "f32[4, 192, 13, 13]" = torch.ops.aten.where.self(le_8, scalar_tensor_8, slice_5);  le_8 = scalar_tensor_8 = slice_5 = None
    sum_9: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(where_8, relu_16, primals_35, [192], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_8 = primals_35 = None
    getitem_30: "f32[4, 48, 13, 13]" = convolution_backward_8[0]
    getitem_31: "f32[192, 48, 1, 1]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    add_2: "f32[4, 48, 13, 13]" = torch.ops.aten.add.Tensor(getitem_27, getitem_30);  getitem_27 = getitem_30 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    alias_54: "f32[4, 48, 13, 13]" = torch.ops.aten.alias.default(relu_16);  relu_16 = None
    alias_55: "f32[4, 48, 13, 13]" = torch.ops.aten.alias.default(alias_54);  alias_54 = None
    le_9: "b8[4, 48, 13, 13]" = torch.ops.aten.le.Scalar(alias_55, 0);  alias_55 = None
    scalar_tensor_9: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_9: "f32[4, 48, 13, 13]" = torch.ops.aten.where.self(le_9, scalar_tensor_9, add_2);  le_9 = scalar_tensor_9 = add_2 = None
    sum_10: "f32[48]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(where_9, cat_4, primals_33, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_9 = cat_4 = primals_33 = None
    getitem_33: "f32[4, 384, 13, 13]" = convolution_backward_9[0]
    getitem_34: "f32[48, 384, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    slice_7: "f32[4, 192, 13, 13]" = torch.ops.aten.slice.Tensor(getitem_33, 1, 0, 192)
    slice_8: "f32[4, 192, 13, 13]" = torch.ops.aten.slice.Tensor(getitem_33, 1, 192, 384);  getitem_33 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    alias_57: "f32[4, 192, 13, 13]" = torch.ops.aten.alias.default(relu_15);  relu_15 = None
    alias_58: "f32[4, 192, 13, 13]" = torch.ops.aten.alias.default(alias_57);  alias_57 = None
    le_10: "b8[4, 192, 13, 13]" = torch.ops.aten.le.Scalar(alias_58, 0);  alias_58 = None
    scalar_tensor_10: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_10: "f32[4, 192, 13, 13]" = torch.ops.aten.where.self(le_10, scalar_tensor_10, slice_8);  le_10 = scalar_tensor_10 = slice_8 = None
    sum_11: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(where_10, relu_13, primals_31, [192], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_10 = primals_31 = None
    getitem_36: "f32[4, 48, 13, 13]" = convolution_backward_10[0]
    getitem_37: "f32[192, 48, 3, 3]" = convolution_backward_10[1];  convolution_backward_10 = None
    alias_60: "f32[4, 192, 13, 13]" = torch.ops.aten.alias.default(relu_14);  relu_14 = None
    alias_61: "f32[4, 192, 13, 13]" = torch.ops.aten.alias.default(alias_60);  alias_60 = None
    le_11: "b8[4, 192, 13, 13]" = torch.ops.aten.le.Scalar(alias_61, 0);  alias_61 = None
    scalar_tensor_11: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_11: "f32[4, 192, 13, 13]" = torch.ops.aten.where.self(le_11, scalar_tensor_11, slice_7);  le_11 = scalar_tensor_11 = slice_7 = None
    sum_12: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(where_11, relu_13, primals_29, [192], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_11 = primals_29 = None
    getitem_39: "f32[4, 48, 13, 13]" = convolution_backward_11[0]
    getitem_40: "f32[192, 48, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    add_3: "f32[4, 48, 13, 13]" = torch.ops.aten.add.Tensor(getitem_36, getitem_39);  getitem_36 = getitem_39 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    alias_63: "f32[4, 48, 13, 13]" = torch.ops.aten.alias.default(relu_13);  relu_13 = None
    alias_64: "f32[4, 48, 13, 13]" = torch.ops.aten.alias.default(alias_63);  alias_63 = None
    le_12: "b8[4, 48, 13, 13]" = torch.ops.aten.le.Scalar(alias_64, 0);  alias_64 = None
    scalar_tensor_12: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_12: "f32[4, 48, 13, 13]" = torch.ops.aten.where.self(le_12, scalar_tensor_12, add_3);  le_12 = scalar_tensor_12 = add_3 = None
    sum_13: "f32[48]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(where_12, getitem_4, primals_27, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_12 = getitem_4 = primals_27 = None
    getitem_42: "f32[4, 256, 13, 13]" = convolution_backward_12[0]
    getitem_43: "f32[48, 256, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:95, code: x = self.features(x)
    max_pool2d_with_indices_backward: "f32[4, 256, 27, 27]" = torch.ops.aten.max_pool2d_with_indices_backward.default(getitem_42, cat_3, [3, 3], [2, 2], [0, 0], [1, 1], True, getitem_5);  getitem_42 = cat_3 = getitem_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    slice_9: "f32[4, 128, 27, 27]" = torch.ops.aten.slice.Tensor(max_pool2d_with_indices_backward, 1, 0, 128)
    slice_10: "f32[4, 128, 27, 27]" = torch.ops.aten.slice.Tensor(max_pool2d_with_indices_backward, 1, 128, 256);  max_pool2d_with_indices_backward = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    alias_66: "f32[4, 128, 27, 27]" = torch.ops.aten.alias.default(relu_12);  relu_12 = None
    alias_67: "f32[4, 128, 27, 27]" = torch.ops.aten.alias.default(alias_66);  alias_66 = None
    le_13: "b8[4, 128, 27, 27]" = torch.ops.aten.le.Scalar(alias_67, 0);  alias_67 = None
    scalar_tensor_13: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_13: "f32[4, 128, 27, 27]" = torch.ops.aten.where.self(le_13, scalar_tensor_13, slice_10);  le_13 = scalar_tensor_13 = slice_10 = None
    sum_14: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(where_13, relu_10, primals_25, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_13 = primals_25 = None
    getitem_45: "f32[4, 32, 27, 27]" = convolution_backward_13[0]
    getitem_46: "f32[128, 32, 3, 3]" = convolution_backward_13[1];  convolution_backward_13 = None
    alias_69: "f32[4, 128, 27, 27]" = torch.ops.aten.alias.default(relu_11);  relu_11 = None
    alias_70: "f32[4, 128, 27, 27]" = torch.ops.aten.alias.default(alias_69);  alias_69 = None
    le_14: "b8[4, 128, 27, 27]" = torch.ops.aten.le.Scalar(alias_70, 0);  alias_70 = None
    scalar_tensor_14: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_14: "f32[4, 128, 27, 27]" = torch.ops.aten.where.self(le_14, scalar_tensor_14, slice_9);  le_14 = scalar_tensor_14 = slice_9 = None
    sum_15: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(where_14, relu_10, primals_23, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_14 = primals_23 = None
    getitem_48: "f32[4, 32, 27, 27]" = convolution_backward_14[0]
    getitem_49: "f32[128, 32, 1, 1]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    add_4: "f32[4, 32, 27, 27]" = torch.ops.aten.add.Tensor(getitem_45, getitem_48);  getitem_45 = getitem_48 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    alias_72: "f32[4, 32, 27, 27]" = torch.ops.aten.alias.default(relu_10);  relu_10 = None
    alias_73: "f32[4, 32, 27, 27]" = torch.ops.aten.alias.default(alias_72);  alias_72 = None
    le_15: "b8[4, 32, 27, 27]" = torch.ops.aten.le.Scalar(alias_73, 0);  alias_73 = None
    scalar_tensor_15: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_15: "f32[4, 32, 27, 27]" = torch.ops.aten.where.self(le_15, scalar_tensor_15, add_4);  le_15 = scalar_tensor_15 = add_4 = None
    sum_16: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(where_15, cat_2, primals_21, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_15 = cat_2 = primals_21 = None
    getitem_51: "f32[4, 256, 27, 27]" = convolution_backward_15[0]
    getitem_52: "f32[32, 256, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    slice_11: "f32[4, 128, 27, 27]" = torch.ops.aten.slice.Tensor(getitem_51, 1, 0, 128)
    slice_12: "f32[4, 128, 27, 27]" = torch.ops.aten.slice.Tensor(getitem_51, 1, 128, 256);  getitem_51 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    alias_75: "f32[4, 128, 27, 27]" = torch.ops.aten.alias.default(relu_9);  relu_9 = None
    alias_76: "f32[4, 128, 27, 27]" = torch.ops.aten.alias.default(alias_75);  alias_75 = None
    le_16: "b8[4, 128, 27, 27]" = torch.ops.aten.le.Scalar(alias_76, 0);  alias_76 = None
    scalar_tensor_16: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_16: "f32[4, 128, 27, 27]" = torch.ops.aten.where.self(le_16, scalar_tensor_16, slice_12);  le_16 = scalar_tensor_16 = slice_12 = None
    sum_17: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(where_16, relu_7, primals_19, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_16 = primals_19 = None
    getitem_54: "f32[4, 32, 27, 27]" = convolution_backward_16[0]
    getitem_55: "f32[128, 32, 3, 3]" = convolution_backward_16[1];  convolution_backward_16 = None
    alias_78: "f32[4, 128, 27, 27]" = torch.ops.aten.alias.default(relu_8);  relu_8 = None
    alias_79: "f32[4, 128, 27, 27]" = torch.ops.aten.alias.default(alias_78);  alias_78 = None
    le_17: "b8[4, 128, 27, 27]" = torch.ops.aten.le.Scalar(alias_79, 0);  alias_79 = None
    scalar_tensor_17: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_17: "f32[4, 128, 27, 27]" = torch.ops.aten.where.self(le_17, scalar_tensor_17, slice_11);  le_17 = scalar_tensor_17 = slice_11 = None
    sum_18: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(where_17, relu_7, primals_17, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_17 = primals_17 = None
    getitem_57: "f32[4, 32, 27, 27]" = convolution_backward_17[0]
    getitem_58: "f32[128, 32, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    add_5: "f32[4, 32, 27, 27]" = torch.ops.aten.add.Tensor(getitem_54, getitem_57);  getitem_54 = getitem_57 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    alias_81: "f32[4, 32, 27, 27]" = torch.ops.aten.alias.default(relu_7);  relu_7 = None
    alias_82: "f32[4, 32, 27, 27]" = torch.ops.aten.alias.default(alias_81);  alias_81 = None
    le_18: "b8[4, 32, 27, 27]" = torch.ops.aten.le.Scalar(alias_82, 0);  alias_82 = None
    scalar_tensor_18: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_18: "f32[4, 32, 27, 27]" = torch.ops.aten.where.self(le_18, scalar_tensor_18, add_5);  le_18 = scalar_tensor_18 = add_5 = None
    sum_19: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(where_18, getitem_2, primals_15, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_18 = getitem_2 = primals_15 = None
    getitem_60: "f32[4, 128, 27, 27]" = convolution_backward_18[0]
    getitem_61: "f32[32, 128, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:95, code: x = self.features(x)
    max_pool2d_with_indices_backward_1: "f32[4, 128, 55, 55]" = torch.ops.aten.max_pool2d_with_indices_backward.default(getitem_60, cat_1, [3, 3], [2, 2], [0, 0], [1, 1], True, getitem_3);  getitem_60 = cat_1 = getitem_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    slice_13: "f32[4, 64, 55, 55]" = torch.ops.aten.slice.Tensor(max_pool2d_with_indices_backward_1, 1, 0, 64)
    slice_14: "f32[4, 64, 55, 55]" = torch.ops.aten.slice.Tensor(max_pool2d_with_indices_backward_1, 1, 64, 128);  max_pool2d_with_indices_backward_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    alias_84: "f32[4, 64, 55, 55]" = torch.ops.aten.alias.default(relu_6);  relu_6 = None
    alias_85: "f32[4, 64, 55, 55]" = torch.ops.aten.alias.default(alias_84);  alias_84 = None
    le_19: "b8[4, 64, 55, 55]" = torch.ops.aten.le.Scalar(alias_85, 0);  alias_85 = None
    scalar_tensor_19: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_19: "f32[4, 64, 55, 55]" = torch.ops.aten.where.self(le_19, scalar_tensor_19, slice_14);  le_19 = scalar_tensor_19 = slice_14 = None
    sum_20: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(where_19, relu_4, primals_13, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_19 = primals_13 = None
    getitem_63: "f32[4, 16, 55, 55]" = convolution_backward_19[0]
    getitem_64: "f32[64, 16, 3, 3]" = convolution_backward_19[1];  convolution_backward_19 = None
    alias_87: "f32[4, 64, 55, 55]" = torch.ops.aten.alias.default(relu_5);  relu_5 = None
    alias_88: "f32[4, 64, 55, 55]" = torch.ops.aten.alias.default(alias_87);  alias_87 = None
    le_20: "b8[4, 64, 55, 55]" = torch.ops.aten.le.Scalar(alias_88, 0);  alias_88 = None
    scalar_tensor_20: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_20: "f32[4, 64, 55, 55]" = torch.ops.aten.where.self(le_20, scalar_tensor_20, slice_13);  le_20 = scalar_tensor_20 = slice_13 = None
    sum_21: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(where_20, relu_4, primals_11, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_20 = primals_11 = None
    getitem_66: "f32[4, 16, 55, 55]" = convolution_backward_20[0]
    getitem_67: "f32[64, 16, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    add_6: "f32[4, 16, 55, 55]" = torch.ops.aten.add.Tensor(getitem_63, getitem_66);  getitem_63 = getitem_66 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    alias_90: "f32[4, 16, 55, 55]" = torch.ops.aten.alias.default(relu_4);  relu_4 = None
    alias_91: "f32[4, 16, 55, 55]" = torch.ops.aten.alias.default(alias_90);  alias_90 = None
    le_21: "b8[4, 16, 55, 55]" = torch.ops.aten.le.Scalar(alias_91, 0);  alias_91 = None
    scalar_tensor_21: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_21: "f32[4, 16, 55, 55]" = torch.ops.aten.where.self(le_21, scalar_tensor_21, add_6);  le_21 = scalar_tensor_21 = add_6 = None
    sum_22: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(where_21, cat, primals_9, [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_21 = cat = primals_9 = None
    getitem_69: "f32[4, 128, 55, 55]" = convolution_backward_21[0]
    getitem_70: "f32[16, 128, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    slice_15: "f32[4, 64, 55, 55]" = torch.ops.aten.slice.Tensor(getitem_69, 1, 0, 64)
    slice_16: "f32[4, 64, 55, 55]" = torch.ops.aten.slice.Tensor(getitem_69, 1, 64, 128);  getitem_69 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    alias_93: "f32[4, 64, 55, 55]" = torch.ops.aten.alias.default(relu_3);  relu_3 = None
    alias_94: "f32[4, 64, 55, 55]" = torch.ops.aten.alias.default(alias_93);  alias_93 = None
    le_22: "b8[4, 64, 55, 55]" = torch.ops.aten.le.Scalar(alias_94, 0);  alias_94 = None
    scalar_tensor_22: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_22: "f32[4, 64, 55, 55]" = torch.ops.aten.where.self(le_22, scalar_tensor_22, slice_16);  le_22 = scalar_tensor_22 = slice_16 = None
    sum_23: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(where_22, relu_1, primals_7, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_22 = primals_7 = None
    getitem_72: "f32[4, 16, 55, 55]" = convolution_backward_22[0]
    getitem_73: "f32[64, 16, 3, 3]" = convolution_backward_22[1];  convolution_backward_22 = None
    alias_96: "f32[4, 64, 55, 55]" = torch.ops.aten.alias.default(relu_2);  relu_2 = None
    alias_97: "f32[4, 64, 55, 55]" = torch.ops.aten.alias.default(alias_96);  alias_96 = None
    le_23: "b8[4, 64, 55, 55]" = torch.ops.aten.le.Scalar(alias_97, 0);  alias_97 = None
    scalar_tensor_23: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_23: "f32[4, 64, 55, 55]" = torch.ops.aten.where.self(le_23, scalar_tensor_23, slice_15);  le_23 = scalar_tensor_23 = slice_15 = None
    sum_24: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(where_23, relu_1, primals_5, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_23 = primals_5 = None
    getitem_75: "f32[4, 16, 55, 55]" = convolution_backward_23[0]
    getitem_76: "f32[64, 16, 1, 1]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    add_7: "f32[4, 16, 55, 55]" = torch.ops.aten.add.Tensor(getitem_72, getitem_75);  getitem_72 = getitem_75 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    alias_99: "f32[4, 16, 55, 55]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_100: "f32[4, 16, 55, 55]" = torch.ops.aten.alias.default(alias_99);  alias_99 = None
    le_24: "b8[4, 16, 55, 55]" = torch.ops.aten.le.Scalar(alias_100, 0);  alias_100 = None
    scalar_tensor_24: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_24: "f32[4, 16, 55, 55]" = torch.ops.aten.where.self(le_24, scalar_tensor_24, add_7);  le_24 = scalar_tensor_24 = add_7 = None
    sum_25: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(where_24, getitem, primals_3, [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_24 = getitem = primals_3 = None
    getitem_78: "f32[4, 64, 55, 55]" = convolution_backward_24[0]
    getitem_79: "f32[16, 64, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:95, code: x = self.features(x)
    max_pool2d_with_indices_backward_2: "f32[4, 64, 111, 111]" = torch.ops.aten.max_pool2d_with_indices_backward.default(getitem_78, relu, [3, 3], [2, 2], [0, 0], [1, 1], True, getitem_1);  getitem_78 = getitem_1 = None
    alias_102: "f32[4, 64, 111, 111]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_103: "f32[4, 64, 111, 111]" = torch.ops.aten.alias.default(alias_102);  alias_102 = None
    le_25: "b8[4, 64, 111, 111]" = torch.ops.aten.le.Scalar(alias_103, 0);  alias_103 = None
    scalar_tensor_25: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_25: "f32[4, 64, 111, 111]" = torch.ops.aten.where.self(le_25, scalar_tensor_25, max_pool2d_with_indices_backward_2);  le_25 = scalar_tensor_25 = max_pool2d_with_indices_backward_2 = None
    sum_26: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(where_25, primals_53, primals_1, [64], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  where_25 = primals_53 = primals_1 = None
    getitem_82: "f32[64, 3, 3, 3]" = convolution_backward_25[1];  convolution_backward_25 = None
    return pytree.tree_unflatten([view, getitem_82, sum_26, getitem_79, sum_25, getitem_76, sum_24, getitem_73, sum_23, getitem_70, sum_22, getitem_67, sum_21, getitem_64, sum_20, getitem_61, sum_19, getitem_58, sum_18, getitem_55, sum_17, getitem_52, sum_16, getitem_49, sum_15, getitem_46, sum_14, getitem_43, sum_13, getitem_40, sum_12, getitem_37, sum_11, getitem_34, sum_10, getitem_31, sum_9, getitem_28, sum_8, getitem_25, sum_7, getitem_22, sum_6, getitem_19, sum_5, getitem_16, sum_4, getitem_13, sum_3, getitem_10, sum_2, getitem_7, sum_1, None], self._out_spec)
    