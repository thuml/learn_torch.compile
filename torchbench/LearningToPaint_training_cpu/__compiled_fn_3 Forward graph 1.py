from __future__ import annotations



def forward(self, primals_1: "f32[64, 9, 3, 3]", primals_2: "f32[64]", primals_3: "f32[64]", primals_4: "f32[64, 64, 3, 3]", primals_5: "f32[64]", primals_6: "f32[64]", primals_7: "f32[64, 64, 3, 3]", primals_8: "f32[64]", primals_9: "f32[64]", primals_10: "f32[64, 64, 1, 1]", primals_11: "f32[64]", primals_12: "f32[64]", primals_13: "f32[64, 64, 3, 3]", primals_14: "f32[64]", primals_15: "f32[64]", primals_16: "f32[64, 64, 3, 3]", primals_17: "f32[64]", primals_18: "f32[64]", primals_19: "f32[128, 64, 3, 3]", primals_20: "f32[128]", primals_21: "f32[128]", primals_22: "f32[128, 128, 3, 3]", primals_23: "f32[128]", primals_24: "f32[128]", primals_25: "f32[128, 64, 1, 1]", primals_26: "f32[128]", primals_27: "f32[128]", primals_28: "f32[128, 128, 3, 3]", primals_29: "f32[128]", primals_30: "f32[128]", primals_31: "f32[128, 128, 3, 3]", primals_32: "f32[128]", primals_33: "f32[128]", primals_34: "f32[256, 128, 3, 3]", primals_35: "f32[256]", primals_36: "f32[256]", primals_37: "f32[256, 256, 3, 3]", primals_38: "f32[256]", primals_39: "f32[256]", primals_40: "f32[256, 128, 1, 1]", primals_41: "f32[256]", primals_42: "f32[256]", primals_43: "f32[256, 256, 3, 3]", primals_44: "f32[256]", primals_45: "f32[256]", primals_46: "f32[256, 256, 3, 3]", primals_47: "f32[256]", primals_48: "f32[256]", primals_49: "f32[512, 256, 3, 3]", primals_50: "f32[512]", primals_51: "f32[512]", primals_52: "f32[512, 512, 3, 3]", primals_53: "f32[512]", primals_54: "f32[512]", primals_55: "f32[512, 256, 1, 1]", primals_56: "f32[512]", primals_57: "f32[512]", primals_58: "f32[512, 512, 3, 3]", primals_59: "f32[512]", primals_60: "f32[512]", primals_61: "f32[512, 512, 3, 3]", primals_62: "f32[512]", primals_63: "f32[512]", primals_64: "f32[65, 512]", primals_65: "f32[65]", primals_66: "f32[64]", primals_67: "f32[64]", primals_68: "i64[]", primals_69: "f32[64]", primals_70: "f32[64]", primals_71: "i64[]", primals_72: "f32[64]", primals_73: "f32[64]", primals_74: "i64[]", primals_75: "f32[64]", primals_76: "f32[64]", primals_77: "i64[]", primals_78: "f32[64]", primals_79: "f32[64]", primals_80: "i64[]", primals_81: "f32[64]", primals_82: "f32[64]", primals_83: "i64[]", primals_84: "f32[128]", primals_85: "f32[128]", primals_86: "i64[]", primals_87: "f32[128]", primals_88: "f32[128]", primals_89: "i64[]", primals_90: "f32[128]", primals_91: "f32[128]", primals_92: "i64[]", primals_93: "f32[128]", primals_94: "f32[128]", primals_95: "i64[]", primals_96: "f32[128]", primals_97: "f32[128]", primals_98: "i64[]", primals_99: "f32[256]", primals_100: "f32[256]", primals_101: "i64[]", primals_102: "f32[256]", primals_103: "f32[256]", primals_104: "i64[]", primals_105: "f32[256]", primals_106: "f32[256]", primals_107: "i64[]", primals_108: "f32[256]", primals_109: "f32[256]", primals_110: "i64[]", primals_111: "f32[256]", primals_112: "f32[256]", primals_113: "i64[]", primals_114: "f32[512]", primals_115: "f32[512]", primals_116: "i64[]", primals_117: "f32[512]", primals_118: "f32[512]", primals_119: "i64[]", primals_120: "f32[512]", primals_121: "f32[512]", primals_122: "i64[]", primals_123: "f32[512]", primals_124: "f32[512]", primals_125: "i64[]", primals_126: "f32[512]", primals_127: "f32[512]", primals_128: "i64[]", primals_129: "f32[4, 9, 128, 128]"):
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:105, code: x = F.relu(self.bn1(self.conv1(x)))
    convolution: "f32[4, 64, 64, 64]" = torch.ops.aten.convolution.default(primals_129, primals_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    _native_batch_norm_legit_no_training = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution, primals_2, primals_3, primals_66, primals_67, 0.1, 1e-05);  primals_3 = None
    getitem: "f32[4, 64, 64, 64]" = _native_batch_norm_legit_no_training[0]
    getitem_1: "f32[0]" = _native_batch_norm_legit_no_training[1]
    getitem_2: "f32[0]" = _native_batch_norm_legit_no_training[2];  _native_batch_norm_legit_no_training = None
    relu: "f32[4, 64, 64, 64]" = torch.ops.aten.relu.default(getitem);  getitem = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    convolution_1: "f32[4, 64, 32, 32]" = torch.ops.aten.convolution.default(relu, primals_4, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    _native_batch_norm_legit_no_training_1 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_1, primals_5, primals_6, primals_69, primals_70, 0.1, 1e-05);  primals_6 = None
    getitem_3: "f32[4, 64, 32, 32]" = _native_batch_norm_legit_no_training_1[0]
    getitem_4: "f32[0]" = _native_batch_norm_legit_no_training_1[1]
    getitem_5: "f32[0]" = _native_batch_norm_legit_no_training_1[2];  _native_batch_norm_legit_no_training_1 = None
    relu_1: "f32[4, 64, 32, 32]" = torch.ops.aten.relu.default(getitem_3);  getitem_3 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:46, code: out = self.bn2(self.conv2(out))
    convolution_2: "f32[4, 64, 32, 32]" = torch.ops.aten.convolution.default(relu_1, primals_7, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    _native_batch_norm_legit_no_training_2 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_2, primals_8, primals_9, primals_72, primals_73, 0.1, 1e-05);  primals_9 = None
    getitem_6: "f32[4, 64, 32, 32]" = _native_batch_norm_legit_no_training_2[0]
    getitem_7: "f32[0]" = _native_batch_norm_legit_no_training_2[1]
    getitem_8: "f32[0]" = _native_batch_norm_legit_no_training_2[2];  _native_batch_norm_legit_no_training_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:47, code: out += self.shortcut(x)
    convolution_3: "f32[4, 64, 32, 32]" = torch.ops.aten.convolution.default(relu, primals_10, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    _native_batch_norm_legit_no_training_3 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_3, primals_11, primals_12, primals_75, primals_76, 0.1, 1e-05);  primals_12 = None
    getitem_9: "f32[4, 64, 32, 32]" = _native_batch_norm_legit_no_training_3[0]
    getitem_10: "f32[0]" = _native_batch_norm_legit_no_training_3[1]
    getitem_11: "f32[0]" = _native_batch_norm_legit_no_training_3[2];  _native_batch_norm_legit_no_training_3 = None
    add: "f32[4, 64, 32, 32]" = torch.ops.aten.add.Tensor(getitem_6, getitem_9);  getitem_6 = getitem_9 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:48, code: out = F.relu(out)
    relu_2: "f32[4, 64, 32, 32]" = torch.ops.aten.relu.default(add);  add = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    convolution_4: "f32[4, 64, 32, 32]" = torch.ops.aten.convolution.default(relu_2, primals_13, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    _native_batch_norm_legit_no_training_4 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_4, primals_14, primals_15, primals_78, primals_79, 0.1, 1e-05);  primals_15 = None
    getitem_12: "f32[4, 64, 32, 32]" = _native_batch_norm_legit_no_training_4[0]
    getitem_13: "f32[0]" = _native_batch_norm_legit_no_training_4[1]
    getitem_14: "f32[0]" = _native_batch_norm_legit_no_training_4[2];  _native_batch_norm_legit_no_training_4 = None
    relu_3: "f32[4, 64, 32, 32]" = torch.ops.aten.relu.default(getitem_12);  getitem_12 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:46, code: out = self.bn2(self.conv2(out))
    convolution_5: "f32[4, 64, 32, 32]" = torch.ops.aten.convolution.default(relu_3, primals_16, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    _native_batch_norm_legit_no_training_5 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_5, primals_17, primals_18, primals_81, primals_82, 0.1, 1e-05);  primals_18 = None
    getitem_15: "f32[4, 64, 32, 32]" = _native_batch_norm_legit_no_training_5[0]
    getitem_16: "f32[0]" = _native_batch_norm_legit_no_training_5[1]
    getitem_17: "f32[0]" = _native_batch_norm_legit_no_training_5[2];  _native_batch_norm_legit_no_training_5 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:47, code: out += self.shortcut(x)
    add_1: "f32[4, 64, 32, 32]" = torch.ops.aten.add.Tensor(getitem_15, relu_2);  getitem_15 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:48, code: out = F.relu(out)
    relu_4: "f32[4, 64, 32, 32]" = torch.ops.aten.relu.default(add_1);  add_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    convolution_6: "f32[4, 128, 16, 16]" = torch.ops.aten.convolution.default(relu_4, primals_19, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    _native_batch_norm_legit_no_training_6 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_6, primals_20, primals_21, primals_84, primals_85, 0.1, 1e-05);  primals_21 = None
    getitem_18: "f32[4, 128, 16, 16]" = _native_batch_norm_legit_no_training_6[0]
    getitem_19: "f32[0]" = _native_batch_norm_legit_no_training_6[1]
    getitem_20: "f32[0]" = _native_batch_norm_legit_no_training_6[2];  _native_batch_norm_legit_no_training_6 = None
    relu_5: "f32[4, 128, 16, 16]" = torch.ops.aten.relu.default(getitem_18);  getitem_18 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:46, code: out = self.bn2(self.conv2(out))
    convolution_7: "f32[4, 128, 16, 16]" = torch.ops.aten.convolution.default(relu_5, primals_22, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    _native_batch_norm_legit_no_training_7 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_7, primals_23, primals_24, primals_87, primals_88, 0.1, 1e-05);  primals_24 = None
    getitem_21: "f32[4, 128, 16, 16]" = _native_batch_norm_legit_no_training_7[0]
    getitem_22: "f32[0]" = _native_batch_norm_legit_no_training_7[1]
    getitem_23: "f32[0]" = _native_batch_norm_legit_no_training_7[2];  _native_batch_norm_legit_no_training_7 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:47, code: out += self.shortcut(x)
    convolution_8: "f32[4, 128, 16, 16]" = torch.ops.aten.convolution.default(relu_4, primals_25, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    _native_batch_norm_legit_no_training_8 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_8, primals_26, primals_27, primals_90, primals_91, 0.1, 1e-05);  primals_27 = None
    getitem_24: "f32[4, 128, 16, 16]" = _native_batch_norm_legit_no_training_8[0]
    getitem_25: "f32[0]" = _native_batch_norm_legit_no_training_8[1]
    getitem_26: "f32[0]" = _native_batch_norm_legit_no_training_8[2];  _native_batch_norm_legit_no_training_8 = None
    add_2: "f32[4, 128, 16, 16]" = torch.ops.aten.add.Tensor(getitem_21, getitem_24);  getitem_21 = getitem_24 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:48, code: out = F.relu(out)
    relu_6: "f32[4, 128, 16, 16]" = torch.ops.aten.relu.default(add_2);  add_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    convolution_9: "f32[4, 128, 16, 16]" = torch.ops.aten.convolution.default(relu_6, primals_28, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    _native_batch_norm_legit_no_training_9 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_9, primals_29, primals_30, primals_93, primals_94, 0.1, 1e-05);  primals_30 = None
    getitem_27: "f32[4, 128, 16, 16]" = _native_batch_norm_legit_no_training_9[0]
    getitem_28: "f32[0]" = _native_batch_norm_legit_no_training_9[1]
    getitem_29: "f32[0]" = _native_batch_norm_legit_no_training_9[2];  _native_batch_norm_legit_no_training_9 = None
    relu_7: "f32[4, 128, 16, 16]" = torch.ops.aten.relu.default(getitem_27);  getitem_27 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:46, code: out = self.bn2(self.conv2(out))
    convolution_10: "f32[4, 128, 16, 16]" = torch.ops.aten.convolution.default(relu_7, primals_31, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    _native_batch_norm_legit_no_training_10 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_10, primals_32, primals_33, primals_96, primals_97, 0.1, 1e-05);  primals_33 = None
    getitem_30: "f32[4, 128, 16, 16]" = _native_batch_norm_legit_no_training_10[0]
    getitem_31: "f32[0]" = _native_batch_norm_legit_no_training_10[1]
    getitem_32: "f32[0]" = _native_batch_norm_legit_no_training_10[2];  _native_batch_norm_legit_no_training_10 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:47, code: out += self.shortcut(x)
    add_3: "f32[4, 128, 16, 16]" = torch.ops.aten.add.Tensor(getitem_30, relu_6);  getitem_30 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:48, code: out = F.relu(out)
    relu_8: "f32[4, 128, 16, 16]" = torch.ops.aten.relu.default(add_3);  add_3 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    convolution_11: "f32[4, 256, 8, 8]" = torch.ops.aten.convolution.default(relu_8, primals_34, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    _native_batch_norm_legit_no_training_11 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_11, primals_35, primals_36, primals_99, primals_100, 0.1, 1e-05);  primals_36 = None
    getitem_33: "f32[4, 256, 8, 8]" = _native_batch_norm_legit_no_training_11[0]
    getitem_34: "f32[0]" = _native_batch_norm_legit_no_training_11[1]
    getitem_35: "f32[0]" = _native_batch_norm_legit_no_training_11[2];  _native_batch_norm_legit_no_training_11 = None
    relu_9: "f32[4, 256, 8, 8]" = torch.ops.aten.relu.default(getitem_33);  getitem_33 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:46, code: out = self.bn2(self.conv2(out))
    convolution_12: "f32[4, 256, 8, 8]" = torch.ops.aten.convolution.default(relu_9, primals_37, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    _native_batch_norm_legit_no_training_12 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_12, primals_38, primals_39, primals_102, primals_103, 0.1, 1e-05);  primals_39 = None
    getitem_36: "f32[4, 256, 8, 8]" = _native_batch_norm_legit_no_training_12[0]
    getitem_37: "f32[0]" = _native_batch_norm_legit_no_training_12[1]
    getitem_38: "f32[0]" = _native_batch_norm_legit_no_training_12[2];  _native_batch_norm_legit_no_training_12 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:47, code: out += self.shortcut(x)
    convolution_13: "f32[4, 256, 8, 8]" = torch.ops.aten.convolution.default(relu_8, primals_40, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    _native_batch_norm_legit_no_training_13 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_13, primals_41, primals_42, primals_105, primals_106, 0.1, 1e-05);  primals_42 = None
    getitem_39: "f32[4, 256, 8, 8]" = _native_batch_norm_legit_no_training_13[0]
    getitem_40: "f32[0]" = _native_batch_norm_legit_no_training_13[1]
    getitem_41: "f32[0]" = _native_batch_norm_legit_no_training_13[2];  _native_batch_norm_legit_no_training_13 = None
    add_4: "f32[4, 256, 8, 8]" = torch.ops.aten.add.Tensor(getitem_36, getitem_39);  getitem_36 = getitem_39 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:48, code: out = F.relu(out)
    relu_10: "f32[4, 256, 8, 8]" = torch.ops.aten.relu.default(add_4);  add_4 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    convolution_14: "f32[4, 256, 8, 8]" = torch.ops.aten.convolution.default(relu_10, primals_43, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    _native_batch_norm_legit_no_training_14 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_14, primals_44, primals_45, primals_108, primals_109, 0.1, 1e-05);  primals_45 = None
    getitem_42: "f32[4, 256, 8, 8]" = _native_batch_norm_legit_no_training_14[0]
    getitem_43: "f32[0]" = _native_batch_norm_legit_no_training_14[1]
    getitem_44: "f32[0]" = _native_batch_norm_legit_no_training_14[2];  _native_batch_norm_legit_no_training_14 = None
    relu_11: "f32[4, 256, 8, 8]" = torch.ops.aten.relu.default(getitem_42);  getitem_42 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:46, code: out = self.bn2(self.conv2(out))
    convolution_15: "f32[4, 256, 8, 8]" = torch.ops.aten.convolution.default(relu_11, primals_46, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    _native_batch_norm_legit_no_training_15 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_15, primals_47, primals_48, primals_111, primals_112, 0.1, 1e-05);  primals_48 = None
    getitem_45: "f32[4, 256, 8, 8]" = _native_batch_norm_legit_no_training_15[0]
    getitem_46: "f32[0]" = _native_batch_norm_legit_no_training_15[1]
    getitem_47: "f32[0]" = _native_batch_norm_legit_no_training_15[2];  _native_batch_norm_legit_no_training_15 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:47, code: out += self.shortcut(x)
    add_5: "f32[4, 256, 8, 8]" = torch.ops.aten.add.Tensor(getitem_45, relu_10);  getitem_45 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:48, code: out = F.relu(out)
    relu_12: "f32[4, 256, 8, 8]" = torch.ops.aten.relu.default(add_5);  add_5 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    convolution_16: "f32[4, 512, 4, 4]" = torch.ops.aten.convolution.default(relu_12, primals_49, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    _native_batch_norm_legit_no_training_16 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_16, primals_50, primals_51, primals_114, primals_115, 0.1, 1e-05);  primals_51 = None
    getitem_48: "f32[4, 512, 4, 4]" = _native_batch_norm_legit_no_training_16[0]
    getitem_49: "f32[0]" = _native_batch_norm_legit_no_training_16[1]
    getitem_50: "f32[0]" = _native_batch_norm_legit_no_training_16[2];  _native_batch_norm_legit_no_training_16 = None
    relu_13: "f32[4, 512, 4, 4]" = torch.ops.aten.relu.default(getitem_48);  getitem_48 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:46, code: out = self.bn2(self.conv2(out))
    convolution_17: "f32[4, 512, 4, 4]" = torch.ops.aten.convolution.default(relu_13, primals_52, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    _native_batch_norm_legit_no_training_17 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_17, primals_53, primals_54, primals_117, primals_118, 0.1, 1e-05);  primals_54 = None
    getitem_51: "f32[4, 512, 4, 4]" = _native_batch_norm_legit_no_training_17[0]
    getitem_52: "f32[0]" = _native_batch_norm_legit_no_training_17[1]
    getitem_53: "f32[0]" = _native_batch_norm_legit_no_training_17[2];  _native_batch_norm_legit_no_training_17 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:47, code: out += self.shortcut(x)
    convolution_18: "f32[4, 512, 4, 4]" = torch.ops.aten.convolution.default(relu_12, primals_55, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    _native_batch_norm_legit_no_training_18 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_18, primals_56, primals_57, primals_120, primals_121, 0.1, 1e-05);  primals_57 = None
    getitem_54: "f32[4, 512, 4, 4]" = _native_batch_norm_legit_no_training_18[0]
    getitem_55: "f32[0]" = _native_batch_norm_legit_no_training_18[1]
    getitem_56: "f32[0]" = _native_batch_norm_legit_no_training_18[2];  _native_batch_norm_legit_no_training_18 = None
    add_6: "f32[4, 512, 4, 4]" = torch.ops.aten.add.Tensor(getitem_51, getitem_54);  getitem_51 = getitem_54 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:48, code: out = F.relu(out)
    relu_14: "f32[4, 512, 4, 4]" = torch.ops.aten.relu.default(add_6);  add_6 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    convolution_19: "f32[4, 512, 4, 4]" = torch.ops.aten.convolution.default(relu_14, primals_58, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    _native_batch_norm_legit_no_training_19 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_19, primals_59, primals_60, primals_123, primals_124, 0.1, 1e-05);  primals_60 = None
    getitem_57: "f32[4, 512, 4, 4]" = _native_batch_norm_legit_no_training_19[0]
    getitem_58: "f32[0]" = _native_batch_norm_legit_no_training_19[1]
    getitem_59: "f32[0]" = _native_batch_norm_legit_no_training_19[2];  _native_batch_norm_legit_no_training_19 = None
    relu_15: "f32[4, 512, 4, 4]" = torch.ops.aten.relu.default(getitem_57);  getitem_57 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:46, code: out = self.bn2(self.conv2(out))
    convolution_20: "f32[4, 512, 4, 4]" = torch.ops.aten.convolution.default(relu_15, primals_61, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    _native_batch_norm_legit_no_training_20 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_20, primals_62, primals_63, primals_126, primals_127, 0.1, 1e-05);  primals_63 = None
    getitem_60: "f32[4, 512, 4, 4]" = _native_batch_norm_legit_no_training_20[0]
    getitem_61: "f32[0]" = _native_batch_norm_legit_no_training_20[1]
    getitem_62: "f32[0]" = _native_batch_norm_legit_no_training_20[2];  _native_batch_norm_legit_no_training_20 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:47, code: out += self.shortcut(x)
    add_7: "f32[4, 512, 4, 4]" = torch.ops.aten.add.Tensor(getitem_60, relu_14);  getitem_60 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:48, code: out = F.relu(out)
    relu_16: "f32[4, 512, 4, 4]" = torch.ops.aten.relu.default(add_7);  add_7 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:110, code: x = F.avg_pool2d(x, 4)
    avg_pool2d: "f32[4, 512, 1, 1]" = torch.ops.aten.avg_pool2d.default(relu_16, [4, 4])
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:111, code: x = x.view(x.size(0), -1)
    view: "f32[4, 512]" = torch.ops.aten.view.default(avg_pool2d, [4, -1]);  avg_pool2d = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:112, code: x = self.fc(x)
    t: "f32[512, 65]" = torch.ops.aten.t.default(primals_64);  primals_64 = None
    addmm: "f32[4, 65]" = torch.ops.aten.addmm.default(primals_65, view, t);  primals_65 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:113, code: x = torch.sigmoid(x)
    sigmoid: "f32[4, 65]" = torch.ops.aten.sigmoid.default(addmm);  addmm = None
    detach_17: "f32[4, 65]" = torch.ops.aten.detach.default(sigmoid)
    detach_18: "f32[4, 65]" = torch.ops.aten.detach.default(detach_17);  detach_17 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:112, code: x = self.fc(x)
    t_1: "f32[65, 512]" = torch.ops.aten.t.default(t);  t = None
    return [sigmoid, primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_66, primals_67, primals_69, primals_70, primals_72, primals_73, primals_75, primals_76, primals_78, primals_79, primals_81, primals_82, primals_84, primals_85, primals_87, primals_88, primals_90, primals_91, primals_93, primals_94, primals_96, primals_97, primals_99, primals_100, primals_102, primals_103, primals_105, primals_106, primals_108, primals_109, primals_111, primals_112, primals_114, primals_115, primals_117, primals_118, primals_120, primals_121, primals_123, primals_124, primals_126, primals_127, primals_129, convolution, getitem_1, getitem_2, relu, convolution_1, getitem_4, getitem_5, relu_1, convolution_2, getitem_7, getitem_8, convolution_3, getitem_10, getitem_11, relu_2, convolution_4, getitem_13, getitem_14, relu_3, convolution_5, getitem_16, getitem_17, relu_4, convolution_6, getitem_19, getitem_20, relu_5, convolution_7, getitem_22, getitem_23, convolution_8, getitem_25, getitem_26, relu_6, convolution_9, getitem_28, getitem_29, relu_7, convolution_10, getitem_31, getitem_32, relu_8, convolution_11, getitem_34, getitem_35, relu_9, convolution_12, getitem_37, getitem_38, convolution_13, getitem_40, getitem_41, relu_10, convolution_14, getitem_43, getitem_44, relu_11, convolution_15, getitem_46, getitem_47, relu_12, convolution_16, getitem_49, getitem_50, relu_13, convolution_17, getitem_52, getitem_53, convolution_18, getitem_55, getitem_56, relu_14, convolution_19, getitem_58, getitem_59, relu_15, convolution_20, getitem_61, getitem_62, relu_16, view, detach_18, t_1]
    