from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[64, 9, 3, 3]"; primals_2: "f32[64]"; primals_3: "f32[64]"; primals_4: "f32[64, 64, 3, 3]"; primals_5: "f32[64]"; primals_6: "f32[64]"; primals_7: "f32[64, 64, 3, 3]"; primals_8: "f32[64]"; primals_9: "f32[64]"; primals_10: "f32[64, 64, 1, 1]"; primals_11: "f32[64]"; primals_12: "f32[64]"; primals_13: "f32[64, 64, 3, 3]"; primals_14: "f32[64]"; primals_15: "f32[64]"; primals_16: "f32[64, 64, 3, 3]"; primals_17: "f32[64]"; primals_18: "f32[64]"; primals_19: "f32[128, 64, 3, 3]"; primals_20: "f32[128]"; primals_21: "f32[128]"; primals_22: "f32[128, 128, 3, 3]"; primals_23: "f32[128]"; primals_24: "f32[128]"; primals_25: "f32[128, 64, 1, 1]"; primals_26: "f32[128]"; primals_27: "f32[128]"; primals_28: "f32[128, 128, 3, 3]"; primals_29: "f32[128]"; primals_30: "f32[128]"; primals_31: "f32[128, 128, 3, 3]"; primals_32: "f32[128]"; primals_33: "f32[128]"; primals_34: "f32[256, 128, 3, 3]"; primals_35: "f32[256]"; primals_36: "f32[256]"; primals_37: "f32[256, 256, 3, 3]"; primals_38: "f32[256]"; primals_39: "f32[256]"; primals_40: "f32[256, 128, 1, 1]"; primals_41: "f32[256]"; primals_42: "f32[256]"; primals_43: "f32[256, 256, 3, 3]"; primals_44: "f32[256]"; primals_45: "f32[256]"; primals_46: "f32[256, 256, 3, 3]"; primals_47: "f32[256]"; primals_48: "f32[256]"; primals_49: "f32[512, 256, 3, 3]"; primals_50: "f32[512]"; primals_51: "f32[512]"; primals_52: "f32[512, 512, 3, 3]"; primals_53: "f32[512]"; primals_54: "f32[512]"; primals_55: "f32[512, 256, 1, 1]"; primals_56: "f32[512]"; primals_57: "f32[512]"; primals_58: "f32[512, 512, 3, 3]"; primals_59: "f32[512]"; primals_60: "f32[512]"; primals_61: "f32[512, 512, 3, 3]"; primals_62: "f32[512]"; primals_63: "f32[512]"; primals_64: "f32[65, 512]"; primals_65: "f32[65]"; primals_66: "f32[64]"; primals_67: "f32[64]"; primals_68: "i64[]"; primals_69: "f32[64]"; primals_70: "f32[64]"; primals_71: "i64[]"; primals_72: "f32[64]"; primals_73: "f32[64]"; primals_74: "i64[]"; primals_75: "f32[64]"; primals_76: "f32[64]"; primals_77: "i64[]"; primals_78: "f32[64]"; primals_79: "f32[64]"; primals_80: "i64[]"; primals_81: "f32[64]"; primals_82: "f32[64]"; primals_83: "i64[]"; primals_84: "f32[128]"; primals_85: "f32[128]"; primals_86: "i64[]"; primals_87: "f32[128]"; primals_88: "f32[128]"; primals_89: "i64[]"; primals_90: "f32[128]"; primals_91: "f32[128]"; primals_92: "i64[]"; primals_93: "f32[128]"; primals_94: "f32[128]"; primals_95: "i64[]"; primals_96: "f32[128]"; primals_97: "f32[128]"; primals_98: "i64[]"; primals_99: "f32[256]"; primals_100: "f32[256]"; primals_101: "i64[]"; primals_102: "f32[256]"; primals_103: "f32[256]"; primals_104: "i64[]"; primals_105: "f32[256]"; primals_106: "f32[256]"; primals_107: "i64[]"; primals_108: "f32[256]"; primals_109: "f32[256]"; primals_110: "i64[]"; primals_111: "f32[256]"; primals_112: "f32[256]"; primals_113: "i64[]"; primals_114: "f32[512]"; primals_115: "f32[512]"; primals_116: "i64[]"; primals_117: "f32[512]"; primals_118: "f32[512]"; primals_119: "i64[]"; primals_120: "f32[512]"; primals_121: "f32[512]"; primals_122: "i64[]"; primals_123: "f32[512]"; primals_124: "f32[512]"; primals_125: "i64[]"; primals_126: "f32[512]"; primals_127: "f32[512]"; primals_128: "i64[]"; primals_129: "f32[4, 9, 128, 128]"; tangents_1: "f32[4, 65]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:105, code: x = F.relu(self.bn1(self.conv1(x)))
    convolution: "f32[4, 64, 64, 64]" = torch.ops.aten.convolution.default(primals_129, primals_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    _native_batch_norm_legit_no_training = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution, primals_2, primals_3, primals_66, primals_67, 0.1, 1e-05);  primals_3 = None
    getitem: "f32[4, 64, 64, 64]" = _native_batch_norm_legit_no_training[0]
    getitem_1: "f32[0]" = _native_batch_norm_legit_no_training[1]
    getitem_2: "f32[0]" = _native_batch_norm_legit_no_training[2];  _native_batch_norm_legit_no_training = None
    relu: "f32[4, 64, 64, 64]" = torch.ops.aten.relu.default(getitem);  getitem = None
    detach: "f32[4, 64, 64, 64]" = torch.ops.aten.detach.default(relu)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    convolution_1: "f32[4, 64, 32, 32]" = torch.ops.aten.convolution.default(relu, primals_4, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    _native_batch_norm_legit_no_training_1 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_1, primals_5, primals_6, primals_69, primals_70, 0.1, 1e-05);  primals_6 = None
    getitem_3: "f32[4, 64, 32, 32]" = _native_batch_norm_legit_no_training_1[0]
    getitem_4: "f32[0]" = _native_batch_norm_legit_no_training_1[1]
    getitem_5: "f32[0]" = _native_batch_norm_legit_no_training_1[2];  _native_batch_norm_legit_no_training_1 = None
    relu_1: "f32[4, 64, 32, 32]" = torch.ops.aten.relu.default(getitem_3);  getitem_3 = None
    detach_1: "f32[4, 64, 32, 32]" = torch.ops.aten.detach.default(relu_1)
    
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
    detach_2: "f32[4, 64, 32, 32]" = torch.ops.aten.detach.default(relu_2)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    convolution_4: "f32[4, 64, 32, 32]" = torch.ops.aten.convolution.default(relu_2, primals_13, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    _native_batch_norm_legit_no_training_4 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_4, primals_14, primals_15, primals_78, primals_79, 0.1, 1e-05);  primals_15 = None
    getitem_12: "f32[4, 64, 32, 32]" = _native_batch_norm_legit_no_training_4[0]
    getitem_13: "f32[0]" = _native_batch_norm_legit_no_training_4[1]
    getitem_14: "f32[0]" = _native_batch_norm_legit_no_training_4[2];  _native_batch_norm_legit_no_training_4 = None
    relu_3: "f32[4, 64, 32, 32]" = torch.ops.aten.relu.default(getitem_12);  getitem_12 = None
    detach_3: "f32[4, 64, 32, 32]" = torch.ops.aten.detach.default(relu_3)
    
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
    detach_4: "f32[4, 64, 32, 32]" = torch.ops.aten.detach.default(relu_4)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    convolution_6: "f32[4, 128, 16, 16]" = torch.ops.aten.convolution.default(relu_4, primals_19, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    _native_batch_norm_legit_no_training_6 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_6, primals_20, primals_21, primals_84, primals_85, 0.1, 1e-05);  primals_21 = None
    getitem_18: "f32[4, 128, 16, 16]" = _native_batch_norm_legit_no_training_6[0]
    getitem_19: "f32[0]" = _native_batch_norm_legit_no_training_6[1]
    getitem_20: "f32[0]" = _native_batch_norm_legit_no_training_6[2];  _native_batch_norm_legit_no_training_6 = None
    relu_5: "f32[4, 128, 16, 16]" = torch.ops.aten.relu.default(getitem_18);  getitem_18 = None
    detach_5: "f32[4, 128, 16, 16]" = torch.ops.aten.detach.default(relu_5)
    
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
    detach_6: "f32[4, 128, 16, 16]" = torch.ops.aten.detach.default(relu_6)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    convolution_9: "f32[4, 128, 16, 16]" = torch.ops.aten.convolution.default(relu_6, primals_28, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    _native_batch_norm_legit_no_training_9 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_9, primals_29, primals_30, primals_93, primals_94, 0.1, 1e-05);  primals_30 = None
    getitem_27: "f32[4, 128, 16, 16]" = _native_batch_norm_legit_no_training_9[0]
    getitem_28: "f32[0]" = _native_batch_norm_legit_no_training_9[1]
    getitem_29: "f32[0]" = _native_batch_norm_legit_no_training_9[2];  _native_batch_norm_legit_no_training_9 = None
    relu_7: "f32[4, 128, 16, 16]" = torch.ops.aten.relu.default(getitem_27);  getitem_27 = None
    detach_7: "f32[4, 128, 16, 16]" = torch.ops.aten.detach.default(relu_7)
    
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
    detach_8: "f32[4, 128, 16, 16]" = torch.ops.aten.detach.default(relu_8)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    convolution_11: "f32[4, 256, 8, 8]" = torch.ops.aten.convolution.default(relu_8, primals_34, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    _native_batch_norm_legit_no_training_11 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_11, primals_35, primals_36, primals_99, primals_100, 0.1, 1e-05);  primals_36 = None
    getitem_33: "f32[4, 256, 8, 8]" = _native_batch_norm_legit_no_training_11[0]
    getitem_34: "f32[0]" = _native_batch_norm_legit_no_training_11[1]
    getitem_35: "f32[0]" = _native_batch_norm_legit_no_training_11[2];  _native_batch_norm_legit_no_training_11 = None
    relu_9: "f32[4, 256, 8, 8]" = torch.ops.aten.relu.default(getitem_33);  getitem_33 = None
    detach_9: "f32[4, 256, 8, 8]" = torch.ops.aten.detach.default(relu_9)
    
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
    detach_10: "f32[4, 256, 8, 8]" = torch.ops.aten.detach.default(relu_10)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    convolution_14: "f32[4, 256, 8, 8]" = torch.ops.aten.convolution.default(relu_10, primals_43, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    _native_batch_norm_legit_no_training_14 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_14, primals_44, primals_45, primals_108, primals_109, 0.1, 1e-05);  primals_45 = None
    getitem_42: "f32[4, 256, 8, 8]" = _native_batch_norm_legit_no_training_14[0]
    getitem_43: "f32[0]" = _native_batch_norm_legit_no_training_14[1]
    getitem_44: "f32[0]" = _native_batch_norm_legit_no_training_14[2];  _native_batch_norm_legit_no_training_14 = None
    relu_11: "f32[4, 256, 8, 8]" = torch.ops.aten.relu.default(getitem_42);  getitem_42 = None
    detach_11: "f32[4, 256, 8, 8]" = torch.ops.aten.detach.default(relu_11)
    
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
    detach_12: "f32[4, 256, 8, 8]" = torch.ops.aten.detach.default(relu_12)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    convolution_16: "f32[4, 512, 4, 4]" = torch.ops.aten.convolution.default(relu_12, primals_49, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    _native_batch_norm_legit_no_training_16 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_16, primals_50, primals_51, primals_114, primals_115, 0.1, 1e-05);  primals_51 = None
    getitem_48: "f32[4, 512, 4, 4]" = _native_batch_norm_legit_no_training_16[0]
    getitem_49: "f32[0]" = _native_batch_norm_legit_no_training_16[1]
    getitem_50: "f32[0]" = _native_batch_norm_legit_no_training_16[2];  _native_batch_norm_legit_no_training_16 = None
    relu_13: "f32[4, 512, 4, 4]" = torch.ops.aten.relu.default(getitem_48);  getitem_48 = None
    detach_13: "f32[4, 512, 4, 4]" = torch.ops.aten.detach.default(relu_13)
    
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
    detach_14: "f32[4, 512, 4, 4]" = torch.ops.aten.detach.default(relu_14)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    convolution_19: "f32[4, 512, 4, 4]" = torch.ops.aten.convolution.default(relu_14, primals_58, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    _native_batch_norm_legit_no_training_19 = torch.ops.aten._native_batch_norm_legit_no_training.default(convolution_19, primals_59, primals_60, primals_123, primals_124, 0.1, 1e-05);  primals_60 = None
    getitem_57: "f32[4, 512, 4, 4]" = _native_batch_norm_legit_no_training_19[0]
    getitem_58: "f32[0]" = _native_batch_norm_legit_no_training_19[1]
    getitem_59: "f32[0]" = _native_batch_norm_legit_no_training_19[2];  _native_batch_norm_legit_no_training_19 = None
    relu_15: "f32[4, 512, 4, 4]" = torch.ops.aten.relu.default(getitem_57);  getitem_57 = None
    detach_15: "f32[4, 512, 4, 4]" = torch.ops.aten.detach.default(relu_15)
    
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
    detach_16: "f32[4, 512, 4, 4]" = torch.ops.aten.detach.default(relu_16)
    
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
    sigmoid_backward: "f32[4, 65]" = torch.ops.aten.sigmoid_backward.default(tangents_1, detach_18);  tangents_1 = detach_18 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:112, code: x = self.fc(x)
    t_1: "f32[65, 512]" = torch.ops.aten.t.default(t);  t = None
    mm: "f32[4, 512]" = torch.ops.aten.mm.default(sigmoid_backward, t_1);  t_1 = None
    t_2: "f32[65, 4]" = torch.ops.aten.t.default(sigmoid_backward)
    mm_1: "f32[65, 512]" = torch.ops.aten.mm.default(t_2, view);  t_2 = view = None
    t_3: "f32[512, 65]" = torch.ops.aten.t.default(mm_1);  mm_1 = None
    sum_1: "f32[1, 65]" = torch.ops.aten.sum.dim_IntList(sigmoid_backward, [0], True);  sigmoid_backward = None
    view_1: "f32[65]" = torch.ops.aten.view.default(sum_1, [65]);  sum_1 = None
    t_4: "f32[65, 512]" = torch.ops.aten.t.default(t_3);  t_3 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:111, code: x = x.view(x.size(0), -1)
    view_2: "f32[4, 512, 1, 1]" = torch.ops.aten.view.default(mm, [4, 512, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:110, code: x = F.avg_pool2d(x, 4)
    avg_pool2d_backward: "f32[4, 512, 4, 4]" = torch.ops.aten.avg_pool2d_backward.default(view_2, relu_16, [4, 4], [], [0, 0], False, True, None);  view_2 = relu_16 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:48, code: out = F.relu(out)
    detach_19: "f32[4, 512, 4, 4]" = torch.ops.aten.detach.default(detach_16);  detach_16 = None
    threshold_backward: "f32[4, 512, 4, 4]" = torch.ops.aten.threshold_backward.default(avg_pool2d_backward, detach_19, 0);  avg_pool2d_backward = detach_19 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:46, code: out = self.bn2(self.conv2(out))
    native_batch_norm_backward = torch.ops.aten.native_batch_norm_backward.default(threshold_backward, convolution_20, primals_62, primals_126, primals_127, getitem_61, getitem_62, False, 1e-05, [True, True, True]);  convolution_20 = primals_62 = primals_126 = primals_127 = getitem_61 = getitem_62 = None
    getitem_63: "f32[4, 512, 4, 4]" = native_batch_norm_backward[0]
    getitem_64: "f32[512]" = native_batch_norm_backward[1]
    getitem_65: "f32[512]" = native_batch_norm_backward[2];  native_batch_norm_backward = None
    convolution_backward = torch.ops.aten.convolution_backward.default(getitem_63, relu_15, primals_61, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_63 = relu_15 = primals_61 = None
    getitem_66: "f32[4, 512, 4, 4]" = convolution_backward[0]
    getitem_67: "f32[512, 512, 3, 3]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    detach_20: "f32[4, 512, 4, 4]" = torch.ops.aten.detach.default(detach_15);  detach_15 = None
    threshold_backward_1: "f32[4, 512, 4, 4]" = torch.ops.aten.threshold_backward.default(getitem_66, detach_20, 0);  getitem_66 = detach_20 = None
    native_batch_norm_backward_1 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_1, convolution_19, primals_59, primals_123, primals_124, getitem_58, getitem_59, False, 1e-05, [True, True, True]);  threshold_backward_1 = convolution_19 = primals_59 = primals_123 = primals_124 = getitem_58 = getitem_59 = None
    getitem_69: "f32[4, 512, 4, 4]" = native_batch_norm_backward_1[0]
    getitem_70: "f32[512]" = native_batch_norm_backward_1[1]
    getitem_71: "f32[512]" = native_batch_norm_backward_1[2];  native_batch_norm_backward_1 = None
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(getitem_69, relu_14, primals_58, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_69 = relu_14 = primals_58 = None
    getitem_72: "f32[4, 512, 4, 4]" = convolution_backward_1[0]
    getitem_73: "f32[512, 512, 3, 3]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    add_8: "f32[4, 512, 4, 4]" = torch.ops.aten.add.Tensor(threshold_backward, getitem_72);  threshold_backward = getitem_72 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:48, code: out = F.relu(out)
    detach_21: "f32[4, 512, 4, 4]" = torch.ops.aten.detach.default(detach_14);  detach_14 = None
    threshold_backward_2: "f32[4, 512, 4, 4]" = torch.ops.aten.threshold_backward.default(add_8, detach_21, 0);  add_8 = detach_21 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:47, code: out += self.shortcut(x)
    native_batch_norm_backward_2 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_2, convolution_18, primals_56, primals_120, primals_121, getitem_55, getitem_56, False, 1e-05, [True, True, True]);  convolution_18 = primals_56 = primals_120 = primals_121 = getitem_55 = getitem_56 = None
    getitem_75: "f32[4, 512, 4, 4]" = native_batch_norm_backward_2[0]
    getitem_76: "f32[512]" = native_batch_norm_backward_2[1]
    getitem_77: "f32[512]" = native_batch_norm_backward_2[2];  native_batch_norm_backward_2 = None
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(getitem_75, relu_12, primals_55, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_75 = primals_55 = None
    getitem_78: "f32[4, 256, 8, 8]" = convolution_backward_2[0]
    getitem_79: "f32[512, 256, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:46, code: out = self.bn2(self.conv2(out))
    native_batch_norm_backward_3 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_2, convolution_17, primals_53, primals_117, primals_118, getitem_52, getitem_53, False, 1e-05, [True, True, True]);  threshold_backward_2 = convolution_17 = primals_53 = primals_117 = primals_118 = getitem_52 = getitem_53 = None
    getitem_81: "f32[4, 512, 4, 4]" = native_batch_norm_backward_3[0]
    getitem_82: "f32[512]" = native_batch_norm_backward_3[1]
    getitem_83: "f32[512]" = native_batch_norm_backward_3[2];  native_batch_norm_backward_3 = None
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(getitem_81, relu_13, primals_52, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_81 = relu_13 = primals_52 = None
    getitem_84: "f32[4, 512, 4, 4]" = convolution_backward_3[0]
    getitem_85: "f32[512, 512, 3, 3]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    detach_22: "f32[4, 512, 4, 4]" = torch.ops.aten.detach.default(detach_13);  detach_13 = None
    threshold_backward_3: "f32[4, 512, 4, 4]" = torch.ops.aten.threshold_backward.default(getitem_84, detach_22, 0);  getitem_84 = detach_22 = None
    native_batch_norm_backward_4 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_3, convolution_16, primals_50, primals_114, primals_115, getitem_49, getitem_50, False, 1e-05, [True, True, True]);  threshold_backward_3 = convolution_16 = primals_50 = primals_114 = primals_115 = getitem_49 = getitem_50 = None
    getitem_87: "f32[4, 512, 4, 4]" = native_batch_norm_backward_4[0]
    getitem_88: "f32[512]" = native_batch_norm_backward_4[1]
    getitem_89: "f32[512]" = native_batch_norm_backward_4[2];  native_batch_norm_backward_4 = None
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(getitem_87, relu_12, primals_49, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_87 = relu_12 = primals_49 = None
    getitem_90: "f32[4, 256, 8, 8]" = convolution_backward_4[0]
    getitem_91: "f32[512, 256, 3, 3]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    add_9: "f32[4, 256, 8, 8]" = torch.ops.aten.add.Tensor(getitem_78, getitem_90);  getitem_78 = getitem_90 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:48, code: out = F.relu(out)
    detach_23: "f32[4, 256, 8, 8]" = torch.ops.aten.detach.default(detach_12);  detach_12 = None
    threshold_backward_4: "f32[4, 256, 8, 8]" = torch.ops.aten.threshold_backward.default(add_9, detach_23, 0);  add_9 = detach_23 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:46, code: out = self.bn2(self.conv2(out))
    native_batch_norm_backward_5 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_4, convolution_15, primals_47, primals_111, primals_112, getitem_46, getitem_47, False, 1e-05, [True, True, True]);  convolution_15 = primals_47 = primals_111 = primals_112 = getitem_46 = getitem_47 = None
    getitem_93: "f32[4, 256, 8, 8]" = native_batch_norm_backward_5[0]
    getitem_94: "f32[256]" = native_batch_norm_backward_5[1]
    getitem_95: "f32[256]" = native_batch_norm_backward_5[2];  native_batch_norm_backward_5 = None
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(getitem_93, relu_11, primals_46, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_93 = relu_11 = primals_46 = None
    getitem_96: "f32[4, 256, 8, 8]" = convolution_backward_5[0]
    getitem_97: "f32[256, 256, 3, 3]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    detach_24: "f32[4, 256, 8, 8]" = torch.ops.aten.detach.default(detach_11);  detach_11 = None
    threshold_backward_5: "f32[4, 256, 8, 8]" = torch.ops.aten.threshold_backward.default(getitem_96, detach_24, 0);  getitem_96 = detach_24 = None
    native_batch_norm_backward_6 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_5, convolution_14, primals_44, primals_108, primals_109, getitem_43, getitem_44, False, 1e-05, [True, True, True]);  threshold_backward_5 = convolution_14 = primals_44 = primals_108 = primals_109 = getitem_43 = getitem_44 = None
    getitem_99: "f32[4, 256, 8, 8]" = native_batch_norm_backward_6[0]
    getitem_100: "f32[256]" = native_batch_norm_backward_6[1]
    getitem_101: "f32[256]" = native_batch_norm_backward_6[2];  native_batch_norm_backward_6 = None
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(getitem_99, relu_10, primals_43, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_99 = relu_10 = primals_43 = None
    getitem_102: "f32[4, 256, 8, 8]" = convolution_backward_6[0]
    getitem_103: "f32[256, 256, 3, 3]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    add_10: "f32[4, 256, 8, 8]" = torch.ops.aten.add.Tensor(threshold_backward_4, getitem_102);  threshold_backward_4 = getitem_102 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:48, code: out = F.relu(out)
    detach_25: "f32[4, 256, 8, 8]" = torch.ops.aten.detach.default(detach_10);  detach_10 = None
    threshold_backward_6: "f32[4, 256, 8, 8]" = torch.ops.aten.threshold_backward.default(add_10, detach_25, 0);  add_10 = detach_25 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:47, code: out += self.shortcut(x)
    native_batch_norm_backward_7 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_6, convolution_13, primals_41, primals_105, primals_106, getitem_40, getitem_41, False, 1e-05, [True, True, True]);  convolution_13 = primals_41 = primals_105 = primals_106 = getitem_40 = getitem_41 = None
    getitem_105: "f32[4, 256, 8, 8]" = native_batch_norm_backward_7[0]
    getitem_106: "f32[256]" = native_batch_norm_backward_7[1]
    getitem_107: "f32[256]" = native_batch_norm_backward_7[2];  native_batch_norm_backward_7 = None
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(getitem_105, relu_8, primals_40, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_105 = primals_40 = None
    getitem_108: "f32[4, 128, 16, 16]" = convolution_backward_7[0]
    getitem_109: "f32[256, 128, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:46, code: out = self.bn2(self.conv2(out))
    native_batch_norm_backward_8 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_6, convolution_12, primals_38, primals_102, primals_103, getitem_37, getitem_38, False, 1e-05, [True, True, True]);  threshold_backward_6 = convolution_12 = primals_38 = primals_102 = primals_103 = getitem_37 = getitem_38 = None
    getitem_111: "f32[4, 256, 8, 8]" = native_batch_norm_backward_8[0]
    getitem_112: "f32[256]" = native_batch_norm_backward_8[1]
    getitem_113: "f32[256]" = native_batch_norm_backward_8[2];  native_batch_norm_backward_8 = None
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(getitem_111, relu_9, primals_37, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_111 = relu_9 = primals_37 = None
    getitem_114: "f32[4, 256, 8, 8]" = convolution_backward_8[0]
    getitem_115: "f32[256, 256, 3, 3]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    detach_26: "f32[4, 256, 8, 8]" = torch.ops.aten.detach.default(detach_9);  detach_9 = None
    threshold_backward_7: "f32[4, 256, 8, 8]" = torch.ops.aten.threshold_backward.default(getitem_114, detach_26, 0);  getitem_114 = detach_26 = None
    native_batch_norm_backward_9 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_7, convolution_11, primals_35, primals_99, primals_100, getitem_34, getitem_35, False, 1e-05, [True, True, True]);  threshold_backward_7 = convolution_11 = primals_35 = primals_99 = primals_100 = getitem_34 = getitem_35 = None
    getitem_117: "f32[4, 256, 8, 8]" = native_batch_norm_backward_9[0]
    getitem_118: "f32[256]" = native_batch_norm_backward_9[1]
    getitem_119: "f32[256]" = native_batch_norm_backward_9[2];  native_batch_norm_backward_9 = None
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(getitem_117, relu_8, primals_34, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_117 = relu_8 = primals_34 = None
    getitem_120: "f32[4, 128, 16, 16]" = convolution_backward_9[0]
    getitem_121: "f32[256, 128, 3, 3]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    add_11: "f32[4, 128, 16, 16]" = torch.ops.aten.add.Tensor(getitem_108, getitem_120);  getitem_108 = getitem_120 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:48, code: out = F.relu(out)
    detach_27: "f32[4, 128, 16, 16]" = torch.ops.aten.detach.default(detach_8);  detach_8 = None
    threshold_backward_8: "f32[4, 128, 16, 16]" = torch.ops.aten.threshold_backward.default(add_11, detach_27, 0);  add_11 = detach_27 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:46, code: out = self.bn2(self.conv2(out))
    native_batch_norm_backward_10 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_8, convolution_10, primals_32, primals_96, primals_97, getitem_31, getitem_32, False, 1e-05, [True, True, True]);  convolution_10 = primals_32 = primals_96 = primals_97 = getitem_31 = getitem_32 = None
    getitem_123: "f32[4, 128, 16, 16]" = native_batch_norm_backward_10[0]
    getitem_124: "f32[128]" = native_batch_norm_backward_10[1]
    getitem_125: "f32[128]" = native_batch_norm_backward_10[2];  native_batch_norm_backward_10 = None
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(getitem_123, relu_7, primals_31, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_123 = relu_7 = primals_31 = None
    getitem_126: "f32[4, 128, 16, 16]" = convolution_backward_10[0]
    getitem_127: "f32[128, 128, 3, 3]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    detach_28: "f32[4, 128, 16, 16]" = torch.ops.aten.detach.default(detach_7);  detach_7 = None
    threshold_backward_9: "f32[4, 128, 16, 16]" = torch.ops.aten.threshold_backward.default(getitem_126, detach_28, 0);  getitem_126 = detach_28 = None
    native_batch_norm_backward_11 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_9, convolution_9, primals_29, primals_93, primals_94, getitem_28, getitem_29, False, 1e-05, [True, True, True]);  threshold_backward_9 = convolution_9 = primals_29 = primals_93 = primals_94 = getitem_28 = getitem_29 = None
    getitem_129: "f32[4, 128, 16, 16]" = native_batch_norm_backward_11[0]
    getitem_130: "f32[128]" = native_batch_norm_backward_11[1]
    getitem_131: "f32[128]" = native_batch_norm_backward_11[2];  native_batch_norm_backward_11 = None
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(getitem_129, relu_6, primals_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_129 = relu_6 = primals_28 = None
    getitem_132: "f32[4, 128, 16, 16]" = convolution_backward_11[0]
    getitem_133: "f32[128, 128, 3, 3]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    add_12: "f32[4, 128, 16, 16]" = torch.ops.aten.add.Tensor(threshold_backward_8, getitem_132);  threshold_backward_8 = getitem_132 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:48, code: out = F.relu(out)
    detach_29: "f32[4, 128, 16, 16]" = torch.ops.aten.detach.default(detach_6);  detach_6 = None
    threshold_backward_10: "f32[4, 128, 16, 16]" = torch.ops.aten.threshold_backward.default(add_12, detach_29, 0);  add_12 = detach_29 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:47, code: out += self.shortcut(x)
    native_batch_norm_backward_12 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_10, convolution_8, primals_26, primals_90, primals_91, getitem_25, getitem_26, False, 1e-05, [True, True, True]);  convolution_8 = primals_26 = primals_90 = primals_91 = getitem_25 = getitem_26 = None
    getitem_135: "f32[4, 128, 16, 16]" = native_batch_norm_backward_12[0]
    getitem_136: "f32[128]" = native_batch_norm_backward_12[1]
    getitem_137: "f32[128]" = native_batch_norm_backward_12[2];  native_batch_norm_backward_12 = None
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(getitem_135, relu_4, primals_25, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_135 = primals_25 = None
    getitem_138: "f32[4, 64, 32, 32]" = convolution_backward_12[0]
    getitem_139: "f32[128, 64, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:46, code: out = self.bn2(self.conv2(out))
    native_batch_norm_backward_13 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_10, convolution_7, primals_23, primals_87, primals_88, getitem_22, getitem_23, False, 1e-05, [True, True, True]);  threshold_backward_10 = convolution_7 = primals_23 = primals_87 = primals_88 = getitem_22 = getitem_23 = None
    getitem_141: "f32[4, 128, 16, 16]" = native_batch_norm_backward_13[0]
    getitem_142: "f32[128]" = native_batch_norm_backward_13[1]
    getitem_143: "f32[128]" = native_batch_norm_backward_13[2];  native_batch_norm_backward_13 = None
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(getitem_141, relu_5, primals_22, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_141 = relu_5 = primals_22 = None
    getitem_144: "f32[4, 128, 16, 16]" = convolution_backward_13[0]
    getitem_145: "f32[128, 128, 3, 3]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    detach_30: "f32[4, 128, 16, 16]" = torch.ops.aten.detach.default(detach_5);  detach_5 = None
    threshold_backward_11: "f32[4, 128, 16, 16]" = torch.ops.aten.threshold_backward.default(getitem_144, detach_30, 0);  getitem_144 = detach_30 = None
    native_batch_norm_backward_14 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_11, convolution_6, primals_20, primals_84, primals_85, getitem_19, getitem_20, False, 1e-05, [True, True, True]);  threshold_backward_11 = convolution_6 = primals_20 = primals_84 = primals_85 = getitem_19 = getitem_20 = None
    getitem_147: "f32[4, 128, 16, 16]" = native_batch_norm_backward_14[0]
    getitem_148: "f32[128]" = native_batch_norm_backward_14[1]
    getitem_149: "f32[128]" = native_batch_norm_backward_14[2];  native_batch_norm_backward_14 = None
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(getitem_147, relu_4, primals_19, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_147 = relu_4 = primals_19 = None
    getitem_150: "f32[4, 64, 32, 32]" = convolution_backward_14[0]
    getitem_151: "f32[128, 64, 3, 3]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    add_13: "f32[4, 64, 32, 32]" = torch.ops.aten.add.Tensor(getitem_138, getitem_150);  getitem_138 = getitem_150 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:48, code: out = F.relu(out)
    detach_31: "f32[4, 64, 32, 32]" = torch.ops.aten.detach.default(detach_4);  detach_4 = None
    threshold_backward_12: "f32[4, 64, 32, 32]" = torch.ops.aten.threshold_backward.default(add_13, detach_31, 0);  add_13 = detach_31 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:46, code: out = self.bn2(self.conv2(out))
    native_batch_norm_backward_15 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_12, convolution_5, primals_17, primals_81, primals_82, getitem_16, getitem_17, False, 1e-05, [True, True, True]);  convolution_5 = primals_17 = primals_81 = primals_82 = getitem_16 = getitem_17 = None
    getitem_153: "f32[4, 64, 32, 32]" = native_batch_norm_backward_15[0]
    getitem_154: "f32[64]" = native_batch_norm_backward_15[1]
    getitem_155: "f32[64]" = native_batch_norm_backward_15[2];  native_batch_norm_backward_15 = None
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(getitem_153, relu_3, primals_16, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_153 = relu_3 = primals_16 = None
    getitem_156: "f32[4, 64, 32, 32]" = convolution_backward_15[0]
    getitem_157: "f32[64, 64, 3, 3]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    detach_32: "f32[4, 64, 32, 32]" = torch.ops.aten.detach.default(detach_3);  detach_3 = None
    threshold_backward_13: "f32[4, 64, 32, 32]" = torch.ops.aten.threshold_backward.default(getitem_156, detach_32, 0);  getitem_156 = detach_32 = None
    native_batch_norm_backward_16 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_13, convolution_4, primals_14, primals_78, primals_79, getitem_13, getitem_14, False, 1e-05, [True, True, True]);  threshold_backward_13 = convolution_4 = primals_14 = primals_78 = primals_79 = getitem_13 = getitem_14 = None
    getitem_159: "f32[4, 64, 32, 32]" = native_batch_norm_backward_16[0]
    getitem_160: "f32[64]" = native_batch_norm_backward_16[1]
    getitem_161: "f32[64]" = native_batch_norm_backward_16[2];  native_batch_norm_backward_16 = None
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(getitem_159, relu_2, primals_13, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_159 = relu_2 = primals_13 = None
    getitem_162: "f32[4, 64, 32, 32]" = convolution_backward_16[0]
    getitem_163: "f32[64, 64, 3, 3]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    add_14: "f32[4, 64, 32, 32]" = torch.ops.aten.add.Tensor(threshold_backward_12, getitem_162);  threshold_backward_12 = getitem_162 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:48, code: out = F.relu(out)
    detach_33: "f32[4, 64, 32, 32]" = torch.ops.aten.detach.default(detach_2);  detach_2 = None
    threshold_backward_14: "f32[4, 64, 32, 32]" = torch.ops.aten.threshold_backward.default(add_14, detach_33, 0);  add_14 = detach_33 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:47, code: out += self.shortcut(x)
    native_batch_norm_backward_17 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_14, convolution_3, primals_11, primals_75, primals_76, getitem_10, getitem_11, False, 1e-05, [True, True, True]);  convolution_3 = primals_11 = primals_75 = primals_76 = getitem_10 = getitem_11 = None
    getitem_165: "f32[4, 64, 32, 32]" = native_batch_norm_backward_17[0]
    getitem_166: "f32[64]" = native_batch_norm_backward_17[1]
    getitem_167: "f32[64]" = native_batch_norm_backward_17[2];  native_batch_norm_backward_17 = None
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(getitem_165, relu, primals_10, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_165 = primals_10 = None
    getitem_168: "f32[4, 64, 64, 64]" = convolution_backward_17[0]
    getitem_169: "f32[64, 64, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:46, code: out = self.bn2(self.conv2(out))
    native_batch_norm_backward_18 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_14, convolution_2, primals_8, primals_72, primals_73, getitem_7, getitem_8, False, 1e-05, [True, True, True]);  threshold_backward_14 = convolution_2 = primals_8 = primals_72 = primals_73 = getitem_7 = getitem_8 = None
    getitem_171: "f32[4, 64, 32, 32]" = native_batch_norm_backward_18[0]
    getitem_172: "f32[64]" = native_batch_norm_backward_18[1]
    getitem_173: "f32[64]" = native_batch_norm_backward_18[2];  native_batch_norm_backward_18 = None
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(getitem_171, relu_1, primals_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_171 = relu_1 = primals_7 = None
    getitem_174: "f32[4, 64, 32, 32]" = convolution_backward_18[0]
    getitem_175: "f32[64, 64, 3, 3]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    detach_34: "f32[4, 64, 32, 32]" = torch.ops.aten.detach.default(detach_1);  detach_1 = None
    threshold_backward_15: "f32[4, 64, 32, 32]" = torch.ops.aten.threshold_backward.default(getitem_174, detach_34, 0);  getitem_174 = detach_34 = None
    native_batch_norm_backward_19 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_15, convolution_1, primals_5, primals_69, primals_70, getitem_4, getitem_5, False, 1e-05, [True, True, True]);  threshold_backward_15 = convolution_1 = primals_5 = primals_69 = primals_70 = getitem_4 = getitem_5 = None
    getitem_177: "f32[4, 64, 32, 32]" = native_batch_norm_backward_19[0]
    getitem_178: "f32[64]" = native_batch_norm_backward_19[1]
    getitem_179: "f32[64]" = native_batch_norm_backward_19[2];  native_batch_norm_backward_19 = None
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(getitem_177, relu, primals_4, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_177 = relu = primals_4 = None
    getitem_180: "f32[4, 64, 64, 64]" = convolution_backward_19[0]
    getitem_181: "f32[64, 64, 3, 3]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    add_15: "f32[4, 64, 64, 64]" = torch.ops.aten.add.Tensor(getitem_168, getitem_180);  getitem_168 = getitem_180 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:105, code: x = F.relu(self.bn1(self.conv1(x)))
    detach_35: "f32[4, 64, 64, 64]" = torch.ops.aten.detach.default(detach);  detach = None
    threshold_backward_16: "f32[4, 64, 64, 64]" = torch.ops.aten.threshold_backward.default(add_15, detach_35, 0);  add_15 = detach_35 = None
    native_batch_norm_backward_20 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_16, convolution, primals_2, primals_66, primals_67, getitem_1, getitem_2, False, 1e-05, [True, True, True]);  threshold_backward_16 = convolution = primals_2 = primals_66 = primals_67 = getitem_1 = getitem_2 = None
    getitem_183: "f32[4, 64, 64, 64]" = native_batch_norm_backward_20[0]
    getitem_184: "f32[64]" = native_batch_norm_backward_20[1]
    getitem_185: "f32[64]" = native_batch_norm_backward_20[2];  native_batch_norm_backward_20 = None
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(getitem_183, primals_129, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  getitem_183 = primals_129 = primals_1 = None
    getitem_187: "f32[64, 9, 3, 3]" = convolution_backward_20[1];  convolution_backward_20 = None
    return pytree.tree_unflatten([sigmoid, getitem_187, getitem_184, getitem_185, getitem_181, getitem_178, getitem_179, getitem_175, getitem_172, getitem_173, getitem_169, getitem_166, getitem_167, getitem_163, getitem_160, getitem_161, getitem_157, getitem_154, getitem_155, getitem_151, getitem_148, getitem_149, getitem_145, getitem_142, getitem_143, getitem_139, getitem_136, getitem_137, getitem_133, getitem_130, getitem_131, getitem_127, getitem_124, getitem_125, getitem_121, getitem_118, getitem_119, getitem_115, getitem_112, getitem_113, getitem_109, getitem_106, getitem_107, getitem_103, getitem_100, getitem_101, getitem_97, getitem_94, getitem_95, getitem_91, getitem_88, getitem_89, getitem_85, getitem_82, getitem_83, getitem_79, getitem_76, getitem_77, getitem_73, getitem_70, getitem_71, getitem_67, getitem_64, getitem_65, t_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], self._out_spec)
    