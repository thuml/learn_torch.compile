from __future__ import annotations



def forward(self, primals_1: "f32[768, 3, 7, 7]", primals_3: "f32[768]", primals_5: "f32[768, 1, 7, 7]", primals_7: "f32[768]", primals_9: "f32[768, 768, 1, 1]", primals_11: "f32[768]", primals_13: "f32[768, 1, 7, 7]", primals_15: "f32[768]", primals_17: "f32[768, 768, 1, 1]", primals_19: "f32[768]", primals_21: "f32[768, 1, 7, 7]", primals_23: "f32[768]", primals_25: "f32[768, 768, 1, 1]", primals_27: "f32[768]", primals_29: "f32[768, 1, 7, 7]", primals_31: "f32[768]", primals_33: "f32[768, 768, 1, 1]", primals_35: "f32[768]", primals_37: "f32[768, 1, 7, 7]", primals_39: "f32[768]", primals_41: "f32[768, 768, 1, 1]", primals_43: "f32[768]", primals_45: "f32[768, 1, 7, 7]", primals_47: "f32[768]", primals_49: "f32[768, 768, 1, 1]", primals_51: "f32[768]", primals_53: "f32[768, 1, 7, 7]", primals_55: "f32[768]", primals_57: "f32[768, 768, 1, 1]", primals_59: "f32[768]", primals_61: "f32[768, 1, 7, 7]", primals_63: "f32[768]", primals_65: "f32[768, 768, 1, 1]", primals_67: "f32[768]", primals_69: "f32[768, 1, 7, 7]", primals_71: "f32[768]", primals_73: "f32[768, 768, 1, 1]", primals_75: "f32[768]", primals_77: "f32[768, 1, 7, 7]", primals_79: "f32[768]", primals_81: "f32[768, 768, 1, 1]", primals_83: "f32[768]", primals_85: "f32[768, 1, 7, 7]", primals_87: "f32[768]", primals_89: "f32[768, 768, 1, 1]", primals_91: "f32[768]", primals_93: "f32[768, 1, 7, 7]", primals_95: "f32[768]", primals_97: "f32[768, 768, 1, 1]", primals_99: "f32[768]", primals_101: "f32[768, 1, 7, 7]", primals_103: "f32[768]", primals_105: "f32[768, 768, 1, 1]", primals_107: "f32[768]", primals_109: "f32[768, 1, 7, 7]", primals_111: "f32[768]", primals_113: "f32[768, 768, 1, 1]", primals_115: "f32[768]", primals_117: "f32[768, 1, 7, 7]", primals_119: "f32[768]", primals_121: "f32[768, 768, 1, 1]", primals_123: "f32[768]", primals_125: "f32[768, 1, 7, 7]", primals_127: "f32[768]", primals_129: "f32[768, 768, 1, 1]", primals_131: "f32[768]", primals_133: "f32[768, 1, 7, 7]", primals_135: "f32[768]", primals_137: "f32[768, 768, 1, 1]", primals_139: "f32[768]", primals_141: "f32[768, 1, 7, 7]", primals_143: "f32[768]", primals_145: "f32[768, 768, 1, 1]", primals_147: "f32[768]", primals_149: "f32[768, 1, 7, 7]", primals_151: "f32[768]", primals_153: "f32[768, 768, 1, 1]", primals_155: "f32[768]", primals_157: "f32[768, 1, 7, 7]", primals_159: "f32[768]", primals_161: "f32[768, 768, 1, 1]", primals_163: "f32[768]", primals_165: "f32[768, 1, 7, 7]", primals_167: "f32[768]", primals_169: "f32[768, 768, 1, 1]", primals_171: "f32[768]", primals_173: "f32[768, 1, 7, 7]", primals_175: "f32[768]", primals_177: "f32[768, 768, 1, 1]", primals_179: "f32[768]", primals_181: "f32[768, 1, 7, 7]", primals_183: "f32[768]", primals_185: "f32[768, 768, 1, 1]", primals_187: "f32[768]", primals_189: "f32[768, 1, 7, 7]", primals_191: "f32[768]", primals_193: "f32[768, 768, 1, 1]", primals_195: "f32[768]", primals_197: "f32[768, 1, 7, 7]", primals_199: "f32[768]", primals_201: "f32[768, 768, 1, 1]", primals_203: "f32[768]", primals_205: "f32[768, 1, 7, 7]", primals_207: "f32[768]", primals_209: "f32[768, 768, 1, 1]", primals_211: "f32[768]", primals_213: "f32[768, 1, 7, 7]", primals_215: "f32[768]", primals_217: "f32[768, 768, 1, 1]", primals_219: "f32[768]", primals_221: "f32[768, 1, 7, 7]", primals_223: "f32[768]", primals_225: "f32[768, 768, 1, 1]", primals_227: "f32[768]", primals_229: "f32[768, 1, 7, 7]", primals_231: "f32[768]", primals_233: "f32[768, 768, 1, 1]", primals_235: "f32[768]", primals_237: "f32[768, 1, 7, 7]", primals_239: "f32[768]", primals_241: "f32[768, 768, 1, 1]", primals_243: "f32[768]", primals_245: "f32[768, 1, 7, 7]", primals_247: "f32[768]", primals_249: "f32[768, 768, 1, 1]", primals_251: "f32[768]", primals_253: "f32[768, 1, 7, 7]", primals_255: "f32[768]", primals_257: "f32[768, 768, 1, 1]", primals_259: "f32[768]", primals_458: "f32[8, 3, 224, 224]", convolution: "f32[8, 768, 32, 32]", squeeze_1: "f32[768]", add_4: "f32[8, 768, 32, 32]", convolution_1: "f32[8, 768, 32, 32]", squeeze_4: "f32[768]", add_10: "f32[8, 768, 32, 32]", convolution_2: "f32[8, 768, 32, 32]", squeeze_7: "f32[768]", add_15: "f32[8, 768, 32, 32]", convolution_3: "f32[8, 768, 32, 32]", squeeze_10: "f32[768]", add_21: "f32[8, 768, 32, 32]", convolution_4: "f32[8, 768, 32, 32]", squeeze_13: "f32[768]", add_26: "f32[8, 768, 32, 32]", convolution_5: "f32[8, 768, 32, 32]", squeeze_16: "f32[768]", add_32: "f32[8, 768, 32, 32]", convolution_6: "f32[8, 768, 32, 32]", squeeze_19: "f32[768]", add_37: "f32[8, 768, 32, 32]", convolution_7: "f32[8, 768, 32, 32]", squeeze_22: "f32[768]", add_43: "f32[8, 768, 32, 32]", convolution_8: "f32[8, 768, 32, 32]", squeeze_25: "f32[768]", add_48: "f32[8, 768, 32, 32]", convolution_9: "f32[8, 768, 32, 32]", squeeze_28: "f32[768]", add_54: "f32[8, 768, 32, 32]", convolution_10: "f32[8, 768, 32, 32]", squeeze_31: "f32[768]", add_59: "f32[8, 768, 32, 32]", convolution_11: "f32[8, 768, 32, 32]", squeeze_34: "f32[768]", add_65: "f32[8, 768, 32, 32]", convolution_12: "f32[8, 768, 32, 32]", squeeze_37: "f32[768]", add_70: "f32[8, 768, 32, 32]", convolution_13: "f32[8, 768, 32, 32]", squeeze_40: "f32[768]", add_76: "f32[8, 768, 32, 32]", convolution_14: "f32[8, 768, 32, 32]", squeeze_43: "f32[768]", add_81: "f32[8, 768, 32, 32]", convolution_15: "f32[8, 768, 32, 32]", squeeze_46: "f32[768]", add_87: "f32[8, 768, 32, 32]", convolution_16: "f32[8, 768, 32, 32]", squeeze_49: "f32[768]", add_92: "f32[8, 768, 32, 32]", convolution_17: "f32[8, 768, 32, 32]", squeeze_52: "f32[768]", add_98: "f32[8, 768, 32, 32]", convolution_18: "f32[8, 768, 32, 32]", squeeze_55: "f32[768]", add_103: "f32[8, 768, 32, 32]", convolution_19: "f32[8, 768, 32, 32]", squeeze_58: "f32[768]", add_109: "f32[8, 768, 32, 32]", convolution_20: "f32[8, 768, 32, 32]", squeeze_61: "f32[768]", add_114: "f32[8, 768, 32, 32]", convolution_21: "f32[8, 768, 32, 32]", squeeze_64: "f32[768]", add_120: "f32[8, 768, 32, 32]", convolution_22: "f32[8, 768, 32, 32]", squeeze_67: "f32[768]", add_125: "f32[8, 768, 32, 32]", convolution_23: "f32[8, 768, 32, 32]", squeeze_70: "f32[768]", add_131: "f32[8, 768, 32, 32]", convolution_24: "f32[8, 768, 32, 32]", squeeze_73: "f32[768]", add_136: "f32[8, 768, 32, 32]", convolution_25: "f32[8, 768, 32, 32]", squeeze_76: "f32[768]", add_142: "f32[8, 768, 32, 32]", convolution_26: "f32[8, 768, 32, 32]", squeeze_79: "f32[768]", add_147: "f32[8, 768, 32, 32]", convolution_27: "f32[8, 768, 32, 32]", squeeze_82: "f32[768]", add_153: "f32[8, 768, 32, 32]", convolution_28: "f32[8, 768, 32, 32]", squeeze_85: "f32[768]", add_158: "f32[8, 768, 32, 32]", convolution_29: "f32[8, 768, 32, 32]", squeeze_88: "f32[768]", add_164: "f32[8, 768, 32, 32]", convolution_30: "f32[8, 768, 32, 32]", squeeze_91: "f32[768]", add_169: "f32[8, 768, 32, 32]", convolution_31: "f32[8, 768, 32, 32]", squeeze_94: "f32[768]", add_175: "f32[8, 768, 32, 32]", convolution_32: "f32[8, 768, 32, 32]", squeeze_97: "f32[768]", add_180: "f32[8, 768, 32, 32]", convolution_33: "f32[8, 768, 32, 32]", squeeze_100: "f32[768]", add_186: "f32[8, 768, 32, 32]", convolution_34: "f32[8, 768, 32, 32]", squeeze_103: "f32[768]", add_191: "f32[8, 768, 32, 32]", convolution_35: "f32[8, 768, 32, 32]", squeeze_106: "f32[768]", add_197: "f32[8, 768, 32, 32]", convolution_36: "f32[8, 768, 32, 32]", squeeze_109: "f32[768]", add_202: "f32[8, 768, 32, 32]", convolution_37: "f32[8, 768, 32, 32]", squeeze_112: "f32[768]", add_208: "f32[8, 768, 32, 32]", convolution_38: "f32[8, 768, 32, 32]", squeeze_115: "f32[768]", add_213: "f32[8, 768, 32, 32]", convolution_39: "f32[8, 768, 32, 32]", squeeze_118: "f32[768]", add_219: "f32[8, 768, 32, 32]", convolution_40: "f32[8, 768, 32, 32]", squeeze_121: "f32[768]", add_224: "f32[8, 768, 32, 32]", convolution_41: "f32[8, 768, 32, 32]", squeeze_124: "f32[768]", add_230: "f32[8, 768, 32, 32]", convolution_42: "f32[8, 768, 32, 32]", squeeze_127: "f32[768]", add_235: "f32[8, 768, 32, 32]", convolution_43: "f32[8, 768, 32, 32]", squeeze_130: "f32[768]", add_241: "f32[8, 768, 32, 32]", convolution_44: "f32[8, 768, 32, 32]", squeeze_133: "f32[768]", add_246: "f32[8, 768, 32, 32]", convolution_45: "f32[8, 768, 32, 32]", squeeze_136: "f32[768]", add_252: "f32[8, 768, 32, 32]", convolution_46: "f32[8, 768, 32, 32]", squeeze_139: "f32[768]", add_257: "f32[8, 768, 32, 32]", convolution_47: "f32[8, 768, 32, 32]", squeeze_142: "f32[768]", add_263: "f32[8, 768, 32, 32]", convolution_48: "f32[8, 768, 32, 32]", squeeze_145: "f32[768]", add_268: "f32[8, 768, 32, 32]", convolution_49: "f32[8, 768, 32, 32]", squeeze_148: "f32[768]", add_274: "f32[8, 768, 32, 32]", convolution_50: "f32[8, 768, 32, 32]", squeeze_151: "f32[768]", add_279: "f32[8, 768, 32, 32]", convolution_51: "f32[8, 768, 32, 32]", squeeze_154: "f32[768]", add_285: "f32[8, 768, 32, 32]", convolution_52: "f32[8, 768, 32, 32]", squeeze_157: "f32[768]", add_290: "f32[8, 768, 32, 32]", convolution_53: "f32[8, 768, 32, 32]", squeeze_160: "f32[768]", add_296: "f32[8, 768, 32, 32]", convolution_54: "f32[8, 768, 32, 32]", squeeze_163: "f32[768]", add_301: "f32[8, 768, 32, 32]", convolution_55: "f32[8, 768, 32, 32]", squeeze_166: "f32[768]", add_307: "f32[8, 768, 32, 32]", convolution_56: "f32[8, 768, 32, 32]", squeeze_169: "f32[768]", add_312: "f32[8, 768, 32, 32]", convolution_57: "f32[8, 768, 32, 32]", squeeze_172: "f32[768]", add_318: "f32[8, 768, 32, 32]", convolution_58: "f32[8, 768, 32, 32]", squeeze_175: "f32[768]", add_323: "f32[8, 768, 32, 32]", convolution_59: "f32[8, 768, 32, 32]", squeeze_178: "f32[768]", add_329: "f32[8, 768, 32, 32]", convolution_60: "f32[8, 768, 32, 32]", squeeze_181: "f32[768]", add_334: "f32[8, 768, 32, 32]", convolution_61: "f32[8, 768, 32, 32]", squeeze_184: "f32[768]", add_340: "f32[8, 768, 32, 32]", convolution_62: "f32[8, 768, 32, 32]", squeeze_187: "f32[768]", add_345: "f32[8, 768, 32, 32]", convolution_63: "f32[8, 768, 32, 32]", squeeze_190: "f32[768]", add_351: "f32[8, 768, 32, 32]", convolution_64: "f32[8, 768, 32, 32]", squeeze_193: "f32[768]", clone: "f32[8, 768]", permute_1: "f32[1000, 768]", unsqueeze_262: "f32[1, 768, 1, 1]", unsqueeze_274: "f32[1, 768, 1, 1]", unsqueeze_286: "f32[1, 768, 1, 1]", unsqueeze_298: "f32[1, 768, 1, 1]", unsqueeze_310: "f32[1, 768, 1, 1]", unsqueeze_322: "f32[1, 768, 1, 1]", unsqueeze_334: "f32[1, 768, 1, 1]", unsqueeze_346: "f32[1, 768, 1, 1]", unsqueeze_358: "f32[1, 768, 1, 1]", unsqueeze_370: "f32[1, 768, 1, 1]", unsqueeze_382: "f32[1, 768, 1, 1]", unsqueeze_394: "f32[1, 768, 1, 1]", unsqueeze_406: "f32[1, 768, 1, 1]", unsqueeze_418: "f32[1, 768, 1, 1]", unsqueeze_430: "f32[1, 768, 1, 1]", unsqueeze_442: "f32[1, 768, 1, 1]", unsqueeze_454: "f32[1, 768, 1, 1]", unsqueeze_466: "f32[1, 768, 1, 1]", unsqueeze_478: "f32[1, 768, 1, 1]", unsqueeze_490: "f32[1, 768, 1, 1]", unsqueeze_502: "f32[1, 768, 1, 1]", unsqueeze_514: "f32[1, 768, 1, 1]", unsqueeze_526: "f32[1, 768, 1, 1]", unsqueeze_538: "f32[1, 768, 1, 1]", unsqueeze_550: "f32[1, 768, 1, 1]", unsqueeze_562: "f32[1, 768, 1, 1]", unsqueeze_574: "f32[1, 768, 1, 1]", unsqueeze_586: "f32[1, 768, 1, 1]", unsqueeze_598: "f32[1, 768, 1, 1]", unsqueeze_610: "f32[1, 768, 1, 1]", unsqueeze_622: "f32[1, 768, 1, 1]", unsqueeze_634: "f32[1, 768, 1, 1]", unsqueeze_646: "f32[1, 768, 1, 1]", unsqueeze_658: "f32[1, 768, 1, 1]", unsqueeze_670: "f32[1, 768, 1, 1]", unsqueeze_682: "f32[1, 768, 1, 1]", unsqueeze_694: "f32[1, 768, 1, 1]", unsqueeze_706: "f32[1, 768, 1, 1]", unsqueeze_718: "f32[1, 768, 1, 1]", unsqueeze_730: "f32[1, 768, 1, 1]", unsqueeze_742: "f32[1, 768, 1, 1]", unsqueeze_754: "f32[1, 768, 1, 1]", unsqueeze_766: "f32[1, 768, 1, 1]", unsqueeze_778: "f32[1, 768, 1, 1]", unsqueeze_790: "f32[1, 768, 1, 1]", unsqueeze_802: "f32[1, 768, 1, 1]", unsqueeze_814: "f32[1, 768, 1, 1]", unsqueeze_826: "f32[1, 768, 1, 1]", unsqueeze_838: "f32[1, 768, 1, 1]", unsqueeze_850: "f32[1, 768, 1, 1]", unsqueeze_862: "f32[1, 768, 1, 1]", unsqueeze_874: "f32[1, 768, 1, 1]", unsqueeze_886: "f32[1, 768, 1, 1]", unsqueeze_898: "f32[1, 768, 1, 1]", unsqueeze_910: "f32[1, 768, 1, 1]", unsqueeze_922: "f32[1, 768, 1, 1]", unsqueeze_934: "f32[1, 768, 1, 1]", unsqueeze_946: "f32[1, 768, 1, 1]", unsqueeze_958: "f32[1, 768, 1, 1]", unsqueeze_970: "f32[1, 768, 1, 1]", unsqueeze_982: "f32[1, 768, 1, 1]", unsqueeze_994: "f32[1, 768, 1, 1]", unsqueeze_1006: "f32[1, 768, 1, 1]", unsqueeze_1018: "f32[1, 768, 1, 1]", unsqueeze_1030: "f32[1, 768, 1, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:85, code: x = self.stem(x)
    relu: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution);  convolution = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    relu_1: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    relu_2: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_2);  convolution_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    relu_3: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    relu_4: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_4);  convolution_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    relu_5: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_5);  convolution_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    relu_6: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_6);  convolution_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    relu_7: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_7);  convolution_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    relu_8: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_8);  convolution_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    relu_9: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    relu_10: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_10);  convolution_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    relu_11: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_11);  convolution_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    relu_12: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_12);  convolution_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    relu_13: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_13);  convolution_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    relu_14: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_14);  convolution_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    relu_15: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_15);  convolution_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    relu_16: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_16);  convolution_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    relu_17: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_17);  convolution_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    relu_18: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_18);  convolution_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    relu_19: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_19);  convolution_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    relu_20: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_20);  convolution_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    relu_21: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_21);  convolution_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    relu_22: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_22);  convolution_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    relu_23: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_23);  convolution_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    relu_24: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_24);  convolution_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    relu_25: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_25);  convolution_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    relu_26: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_26);  convolution_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    relu_27: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_27);  convolution_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    relu_28: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_28);  convolution_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    relu_29: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_29);  convolution_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    relu_30: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_30);  convolution_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    relu_31: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_31);  convolution_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    relu_32: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_32);  convolution_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    relu_33: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_33);  convolution_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    relu_34: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_34);  convolution_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    relu_35: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_35);  convolution_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    relu_36: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_36);  convolution_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    relu_37: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_37);  convolution_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    relu_38: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_38);  convolution_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    relu_39: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_39);  convolution_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    relu_40: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_40);  convolution_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    relu_41: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_41);  convolution_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    relu_42: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_42);  convolution_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    relu_43: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_43);  convolution_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    relu_44: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_44);  convolution_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    relu_45: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_45);  convolution_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    relu_46: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_46);  convolution_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    relu_47: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_47);  convolution_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    relu_48: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_48);  convolution_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    relu_49: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_49);  convolution_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    relu_50: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_50);  convolution_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    relu_51: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_51);  convolution_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    relu_52: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_52);  convolution_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    relu_53: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_53);  convolution_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    relu_54: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_54);  convolution_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    relu_55: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_55);  convolution_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    relu_56: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_56);  convolution_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    relu_57: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_57);  convolution_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    relu_58: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_58);  convolution_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    relu_59: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_59);  convolution_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    relu_60: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_60);  convolution_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    relu_61: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_61);  convolution_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    relu_62: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_62);  convolution_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    relu_63: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_63);  convolution_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    relu_64: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_64);  convolution_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:95, code: return x if pre_logits else self.head(x)
    mm: "f32[8, 768]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 768]" = torch.ops.aten.mm.default(permute_2, clone);  permute_2 = clone = None
    permute_3: "f32[768, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_1: "f32[1000]" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_2: "f32[8, 768, 1, 1]" = torch.ops.aten.reshape.default(mm, [8, 768, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[8, 768, 32, 32]" = torch.ops.aten.expand.default(view_2, [8, 768, 32, 32]);  view_2 = None
    div: "f32[8, 768, 32, 32]" = torch.ops.aten.div.Scalar(expand, 1024);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    sum_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(div, [0, 2, 3])
    sub_65: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_64, unsqueeze_262);  unsqueeze_262 = None
    mul_455: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(div, sub_65)
    sum_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_455, [0, 2, 3]);  mul_455 = None
    mul_456: "f32[768]" = torch.ops.aten.mul.Tensor(sum_2, 0.0001220703125)
    unsqueeze_263: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_456, 0);  mul_456 = None
    unsqueeze_264: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 2);  unsqueeze_263 = None
    unsqueeze_265: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, 3);  unsqueeze_264 = None
    mul_457: "f32[768]" = torch.ops.aten.mul.Tensor(sum_3, 0.0001220703125)
    mul_458: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_193, squeeze_193)
    mul_459: "f32[768]" = torch.ops.aten.mul.Tensor(mul_457, mul_458);  mul_457 = mul_458 = None
    unsqueeze_266: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_459, 0);  mul_459 = None
    unsqueeze_267: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 2);  unsqueeze_266 = None
    unsqueeze_268: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 3);  unsqueeze_267 = None
    mul_460: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_193, primals_259);  primals_259 = None
    unsqueeze_269: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_460, 0);  mul_460 = None
    unsqueeze_270: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 2);  unsqueeze_269 = None
    unsqueeze_271: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, 3);  unsqueeze_270 = None
    mul_461: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_268);  sub_65 = unsqueeze_268 = None
    sub_67: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(div, mul_461);  div = mul_461 = None
    sub_68: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_67, unsqueeze_265);  sub_67 = unsqueeze_265 = None
    mul_462: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_271);  sub_68 = unsqueeze_271 = None
    mul_463: "f32[768]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_193);  sum_3 = squeeze_193 = None
    le: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_64, 0);  relu_64 = None
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le, full_default, mul_462);  le = mul_462 = None
    sum_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    convolution_backward = torch.ops.aten.convolution_backward.default(where, add_351, primals_257, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where = add_351 = primals_257 = None
    getitem_130: "f32[8, 768, 32, 32]" = convolution_backward[0]
    getitem_131: "f32[768, 768, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    sum_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_130, [0, 2, 3])
    sub_69: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_63, unsqueeze_274);  unsqueeze_274 = None
    mul_464: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_130, sub_69)
    sum_6: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_464, [0, 2, 3]);  mul_464 = None
    mul_465: "f32[768]" = torch.ops.aten.mul.Tensor(sum_5, 0.0001220703125)
    unsqueeze_275: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_465, 0);  mul_465 = None
    unsqueeze_276: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 2);  unsqueeze_275 = None
    unsqueeze_277: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 3);  unsqueeze_276 = None
    mul_466: "f32[768]" = torch.ops.aten.mul.Tensor(sum_6, 0.0001220703125)
    mul_467: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_190, squeeze_190)
    mul_468: "f32[768]" = torch.ops.aten.mul.Tensor(mul_466, mul_467);  mul_466 = mul_467 = None
    unsqueeze_278: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_468, 0);  mul_468 = None
    unsqueeze_279: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 2);  unsqueeze_278 = None
    unsqueeze_280: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 3);  unsqueeze_279 = None
    mul_469: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_190, primals_255);  primals_255 = None
    unsqueeze_281: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_469, 0);  mul_469 = None
    unsqueeze_282: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 2);  unsqueeze_281 = None
    unsqueeze_283: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, 3);  unsqueeze_282 = None
    mul_470: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_280);  sub_69 = unsqueeze_280 = None
    sub_71: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(getitem_130, mul_470);  mul_470 = None
    sub_72: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_71, unsqueeze_277);  sub_71 = unsqueeze_277 = None
    mul_471: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_283);  sub_72 = unsqueeze_283 = None
    mul_472: "f32[768]" = torch.ops.aten.mul.Tensor(sum_6, squeeze_190);  sum_6 = squeeze_190 = None
    le_1: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_63, 0);  relu_63 = None
    where_1: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_1, full_default, mul_471);  le_1 = mul_471 = None
    sum_7: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(where_1, add_345, primals_253, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False]);  where_1 = add_345 = primals_253 = None
    getitem_133: "f32[8, 768, 32, 32]" = convolution_backward_1[0]
    getitem_134: "f32[768, 1, 7, 7]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    add_357: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(getitem_130, getitem_133);  getitem_130 = getitem_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    sum_8: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_357, [0, 2, 3])
    sub_73: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_62, unsqueeze_286);  unsqueeze_286 = None
    mul_473: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(add_357, sub_73)
    sum_9: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_473, [0, 2, 3]);  mul_473 = None
    mul_474: "f32[768]" = torch.ops.aten.mul.Tensor(sum_8, 0.0001220703125)
    unsqueeze_287: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_474, 0);  mul_474 = None
    unsqueeze_288: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 2);  unsqueeze_287 = None
    unsqueeze_289: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 3);  unsqueeze_288 = None
    mul_475: "f32[768]" = torch.ops.aten.mul.Tensor(sum_9, 0.0001220703125)
    mul_476: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_187, squeeze_187)
    mul_477: "f32[768]" = torch.ops.aten.mul.Tensor(mul_475, mul_476);  mul_475 = mul_476 = None
    unsqueeze_290: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_477, 0);  mul_477 = None
    unsqueeze_291: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 2);  unsqueeze_290 = None
    unsqueeze_292: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 3);  unsqueeze_291 = None
    mul_478: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_187, primals_251);  primals_251 = None
    unsqueeze_293: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_478, 0);  mul_478 = None
    unsqueeze_294: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 2);  unsqueeze_293 = None
    unsqueeze_295: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, 3);  unsqueeze_294 = None
    mul_479: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_292);  sub_73 = unsqueeze_292 = None
    sub_75: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(add_357, mul_479);  add_357 = mul_479 = None
    sub_76: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_75, unsqueeze_289);  sub_75 = unsqueeze_289 = None
    mul_480: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_295);  sub_76 = unsqueeze_295 = None
    mul_481: "f32[768]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_187);  sum_9 = squeeze_187 = None
    le_2: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_62, 0);  relu_62 = None
    where_2: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_2, full_default, mul_480);  le_2 = mul_480 = None
    sum_10: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(where_2, add_340, primals_249, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_2 = add_340 = primals_249 = None
    getitem_136: "f32[8, 768, 32, 32]" = convolution_backward_2[0]
    getitem_137: "f32[768, 768, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    sum_11: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_136, [0, 2, 3])
    sub_77: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_61, unsqueeze_298);  unsqueeze_298 = None
    mul_482: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_136, sub_77)
    sum_12: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_482, [0, 2, 3]);  mul_482 = None
    mul_483: "f32[768]" = torch.ops.aten.mul.Tensor(sum_11, 0.0001220703125)
    unsqueeze_299: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_483, 0);  mul_483 = None
    unsqueeze_300: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 2);  unsqueeze_299 = None
    unsqueeze_301: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, 3);  unsqueeze_300 = None
    mul_484: "f32[768]" = torch.ops.aten.mul.Tensor(sum_12, 0.0001220703125)
    mul_485: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_184, squeeze_184)
    mul_486: "f32[768]" = torch.ops.aten.mul.Tensor(mul_484, mul_485);  mul_484 = mul_485 = None
    unsqueeze_302: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_486, 0);  mul_486 = None
    unsqueeze_303: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 2);  unsqueeze_302 = None
    unsqueeze_304: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 3);  unsqueeze_303 = None
    mul_487: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_184, primals_247);  primals_247 = None
    unsqueeze_305: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_487, 0);  mul_487 = None
    unsqueeze_306: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 2);  unsqueeze_305 = None
    unsqueeze_307: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, 3);  unsqueeze_306 = None
    mul_488: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_304);  sub_77 = unsqueeze_304 = None
    sub_79: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(getitem_136, mul_488);  mul_488 = None
    sub_80: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_79, unsqueeze_301);  sub_79 = unsqueeze_301 = None
    mul_489: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_307);  sub_80 = unsqueeze_307 = None
    mul_490: "f32[768]" = torch.ops.aten.mul.Tensor(sum_12, squeeze_184);  sum_12 = squeeze_184 = None
    le_3: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_61, 0);  relu_61 = None
    where_3: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_3, full_default, mul_489);  le_3 = mul_489 = None
    sum_13: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(where_3, add_334, primals_245, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False]);  where_3 = add_334 = primals_245 = None
    getitem_139: "f32[8, 768, 32, 32]" = convolution_backward_3[0]
    getitem_140: "f32[768, 1, 7, 7]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    add_358: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(getitem_136, getitem_139);  getitem_136 = getitem_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    sum_14: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_358, [0, 2, 3])
    sub_81: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_60, unsqueeze_310);  unsqueeze_310 = None
    mul_491: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(add_358, sub_81)
    sum_15: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_491, [0, 2, 3]);  mul_491 = None
    mul_492: "f32[768]" = torch.ops.aten.mul.Tensor(sum_14, 0.0001220703125)
    unsqueeze_311: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_492, 0);  mul_492 = None
    unsqueeze_312: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 2);  unsqueeze_311 = None
    unsqueeze_313: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, 3);  unsqueeze_312 = None
    mul_493: "f32[768]" = torch.ops.aten.mul.Tensor(sum_15, 0.0001220703125)
    mul_494: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_181, squeeze_181)
    mul_495: "f32[768]" = torch.ops.aten.mul.Tensor(mul_493, mul_494);  mul_493 = mul_494 = None
    unsqueeze_314: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_495, 0);  mul_495 = None
    unsqueeze_315: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 2);  unsqueeze_314 = None
    unsqueeze_316: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 3);  unsqueeze_315 = None
    mul_496: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_181, primals_243);  primals_243 = None
    unsqueeze_317: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_496, 0);  mul_496 = None
    unsqueeze_318: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 2);  unsqueeze_317 = None
    unsqueeze_319: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, 3);  unsqueeze_318 = None
    mul_497: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_316);  sub_81 = unsqueeze_316 = None
    sub_83: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(add_358, mul_497);  add_358 = mul_497 = None
    sub_84: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_83, unsqueeze_313);  sub_83 = unsqueeze_313 = None
    mul_498: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_319);  sub_84 = unsqueeze_319 = None
    mul_499: "f32[768]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_181);  sum_15 = squeeze_181 = None
    le_4: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_60, 0);  relu_60 = None
    where_4: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_4, full_default, mul_498);  le_4 = mul_498 = None
    sum_16: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(where_4, add_329, primals_241, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_4 = add_329 = primals_241 = None
    getitem_142: "f32[8, 768, 32, 32]" = convolution_backward_4[0]
    getitem_143: "f32[768, 768, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    sum_17: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_142, [0, 2, 3])
    sub_85: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_59, unsqueeze_322);  unsqueeze_322 = None
    mul_500: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_142, sub_85)
    sum_18: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_500, [0, 2, 3]);  mul_500 = None
    mul_501: "f32[768]" = torch.ops.aten.mul.Tensor(sum_17, 0.0001220703125)
    unsqueeze_323: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_501, 0);  mul_501 = None
    unsqueeze_324: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 2);  unsqueeze_323 = None
    unsqueeze_325: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, 3);  unsqueeze_324 = None
    mul_502: "f32[768]" = torch.ops.aten.mul.Tensor(sum_18, 0.0001220703125)
    mul_503: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_178, squeeze_178)
    mul_504: "f32[768]" = torch.ops.aten.mul.Tensor(mul_502, mul_503);  mul_502 = mul_503 = None
    unsqueeze_326: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_504, 0);  mul_504 = None
    unsqueeze_327: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 2);  unsqueeze_326 = None
    unsqueeze_328: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_327, 3);  unsqueeze_327 = None
    mul_505: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_178, primals_239);  primals_239 = None
    unsqueeze_329: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_505, 0);  mul_505 = None
    unsqueeze_330: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 2);  unsqueeze_329 = None
    unsqueeze_331: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, 3);  unsqueeze_330 = None
    mul_506: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_328);  sub_85 = unsqueeze_328 = None
    sub_87: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(getitem_142, mul_506);  mul_506 = None
    sub_88: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_87, unsqueeze_325);  sub_87 = unsqueeze_325 = None
    mul_507: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_331);  sub_88 = unsqueeze_331 = None
    mul_508: "f32[768]" = torch.ops.aten.mul.Tensor(sum_18, squeeze_178);  sum_18 = squeeze_178 = None
    le_5: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_59, 0);  relu_59 = None
    where_5: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_5, full_default, mul_507);  le_5 = mul_507 = None
    sum_19: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(where_5, add_323, primals_237, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False]);  where_5 = add_323 = primals_237 = None
    getitem_145: "f32[8, 768, 32, 32]" = convolution_backward_5[0]
    getitem_146: "f32[768, 1, 7, 7]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    add_359: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(getitem_142, getitem_145);  getitem_142 = getitem_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    sum_20: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_359, [0, 2, 3])
    sub_89: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_58, unsqueeze_334);  unsqueeze_334 = None
    mul_509: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(add_359, sub_89)
    sum_21: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_509, [0, 2, 3]);  mul_509 = None
    mul_510: "f32[768]" = torch.ops.aten.mul.Tensor(sum_20, 0.0001220703125)
    unsqueeze_335: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_510, 0);  mul_510 = None
    unsqueeze_336: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 2);  unsqueeze_335 = None
    unsqueeze_337: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, 3);  unsqueeze_336 = None
    mul_511: "f32[768]" = torch.ops.aten.mul.Tensor(sum_21, 0.0001220703125)
    mul_512: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_175, squeeze_175)
    mul_513: "f32[768]" = torch.ops.aten.mul.Tensor(mul_511, mul_512);  mul_511 = mul_512 = None
    unsqueeze_338: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_513, 0);  mul_513 = None
    unsqueeze_339: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 2);  unsqueeze_338 = None
    unsqueeze_340: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 3);  unsqueeze_339 = None
    mul_514: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_175, primals_235);  primals_235 = None
    unsqueeze_341: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_514, 0);  mul_514 = None
    unsqueeze_342: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 2);  unsqueeze_341 = None
    unsqueeze_343: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, 3);  unsqueeze_342 = None
    mul_515: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_340);  sub_89 = unsqueeze_340 = None
    sub_91: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(add_359, mul_515);  add_359 = mul_515 = None
    sub_92: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_91, unsqueeze_337);  sub_91 = unsqueeze_337 = None
    mul_516: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_343);  sub_92 = unsqueeze_343 = None
    mul_517: "f32[768]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_175);  sum_21 = squeeze_175 = None
    le_6: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_58, 0);  relu_58 = None
    where_6: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_6, full_default, mul_516);  le_6 = mul_516 = None
    sum_22: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(where_6, add_318, primals_233, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_6 = add_318 = primals_233 = None
    getitem_148: "f32[8, 768, 32, 32]" = convolution_backward_6[0]
    getitem_149: "f32[768, 768, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    sum_23: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_148, [0, 2, 3])
    sub_93: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_57, unsqueeze_346);  unsqueeze_346 = None
    mul_518: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_148, sub_93)
    sum_24: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_518, [0, 2, 3]);  mul_518 = None
    mul_519: "f32[768]" = torch.ops.aten.mul.Tensor(sum_23, 0.0001220703125)
    unsqueeze_347: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_519, 0);  mul_519 = None
    unsqueeze_348: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 2);  unsqueeze_347 = None
    unsqueeze_349: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, 3);  unsqueeze_348 = None
    mul_520: "f32[768]" = torch.ops.aten.mul.Tensor(sum_24, 0.0001220703125)
    mul_521: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_172, squeeze_172)
    mul_522: "f32[768]" = torch.ops.aten.mul.Tensor(mul_520, mul_521);  mul_520 = mul_521 = None
    unsqueeze_350: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_522, 0);  mul_522 = None
    unsqueeze_351: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 2);  unsqueeze_350 = None
    unsqueeze_352: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 3);  unsqueeze_351 = None
    mul_523: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_172, primals_231);  primals_231 = None
    unsqueeze_353: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_523, 0);  mul_523 = None
    unsqueeze_354: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 2);  unsqueeze_353 = None
    unsqueeze_355: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, 3);  unsqueeze_354 = None
    mul_524: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_352);  sub_93 = unsqueeze_352 = None
    sub_95: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(getitem_148, mul_524);  mul_524 = None
    sub_96: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_95, unsqueeze_349);  sub_95 = unsqueeze_349 = None
    mul_525: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_355);  sub_96 = unsqueeze_355 = None
    mul_526: "f32[768]" = torch.ops.aten.mul.Tensor(sum_24, squeeze_172);  sum_24 = squeeze_172 = None
    le_7: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_57, 0);  relu_57 = None
    where_7: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_7, full_default, mul_525);  le_7 = mul_525 = None
    sum_25: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(where_7, add_312, primals_229, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False]);  where_7 = add_312 = primals_229 = None
    getitem_151: "f32[8, 768, 32, 32]" = convolution_backward_7[0]
    getitem_152: "f32[768, 1, 7, 7]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    add_360: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(getitem_148, getitem_151);  getitem_148 = getitem_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    sum_26: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_360, [0, 2, 3])
    sub_97: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_56, unsqueeze_358);  unsqueeze_358 = None
    mul_527: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(add_360, sub_97)
    sum_27: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_527, [0, 2, 3]);  mul_527 = None
    mul_528: "f32[768]" = torch.ops.aten.mul.Tensor(sum_26, 0.0001220703125)
    unsqueeze_359: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_528, 0);  mul_528 = None
    unsqueeze_360: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 2);  unsqueeze_359 = None
    unsqueeze_361: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, 3);  unsqueeze_360 = None
    mul_529: "f32[768]" = torch.ops.aten.mul.Tensor(sum_27, 0.0001220703125)
    mul_530: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_169, squeeze_169)
    mul_531: "f32[768]" = torch.ops.aten.mul.Tensor(mul_529, mul_530);  mul_529 = mul_530 = None
    unsqueeze_362: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_531, 0);  mul_531 = None
    unsqueeze_363: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 2);  unsqueeze_362 = None
    unsqueeze_364: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 3);  unsqueeze_363 = None
    mul_532: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_169, primals_227);  primals_227 = None
    unsqueeze_365: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_532, 0);  mul_532 = None
    unsqueeze_366: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 2);  unsqueeze_365 = None
    unsqueeze_367: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, 3);  unsqueeze_366 = None
    mul_533: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_364);  sub_97 = unsqueeze_364 = None
    sub_99: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(add_360, mul_533);  add_360 = mul_533 = None
    sub_100: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_99, unsqueeze_361);  sub_99 = unsqueeze_361 = None
    mul_534: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_367);  sub_100 = unsqueeze_367 = None
    mul_535: "f32[768]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_169);  sum_27 = squeeze_169 = None
    le_8: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_56, 0);  relu_56 = None
    where_8: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_8, full_default, mul_534);  le_8 = mul_534 = None
    sum_28: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(where_8, add_307, primals_225, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_8 = add_307 = primals_225 = None
    getitem_154: "f32[8, 768, 32, 32]" = convolution_backward_8[0]
    getitem_155: "f32[768, 768, 1, 1]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    sum_29: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_154, [0, 2, 3])
    sub_101: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_55, unsqueeze_370);  unsqueeze_370 = None
    mul_536: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_154, sub_101)
    sum_30: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_536, [0, 2, 3]);  mul_536 = None
    mul_537: "f32[768]" = torch.ops.aten.mul.Tensor(sum_29, 0.0001220703125)
    unsqueeze_371: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_537, 0);  mul_537 = None
    unsqueeze_372: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 2);  unsqueeze_371 = None
    unsqueeze_373: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, 3);  unsqueeze_372 = None
    mul_538: "f32[768]" = torch.ops.aten.mul.Tensor(sum_30, 0.0001220703125)
    mul_539: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
    mul_540: "f32[768]" = torch.ops.aten.mul.Tensor(mul_538, mul_539);  mul_538 = mul_539 = None
    unsqueeze_374: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_540, 0);  mul_540 = None
    unsqueeze_375: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 2);  unsqueeze_374 = None
    unsqueeze_376: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_375, 3);  unsqueeze_375 = None
    mul_541: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_166, primals_223);  primals_223 = None
    unsqueeze_377: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_541, 0);  mul_541 = None
    unsqueeze_378: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 2);  unsqueeze_377 = None
    unsqueeze_379: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, 3);  unsqueeze_378 = None
    mul_542: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_376);  sub_101 = unsqueeze_376 = None
    sub_103: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(getitem_154, mul_542);  mul_542 = None
    sub_104: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_103, unsqueeze_373);  sub_103 = unsqueeze_373 = None
    mul_543: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_379);  sub_104 = unsqueeze_379 = None
    mul_544: "f32[768]" = torch.ops.aten.mul.Tensor(sum_30, squeeze_166);  sum_30 = squeeze_166 = None
    le_9: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_55, 0);  relu_55 = None
    where_9: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_9, full_default, mul_543);  le_9 = mul_543 = None
    sum_31: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(where_9, add_301, primals_221, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False]);  where_9 = add_301 = primals_221 = None
    getitem_157: "f32[8, 768, 32, 32]" = convolution_backward_9[0]
    getitem_158: "f32[768, 1, 7, 7]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    add_361: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(getitem_154, getitem_157);  getitem_154 = getitem_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    sum_32: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_361, [0, 2, 3])
    sub_105: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_54, unsqueeze_382);  unsqueeze_382 = None
    mul_545: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(add_361, sub_105)
    sum_33: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_545, [0, 2, 3]);  mul_545 = None
    mul_546: "f32[768]" = torch.ops.aten.mul.Tensor(sum_32, 0.0001220703125)
    unsqueeze_383: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_546, 0);  mul_546 = None
    unsqueeze_384: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 2);  unsqueeze_383 = None
    unsqueeze_385: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, 3);  unsqueeze_384 = None
    mul_547: "f32[768]" = torch.ops.aten.mul.Tensor(sum_33, 0.0001220703125)
    mul_548: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
    mul_549: "f32[768]" = torch.ops.aten.mul.Tensor(mul_547, mul_548);  mul_547 = mul_548 = None
    unsqueeze_386: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_549, 0);  mul_549 = None
    unsqueeze_387: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 2);  unsqueeze_386 = None
    unsqueeze_388: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_387, 3);  unsqueeze_387 = None
    mul_550: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_163, primals_219);  primals_219 = None
    unsqueeze_389: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_550, 0);  mul_550 = None
    unsqueeze_390: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 2);  unsqueeze_389 = None
    unsqueeze_391: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, 3);  unsqueeze_390 = None
    mul_551: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_388);  sub_105 = unsqueeze_388 = None
    sub_107: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(add_361, mul_551);  add_361 = mul_551 = None
    sub_108: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_107, unsqueeze_385);  sub_107 = unsqueeze_385 = None
    mul_552: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_391);  sub_108 = unsqueeze_391 = None
    mul_553: "f32[768]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_163);  sum_33 = squeeze_163 = None
    le_10: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_54, 0);  relu_54 = None
    where_10: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_10, full_default, mul_552);  le_10 = mul_552 = None
    sum_34: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(where_10, add_296, primals_217, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_10 = add_296 = primals_217 = None
    getitem_160: "f32[8, 768, 32, 32]" = convolution_backward_10[0]
    getitem_161: "f32[768, 768, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    sum_35: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_160, [0, 2, 3])
    sub_109: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_53, unsqueeze_394);  unsqueeze_394 = None
    mul_554: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_160, sub_109)
    sum_36: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_554, [0, 2, 3]);  mul_554 = None
    mul_555: "f32[768]" = torch.ops.aten.mul.Tensor(sum_35, 0.0001220703125)
    unsqueeze_395: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_555, 0);  mul_555 = None
    unsqueeze_396: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 2);  unsqueeze_395 = None
    unsqueeze_397: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, 3);  unsqueeze_396 = None
    mul_556: "f32[768]" = torch.ops.aten.mul.Tensor(sum_36, 0.0001220703125)
    mul_557: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
    mul_558: "f32[768]" = torch.ops.aten.mul.Tensor(mul_556, mul_557);  mul_556 = mul_557 = None
    unsqueeze_398: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_558, 0);  mul_558 = None
    unsqueeze_399: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 2);  unsqueeze_398 = None
    unsqueeze_400: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_399, 3);  unsqueeze_399 = None
    mul_559: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_160, primals_215);  primals_215 = None
    unsqueeze_401: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_559, 0);  mul_559 = None
    unsqueeze_402: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 2);  unsqueeze_401 = None
    unsqueeze_403: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, 3);  unsqueeze_402 = None
    mul_560: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_400);  sub_109 = unsqueeze_400 = None
    sub_111: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(getitem_160, mul_560);  mul_560 = None
    sub_112: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_111, unsqueeze_397);  sub_111 = unsqueeze_397 = None
    mul_561: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_403);  sub_112 = unsqueeze_403 = None
    mul_562: "f32[768]" = torch.ops.aten.mul.Tensor(sum_36, squeeze_160);  sum_36 = squeeze_160 = None
    le_11: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_53, 0);  relu_53 = None
    where_11: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_11, full_default, mul_561);  le_11 = mul_561 = None
    sum_37: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(where_11, add_290, primals_213, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False]);  where_11 = add_290 = primals_213 = None
    getitem_163: "f32[8, 768, 32, 32]" = convolution_backward_11[0]
    getitem_164: "f32[768, 1, 7, 7]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    add_362: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(getitem_160, getitem_163);  getitem_160 = getitem_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    sum_38: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_362, [0, 2, 3])
    sub_113: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_52, unsqueeze_406);  unsqueeze_406 = None
    mul_563: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(add_362, sub_113)
    sum_39: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_563, [0, 2, 3]);  mul_563 = None
    mul_564: "f32[768]" = torch.ops.aten.mul.Tensor(sum_38, 0.0001220703125)
    unsqueeze_407: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_564, 0);  mul_564 = None
    unsqueeze_408: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 2);  unsqueeze_407 = None
    unsqueeze_409: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, 3);  unsqueeze_408 = None
    mul_565: "f32[768]" = torch.ops.aten.mul.Tensor(sum_39, 0.0001220703125)
    mul_566: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
    mul_567: "f32[768]" = torch.ops.aten.mul.Tensor(mul_565, mul_566);  mul_565 = mul_566 = None
    unsqueeze_410: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_567, 0);  mul_567 = None
    unsqueeze_411: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 2);  unsqueeze_410 = None
    unsqueeze_412: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 3);  unsqueeze_411 = None
    mul_568: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_157, primals_211);  primals_211 = None
    unsqueeze_413: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_568, 0);  mul_568 = None
    unsqueeze_414: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 2);  unsqueeze_413 = None
    unsqueeze_415: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, 3);  unsqueeze_414 = None
    mul_569: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_412);  sub_113 = unsqueeze_412 = None
    sub_115: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(add_362, mul_569);  add_362 = mul_569 = None
    sub_116: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_115, unsqueeze_409);  sub_115 = unsqueeze_409 = None
    mul_570: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_415);  sub_116 = unsqueeze_415 = None
    mul_571: "f32[768]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_157);  sum_39 = squeeze_157 = None
    le_12: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_52, 0);  relu_52 = None
    where_12: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_12, full_default, mul_570);  le_12 = mul_570 = None
    sum_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(where_12, add_285, primals_209, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_12 = add_285 = primals_209 = None
    getitem_166: "f32[8, 768, 32, 32]" = convolution_backward_12[0]
    getitem_167: "f32[768, 768, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    sum_41: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_166, [0, 2, 3])
    sub_117: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_51, unsqueeze_418);  unsqueeze_418 = None
    mul_572: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_166, sub_117)
    sum_42: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_572, [0, 2, 3]);  mul_572 = None
    mul_573: "f32[768]" = torch.ops.aten.mul.Tensor(sum_41, 0.0001220703125)
    unsqueeze_419: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_573, 0);  mul_573 = None
    unsqueeze_420: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 2);  unsqueeze_419 = None
    unsqueeze_421: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, 3);  unsqueeze_420 = None
    mul_574: "f32[768]" = torch.ops.aten.mul.Tensor(sum_42, 0.0001220703125)
    mul_575: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
    mul_576: "f32[768]" = torch.ops.aten.mul.Tensor(mul_574, mul_575);  mul_574 = mul_575 = None
    unsqueeze_422: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_576, 0);  mul_576 = None
    unsqueeze_423: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 2);  unsqueeze_422 = None
    unsqueeze_424: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 3);  unsqueeze_423 = None
    mul_577: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_154, primals_207);  primals_207 = None
    unsqueeze_425: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_577, 0);  mul_577 = None
    unsqueeze_426: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 2);  unsqueeze_425 = None
    unsqueeze_427: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 3);  unsqueeze_426 = None
    mul_578: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_424);  sub_117 = unsqueeze_424 = None
    sub_119: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(getitem_166, mul_578);  mul_578 = None
    sub_120: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_119, unsqueeze_421);  sub_119 = unsqueeze_421 = None
    mul_579: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_427);  sub_120 = unsqueeze_427 = None
    mul_580: "f32[768]" = torch.ops.aten.mul.Tensor(sum_42, squeeze_154);  sum_42 = squeeze_154 = None
    le_13: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_51, 0);  relu_51 = None
    where_13: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_13, full_default, mul_579);  le_13 = mul_579 = None
    sum_43: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(where_13, add_279, primals_205, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False]);  where_13 = add_279 = primals_205 = None
    getitem_169: "f32[8, 768, 32, 32]" = convolution_backward_13[0]
    getitem_170: "f32[768, 1, 7, 7]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    add_363: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(getitem_166, getitem_169);  getitem_166 = getitem_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    sum_44: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_363, [0, 2, 3])
    sub_121: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_50, unsqueeze_430);  unsqueeze_430 = None
    mul_581: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(add_363, sub_121)
    sum_45: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_581, [0, 2, 3]);  mul_581 = None
    mul_582: "f32[768]" = torch.ops.aten.mul.Tensor(sum_44, 0.0001220703125)
    unsqueeze_431: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_582, 0);  mul_582 = None
    unsqueeze_432: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 2);  unsqueeze_431 = None
    unsqueeze_433: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, 3);  unsqueeze_432 = None
    mul_583: "f32[768]" = torch.ops.aten.mul.Tensor(sum_45, 0.0001220703125)
    mul_584: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
    mul_585: "f32[768]" = torch.ops.aten.mul.Tensor(mul_583, mul_584);  mul_583 = mul_584 = None
    unsqueeze_434: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_585, 0);  mul_585 = None
    unsqueeze_435: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 2);  unsqueeze_434 = None
    unsqueeze_436: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_435, 3);  unsqueeze_435 = None
    mul_586: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_151, primals_203);  primals_203 = None
    unsqueeze_437: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_586, 0);  mul_586 = None
    unsqueeze_438: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 2);  unsqueeze_437 = None
    unsqueeze_439: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 3);  unsqueeze_438 = None
    mul_587: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_436);  sub_121 = unsqueeze_436 = None
    sub_123: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(add_363, mul_587);  add_363 = mul_587 = None
    sub_124: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_123, unsqueeze_433);  sub_123 = unsqueeze_433 = None
    mul_588: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_439);  sub_124 = unsqueeze_439 = None
    mul_589: "f32[768]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_151);  sum_45 = squeeze_151 = None
    le_14: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_50, 0);  relu_50 = None
    where_14: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_14, full_default, mul_588);  le_14 = mul_588 = None
    sum_46: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(where_14, add_274, primals_201, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_14 = add_274 = primals_201 = None
    getitem_172: "f32[8, 768, 32, 32]" = convolution_backward_14[0]
    getitem_173: "f32[768, 768, 1, 1]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    sum_47: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_172, [0, 2, 3])
    sub_125: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_49, unsqueeze_442);  unsqueeze_442 = None
    mul_590: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_172, sub_125)
    sum_48: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_590, [0, 2, 3]);  mul_590 = None
    mul_591: "f32[768]" = torch.ops.aten.mul.Tensor(sum_47, 0.0001220703125)
    unsqueeze_443: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_591, 0);  mul_591 = None
    unsqueeze_444: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 2);  unsqueeze_443 = None
    unsqueeze_445: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, 3);  unsqueeze_444 = None
    mul_592: "f32[768]" = torch.ops.aten.mul.Tensor(sum_48, 0.0001220703125)
    mul_593: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
    mul_594: "f32[768]" = torch.ops.aten.mul.Tensor(mul_592, mul_593);  mul_592 = mul_593 = None
    unsqueeze_446: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_594, 0);  mul_594 = None
    unsqueeze_447: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 2);  unsqueeze_446 = None
    unsqueeze_448: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_447, 3);  unsqueeze_447 = None
    mul_595: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_148, primals_199);  primals_199 = None
    unsqueeze_449: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_595, 0);  mul_595 = None
    unsqueeze_450: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 2);  unsqueeze_449 = None
    unsqueeze_451: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, 3);  unsqueeze_450 = None
    mul_596: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_448);  sub_125 = unsqueeze_448 = None
    sub_127: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(getitem_172, mul_596);  mul_596 = None
    sub_128: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_127, unsqueeze_445);  sub_127 = unsqueeze_445 = None
    mul_597: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_451);  sub_128 = unsqueeze_451 = None
    mul_598: "f32[768]" = torch.ops.aten.mul.Tensor(sum_48, squeeze_148);  sum_48 = squeeze_148 = None
    le_15: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_49, 0);  relu_49 = None
    where_15: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_15, full_default, mul_597);  le_15 = mul_597 = None
    sum_49: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(where_15, add_268, primals_197, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False]);  where_15 = add_268 = primals_197 = None
    getitem_175: "f32[8, 768, 32, 32]" = convolution_backward_15[0]
    getitem_176: "f32[768, 1, 7, 7]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    add_364: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(getitem_172, getitem_175);  getitem_172 = getitem_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    sum_50: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_364, [0, 2, 3])
    sub_129: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_48, unsqueeze_454);  unsqueeze_454 = None
    mul_599: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(add_364, sub_129)
    sum_51: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_599, [0, 2, 3]);  mul_599 = None
    mul_600: "f32[768]" = torch.ops.aten.mul.Tensor(sum_50, 0.0001220703125)
    unsqueeze_455: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_600, 0);  mul_600 = None
    unsqueeze_456: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 2);  unsqueeze_455 = None
    unsqueeze_457: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, 3);  unsqueeze_456 = None
    mul_601: "f32[768]" = torch.ops.aten.mul.Tensor(sum_51, 0.0001220703125)
    mul_602: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
    mul_603: "f32[768]" = torch.ops.aten.mul.Tensor(mul_601, mul_602);  mul_601 = mul_602 = None
    unsqueeze_458: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_603, 0);  mul_603 = None
    unsqueeze_459: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 2);  unsqueeze_458 = None
    unsqueeze_460: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 3);  unsqueeze_459 = None
    mul_604: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_145, primals_195);  primals_195 = None
    unsqueeze_461: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_604, 0);  mul_604 = None
    unsqueeze_462: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 2);  unsqueeze_461 = None
    unsqueeze_463: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, 3);  unsqueeze_462 = None
    mul_605: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_460);  sub_129 = unsqueeze_460 = None
    sub_131: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(add_364, mul_605);  add_364 = mul_605 = None
    sub_132: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_131, unsqueeze_457);  sub_131 = unsqueeze_457 = None
    mul_606: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_463);  sub_132 = unsqueeze_463 = None
    mul_607: "f32[768]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_145);  sum_51 = squeeze_145 = None
    le_16: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_48, 0);  relu_48 = None
    where_16: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_16, full_default, mul_606);  le_16 = mul_606 = None
    sum_52: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(where_16, add_263, primals_193, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_16 = add_263 = primals_193 = None
    getitem_178: "f32[8, 768, 32, 32]" = convolution_backward_16[0]
    getitem_179: "f32[768, 768, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    sum_53: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_178, [0, 2, 3])
    sub_133: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_47, unsqueeze_466);  unsqueeze_466 = None
    mul_608: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_178, sub_133)
    sum_54: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_608, [0, 2, 3]);  mul_608 = None
    mul_609: "f32[768]" = torch.ops.aten.mul.Tensor(sum_53, 0.0001220703125)
    unsqueeze_467: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_609, 0);  mul_609 = None
    unsqueeze_468: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 2);  unsqueeze_467 = None
    unsqueeze_469: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, 3);  unsqueeze_468 = None
    mul_610: "f32[768]" = torch.ops.aten.mul.Tensor(sum_54, 0.0001220703125)
    mul_611: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
    mul_612: "f32[768]" = torch.ops.aten.mul.Tensor(mul_610, mul_611);  mul_610 = mul_611 = None
    unsqueeze_470: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_612, 0);  mul_612 = None
    unsqueeze_471: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 2);  unsqueeze_470 = None
    unsqueeze_472: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_471, 3);  unsqueeze_471 = None
    mul_613: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_191);  primals_191 = None
    unsqueeze_473: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_613, 0);  mul_613 = None
    unsqueeze_474: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 2);  unsqueeze_473 = None
    unsqueeze_475: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, 3);  unsqueeze_474 = None
    mul_614: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_472);  sub_133 = unsqueeze_472 = None
    sub_135: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(getitem_178, mul_614);  mul_614 = None
    sub_136: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_135, unsqueeze_469);  sub_135 = unsqueeze_469 = None
    mul_615: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_475);  sub_136 = unsqueeze_475 = None
    mul_616: "f32[768]" = torch.ops.aten.mul.Tensor(sum_54, squeeze_142);  sum_54 = squeeze_142 = None
    le_17: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_47, 0);  relu_47 = None
    where_17: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_17, full_default, mul_615);  le_17 = mul_615 = None
    sum_55: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(where_17, add_257, primals_189, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False]);  where_17 = add_257 = primals_189 = None
    getitem_181: "f32[8, 768, 32, 32]" = convolution_backward_17[0]
    getitem_182: "f32[768, 1, 7, 7]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    add_365: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(getitem_178, getitem_181);  getitem_178 = getitem_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    sum_56: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_365, [0, 2, 3])
    sub_137: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_46, unsqueeze_478);  unsqueeze_478 = None
    mul_617: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(add_365, sub_137)
    sum_57: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_617, [0, 2, 3]);  mul_617 = None
    mul_618: "f32[768]" = torch.ops.aten.mul.Tensor(sum_56, 0.0001220703125)
    unsqueeze_479: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_618, 0);  mul_618 = None
    unsqueeze_480: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 2);  unsqueeze_479 = None
    unsqueeze_481: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, 3);  unsqueeze_480 = None
    mul_619: "f32[768]" = torch.ops.aten.mul.Tensor(sum_57, 0.0001220703125)
    mul_620: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
    mul_621: "f32[768]" = torch.ops.aten.mul.Tensor(mul_619, mul_620);  mul_619 = mul_620 = None
    unsqueeze_482: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_621, 0);  mul_621 = None
    unsqueeze_483: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 2);  unsqueeze_482 = None
    unsqueeze_484: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 3);  unsqueeze_483 = None
    mul_622: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_139, primals_187);  primals_187 = None
    unsqueeze_485: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_622, 0);  mul_622 = None
    unsqueeze_486: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 2);  unsqueeze_485 = None
    unsqueeze_487: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, 3);  unsqueeze_486 = None
    mul_623: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_484);  sub_137 = unsqueeze_484 = None
    sub_139: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(add_365, mul_623);  add_365 = mul_623 = None
    sub_140: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_139, unsqueeze_481);  sub_139 = unsqueeze_481 = None
    mul_624: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_487);  sub_140 = unsqueeze_487 = None
    mul_625: "f32[768]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_139);  sum_57 = squeeze_139 = None
    le_18: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_46, 0);  relu_46 = None
    where_18: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_18, full_default, mul_624);  le_18 = mul_624 = None
    sum_58: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(where_18, add_252, primals_185, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_18 = add_252 = primals_185 = None
    getitem_184: "f32[8, 768, 32, 32]" = convolution_backward_18[0]
    getitem_185: "f32[768, 768, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    sum_59: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_184, [0, 2, 3])
    sub_141: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_45, unsqueeze_490);  unsqueeze_490 = None
    mul_626: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_184, sub_141)
    sum_60: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_626, [0, 2, 3]);  mul_626 = None
    mul_627: "f32[768]" = torch.ops.aten.mul.Tensor(sum_59, 0.0001220703125)
    unsqueeze_491: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_627, 0);  mul_627 = None
    unsqueeze_492: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 2);  unsqueeze_491 = None
    unsqueeze_493: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, 3);  unsqueeze_492 = None
    mul_628: "f32[768]" = torch.ops.aten.mul.Tensor(sum_60, 0.0001220703125)
    mul_629: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
    mul_630: "f32[768]" = torch.ops.aten.mul.Tensor(mul_628, mul_629);  mul_628 = mul_629 = None
    unsqueeze_494: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_630, 0);  mul_630 = None
    unsqueeze_495: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 2);  unsqueeze_494 = None
    unsqueeze_496: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_495, 3);  unsqueeze_495 = None
    mul_631: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_136, primals_183);  primals_183 = None
    unsqueeze_497: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_631, 0);  mul_631 = None
    unsqueeze_498: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 2);  unsqueeze_497 = None
    unsqueeze_499: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, 3);  unsqueeze_498 = None
    mul_632: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_496);  sub_141 = unsqueeze_496 = None
    sub_143: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(getitem_184, mul_632);  mul_632 = None
    sub_144: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_143, unsqueeze_493);  sub_143 = unsqueeze_493 = None
    mul_633: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_499);  sub_144 = unsqueeze_499 = None
    mul_634: "f32[768]" = torch.ops.aten.mul.Tensor(sum_60, squeeze_136);  sum_60 = squeeze_136 = None
    le_19: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_45, 0);  relu_45 = None
    where_19: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_19, full_default, mul_633);  le_19 = mul_633 = None
    sum_61: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(where_19, add_246, primals_181, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False]);  where_19 = add_246 = primals_181 = None
    getitem_187: "f32[8, 768, 32, 32]" = convolution_backward_19[0]
    getitem_188: "f32[768, 1, 7, 7]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    add_366: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(getitem_184, getitem_187);  getitem_184 = getitem_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    sum_62: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_366, [0, 2, 3])
    sub_145: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_44, unsqueeze_502);  unsqueeze_502 = None
    mul_635: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(add_366, sub_145)
    sum_63: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_635, [0, 2, 3]);  mul_635 = None
    mul_636: "f32[768]" = torch.ops.aten.mul.Tensor(sum_62, 0.0001220703125)
    unsqueeze_503: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_636, 0);  mul_636 = None
    unsqueeze_504: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 2);  unsqueeze_503 = None
    unsqueeze_505: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, 3);  unsqueeze_504 = None
    mul_637: "f32[768]" = torch.ops.aten.mul.Tensor(sum_63, 0.0001220703125)
    mul_638: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
    mul_639: "f32[768]" = torch.ops.aten.mul.Tensor(mul_637, mul_638);  mul_637 = mul_638 = None
    unsqueeze_506: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_639, 0);  mul_639 = None
    unsqueeze_507: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 2);  unsqueeze_506 = None
    unsqueeze_508: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_507, 3);  unsqueeze_507 = None
    mul_640: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_133, primals_179);  primals_179 = None
    unsqueeze_509: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_640, 0);  mul_640 = None
    unsqueeze_510: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 2);  unsqueeze_509 = None
    unsqueeze_511: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, 3);  unsqueeze_510 = None
    mul_641: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_508);  sub_145 = unsqueeze_508 = None
    sub_147: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(add_366, mul_641);  add_366 = mul_641 = None
    sub_148: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_147, unsqueeze_505);  sub_147 = unsqueeze_505 = None
    mul_642: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_511);  sub_148 = unsqueeze_511 = None
    mul_643: "f32[768]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_133);  sum_63 = squeeze_133 = None
    le_20: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_44, 0);  relu_44 = None
    where_20: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_20, full_default, mul_642);  le_20 = mul_642 = None
    sum_64: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(where_20, add_241, primals_177, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_20 = add_241 = primals_177 = None
    getitem_190: "f32[8, 768, 32, 32]" = convolution_backward_20[0]
    getitem_191: "f32[768, 768, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    sum_65: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_190, [0, 2, 3])
    sub_149: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_43, unsqueeze_514);  unsqueeze_514 = None
    mul_644: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_190, sub_149)
    sum_66: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_644, [0, 2, 3]);  mul_644 = None
    mul_645: "f32[768]" = torch.ops.aten.mul.Tensor(sum_65, 0.0001220703125)
    unsqueeze_515: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_645, 0);  mul_645 = None
    unsqueeze_516: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 2);  unsqueeze_515 = None
    unsqueeze_517: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, 3);  unsqueeze_516 = None
    mul_646: "f32[768]" = torch.ops.aten.mul.Tensor(sum_66, 0.0001220703125)
    mul_647: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
    mul_648: "f32[768]" = torch.ops.aten.mul.Tensor(mul_646, mul_647);  mul_646 = mul_647 = None
    unsqueeze_518: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_648, 0);  mul_648 = None
    unsqueeze_519: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 2);  unsqueeze_518 = None
    unsqueeze_520: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_519, 3);  unsqueeze_519 = None
    mul_649: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_175);  primals_175 = None
    unsqueeze_521: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_649, 0);  mul_649 = None
    unsqueeze_522: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 2);  unsqueeze_521 = None
    unsqueeze_523: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, 3);  unsqueeze_522 = None
    mul_650: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_520);  sub_149 = unsqueeze_520 = None
    sub_151: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(getitem_190, mul_650);  mul_650 = None
    sub_152: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_151, unsqueeze_517);  sub_151 = unsqueeze_517 = None
    mul_651: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_523);  sub_152 = unsqueeze_523 = None
    mul_652: "f32[768]" = torch.ops.aten.mul.Tensor(sum_66, squeeze_130);  sum_66 = squeeze_130 = None
    le_21: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_43, 0);  relu_43 = None
    where_21: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_21, full_default, mul_651);  le_21 = mul_651 = None
    sum_67: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(where_21, add_235, primals_173, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False]);  where_21 = add_235 = primals_173 = None
    getitem_193: "f32[8, 768, 32, 32]" = convolution_backward_21[0]
    getitem_194: "f32[768, 1, 7, 7]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    add_367: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(getitem_190, getitem_193);  getitem_190 = getitem_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    sum_68: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_367, [0, 2, 3])
    sub_153: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_42, unsqueeze_526);  unsqueeze_526 = None
    mul_653: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(add_367, sub_153)
    sum_69: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_653, [0, 2, 3]);  mul_653 = None
    mul_654: "f32[768]" = torch.ops.aten.mul.Tensor(sum_68, 0.0001220703125)
    unsqueeze_527: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_654, 0);  mul_654 = None
    unsqueeze_528: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 2);  unsqueeze_527 = None
    unsqueeze_529: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, 3);  unsqueeze_528 = None
    mul_655: "f32[768]" = torch.ops.aten.mul.Tensor(sum_69, 0.0001220703125)
    mul_656: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
    mul_657: "f32[768]" = torch.ops.aten.mul.Tensor(mul_655, mul_656);  mul_655 = mul_656 = None
    unsqueeze_530: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_657, 0);  mul_657 = None
    unsqueeze_531: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 2);  unsqueeze_530 = None
    unsqueeze_532: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_531, 3);  unsqueeze_531 = None
    mul_658: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_171);  primals_171 = None
    unsqueeze_533: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_658, 0);  mul_658 = None
    unsqueeze_534: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 2);  unsqueeze_533 = None
    unsqueeze_535: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, 3);  unsqueeze_534 = None
    mul_659: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_532);  sub_153 = unsqueeze_532 = None
    sub_155: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(add_367, mul_659);  add_367 = mul_659 = None
    sub_156: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_155, unsqueeze_529);  sub_155 = unsqueeze_529 = None
    mul_660: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_535);  sub_156 = unsqueeze_535 = None
    mul_661: "f32[768]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_127);  sum_69 = squeeze_127 = None
    le_22: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_42, 0);  relu_42 = None
    where_22: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_22, full_default, mul_660);  le_22 = mul_660 = None
    sum_70: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(where_22, add_230, primals_169, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_22 = add_230 = primals_169 = None
    getitem_196: "f32[8, 768, 32, 32]" = convolution_backward_22[0]
    getitem_197: "f32[768, 768, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    sum_71: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_196, [0, 2, 3])
    sub_157: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_41, unsqueeze_538);  unsqueeze_538 = None
    mul_662: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_196, sub_157)
    sum_72: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_662, [0, 2, 3]);  mul_662 = None
    mul_663: "f32[768]" = torch.ops.aten.mul.Tensor(sum_71, 0.0001220703125)
    unsqueeze_539: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_663, 0);  mul_663 = None
    unsqueeze_540: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 2);  unsqueeze_539 = None
    unsqueeze_541: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, 3);  unsqueeze_540 = None
    mul_664: "f32[768]" = torch.ops.aten.mul.Tensor(sum_72, 0.0001220703125)
    mul_665: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
    mul_666: "f32[768]" = torch.ops.aten.mul.Tensor(mul_664, mul_665);  mul_664 = mul_665 = None
    unsqueeze_542: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_666, 0);  mul_666 = None
    unsqueeze_543: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 2);  unsqueeze_542 = None
    unsqueeze_544: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_543, 3);  unsqueeze_543 = None
    mul_667: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_167);  primals_167 = None
    unsqueeze_545: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_667, 0);  mul_667 = None
    unsqueeze_546: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 2);  unsqueeze_545 = None
    unsqueeze_547: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, 3);  unsqueeze_546 = None
    mul_668: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_544);  sub_157 = unsqueeze_544 = None
    sub_159: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(getitem_196, mul_668);  mul_668 = None
    sub_160: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_159, unsqueeze_541);  sub_159 = unsqueeze_541 = None
    mul_669: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_547);  sub_160 = unsqueeze_547 = None
    mul_670: "f32[768]" = torch.ops.aten.mul.Tensor(sum_72, squeeze_124);  sum_72 = squeeze_124 = None
    le_23: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_41, 0);  relu_41 = None
    where_23: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_23, full_default, mul_669);  le_23 = mul_669 = None
    sum_73: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(where_23, add_224, primals_165, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False]);  where_23 = add_224 = primals_165 = None
    getitem_199: "f32[8, 768, 32, 32]" = convolution_backward_23[0]
    getitem_200: "f32[768, 1, 7, 7]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    add_368: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(getitem_196, getitem_199);  getitem_196 = getitem_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    sum_74: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_368, [0, 2, 3])
    sub_161: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_40, unsqueeze_550);  unsqueeze_550 = None
    mul_671: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(add_368, sub_161)
    sum_75: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_671, [0, 2, 3]);  mul_671 = None
    mul_672: "f32[768]" = torch.ops.aten.mul.Tensor(sum_74, 0.0001220703125)
    unsqueeze_551: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_672, 0);  mul_672 = None
    unsqueeze_552: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 2);  unsqueeze_551 = None
    unsqueeze_553: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, 3);  unsqueeze_552 = None
    mul_673: "f32[768]" = torch.ops.aten.mul.Tensor(sum_75, 0.0001220703125)
    mul_674: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
    mul_675: "f32[768]" = torch.ops.aten.mul.Tensor(mul_673, mul_674);  mul_673 = mul_674 = None
    unsqueeze_554: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_675, 0);  mul_675 = None
    unsqueeze_555: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 2);  unsqueeze_554 = None
    unsqueeze_556: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_555, 3);  unsqueeze_555 = None
    mul_676: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_163);  primals_163 = None
    unsqueeze_557: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_676, 0);  mul_676 = None
    unsqueeze_558: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 2);  unsqueeze_557 = None
    unsqueeze_559: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, 3);  unsqueeze_558 = None
    mul_677: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_161, unsqueeze_556);  sub_161 = unsqueeze_556 = None
    sub_163: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(add_368, mul_677);  add_368 = mul_677 = None
    sub_164: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_163, unsqueeze_553);  sub_163 = unsqueeze_553 = None
    mul_678: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_559);  sub_164 = unsqueeze_559 = None
    mul_679: "f32[768]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_121);  sum_75 = squeeze_121 = None
    le_24: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_40, 0);  relu_40 = None
    where_24: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_24, full_default, mul_678);  le_24 = mul_678 = None
    sum_76: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(where_24, add_219, primals_161, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_24 = add_219 = primals_161 = None
    getitem_202: "f32[8, 768, 32, 32]" = convolution_backward_24[0]
    getitem_203: "f32[768, 768, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    sum_77: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_202, [0, 2, 3])
    sub_165: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_39, unsqueeze_562);  unsqueeze_562 = None
    mul_680: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_202, sub_165)
    sum_78: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_680, [0, 2, 3]);  mul_680 = None
    mul_681: "f32[768]" = torch.ops.aten.mul.Tensor(sum_77, 0.0001220703125)
    unsqueeze_563: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_681, 0);  mul_681 = None
    unsqueeze_564: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 2);  unsqueeze_563 = None
    unsqueeze_565: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, 3);  unsqueeze_564 = None
    mul_682: "f32[768]" = torch.ops.aten.mul.Tensor(sum_78, 0.0001220703125)
    mul_683: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
    mul_684: "f32[768]" = torch.ops.aten.mul.Tensor(mul_682, mul_683);  mul_682 = mul_683 = None
    unsqueeze_566: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_684, 0);  mul_684 = None
    unsqueeze_567: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 2);  unsqueeze_566 = None
    unsqueeze_568: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_567, 3);  unsqueeze_567 = None
    mul_685: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_159);  primals_159 = None
    unsqueeze_569: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_685, 0);  mul_685 = None
    unsqueeze_570: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 2);  unsqueeze_569 = None
    unsqueeze_571: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, 3);  unsqueeze_570 = None
    mul_686: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_568);  sub_165 = unsqueeze_568 = None
    sub_167: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(getitem_202, mul_686);  mul_686 = None
    sub_168: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_167, unsqueeze_565);  sub_167 = unsqueeze_565 = None
    mul_687: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_571);  sub_168 = unsqueeze_571 = None
    mul_688: "f32[768]" = torch.ops.aten.mul.Tensor(sum_78, squeeze_118);  sum_78 = squeeze_118 = None
    le_25: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_39, 0);  relu_39 = None
    where_25: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_25, full_default, mul_687);  le_25 = mul_687 = None
    sum_79: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(where_25, add_213, primals_157, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False]);  where_25 = add_213 = primals_157 = None
    getitem_205: "f32[8, 768, 32, 32]" = convolution_backward_25[0]
    getitem_206: "f32[768, 1, 7, 7]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    add_369: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(getitem_202, getitem_205);  getitem_202 = getitem_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    sum_80: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_369, [0, 2, 3])
    sub_169: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_38, unsqueeze_574);  unsqueeze_574 = None
    mul_689: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(add_369, sub_169)
    sum_81: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_689, [0, 2, 3]);  mul_689 = None
    mul_690: "f32[768]" = torch.ops.aten.mul.Tensor(sum_80, 0.0001220703125)
    unsqueeze_575: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_690, 0);  mul_690 = None
    unsqueeze_576: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 2);  unsqueeze_575 = None
    unsqueeze_577: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, 3);  unsqueeze_576 = None
    mul_691: "f32[768]" = torch.ops.aten.mul.Tensor(sum_81, 0.0001220703125)
    mul_692: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
    mul_693: "f32[768]" = torch.ops.aten.mul.Tensor(mul_691, mul_692);  mul_691 = mul_692 = None
    unsqueeze_578: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_693, 0);  mul_693 = None
    unsqueeze_579: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 2);  unsqueeze_578 = None
    unsqueeze_580: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_579, 3);  unsqueeze_579 = None
    mul_694: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_155);  primals_155 = None
    unsqueeze_581: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_694, 0);  mul_694 = None
    unsqueeze_582: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 2);  unsqueeze_581 = None
    unsqueeze_583: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, 3);  unsqueeze_582 = None
    mul_695: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_580);  sub_169 = unsqueeze_580 = None
    sub_171: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(add_369, mul_695);  add_369 = mul_695 = None
    sub_172: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_171, unsqueeze_577);  sub_171 = unsqueeze_577 = None
    mul_696: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_583);  sub_172 = unsqueeze_583 = None
    mul_697: "f32[768]" = torch.ops.aten.mul.Tensor(sum_81, squeeze_115);  sum_81 = squeeze_115 = None
    le_26: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_38, 0);  relu_38 = None
    where_26: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_26, full_default, mul_696);  le_26 = mul_696 = None
    sum_82: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(where_26, add_208, primals_153, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_26 = add_208 = primals_153 = None
    getitem_208: "f32[8, 768, 32, 32]" = convolution_backward_26[0]
    getitem_209: "f32[768, 768, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    sum_83: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_208, [0, 2, 3])
    sub_173: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_37, unsqueeze_586);  unsqueeze_586 = None
    mul_698: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_208, sub_173)
    sum_84: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_698, [0, 2, 3]);  mul_698 = None
    mul_699: "f32[768]" = torch.ops.aten.mul.Tensor(sum_83, 0.0001220703125)
    unsqueeze_587: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_699, 0);  mul_699 = None
    unsqueeze_588: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 2);  unsqueeze_587 = None
    unsqueeze_589: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, 3);  unsqueeze_588 = None
    mul_700: "f32[768]" = torch.ops.aten.mul.Tensor(sum_84, 0.0001220703125)
    mul_701: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
    mul_702: "f32[768]" = torch.ops.aten.mul.Tensor(mul_700, mul_701);  mul_700 = mul_701 = None
    unsqueeze_590: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_702, 0);  mul_702 = None
    unsqueeze_591: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 2);  unsqueeze_590 = None
    unsqueeze_592: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_591, 3);  unsqueeze_591 = None
    mul_703: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_151);  primals_151 = None
    unsqueeze_593: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_703, 0);  mul_703 = None
    unsqueeze_594: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 2);  unsqueeze_593 = None
    unsqueeze_595: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, 3);  unsqueeze_594 = None
    mul_704: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_592);  sub_173 = unsqueeze_592 = None
    sub_175: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(getitem_208, mul_704);  mul_704 = None
    sub_176: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_175, unsqueeze_589);  sub_175 = unsqueeze_589 = None
    mul_705: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_595);  sub_176 = unsqueeze_595 = None
    mul_706: "f32[768]" = torch.ops.aten.mul.Tensor(sum_84, squeeze_112);  sum_84 = squeeze_112 = None
    le_27: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_37, 0);  relu_37 = None
    where_27: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_27, full_default, mul_705);  le_27 = mul_705 = None
    sum_85: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(where_27, add_202, primals_149, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False]);  where_27 = add_202 = primals_149 = None
    getitem_211: "f32[8, 768, 32, 32]" = convolution_backward_27[0]
    getitem_212: "f32[768, 1, 7, 7]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    add_370: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(getitem_208, getitem_211);  getitem_208 = getitem_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    sum_86: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_370, [0, 2, 3])
    sub_177: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_36, unsqueeze_598);  unsqueeze_598 = None
    mul_707: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(add_370, sub_177)
    sum_87: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_707, [0, 2, 3]);  mul_707 = None
    mul_708: "f32[768]" = torch.ops.aten.mul.Tensor(sum_86, 0.0001220703125)
    unsqueeze_599: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_708, 0);  mul_708 = None
    unsqueeze_600: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 2);  unsqueeze_599 = None
    unsqueeze_601: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, 3);  unsqueeze_600 = None
    mul_709: "f32[768]" = torch.ops.aten.mul.Tensor(sum_87, 0.0001220703125)
    mul_710: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_711: "f32[768]" = torch.ops.aten.mul.Tensor(mul_709, mul_710);  mul_709 = mul_710 = None
    unsqueeze_602: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_711, 0);  mul_711 = None
    unsqueeze_603: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 2);  unsqueeze_602 = None
    unsqueeze_604: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_603, 3);  unsqueeze_603 = None
    mul_712: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_147);  primals_147 = None
    unsqueeze_605: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_712, 0);  mul_712 = None
    unsqueeze_606: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 2);  unsqueeze_605 = None
    unsqueeze_607: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, 3);  unsqueeze_606 = None
    mul_713: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_604);  sub_177 = unsqueeze_604 = None
    sub_179: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(add_370, mul_713);  add_370 = mul_713 = None
    sub_180: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_179, unsqueeze_601);  sub_179 = unsqueeze_601 = None
    mul_714: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_607);  sub_180 = unsqueeze_607 = None
    mul_715: "f32[768]" = torch.ops.aten.mul.Tensor(sum_87, squeeze_109);  sum_87 = squeeze_109 = None
    le_28: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_36, 0);  relu_36 = None
    where_28: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_28, full_default, mul_714);  le_28 = mul_714 = None
    sum_88: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(where_28, add_197, primals_145, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_28 = add_197 = primals_145 = None
    getitem_214: "f32[8, 768, 32, 32]" = convolution_backward_28[0]
    getitem_215: "f32[768, 768, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    sum_89: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_214, [0, 2, 3])
    sub_181: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_35, unsqueeze_610);  unsqueeze_610 = None
    mul_716: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_214, sub_181)
    sum_90: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_716, [0, 2, 3]);  mul_716 = None
    mul_717: "f32[768]" = torch.ops.aten.mul.Tensor(sum_89, 0.0001220703125)
    unsqueeze_611: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_717, 0);  mul_717 = None
    unsqueeze_612: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_611, 2);  unsqueeze_611 = None
    unsqueeze_613: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, 3);  unsqueeze_612 = None
    mul_718: "f32[768]" = torch.ops.aten.mul.Tensor(sum_90, 0.0001220703125)
    mul_719: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
    mul_720: "f32[768]" = torch.ops.aten.mul.Tensor(mul_718, mul_719);  mul_718 = mul_719 = None
    unsqueeze_614: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_720, 0);  mul_720 = None
    unsqueeze_615: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 2);  unsqueeze_614 = None
    unsqueeze_616: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_615, 3);  unsqueeze_615 = None
    mul_721: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_143);  primals_143 = None
    unsqueeze_617: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_721, 0);  mul_721 = None
    unsqueeze_618: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 2);  unsqueeze_617 = None
    unsqueeze_619: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, 3);  unsqueeze_618 = None
    mul_722: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_181, unsqueeze_616);  sub_181 = unsqueeze_616 = None
    sub_183: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(getitem_214, mul_722);  mul_722 = None
    sub_184: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_183, unsqueeze_613);  sub_183 = unsqueeze_613 = None
    mul_723: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_619);  sub_184 = unsqueeze_619 = None
    mul_724: "f32[768]" = torch.ops.aten.mul.Tensor(sum_90, squeeze_106);  sum_90 = squeeze_106 = None
    le_29: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_35, 0);  relu_35 = None
    where_29: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_29, full_default, mul_723);  le_29 = mul_723 = None
    sum_91: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(where_29, add_191, primals_141, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False]);  where_29 = add_191 = primals_141 = None
    getitem_217: "f32[8, 768, 32, 32]" = convolution_backward_29[0]
    getitem_218: "f32[768, 1, 7, 7]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    add_371: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(getitem_214, getitem_217);  getitem_214 = getitem_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    sum_92: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_371, [0, 2, 3])
    sub_185: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_34, unsqueeze_622);  unsqueeze_622 = None
    mul_725: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(add_371, sub_185)
    sum_93: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_725, [0, 2, 3]);  mul_725 = None
    mul_726: "f32[768]" = torch.ops.aten.mul.Tensor(sum_92, 0.0001220703125)
    unsqueeze_623: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_726, 0);  mul_726 = None
    unsqueeze_624: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 2);  unsqueeze_623 = None
    unsqueeze_625: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_624, 3);  unsqueeze_624 = None
    mul_727: "f32[768]" = torch.ops.aten.mul.Tensor(sum_93, 0.0001220703125)
    mul_728: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_729: "f32[768]" = torch.ops.aten.mul.Tensor(mul_727, mul_728);  mul_727 = mul_728 = None
    unsqueeze_626: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_729, 0);  mul_729 = None
    unsqueeze_627: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 2);  unsqueeze_626 = None
    unsqueeze_628: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_627, 3);  unsqueeze_627 = None
    mul_730: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_139);  primals_139 = None
    unsqueeze_629: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_730, 0);  mul_730 = None
    unsqueeze_630: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 2);  unsqueeze_629 = None
    unsqueeze_631: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, 3);  unsqueeze_630 = None
    mul_731: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_628);  sub_185 = unsqueeze_628 = None
    sub_187: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(add_371, mul_731);  add_371 = mul_731 = None
    sub_188: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_187, unsqueeze_625);  sub_187 = unsqueeze_625 = None
    mul_732: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_631);  sub_188 = unsqueeze_631 = None
    mul_733: "f32[768]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_103);  sum_93 = squeeze_103 = None
    le_30: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_34, 0);  relu_34 = None
    where_30: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_30, full_default, mul_732);  le_30 = mul_732 = None
    sum_94: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(where_30, add_186, primals_137, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_30 = add_186 = primals_137 = None
    getitem_220: "f32[8, 768, 32, 32]" = convolution_backward_30[0]
    getitem_221: "f32[768, 768, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    sum_95: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_220, [0, 2, 3])
    sub_189: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_33, unsqueeze_634);  unsqueeze_634 = None
    mul_734: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_220, sub_189)
    sum_96: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_734, [0, 2, 3]);  mul_734 = None
    mul_735: "f32[768]" = torch.ops.aten.mul.Tensor(sum_95, 0.0001220703125)
    unsqueeze_635: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_735, 0);  mul_735 = None
    unsqueeze_636: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 2);  unsqueeze_635 = None
    unsqueeze_637: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, 3);  unsqueeze_636 = None
    mul_736: "f32[768]" = torch.ops.aten.mul.Tensor(sum_96, 0.0001220703125)
    mul_737: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
    mul_738: "f32[768]" = torch.ops.aten.mul.Tensor(mul_736, mul_737);  mul_736 = mul_737 = None
    unsqueeze_638: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_738, 0);  mul_738 = None
    unsqueeze_639: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 2);  unsqueeze_638 = None
    unsqueeze_640: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_639, 3);  unsqueeze_639 = None
    mul_739: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_135);  primals_135 = None
    unsqueeze_641: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_739, 0);  mul_739 = None
    unsqueeze_642: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 2);  unsqueeze_641 = None
    unsqueeze_643: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, 3);  unsqueeze_642 = None
    mul_740: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_189, unsqueeze_640);  sub_189 = unsqueeze_640 = None
    sub_191: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(getitem_220, mul_740);  mul_740 = None
    sub_192: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_191, unsqueeze_637);  sub_191 = unsqueeze_637 = None
    mul_741: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_643);  sub_192 = unsqueeze_643 = None
    mul_742: "f32[768]" = torch.ops.aten.mul.Tensor(sum_96, squeeze_100);  sum_96 = squeeze_100 = None
    le_31: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_33, 0);  relu_33 = None
    where_31: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_31, full_default, mul_741);  le_31 = mul_741 = None
    sum_97: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(where_31, add_180, primals_133, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False]);  where_31 = add_180 = primals_133 = None
    getitem_223: "f32[8, 768, 32, 32]" = convolution_backward_31[0]
    getitem_224: "f32[768, 1, 7, 7]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    add_372: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(getitem_220, getitem_223);  getitem_220 = getitem_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    sum_98: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_372, [0, 2, 3])
    sub_193: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_32, unsqueeze_646);  unsqueeze_646 = None
    mul_743: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(add_372, sub_193)
    sum_99: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_743, [0, 2, 3]);  mul_743 = None
    mul_744: "f32[768]" = torch.ops.aten.mul.Tensor(sum_98, 0.0001220703125)
    unsqueeze_647: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_744, 0);  mul_744 = None
    unsqueeze_648: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 2);  unsqueeze_647 = None
    unsqueeze_649: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, 3);  unsqueeze_648 = None
    mul_745: "f32[768]" = torch.ops.aten.mul.Tensor(sum_99, 0.0001220703125)
    mul_746: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_747: "f32[768]" = torch.ops.aten.mul.Tensor(mul_745, mul_746);  mul_745 = mul_746 = None
    unsqueeze_650: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_747, 0);  mul_747 = None
    unsqueeze_651: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 2);  unsqueeze_650 = None
    unsqueeze_652: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_651, 3);  unsqueeze_651 = None
    mul_748: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_131);  primals_131 = None
    unsqueeze_653: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_748, 0);  mul_748 = None
    unsqueeze_654: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 2);  unsqueeze_653 = None
    unsqueeze_655: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, 3);  unsqueeze_654 = None
    mul_749: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_193, unsqueeze_652);  sub_193 = unsqueeze_652 = None
    sub_195: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(add_372, mul_749);  add_372 = mul_749 = None
    sub_196: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_195, unsqueeze_649);  sub_195 = unsqueeze_649 = None
    mul_750: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_655);  sub_196 = unsqueeze_655 = None
    mul_751: "f32[768]" = torch.ops.aten.mul.Tensor(sum_99, squeeze_97);  sum_99 = squeeze_97 = None
    le_32: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_32, 0);  relu_32 = None
    where_32: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_32, full_default, mul_750);  le_32 = mul_750 = None
    sum_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(where_32, add_175, primals_129, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_32 = add_175 = primals_129 = None
    getitem_226: "f32[8, 768, 32, 32]" = convolution_backward_32[0]
    getitem_227: "f32[768, 768, 1, 1]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    sum_101: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_226, [0, 2, 3])
    sub_197: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_31, unsqueeze_658);  unsqueeze_658 = None
    mul_752: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_226, sub_197)
    sum_102: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_752, [0, 2, 3]);  mul_752 = None
    mul_753: "f32[768]" = torch.ops.aten.mul.Tensor(sum_101, 0.0001220703125)
    unsqueeze_659: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_753, 0);  mul_753 = None
    unsqueeze_660: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_659, 2);  unsqueeze_659 = None
    unsqueeze_661: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_660, 3);  unsqueeze_660 = None
    mul_754: "f32[768]" = torch.ops.aten.mul.Tensor(sum_102, 0.0001220703125)
    mul_755: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_756: "f32[768]" = torch.ops.aten.mul.Tensor(mul_754, mul_755);  mul_754 = mul_755 = None
    unsqueeze_662: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_756, 0);  mul_756 = None
    unsqueeze_663: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 2);  unsqueeze_662 = None
    unsqueeze_664: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_663, 3);  unsqueeze_663 = None
    mul_757: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_127);  primals_127 = None
    unsqueeze_665: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_757, 0);  mul_757 = None
    unsqueeze_666: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 2);  unsqueeze_665 = None
    unsqueeze_667: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, 3);  unsqueeze_666 = None
    mul_758: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_197, unsqueeze_664);  sub_197 = unsqueeze_664 = None
    sub_199: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(getitem_226, mul_758);  mul_758 = None
    sub_200: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_199, unsqueeze_661);  sub_199 = unsqueeze_661 = None
    mul_759: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_667);  sub_200 = unsqueeze_667 = None
    mul_760: "f32[768]" = torch.ops.aten.mul.Tensor(sum_102, squeeze_94);  sum_102 = squeeze_94 = None
    le_33: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_31, 0);  relu_31 = None
    where_33: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_33, full_default, mul_759);  le_33 = mul_759 = None
    sum_103: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(where_33, add_169, primals_125, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False]);  where_33 = add_169 = primals_125 = None
    getitem_229: "f32[8, 768, 32, 32]" = convolution_backward_33[0]
    getitem_230: "f32[768, 1, 7, 7]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    add_373: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(getitem_226, getitem_229);  getitem_226 = getitem_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    sum_104: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_373, [0, 2, 3])
    sub_201: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_30, unsqueeze_670);  unsqueeze_670 = None
    mul_761: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(add_373, sub_201)
    sum_105: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_761, [0, 2, 3]);  mul_761 = None
    mul_762: "f32[768]" = torch.ops.aten.mul.Tensor(sum_104, 0.0001220703125)
    unsqueeze_671: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_762, 0);  mul_762 = None
    unsqueeze_672: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_671, 2);  unsqueeze_671 = None
    unsqueeze_673: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_672, 3);  unsqueeze_672 = None
    mul_763: "f32[768]" = torch.ops.aten.mul.Tensor(sum_105, 0.0001220703125)
    mul_764: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_765: "f32[768]" = torch.ops.aten.mul.Tensor(mul_763, mul_764);  mul_763 = mul_764 = None
    unsqueeze_674: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_765, 0);  mul_765 = None
    unsqueeze_675: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 2);  unsqueeze_674 = None
    unsqueeze_676: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_675, 3);  unsqueeze_675 = None
    mul_766: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_123);  primals_123 = None
    unsqueeze_677: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_766, 0);  mul_766 = None
    unsqueeze_678: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 2);  unsqueeze_677 = None
    unsqueeze_679: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, 3);  unsqueeze_678 = None
    mul_767: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_676);  sub_201 = unsqueeze_676 = None
    sub_203: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(add_373, mul_767);  add_373 = mul_767 = None
    sub_204: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_203, unsqueeze_673);  sub_203 = unsqueeze_673 = None
    mul_768: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_679);  sub_204 = unsqueeze_679 = None
    mul_769: "f32[768]" = torch.ops.aten.mul.Tensor(sum_105, squeeze_91);  sum_105 = squeeze_91 = None
    le_34: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_30, 0);  relu_30 = None
    where_34: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_34, full_default, mul_768);  le_34 = mul_768 = None
    sum_106: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(where_34, add_164, primals_121, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_34 = add_164 = primals_121 = None
    getitem_232: "f32[8, 768, 32, 32]" = convolution_backward_34[0]
    getitem_233: "f32[768, 768, 1, 1]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    sum_107: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_232, [0, 2, 3])
    sub_205: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_29, unsqueeze_682);  unsqueeze_682 = None
    mul_770: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_232, sub_205)
    sum_108: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_770, [0, 2, 3]);  mul_770 = None
    mul_771: "f32[768]" = torch.ops.aten.mul.Tensor(sum_107, 0.0001220703125)
    unsqueeze_683: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_771, 0);  mul_771 = None
    unsqueeze_684: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 2);  unsqueeze_683 = None
    unsqueeze_685: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_684, 3);  unsqueeze_684 = None
    mul_772: "f32[768]" = torch.ops.aten.mul.Tensor(sum_108, 0.0001220703125)
    mul_773: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_774: "f32[768]" = torch.ops.aten.mul.Tensor(mul_772, mul_773);  mul_772 = mul_773 = None
    unsqueeze_686: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_774, 0);  mul_774 = None
    unsqueeze_687: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 2);  unsqueeze_686 = None
    unsqueeze_688: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_687, 3);  unsqueeze_687 = None
    mul_775: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_119);  primals_119 = None
    unsqueeze_689: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_775, 0);  mul_775 = None
    unsqueeze_690: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 2);  unsqueeze_689 = None
    unsqueeze_691: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, 3);  unsqueeze_690 = None
    mul_776: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_688);  sub_205 = unsqueeze_688 = None
    sub_207: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(getitem_232, mul_776);  mul_776 = None
    sub_208: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_207, unsqueeze_685);  sub_207 = unsqueeze_685 = None
    mul_777: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_691);  sub_208 = unsqueeze_691 = None
    mul_778: "f32[768]" = torch.ops.aten.mul.Tensor(sum_108, squeeze_88);  sum_108 = squeeze_88 = None
    le_35: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_29, 0);  relu_29 = None
    where_35: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_35, full_default, mul_777);  le_35 = mul_777 = None
    sum_109: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(where_35, add_158, primals_117, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False]);  where_35 = add_158 = primals_117 = None
    getitem_235: "f32[8, 768, 32, 32]" = convolution_backward_35[0]
    getitem_236: "f32[768, 1, 7, 7]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    add_374: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(getitem_232, getitem_235);  getitem_232 = getitem_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    sum_110: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_374, [0, 2, 3])
    sub_209: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_28, unsqueeze_694);  unsqueeze_694 = None
    mul_779: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(add_374, sub_209)
    sum_111: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_779, [0, 2, 3]);  mul_779 = None
    mul_780: "f32[768]" = torch.ops.aten.mul.Tensor(sum_110, 0.0001220703125)
    unsqueeze_695: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_780, 0);  mul_780 = None
    unsqueeze_696: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 2);  unsqueeze_695 = None
    unsqueeze_697: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_696, 3);  unsqueeze_696 = None
    mul_781: "f32[768]" = torch.ops.aten.mul.Tensor(sum_111, 0.0001220703125)
    mul_782: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_783: "f32[768]" = torch.ops.aten.mul.Tensor(mul_781, mul_782);  mul_781 = mul_782 = None
    unsqueeze_698: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_783, 0);  mul_783 = None
    unsqueeze_699: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 2);  unsqueeze_698 = None
    unsqueeze_700: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_699, 3);  unsqueeze_699 = None
    mul_784: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_115);  primals_115 = None
    unsqueeze_701: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_784, 0);  mul_784 = None
    unsqueeze_702: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 2);  unsqueeze_701 = None
    unsqueeze_703: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, 3);  unsqueeze_702 = None
    mul_785: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_700);  sub_209 = unsqueeze_700 = None
    sub_211: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(add_374, mul_785);  add_374 = mul_785 = None
    sub_212: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_211, unsqueeze_697);  sub_211 = unsqueeze_697 = None
    mul_786: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_703);  sub_212 = unsqueeze_703 = None
    mul_787: "f32[768]" = torch.ops.aten.mul.Tensor(sum_111, squeeze_85);  sum_111 = squeeze_85 = None
    le_36: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_28, 0);  relu_28 = None
    where_36: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_36, full_default, mul_786);  le_36 = mul_786 = None
    sum_112: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(where_36, add_153, primals_113, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_36 = add_153 = primals_113 = None
    getitem_238: "f32[8, 768, 32, 32]" = convolution_backward_36[0]
    getitem_239: "f32[768, 768, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    sum_113: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_238, [0, 2, 3])
    sub_213: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_27, unsqueeze_706);  unsqueeze_706 = None
    mul_788: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_238, sub_213)
    sum_114: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_788, [0, 2, 3]);  mul_788 = None
    mul_789: "f32[768]" = torch.ops.aten.mul.Tensor(sum_113, 0.0001220703125)
    unsqueeze_707: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_789, 0);  mul_789 = None
    unsqueeze_708: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_707, 2);  unsqueeze_707 = None
    unsqueeze_709: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_708, 3);  unsqueeze_708 = None
    mul_790: "f32[768]" = torch.ops.aten.mul.Tensor(sum_114, 0.0001220703125)
    mul_791: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_792: "f32[768]" = torch.ops.aten.mul.Tensor(mul_790, mul_791);  mul_790 = mul_791 = None
    unsqueeze_710: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_792, 0);  mul_792 = None
    unsqueeze_711: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 2);  unsqueeze_710 = None
    unsqueeze_712: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_711, 3);  unsqueeze_711 = None
    mul_793: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_111);  primals_111 = None
    unsqueeze_713: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_793, 0);  mul_793 = None
    unsqueeze_714: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 2);  unsqueeze_713 = None
    unsqueeze_715: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_714, 3);  unsqueeze_714 = None
    mul_794: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_712);  sub_213 = unsqueeze_712 = None
    sub_215: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(getitem_238, mul_794);  mul_794 = None
    sub_216: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_215, unsqueeze_709);  sub_215 = unsqueeze_709 = None
    mul_795: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_715);  sub_216 = unsqueeze_715 = None
    mul_796: "f32[768]" = torch.ops.aten.mul.Tensor(sum_114, squeeze_82);  sum_114 = squeeze_82 = None
    le_37: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_27, 0);  relu_27 = None
    where_37: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_37, full_default, mul_795);  le_37 = mul_795 = None
    sum_115: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(where_37, add_147, primals_109, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False]);  where_37 = add_147 = primals_109 = None
    getitem_241: "f32[8, 768, 32, 32]" = convolution_backward_37[0]
    getitem_242: "f32[768, 1, 7, 7]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    add_375: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(getitem_238, getitem_241);  getitem_238 = getitem_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    sum_116: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_375, [0, 2, 3])
    sub_217: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_26, unsqueeze_718);  unsqueeze_718 = None
    mul_797: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(add_375, sub_217)
    sum_117: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_797, [0, 2, 3]);  mul_797 = None
    mul_798: "f32[768]" = torch.ops.aten.mul.Tensor(sum_116, 0.0001220703125)
    unsqueeze_719: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_798, 0);  mul_798 = None
    unsqueeze_720: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_719, 2);  unsqueeze_719 = None
    unsqueeze_721: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_720, 3);  unsqueeze_720 = None
    mul_799: "f32[768]" = torch.ops.aten.mul.Tensor(sum_117, 0.0001220703125)
    mul_800: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_801: "f32[768]" = torch.ops.aten.mul.Tensor(mul_799, mul_800);  mul_799 = mul_800 = None
    unsqueeze_722: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_801, 0);  mul_801 = None
    unsqueeze_723: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 2);  unsqueeze_722 = None
    unsqueeze_724: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_723, 3);  unsqueeze_723 = None
    mul_802: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_107);  primals_107 = None
    unsqueeze_725: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_802, 0);  mul_802 = None
    unsqueeze_726: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 2);  unsqueeze_725 = None
    unsqueeze_727: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, 3);  unsqueeze_726 = None
    mul_803: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_724);  sub_217 = unsqueeze_724 = None
    sub_219: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(add_375, mul_803);  add_375 = mul_803 = None
    sub_220: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_219, unsqueeze_721);  sub_219 = unsqueeze_721 = None
    mul_804: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_727);  sub_220 = unsqueeze_727 = None
    mul_805: "f32[768]" = torch.ops.aten.mul.Tensor(sum_117, squeeze_79);  sum_117 = squeeze_79 = None
    le_38: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_26, 0);  relu_26 = None
    where_38: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_38, full_default, mul_804);  le_38 = mul_804 = None
    sum_118: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_38, [0, 2, 3])
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(where_38, add_142, primals_105, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_38 = add_142 = primals_105 = None
    getitem_244: "f32[8, 768, 32, 32]" = convolution_backward_38[0]
    getitem_245: "f32[768, 768, 1, 1]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    sum_119: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_244, [0, 2, 3])
    sub_221: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_25, unsqueeze_730);  unsqueeze_730 = None
    mul_806: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_244, sub_221)
    sum_120: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_806, [0, 2, 3]);  mul_806 = None
    mul_807: "f32[768]" = torch.ops.aten.mul.Tensor(sum_119, 0.0001220703125)
    unsqueeze_731: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_807, 0);  mul_807 = None
    unsqueeze_732: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_731, 2);  unsqueeze_731 = None
    unsqueeze_733: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_732, 3);  unsqueeze_732 = None
    mul_808: "f32[768]" = torch.ops.aten.mul.Tensor(sum_120, 0.0001220703125)
    mul_809: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_810: "f32[768]" = torch.ops.aten.mul.Tensor(mul_808, mul_809);  mul_808 = mul_809 = None
    unsqueeze_734: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_810, 0);  mul_810 = None
    unsqueeze_735: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 2);  unsqueeze_734 = None
    unsqueeze_736: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_735, 3);  unsqueeze_735 = None
    mul_811: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_103);  primals_103 = None
    unsqueeze_737: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_811, 0);  mul_811 = None
    unsqueeze_738: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 2);  unsqueeze_737 = None
    unsqueeze_739: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, 3);  unsqueeze_738 = None
    mul_812: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_221, unsqueeze_736);  sub_221 = unsqueeze_736 = None
    sub_223: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(getitem_244, mul_812);  mul_812 = None
    sub_224: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_223, unsqueeze_733);  sub_223 = unsqueeze_733 = None
    mul_813: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_739);  sub_224 = unsqueeze_739 = None
    mul_814: "f32[768]" = torch.ops.aten.mul.Tensor(sum_120, squeeze_76);  sum_120 = squeeze_76 = None
    le_39: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_25, 0);  relu_25 = None
    where_39: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_39, full_default, mul_813);  le_39 = mul_813 = None
    sum_121: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(where_39, add_136, primals_101, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False]);  where_39 = add_136 = primals_101 = None
    getitem_247: "f32[8, 768, 32, 32]" = convolution_backward_39[0]
    getitem_248: "f32[768, 1, 7, 7]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    add_376: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(getitem_244, getitem_247);  getitem_244 = getitem_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    sum_122: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_376, [0, 2, 3])
    sub_225: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_24, unsqueeze_742);  unsqueeze_742 = None
    mul_815: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(add_376, sub_225)
    sum_123: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_815, [0, 2, 3]);  mul_815 = None
    mul_816: "f32[768]" = torch.ops.aten.mul.Tensor(sum_122, 0.0001220703125)
    unsqueeze_743: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_816, 0);  mul_816 = None
    unsqueeze_744: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_743, 2);  unsqueeze_743 = None
    unsqueeze_745: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_744, 3);  unsqueeze_744 = None
    mul_817: "f32[768]" = torch.ops.aten.mul.Tensor(sum_123, 0.0001220703125)
    mul_818: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_819: "f32[768]" = torch.ops.aten.mul.Tensor(mul_817, mul_818);  mul_817 = mul_818 = None
    unsqueeze_746: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_819, 0);  mul_819 = None
    unsqueeze_747: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 2);  unsqueeze_746 = None
    unsqueeze_748: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_747, 3);  unsqueeze_747 = None
    mul_820: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_99);  primals_99 = None
    unsqueeze_749: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_820, 0);  mul_820 = None
    unsqueeze_750: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_749, 2);  unsqueeze_749 = None
    unsqueeze_751: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_750, 3);  unsqueeze_750 = None
    mul_821: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_748);  sub_225 = unsqueeze_748 = None
    sub_227: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(add_376, mul_821);  add_376 = mul_821 = None
    sub_228: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_227, unsqueeze_745);  sub_227 = unsqueeze_745 = None
    mul_822: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_751);  sub_228 = unsqueeze_751 = None
    mul_823: "f32[768]" = torch.ops.aten.mul.Tensor(sum_123, squeeze_73);  sum_123 = squeeze_73 = None
    le_40: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_24, 0);  relu_24 = None
    where_40: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_40, full_default, mul_822);  le_40 = mul_822 = None
    sum_124: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_40, [0, 2, 3])
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(where_40, add_131, primals_97, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_40 = add_131 = primals_97 = None
    getitem_250: "f32[8, 768, 32, 32]" = convolution_backward_40[0]
    getitem_251: "f32[768, 768, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    sum_125: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_250, [0, 2, 3])
    sub_229: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_23, unsqueeze_754);  unsqueeze_754 = None
    mul_824: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_250, sub_229)
    sum_126: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_824, [0, 2, 3]);  mul_824 = None
    mul_825: "f32[768]" = torch.ops.aten.mul.Tensor(sum_125, 0.0001220703125)
    unsqueeze_755: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_825, 0);  mul_825 = None
    unsqueeze_756: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_755, 2);  unsqueeze_755 = None
    unsqueeze_757: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_756, 3);  unsqueeze_756 = None
    mul_826: "f32[768]" = torch.ops.aten.mul.Tensor(sum_126, 0.0001220703125)
    mul_827: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_828: "f32[768]" = torch.ops.aten.mul.Tensor(mul_826, mul_827);  mul_826 = mul_827 = None
    unsqueeze_758: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_828, 0);  mul_828 = None
    unsqueeze_759: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 2);  unsqueeze_758 = None
    unsqueeze_760: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_759, 3);  unsqueeze_759 = None
    mul_829: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_95);  primals_95 = None
    unsqueeze_761: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_829, 0);  mul_829 = None
    unsqueeze_762: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_761, 2);  unsqueeze_761 = None
    unsqueeze_763: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_762, 3);  unsqueeze_762 = None
    mul_830: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_760);  sub_229 = unsqueeze_760 = None
    sub_231: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(getitem_250, mul_830);  mul_830 = None
    sub_232: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_231, unsqueeze_757);  sub_231 = unsqueeze_757 = None
    mul_831: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_763);  sub_232 = unsqueeze_763 = None
    mul_832: "f32[768]" = torch.ops.aten.mul.Tensor(sum_126, squeeze_70);  sum_126 = squeeze_70 = None
    le_41: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_23, 0);  relu_23 = None
    where_41: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_41, full_default, mul_831);  le_41 = mul_831 = None
    sum_127: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_41, [0, 2, 3])
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(where_41, add_125, primals_93, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False]);  where_41 = add_125 = primals_93 = None
    getitem_253: "f32[8, 768, 32, 32]" = convolution_backward_41[0]
    getitem_254: "f32[768, 1, 7, 7]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    add_377: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(getitem_250, getitem_253);  getitem_250 = getitem_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    sum_128: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_377, [0, 2, 3])
    sub_233: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_22, unsqueeze_766);  unsqueeze_766 = None
    mul_833: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(add_377, sub_233)
    sum_129: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_833, [0, 2, 3]);  mul_833 = None
    mul_834: "f32[768]" = torch.ops.aten.mul.Tensor(sum_128, 0.0001220703125)
    unsqueeze_767: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_834, 0);  mul_834 = None
    unsqueeze_768: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_767, 2);  unsqueeze_767 = None
    unsqueeze_769: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_768, 3);  unsqueeze_768 = None
    mul_835: "f32[768]" = torch.ops.aten.mul.Tensor(sum_129, 0.0001220703125)
    mul_836: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_837: "f32[768]" = torch.ops.aten.mul.Tensor(mul_835, mul_836);  mul_835 = mul_836 = None
    unsqueeze_770: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_837, 0);  mul_837 = None
    unsqueeze_771: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 2);  unsqueeze_770 = None
    unsqueeze_772: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_771, 3);  unsqueeze_771 = None
    mul_838: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_91);  primals_91 = None
    unsqueeze_773: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_838, 0);  mul_838 = None
    unsqueeze_774: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 2);  unsqueeze_773 = None
    unsqueeze_775: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_774, 3);  unsqueeze_774 = None
    mul_839: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_233, unsqueeze_772);  sub_233 = unsqueeze_772 = None
    sub_235: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(add_377, mul_839);  add_377 = mul_839 = None
    sub_236: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_235, unsqueeze_769);  sub_235 = unsqueeze_769 = None
    mul_840: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_236, unsqueeze_775);  sub_236 = unsqueeze_775 = None
    mul_841: "f32[768]" = torch.ops.aten.mul.Tensor(sum_129, squeeze_67);  sum_129 = squeeze_67 = None
    le_42: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_22, 0);  relu_22 = None
    where_42: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_42, full_default, mul_840);  le_42 = mul_840 = None
    sum_130: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_42, [0, 2, 3])
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(where_42, add_120, primals_89, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_42 = add_120 = primals_89 = None
    getitem_256: "f32[8, 768, 32, 32]" = convolution_backward_42[0]
    getitem_257: "f32[768, 768, 1, 1]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    sum_131: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_256, [0, 2, 3])
    sub_237: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_21, unsqueeze_778);  unsqueeze_778 = None
    mul_842: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_256, sub_237)
    sum_132: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_842, [0, 2, 3]);  mul_842 = None
    mul_843: "f32[768]" = torch.ops.aten.mul.Tensor(sum_131, 0.0001220703125)
    unsqueeze_779: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_843, 0);  mul_843 = None
    unsqueeze_780: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_779, 2);  unsqueeze_779 = None
    unsqueeze_781: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_780, 3);  unsqueeze_780 = None
    mul_844: "f32[768]" = torch.ops.aten.mul.Tensor(sum_132, 0.0001220703125)
    mul_845: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_846: "f32[768]" = torch.ops.aten.mul.Tensor(mul_844, mul_845);  mul_844 = mul_845 = None
    unsqueeze_782: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_846, 0);  mul_846 = None
    unsqueeze_783: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 2);  unsqueeze_782 = None
    unsqueeze_784: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_783, 3);  unsqueeze_783 = None
    mul_847: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_87);  primals_87 = None
    unsqueeze_785: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_847, 0);  mul_847 = None
    unsqueeze_786: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_785, 2);  unsqueeze_785 = None
    unsqueeze_787: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_786, 3);  unsqueeze_786 = None
    mul_848: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_237, unsqueeze_784);  sub_237 = unsqueeze_784 = None
    sub_239: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(getitem_256, mul_848);  mul_848 = None
    sub_240: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_239, unsqueeze_781);  sub_239 = unsqueeze_781 = None
    mul_849: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_787);  sub_240 = unsqueeze_787 = None
    mul_850: "f32[768]" = torch.ops.aten.mul.Tensor(sum_132, squeeze_64);  sum_132 = squeeze_64 = None
    le_43: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_21, 0);  relu_21 = None
    where_43: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_43, full_default, mul_849);  le_43 = mul_849 = None
    sum_133: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_43, [0, 2, 3])
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(where_43, add_114, primals_85, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False]);  where_43 = add_114 = primals_85 = None
    getitem_259: "f32[8, 768, 32, 32]" = convolution_backward_43[0]
    getitem_260: "f32[768, 1, 7, 7]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    add_378: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(getitem_256, getitem_259);  getitem_256 = getitem_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    sum_134: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_378, [0, 2, 3])
    sub_241: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_20, unsqueeze_790);  unsqueeze_790 = None
    mul_851: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(add_378, sub_241)
    sum_135: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_851, [0, 2, 3]);  mul_851 = None
    mul_852: "f32[768]" = torch.ops.aten.mul.Tensor(sum_134, 0.0001220703125)
    unsqueeze_791: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_852, 0);  mul_852 = None
    unsqueeze_792: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_791, 2);  unsqueeze_791 = None
    unsqueeze_793: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_792, 3);  unsqueeze_792 = None
    mul_853: "f32[768]" = torch.ops.aten.mul.Tensor(sum_135, 0.0001220703125)
    mul_854: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_855: "f32[768]" = torch.ops.aten.mul.Tensor(mul_853, mul_854);  mul_853 = mul_854 = None
    unsqueeze_794: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_855, 0);  mul_855 = None
    unsqueeze_795: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, 2);  unsqueeze_794 = None
    unsqueeze_796: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_795, 3);  unsqueeze_795 = None
    mul_856: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_83);  primals_83 = None
    unsqueeze_797: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_856, 0);  mul_856 = None
    unsqueeze_798: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_797, 2);  unsqueeze_797 = None
    unsqueeze_799: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_798, 3);  unsqueeze_798 = None
    mul_857: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_241, unsqueeze_796);  sub_241 = unsqueeze_796 = None
    sub_243: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(add_378, mul_857);  add_378 = mul_857 = None
    sub_244: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_243, unsqueeze_793);  sub_243 = unsqueeze_793 = None
    mul_858: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_799);  sub_244 = unsqueeze_799 = None
    mul_859: "f32[768]" = torch.ops.aten.mul.Tensor(sum_135, squeeze_61);  sum_135 = squeeze_61 = None
    le_44: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None
    where_44: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_44, full_default, mul_858);  le_44 = mul_858 = None
    sum_136: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_44, [0, 2, 3])
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(where_44, add_109, primals_81, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_44 = add_109 = primals_81 = None
    getitem_262: "f32[8, 768, 32, 32]" = convolution_backward_44[0]
    getitem_263: "f32[768, 768, 1, 1]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    sum_137: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_262, [0, 2, 3])
    sub_245: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_19, unsqueeze_802);  unsqueeze_802 = None
    mul_860: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_262, sub_245)
    sum_138: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_860, [0, 2, 3]);  mul_860 = None
    mul_861: "f32[768]" = torch.ops.aten.mul.Tensor(sum_137, 0.0001220703125)
    unsqueeze_803: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_861, 0);  mul_861 = None
    unsqueeze_804: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_803, 2);  unsqueeze_803 = None
    unsqueeze_805: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_804, 3);  unsqueeze_804 = None
    mul_862: "f32[768]" = torch.ops.aten.mul.Tensor(sum_138, 0.0001220703125)
    mul_863: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_864: "f32[768]" = torch.ops.aten.mul.Tensor(mul_862, mul_863);  mul_862 = mul_863 = None
    unsqueeze_806: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_864, 0);  mul_864 = None
    unsqueeze_807: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, 2);  unsqueeze_806 = None
    unsqueeze_808: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_807, 3);  unsqueeze_807 = None
    mul_865: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_79);  primals_79 = None
    unsqueeze_809: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_865, 0);  mul_865 = None
    unsqueeze_810: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_809, 2);  unsqueeze_809 = None
    unsqueeze_811: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_810, 3);  unsqueeze_810 = None
    mul_866: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_245, unsqueeze_808);  sub_245 = unsqueeze_808 = None
    sub_247: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(getitem_262, mul_866);  mul_866 = None
    sub_248: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_247, unsqueeze_805);  sub_247 = unsqueeze_805 = None
    mul_867: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_811);  sub_248 = unsqueeze_811 = None
    mul_868: "f32[768]" = torch.ops.aten.mul.Tensor(sum_138, squeeze_58);  sum_138 = squeeze_58 = None
    le_45: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_19, 0);  relu_19 = None
    where_45: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_45, full_default, mul_867);  le_45 = mul_867 = None
    sum_139: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_45, [0, 2, 3])
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(where_45, add_103, primals_77, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False]);  where_45 = add_103 = primals_77 = None
    getitem_265: "f32[8, 768, 32, 32]" = convolution_backward_45[0]
    getitem_266: "f32[768, 1, 7, 7]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    add_379: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(getitem_262, getitem_265);  getitem_262 = getitem_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    sum_140: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_379, [0, 2, 3])
    sub_249: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_18, unsqueeze_814);  unsqueeze_814 = None
    mul_869: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(add_379, sub_249)
    sum_141: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_869, [0, 2, 3]);  mul_869 = None
    mul_870: "f32[768]" = torch.ops.aten.mul.Tensor(sum_140, 0.0001220703125)
    unsqueeze_815: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_870, 0);  mul_870 = None
    unsqueeze_816: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_815, 2);  unsqueeze_815 = None
    unsqueeze_817: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_816, 3);  unsqueeze_816 = None
    mul_871: "f32[768]" = torch.ops.aten.mul.Tensor(sum_141, 0.0001220703125)
    mul_872: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_873: "f32[768]" = torch.ops.aten.mul.Tensor(mul_871, mul_872);  mul_871 = mul_872 = None
    unsqueeze_818: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_873, 0);  mul_873 = None
    unsqueeze_819: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, 2);  unsqueeze_818 = None
    unsqueeze_820: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_819, 3);  unsqueeze_819 = None
    mul_874: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_75);  primals_75 = None
    unsqueeze_821: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_874, 0);  mul_874 = None
    unsqueeze_822: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_821, 2);  unsqueeze_821 = None
    unsqueeze_823: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_822, 3);  unsqueeze_822 = None
    mul_875: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_249, unsqueeze_820);  sub_249 = unsqueeze_820 = None
    sub_251: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(add_379, mul_875);  add_379 = mul_875 = None
    sub_252: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_251, unsqueeze_817);  sub_251 = unsqueeze_817 = None
    mul_876: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_823);  sub_252 = unsqueeze_823 = None
    mul_877: "f32[768]" = torch.ops.aten.mul.Tensor(sum_141, squeeze_55);  sum_141 = squeeze_55 = None
    le_46: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None
    where_46: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_46, full_default, mul_876);  le_46 = mul_876 = None
    sum_142: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_46, [0, 2, 3])
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(where_46, add_98, primals_73, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_46 = add_98 = primals_73 = None
    getitem_268: "f32[8, 768, 32, 32]" = convolution_backward_46[0]
    getitem_269: "f32[768, 768, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    sum_143: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_268, [0, 2, 3])
    sub_253: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_17, unsqueeze_826);  unsqueeze_826 = None
    mul_878: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_268, sub_253)
    sum_144: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_878, [0, 2, 3]);  mul_878 = None
    mul_879: "f32[768]" = torch.ops.aten.mul.Tensor(sum_143, 0.0001220703125)
    unsqueeze_827: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_879, 0);  mul_879 = None
    unsqueeze_828: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_827, 2);  unsqueeze_827 = None
    unsqueeze_829: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_828, 3);  unsqueeze_828 = None
    mul_880: "f32[768]" = torch.ops.aten.mul.Tensor(sum_144, 0.0001220703125)
    mul_881: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_882: "f32[768]" = torch.ops.aten.mul.Tensor(mul_880, mul_881);  mul_880 = mul_881 = None
    unsqueeze_830: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_882, 0);  mul_882 = None
    unsqueeze_831: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_830, 2);  unsqueeze_830 = None
    unsqueeze_832: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_831, 3);  unsqueeze_831 = None
    mul_883: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_71);  primals_71 = None
    unsqueeze_833: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_883, 0);  mul_883 = None
    unsqueeze_834: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_833, 2);  unsqueeze_833 = None
    unsqueeze_835: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_834, 3);  unsqueeze_834 = None
    mul_884: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_253, unsqueeze_832);  sub_253 = unsqueeze_832 = None
    sub_255: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(getitem_268, mul_884);  mul_884 = None
    sub_256: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_255, unsqueeze_829);  sub_255 = unsqueeze_829 = None
    mul_885: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_835);  sub_256 = unsqueeze_835 = None
    mul_886: "f32[768]" = torch.ops.aten.mul.Tensor(sum_144, squeeze_52);  sum_144 = squeeze_52 = None
    le_47: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
    where_47: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_47, full_default, mul_885);  le_47 = mul_885 = None
    sum_145: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_47, [0, 2, 3])
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(where_47, add_92, primals_69, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False]);  where_47 = add_92 = primals_69 = None
    getitem_271: "f32[8, 768, 32, 32]" = convolution_backward_47[0]
    getitem_272: "f32[768, 1, 7, 7]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    add_380: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(getitem_268, getitem_271);  getitem_268 = getitem_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    sum_146: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_380, [0, 2, 3])
    sub_257: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_16, unsqueeze_838);  unsqueeze_838 = None
    mul_887: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(add_380, sub_257)
    sum_147: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_887, [0, 2, 3]);  mul_887 = None
    mul_888: "f32[768]" = torch.ops.aten.mul.Tensor(sum_146, 0.0001220703125)
    unsqueeze_839: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_888, 0);  mul_888 = None
    unsqueeze_840: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_839, 2);  unsqueeze_839 = None
    unsqueeze_841: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_840, 3);  unsqueeze_840 = None
    mul_889: "f32[768]" = torch.ops.aten.mul.Tensor(sum_147, 0.0001220703125)
    mul_890: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_891: "f32[768]" = torch.ops.aten.mul.Tensor(mul_889, mul_890);  mul_889 = mul_890 = None
    unsqueeze_842: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_891, 0);  mul_891 = None
    unsqueeze_843: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_842, 2);  unsqueeze_842 = None
    unsqueeze_844: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_843, 3);  unsqueeze_843 = None
    mul_892: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_67);  primals_67 = None
    unsqueeze_845: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_892, 0);  mul_892 = None
    unsqueeze_846: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_845, 2);  unsqueeze_845 = None
    unsqueeze_847: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_846, 3);  unsqueeze_846 = None
    mul_893: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_257, unsqueeze_844);  sub_257 = unsqueeze_844 = None
    sub_259: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(add_380, mul_893);  add_380 = mul_893 = None
    sub_260: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_259, unsqueeze_841);  sub_259 = unsqueeze_841 = None
    mul_894: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_847);  sub_260 = unsqueeze_847 = None
    mul_895: "f32[768]" = torch.ops.aten.mul.Tensor(sum_147, squeeze_49);  sum_147 = squeeze_49 = None
    le_48: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
    where_48: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_48, full_default, mul_894);  le_48 = mul_894 = None
    sum_148: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_48, [0, 2, 3])
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(where_48, add_87, primals_65, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_48 = add_87 = primals_65 = None
    getitem_274: "f32[8, 768, 32, 32]" = convolution_backward_48[0]
    getitem_275: "f32[768, 768, 1, 1]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    sum_149: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_274, [0, 2, 3])
    sub_261: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_15, unsqueeze_850);  unsqueeze_850 = None
    mul_896: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_274, sub_261)
    sum_150: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_896, [0, 2, 3]);  mul_896 = None
    mul_897: "f32[768]" = torch.ops.aten.mul.Tensor(sum_149, 0.0001220703125)
    unsqueeze_851: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_897, 0);  mul_897 = None
    unsqueeze_852: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_851, 2);  unsqueeze_851 = None
    unsqueeze_853: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_852, 3);  unsqueeze_852 = None
    mul_898: "f32[768]" = torch.ops.aten.mul.Tensor(sum_150, 0.0001220703125)
    mul_899: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_900: "f32[768]" = torch.ops.aten.mul.Tensor(mul_898, mul_899);  mul_898 = mul_899 = None
    unsqueeze_854: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_900, 0);  mul_900 = None
    unsqueeze_855: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_854, 2);  unsqueeze_854 = None
    unsqueeze_856: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_855, 3);  unsqueeze_855 = None
    mul_901: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_63);  primals_63 = None
    unsqueeze_857: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_901, 0);  mul_901 = None
    unsqueeze_858: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_857, 2);  unsqueeze_857 = None
    unsqueeze_859: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_858, 3);  unsqueeze_858 = None
    mul_902: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_261, unsqueeze_856);  sub_261 = unsqueeze_856 = None
    sub_263: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(getitem_274, mul_902);  mul_902 = None
    sub_264: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_263, unsqueeze_853);  sub_263 = unsqueeze_853 = None
    mul_903: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_859);  sub_264 = unsqueeze_859 = None
    mul_904: "f32[768]" = torch.ops.aten.mul.Tensor(sum_150, squeeze_46);  sum_150 = squeeze_46 = None
    le_49: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
    where_49: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_49, full_default, mul_903);  le_49 = mul_903 = None
    sum_151: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_49, [0, 2, 3])
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(where_49, add_81, primals_61, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False]);  where_49 = add_81 = primals_61 = None
    getitem_277: "f32[8, 768, 32, 32]" = convolution_backward_49[0]
    getitem_278: "f32[768, 1, 7, 7]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    add_381: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(getitem_274, getitem_277);  getitem_274 = getitem_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    sum_152: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_381, [0, 2, 3])
    sub_265: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_14, unsqueeze_862);  unsqueeze_862 = None
    mul_905: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(add_381, sub_265)
    sum_153: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_905, [0, 2, 3]);  mul_905 = None
    mul_906: "f32[768]" = torch.ops.aten.mul.Tensor(sum_152, 0.0001220703125)
    unsqueeze_863: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_906, 0);  mul_906 = None
    unsqueeze_864: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_863, 2);  unsqueeze_863 = None
    unsqueeze_865: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_864, 3);  unsqueeze_864 = None
    mul_907: "f32[768]" = torch.ops.aten.mul.Tensor(sum_153, 0.0001220703125)
    mul_908: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_909: "f32[768]" = torch.ops.aten.mul.Tensor(mul_907, mul_908);  mul_907 = mul_908 = None
    unsqueeze_866: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_909, 0);  mul_909 = None
    unsqueeze_867: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_866, 2);  unsqueeze_866 = None
    unsqueeze_868: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_867, 3);  unsqueeze_867 = None
    mul_910: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_59);  primals_59 = None
    unsqueeze_869: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_910, 0);  mul_910 = None
    unsqueeze_870: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_869, 2);  unsqueeze_869 = None
    unsqueeze_871: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_870, 3);  unsqueeze_870 = None
    mul_911: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_265, unsqueeze_868);  sub_265 = unsqueeze_868 = None
    sub_267: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(add_381, mul_911);  add_381 = mul_911 = None
    sub_268: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_267, unsqueeze_865);  sub_267 = unsqueeze_865 = None
    mul_912: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_268, unsqueeze_871);  sub_268 = unsqueeze_871 = None
    mul_913: "f32[768]" = torch.ops.aten.mul.Tensor(sum_153, squeeze_43);  sum_153 = squeeze_43 = None
    le_50: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
    where_50: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_50, full_default, mul_912);  le_50 = mul_912 = None
    sum_154: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_50, [0, 2, 3])
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(where_50, add_76, primals_57, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_50 = add_76 = primals_57 = None
    getitem_280: "f32[8, 768, 32, 32]" = convolution_backward_50[0]
    getitem_281: "f32[768, 768, 1, 1]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    sum_155: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_280, [0, 2, 3])
    sub_269: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_13, unsqueeze_874);  unsqueeze_874 = None
    mul_914: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_280, sub_269)
    sum_156: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_914, [0, 2, 3]);  mul_914 = None
    mul_915: "f32[768]" = torch.ops.aten.mul.Tensor(sum_155, 0.0001220703125)
    unsqueeze_875: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_915, 0);  mul_915 = None
    unsqueeze_876: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_875, 2);  unsqueeze_875 = None
    unsqueeze_877: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_876, 3);  unsqueeze_876 = None
    mul_916: "f32[768]" = torch.ops.aten.mul.Tensor(sum_156, 0.0001220703125)
    mul_917: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_918: "f32[768]" = torch.ops.aten.mul.Tensor(mul_916, mul_917);  mul_916 = mul_917 = None
    unsqueeze_878: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_918, 0);  mul_918 = None
    unsqueeze_879: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_878, 2);  unsqueeze_878 = None
    unsqueeze_880: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_879, 3);  unsqueeze_879 = None
    mul_919: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_55);  primals_55 = None
    unsqueeze_881: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_919, 0);  mul_919 = None
    unsqueeze_882: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_881, 2);  unsqueeze_881 = None
    unsqueeze_883: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_882, 3);  unsqueeze_882 = None
    mul_920: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_269, unsqueeze_880);  sub_269 = unsqueeze_880 = None
    sub_271: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(getitem_280, mul_920);  mul_920 = None
    sub_272: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_271, unsqueeze_877);  sub_271 = unsqueeze_877 = None
    mul_921: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_883);  sub_272 = unsqueeze_883 = None
    mul_922: "f32[768]" = torch.ops.aten.mul.Tensor(sum_156, squeeze_40);  sum_156 = squeeze_40 = None
    le_51: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
    where_51: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_51, full_default, mul_921);  le_51 = mul_921 = None
    sum_157: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_51, [0, 2, 3])
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(where_51, add_70, primals_53, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False]);  where_51 = add_70 = primals_53 = None
    getitem_283: "f32[8, 768, 32, 32]" = convolution_backward_51[0]
    getitem_284: "f32[768, 1, 7, 7]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    add_382: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(getitem_280, getitem_283);  getitem_280 = getitem_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    sum_158: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_382, [0, 2, 3])
    sub_273: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_12, unsqueeze_886);  unsqueeze_886 = None
    mul_923: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(add_382, sub_273)
    sum_159: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_923, [0, 2, 3]);  mul_923 = None
    mul_924: "f32[768]" = torch.ops.aten.mul.Tensor(sum_158, 0.0001220703125)
    unsqueeze_887: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_924, 0);  mul_924 = None
    unsqueeze_888: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_887, 2);  unsqueeze_887 = None
    unsqueeze_889: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_888, 3);  unsqueeze_888 = None
    mul_925: "f32[768]" = torch.ops.aten.mul.Tensor(sum_159, 0.0001220703125)
    mul_926: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_927: "f32[768]" = torch.ops.aten.mul.Tensor(mul_925, mul_926);  mul_925 = mul_926 = None
    unsqueeze_890: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_927, 0);  mul_927 = None
    unsqueeze_891: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_890, 2);  unsqueeze_890 = None
    unsqueeze_892: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_891, 3);  unsqueeze_891 = None
    mul_928: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_51);  primals_51 = None
    unsqueeze_893: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_928, 0);  mul_928 = None
    unsqueeze_894: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_893, 2);  unsqueeze_893 = None
    unsqueeze_895: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_894, 3);  unsqueeze_894 = None
    mul_929: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_273, unsqueeze_892);  sub_273 = unsqueeze_892 = None
    sub_275: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(add_382, mul_929);  add_382 = mul_929 = None
    sub_276: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_275, unsqueeze_889);  sub_275 = unsqueeze_889 = None
    mul_930: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_276, unsqueeze_895);  sub_276 = unsqueeze_895 = None
    mul_931: "f32[768]" = torch.ops.aten.mul.Tensor(sum_159, squeeze_37);  sum_159 = squeeze_37 = None
    le_52: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
    where_52: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_52, full_default, mul_930);  le_52 = mul_930 = None
    sum_160: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_52, [0, 2, 3])
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(where_52, add_65, primals_49, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_52 = add_65 = primals_49 = None
    getitem_286: "f32[8, 768, 32, 32]" = convolution_backward_52[0]
    getitem_287: "f32[768, 768, 1, 1]" = convolution_backward_52[1];  convolution_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    sum_161: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_286, [0, 2, 3])
    sub_277: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_11, unsqueeze_898);  unsqueeze_898 = None
    mul_932: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_286, sub_277)
    sum_162: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_932, [0, 2, 3]);  mul_932 = None
    mul_933: "f32[768]" = torch.ops.aten.mul.Tensor(sum_161, 0.0001220703125)
    unsqueeze_899: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_933, 0);  mul_933 = None
    unsqueeze_900: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_899, 2);  unsqueeze_899 = None
    unsqueeze_901: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_900, 3);  unsqueeze_900 = None
    mul_934: "f32[768]" = torch.ops.aten.mul.Tensor(sum_162, 0.0001220703125)
    mul_935: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_936: "f32[768]" = torch.ops.aten.mul.Tensor(mul_934, mul_935);  mul_934 = mul_935 = None
    unsqueeze_902: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_936, 0);  mul_936 = None
    unsqueeze_903: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_902, 2);  unsqueeze_902 = None
    unsqueeze_904: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_903, 3);  unsqueeze_903 = None
    mul_937: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_47);  primals_47 = None
    unsqueeze_905: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_937, 0);  mul_937 = None
    unsqueeze_906: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_905, 2);  unsqueeze_905 = None
    unsqueeze_907: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_906, 3);  unsqueeze_906 = None
    mul_938: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_277, unsqueeze_904);  sub_277 = unsqueeze_904 = None
    sub_279: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(getitem_286, mul_938);  mul_938 = None
    sub_280: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_279, unsqueeze_901);  sub_279 = unsqueeze_901 = None
    mul_939: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_280, unsqueeze_907);  sub_280 = unsqueeze_907 = None
    mul_940: "f32[768]" = torch.ops.aten.mul.Tensor(sum_162, squeeze_34);  sum_162 = squeeze_34 = None
    le_53: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
    where_53: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_53, full_default, mul_939);  le_53 = mul_939 = None
    sum_163: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_53, [0, 2, 3])
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(where_53, add_59, primals_45, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False]);  where_53 = add_59 = primals_45 = None
    getitem_289: "f32[8, 768, 32, 32]" = convolution_backward_53[0]
    getitem_290: "f32[768, 1, 7, 7]" = convolution_backward_53[1];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    add_383: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(getitem_286, getitem_289);  getitem_286 = getitem_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    sum_164: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_383, [0, 2, 3])
    sub_281: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_10, unsqueeze_910);  unsqueeze_910 = None
    mul_941: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(add_383, sub_281)
    sum_165: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_941, [0, 2, 3]);  mul_941 = None
    mul_942: "f32[768]" = torch.ops.aten.mul.Tensor(sum_164, 0.0001220703125)
    unsqueeze_911: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_942, 0);  mul_942 = None
    unsqueeze_912: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_911, 2);  unsqueeze_911 = None
    unsqueeze_913: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_912, 3);  unsqueeze_912 = None
    mul_943: "f32[768]" = torch.ops.aten.mul.Tensor(sum_165, 0.0001220703125)
    mul_944: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_945: "f32[768]" = torch.ops.aten.mul.Tensor(mul_943, mul_944);  mul_943 = mul_944 = None
    unsqueeze_914: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_945, 0);  mul_945 = None
    unsqueeze_915: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_914, 2);  unsqueeze_914 = None
    unsqueeze_916: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_915, 3);  unsqueeze_915 = None
    mul_946: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_43);  primals_43 = None
    unsqueeze_917: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_946, 0);  mul_946 = None
    unsqueeze_918: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_917, 2);  unsqueeze_917 = None
    unsqueeze_919: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_918, 3);  unsqueeze_918 = None
    mul_947: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_281, unsqueeze_916);  sub_281 = unsqueeze_916 = None
    sub_283: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(add_383, mul_947);  add_383 = mul_947 = None
    sub_284: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_283, unsqueeze_913);  sub_283 = unsqueeze_913 = None
    mul_948: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_284, unsqueeze_919);  sub_284 = unsqueeze_919 = None
    mul_949: "f32[768]" = torch.ops.aten.mul.Tensor(sum_165, squeeze_31);  sum_165 = squeeze_31 = None
    le_54: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
    where_54: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_54, full_default, mul_948);  le_54 = mul_948 = None
    sum_166: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_54, [0, 2, 3])
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(where_54, add_54, primals_41, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_54 = add_54 = primals_41 = None
    getitem_292: "f32[8, 768, 32, 32]" = convolution_backward_54[0]
    getitem_293: "f32[768, 768, 1, 1]" = convolution_backward_54[1];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    sum_167: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_292, [0, 2, 3])
    sub_285: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_9, unsqueeze_922);  unsqueeze_922 = None
    mul_950: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_292, sub_285)
    sum_168: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_950, [0, 2, 3]);  mul_950 = None
    mul_951: "f32[768]" = torch.ops.aten.mul.Tensor(sum_167, 0.0001220703125)
    unsqueeze_923: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_951, 0);  mul_951 = None
    unsqueeze_924: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_923, 2);  unsqueeze_923 = None
    unsqueeze_925: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_924, 3);  unsqueeze_924 = None
    mul_952: "f32[768]" = torch.ops.aten.mul.Tensor(sum_168, 0.0001220703125)
    mul_953: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_954: "f32[768]" = torch.ops.aten.mul.Tensor(mul_952, mul_953);  mul_952 = mul_953 = None
    unsqueeze_926: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_954, 0);  mul_954 = None
    unsqueeze_927: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_926, 2);  unsqueeze_926 = None
    unsqueeze_928: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_927, 3);  unsqueeze_927 = None
    mul_955: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_39);  primals_39 = None
    unsqueeze_929: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_955, 0);  mul_955 = None
    unsqueeze_930: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_929, 2);  unsqueeze_929 = None
    unsqueeze_931: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_930, 3);  unsqueeze_930 = None
    mul_956: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_285, unsqueeze_928);  sub_285 = unsqueeze_928 = None
    sub_287: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(getitem_292, mul_956);  mul_956 = None
    sub_288: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_287, unsqueeze_925);  sub_287 = unsqueeze_925 = None
    mul_957: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_288, unsqueeze_931);  sub_288 = unsqueeze_931 = None
    mul_958: "f32[768]" = torch.ops.aten.mul.Tensor(sum_168, squeeze_28);  sum_168 = squeeze_28 = None
    le_55: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
    where_55: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_55, full_default, mul_957);  le_55 = mul_957 = None
    sum_169: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_55, [0, 2, 3])
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(where_55, add_48, primals_37, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False]);  where_55 = add_48 = primals_37 = None
    getitem_295: "f32[8, 768, 32, 32]" = convolution_backward_55[0]
    getitem_296: "f32[768, 1, 7, 7]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    add_384: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(getitem_292, getitem_295);  getitem_292 = getitem_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    sum_170: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_384, [0, 2, 3])
    sub_289: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_8, unsqueeze_934);  unsqueeze_934 = None
    mul_959: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(add_384, sub_289)
    sum_171: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_959, [0, 2, 3]);  mul_959 = None
    mul_960: "f32[768]" = torch.ops.aten.mul.Tensor(sum_170, 0.0001220703125)
    unsqueeze_935: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_960, 0);  mul_960 = None
    unsqueeze_936: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_935, 2);  unsqueeze_935 = None
    unsqueeze_937: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_936, 3);  unsqueeze_936 = None
    mul_961: "f32[768]" = torch.ops.aten.mul.Tensor(sum_171, 0.0001220703125)
    mul_962: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_963: "f32[768]" = torch.ops.aten.mul.Tensor(mul_961, mul_962);  mul_961 = mul_962 = None
    unsqueeze_938: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_963, 0);  mul_963 = None
    unsqueeze_939: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_938, 2);  unsqueeze_938 = None
    unsqueeze_940: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_939, 3);  unsqueeze_939 = None
    mul_964: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_35);  primals_35 = None
    unsqueeze_941: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_964, 0);  mul_964 = None
    unsqueeze_942: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_941, 2);  unsqueeze_941 = None
    unsqueeze_943: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_942, 3);  unsqueeze_942 = None
    mul_965: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_289, unsqueeze_940);  sub_289 = unsqueeze_940 = None
    sub_291: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(add_384, mul_965);  add_384 = mul_965 = None
    sub_292: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_291, unsqueeze_937);  sub_291 = unsqueeze_937 = None
    mul_966: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_292, unsqueeze_943);  sub_292 = unsqueeze_943 = None
    mul_967: "f32[768]" = torch.ops.aten.mul.Tensor(sum_171, squeeze_25);  sum_171 = squeeze_25 = None
    le_56: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
    where_56: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_56, full_default, mul_966);  le_56 = mul_966 = None
    sum_172: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_56, [0, 2, 3])
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(where_56, add_43, primals_33, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_56 = add_43 = primals_33 = None
    getitem_298: "f32[8, 768, 32, 32]" = convolution_backward_56[0]
    getitem_299: "f32[768, 768, 1, 1]" = convolution_backward_56[1];  convolution_backward_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    sum_173: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_298, [0, 2, 3])
    sub_293: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_7, unsqueeze_946);  unsqueeze_946 = None
    mul_968: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_298, sub_293)
    sum_174: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_968, [0, 2, 3]);  mul_968 = None
    mul_969: "f32[768]" = torch.ops.aten.mul.Tensor(sum_173, 0.0001220703125)
    unsqueeze_947: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_969, 0);  mul_969 = None
    unsqueeze_948: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_947, 2);  unsqueeze_947 = None
    unsqueeze_949: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_948, 3);  unsqueeze_948 = None
    mul_970: "f32[768]" = torch.ops.aten.mul.Tensor(sum_174, 0.0001220703125)
    mul_971: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_972: "f32[768]" = torch.ops.aten.mul.Tensor(mul_970, mul_971);  mul_970 = mul_971 = None
    unsqueeze_950: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_972, 0);  mul_972 = None
    unsqueeze_951: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_950, 2);  unsqueeze_950 = None
    unsqueeze_952: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_951, 3);  unsqueeze_951 = None
    mul_973: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_31);  primals_31 = None
    unsqueeze_953: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_973, 0);  mul_973 = None
    unsqueeze_954: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_953, 2);  unsqueeze_953 = None
    unsqueeze_955: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_954, 3);  unsqueeze_954 = None
    mul_974: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_293, unsqueeze_952);  sub_293 = unsqueeze_952 = None
    sub_295: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(getitem_298, mul_974);  mul_974 = None
    sub_296: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_295, unsqueeze_949);  sub_295 = unsqueeze_949 = None
    mul_975: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_296, unsqueeze_955);  sub_296 = unsqueeze_955 = None
    mul_976: "f32[768]" = torch.ops.aten.mul.Tensor(sum_174, squeeze_22);  sum_174 = squeeze_22 = None
    le_57: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
    where_57: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_57, full_default, mul_975);  le_57 = mul_975 = None
    sum_175: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_57, [0, 2, 3])
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(where_57, add_37, primals_29, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False]);  where_57 = add_37 = primals_29 = None
    getitem_301: "f32[8, 768, 32, 32]" = convolution_backward_57[0]
    getitem_302: "f32[768, 1, 7, 7]" = convolution_backward_57[1];  convolution_backward_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    add_385: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(getitem_298, getitem_301);  getitem_298 = getitem_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    sum_176: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_385, [0, 2, 3])
    sub_297: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_6, unsqueeze_958);  unsqueeze_958 = None
    mul_977: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(add_385, sub_297)
    sum_177: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_977, [0, 2, 3]);  mul_977 = None
    mul_978: "f32[768]" = torch.ops.aten.mul.Tensor(sum_176, 0.0001220703125)
    unsqueeze_959: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_978, 0);  mul_978 = None
    unsqueeze_960: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_959, 2);  unsqueeze_959 = None
    unsqueeze_961: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_960, 3);  unsqueeze_960 = None
    mul_979: "f32[768]" = torch.ops.aten.mul.Tensor(sum_177, 0.0001220703125)
    mul_980: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_981: "f32[768]" = torch.ops.aten.mul.Tensor(mul_979, mul_980);  mul_979 = mul_980 = None
    unsqueeze_962: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_981, 0);  mul_981 = None
    unsqueeze_963: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_962, 2);  unsqueeze_962 = None
    unsqueeze_964: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_963, 3);  unsqueeze_963 = None
    mul_982: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_27);  primals_27 = None
    unsqueeze_965: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_982, 0);  mul_982 = None
    unsqueeze_966: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_965, 2);  unsqueeze_965 = None
    unsqueeze_967: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_966, 3);  unsqueeze_966 = None
    mul_983: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_297, unsqueeze_964);  sub_297 = unsqueeze_964 = None
    sub_299: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(add_385, mul_983);  add_385 = mul_983 = None
    sub_300: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_299, unsqueeze_961);  sub_299 = unsqueeze_961 = None
    mul_984: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_300, unsqueeze_967);  sub_300 = unsqueeze_967 = None
    mul_985: "f32[768]" = torch.ops.aten.mul.Tensor(sum_177, squeeze_19);  sum_177 = squeeze_19 = None
    le_58: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
    where_58: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_58, full_default, mul_984);  le_58 = mul_984 = None
    sum_178: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_58, [0, 2, 3])
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(where_58, add_32, primals_25, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_58 = add_32 = primals_25 = None
    getitem_304: "f32[8, 768, 32, 32]" = convolution_backward_58[0]
    getitem_305: "f32[768, 768, 1, 1]" = convolution_backward_58[1];  convolution_backward_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    sum_179: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_304, [0, 2, 3])
    sub_301: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_5, unsqueeze_970);  unsqueeze_970 = None
    mul_986: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_304, sub_301)
    sum_180: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_986, [0, 2, 3]);  mul_986 = None
    mul_987: "f32[768]" = torch.ops.aten.mul.Tensor(sum_179, 0.0001220703125)
    unsqueeze_971: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_987, 0);  mul_987 = None
    unsqueeze_972: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_971, 2);  unsqueeze_971 = None
    unsqueeze_973: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_972, 3);  unsqueeze_972 = None
    mul_988: "f32[768]" = torch.ops.aten.mul.Tensor(sum_180, 0.0001220703125)
    mul_989: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_990: "f32[768]" = torch.ops.aten.mul.Tensor(mul_988, mul_989);  mul_988 = mul_989 = None
    unsqueeze_974: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_990, 0);  mul_990 = None
    unsqueeze_975: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_974, 2);  unsqueeze_974 = None
    unsqueeze_976: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_975, 3);  unsqueeze_975 = None
    mul_991: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_23);  primals_23 = None
    unsqueeze_977: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_991, 0);  mul_991 = None
    unsqueeze_978: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_977, 2);  unsqueeze_977 = None
    unsqueeze_979: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_978, 3);  unsqueeze_978 = None
    mul_992: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_301, unsqueeze_976);  sub_301 = unsqueeze_976 = None
    sub_303: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(getitem_304, mul_992);  mul_992 = None
    sub_304: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_303, unsqueeze_973);  sub_303 = unsqueeze_973 = None
    mul_993: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_304, unsqueeze_979);  sub_304 = unsqueeze_979 = None
    mul_994: "f32[768]" = torch.ops.aten.mul.Tensor(sum_180, squeeze_16);  sum_180 = squeeze_16 = None
    le_59: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
    where_59: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_59, full_default, mul_993);  le_59 = mul_993 = None
    sum_181: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_59, [0, 2, 3])
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(where_59, add_26, primals_21, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False]);  where_59 = add_26 = primals_21 = None
    getitem_307: "f32[8, 768, 32, 32]" = convolution_backward_59[0]
    getitem_308: "f32[768, 1, 7, 7]" = convolution_backward_59[1];  convolution_backward_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    add_386: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(getitem_304, getitem_307);  getitem_304 = getitem_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    sum_182: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_386, [0, 2, 3])
    sub_305: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_4, unsqueeze_982);  unsqueeze_982 = None
    mul_995: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(add_386, sub_305)
    sum_183: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_995, [0, 2, 3]);  mul_995 = None
    mul_996: "f32[768]" = torch.ops.aten.mul.Tensor(sum_182, 0.0001220703125)
    unsqueeze_983: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_996, 0);  mul_996 = None
    unsqueeze_984: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_983, 2);  unsqueeze_983 = None
    unsqueeze_985: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_984, 3);  unsqueeze_984 = None
    mul_997: "f32[768]" = torch.ops.aten.mul.Tensor(sum_183, 0.0001220703125)
    mul_998: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_999: "f32[768]" = torch.ops.aten.mul.Tensor(mul_997, mul_998);  mul_997 = mul_998 = None
    unsqueeze_986: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_999, 0);  mul_999 = None
    unsqueeze_987: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_986, 2);  unsqueeze_986 = None
    unsqueeze_988: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_987, 3);  unsqueeze_987 = None
    mul_1000: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_19);  primals_19 = None
    unsqueeze_989: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1000, 0);  mul_1000 = None
    unsqueeze_990: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_989, 2);  unsqueeze_989 = None
    unsqueeze_991: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_990, 3);  unsqueeze_990 = None
    mul_1001: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_305, unsqueeze_988);  sub_305 = unsqueeze_988 = None
    sub_307: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(add_386, mul_1001);  add_386 = mul_1001 = None
    sub_308: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_307, unsqueeze_985);  sub_307 = unsqueeze_985 = None
    mul_1002: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_308, unsqueeze_991);  sub_308 = unsqueeze_991 = None
    mul_1003: "f32[768]" = torch.ops.aten.mul.Tensor(sum_183, squeeze_13);  sum_183 = squeeze_13 = None
    le_60: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
    where_60: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_60, full_default, mul_1002);  le_60 = mul_1002 = None
    sum_184: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_60, [0, 2, 3])
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(where_60, add_21, primals_17, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_60 = add_21 = primals_17 = None
    getitem_310: "f32[8, 768, 32, 32]" = convolution_backward_60[0]
    getitem_311: "f32[768, 768, 1, 1]" = convolution_backward_60[1];  convolution_backward_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    sum_185: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_310, [0, 2, 3])
    sub_309: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_3, unsqueeze_994);  unsqueeze_994 = None
    mul_1004: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_310, sub_309)
    sum_186: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1004, [0, 2, 3]);  mul_1004 = None
    mul_1005: "f32[768]" = torch.ops.aten.mul.Tensor(sum_185, 0.0001220703125)
    unsqueeze_995: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1005, 0);  mul_1005 = None
    unsqueeze_996: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_995, 2);  unsqueeze_995 = None
    unsqueeze_997: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_996, 3);  unsqueeze_996 = None
    mul_1006: "f32[768]" = torch.ops.aten.mul.Tensor(sum_186, 0.0001220703125)
    mul_1007: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_1008: "f32[768]" = torch.ops.aten.mul.Tensor(mul_1006, mul_1007);  mul_1006 = mul_1007 = None
    unsqueeze_998: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1008, 0);  mul_1008 = None
    unsqueeze_999: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_998, 2);  unsqueeze_998 = None
    unsqueeze_1000: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_999, 3);  unsqueeze_999 = None
    mul_1009: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_15);  primals_15 = None
    unsqueeze_1001: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1009, 0);  mul_1009 = None
    unsqueeze_1002: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1001, 2);  unsqueeze_1001 = None
    unsqueeze_1003: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1002, 3);  unsqueeze_1002 = None
    mul_1010: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_309, unsqueeze_1000);  sub_309 = unsqueeze_1000 = None
    sub_311: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(getitem_310, mul_1010);  mul_1010 = None
    sub_312: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_311, unsqueeze_997);  sub_311 = unsqueeze_997 = None
    mul_1011: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_312, unsqueeze_1003);  sub_312 = unsqueeze_1003 = None
    mul_1012: "f32[768]" = torch.ops.aten.mul.Tensor(sum_186, squeeze_10);  sum_186 = squeeze_10 = None
    le_61: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
    where_61: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_61, full_default, mul_1011);  le_61 = mul_1011 = None
    sum_187: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_61, [0, 2, 3])
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(where_61, add_15, primals_13, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False]);  where_61 = add_15 = primals_13 = None
    getitem_313: "f32[8, 768, 32, 32]" = convolution_backward_61[0]
    getitem_314: "f32[768, 1, 7, 7]" = convolution_backward_61[1];  convolution_backward_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    add_387: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(getitem_310, getitem_313);  getitem_310 = getitem_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    sum_188: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_387, [0, 2, 3])
    sub_313: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_2, unsqueeze_1006);  unsqueeze_1006 = None
    mul_1013: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(add_387, sub_313)
    sum_189: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1013, [0, 2, 3]);  mul_1013 = None
    mul_1014: "f32[768]" = torch.ops.aten.mul.Tensor(sum_188, 0.0001220703125)
    unsqueeze_1007: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1014, 0);  mul_1014 = None
    unsqueeze_1008: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1007, 2);  unsqueeze_1007 = None
    unsqueeze_1009: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1008, 3);  unsqueeze_1008 = None
    mul_1015: "f32[768]" = torch.ops.aten.mul.Tensor(sum_189, 0.0001220703125)
    mul_1016: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_1017: "f32[768]" = torch.ops.aten.mul.Tensor(mul_1015, mul_1016);  mul_1015 = mul_1016 = None
    unsqueeze_1010: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1017, 0);  mul_1017 = None
    unsqueeze_1011: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1010, 2);  unsqueeze_1010 = None
    unsqueeze_1012: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1011, 3);  unsqueeze_1011 = None
    mul_1018: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_11);  primals_11 = None
    unsqueeze_1013: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1018, 0);  mul_1018 = None
    unsqueeze_1014: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1013, 2);  unsqueeze_1013 = None
    unsqueeze_1015: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1014, 3);  unsqueeze_1014 = None
    mul_1019: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_313, unsqueeze_1012);  sub_313 = unsqueeze_1012 = None
    sub_315: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(add_387, mul_1019);  add_387 = mul_1019 = None
    sub_316: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_315, unsqueeze_1009);  sub_315 = unsqueeze_1009 = None
    mul_1020: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_316, unsqueeze_1015);  sub_316 = unsqueeze_1015 = None
    mul_1021: "f32[768]" = torch.ops.aten.mul.Tensor(sum_189, squeeze_7);  sum_189 = squeeze_7 = None
    le_62: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
    where_62: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_62, full_default, mul_1020);  le_62 = mul_1020 = None
    sum_190: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_62, [0, 2, 3])
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(where_62, add_10, primals_9, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_62 = add_10 = primals_9 = None
    getitem_316: "f32[8, 768, 32, 32]" = convolution_backward_62[0]
    getitem_317: "f32[768, 768, 1, 1]" = convolution_backward_62[1];  convolution_backward_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    sum_191: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_316, [0, 2, 3])
    sub_317: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_1, unsqueeze_1018);  unsqueeze_1018 = None
    mul_1022: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_316, sub_317)
    sum_192: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1022, [0, 2, 3]);  mul_1022 = None
    mul_1023: "f32[768]" = torch.ops.aten.mul.Tensor(sum_191, 0.0001220703125)
    unsqueeze_1019: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1023, 0);  mul_1023 = None
    unsqueeze_1020: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1019, 2);  unsqueeze_1019 = None
    unsqueeze_1021: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1020, 3);  unsqueeze_1020 = None
    mul_1024: "f32[768]" = torch.ops.aten.mul.Tensor(sum_192, 0.0001220703125)
    mul_1025: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_1026: "f32[768]" = torch.ops.aten.mul.Tensor(mul_1024, mul_1025);  mul_1024 = mul_1025 = None
    unsqueeze_1022: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1026, 0);  mul_1026 = None
    unsqueeze_1023: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1022, 2);  unsqueeze_1022 = None
    unsqueeze_1024: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1023, 3);  unsqueeze_1023 = None
    mul_1027: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_7);  primals_7 = None
    unsqueeze_1025: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1027, 0);  mul_1027 = None
    unsqueeze_1026: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1025, 2);  unsqueeze_1025 = None
    unsqueeze_1027: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1026, 3);  unsqueeze_1026 = None
    mul_1028: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_317, unsqueeze_1024);  sub_317 = unsqueeze_1024 = None
    sub_319: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(getitem_316, mul_1028);  mul_1028 = None
    sub_320: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_319, unsqueeze_1021);  sub_319 = unsqueeze_1021 = None
    mul_1029: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_320, unsqueeze_1027);  sub_320 = unsqueeze_1027 = None
    mul_1030: "f32[768]" = torch.ops.aten.mul.Tensor(sum_192, squeeze_4);  sum_192 = squeeze_4 = None
    le_63: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
    where_63: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_63, full_default, mul_1029);  le_63 = mul_1029 = None
    sum_193: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_63, [0, 2, 3])
    convolution_backward_63 = torch.ops.aten.convolution_backward.default(where_63, add_4, primals_5, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, False]);  where_63 = add_4 = primals_5 = None
    getitem_319: "f32[8, 768, 32, 32]" = convolution_backward_63[0]
    getitem_320: "f32[768, 1, 7, 7]" = convolution_backward_63[1];  convolution_backward_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    add_388: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(getitem_316, getitem_319);  getitem_316 = getitem_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:85, code: x = self.stem(x)
    sum_194: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_388, [0, 2, 3])
    sub_321: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu, unsqueeze_1030);  unsqueeze_1030 = None
    mul_1031: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(add_388, sub_321)
    sum_195: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1031, [0, 2, 3]);  mul_1031 = None
    mul_1032: "f32[768]" = torch.ops.aten.mul.Tensor(sum_194, 0.0001220703125)
    unsqueeze_1031: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1032, 0);  mul_1032 = None
    unsqueeze_1032: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1031, 2);  unsqueeze_1031 = None
    unsqueeze_1033: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1032, 3);  unsqueeze_1032 = None
    mul_1033: "f32[768]" = torch.ops.aten.mul.Tensor(sum_195, 0.0001220703125)
    mul_1034: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_1035: "f32[768]" = torch.ops.aten.mul.Tensor(mul_1033, mul_1034);  mul_1033 = mul_1034 = None
    unsqueeze_1034: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1035, 0);  mul_1035 = None
    unsqueeze_1035: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1034, 2);  unsqueeze_1034 = None
    unsqueeze_1036: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1035, 3);  unsqueeze_1035 = None
    mul_1036: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_3);  primals_3 = None
    unsqueeze_1037: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1036, 0);  mul_1036 = None
    unsqueeze_1038: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1037, 2);  unsqueeze_1037 = None
    unsqueeze_1039: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1038, 3);  unsqueeze_1038 = None
    mul_1037: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_321, unsqueeze_1036);  sub_321 = unsqueeze_1036 = None
    sub_323: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(add_388, mul_1037);  add_388 = mul_1037 = None
    sub_324: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(sub_323, unsqueeze_1033);  sub_323 = unsqueeze_1033 = None
    mul_1038: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_324, unsqueeze_1039);  sub_324 = unsqueeze_1039 = None
    mul_1039: "f32[768]" = torch.ops.aten.mul.Tensor(sum_195, squeeze_1);  sum_195 = squeeze_1 = None
    le_64: "b8[8, 768, 32, 32]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
    where_64: "f32[8, 768, 32, 32]" = torch.ops.aten.where.self(le_64, full_default, mul_1038);  le_64 = full_default = mul_1038 = None
    sum_196: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_64, [0, 2, 3])
    convolution_backward_64 = torch.ops.aten.convolution_backward.default(where_64, primals_458, primals_1, [768], [7, 7], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  where_64 = primals_458 = primals_1 = None
    getitem_323: "f32[768, 3, 7, 7]" = convolution_backward_64[1];  convolution_backward_64 = None
    return [getitem_323, sum_196, mul_1039, sum_194, getitem_320, sum_193, mul_1030, sum_191, getitem_317, sum_190, mul_1021, sum_188, getitem_314, sum_187, mul_1012, sum_185, getitem_311, sum_184, mul_1003, sum_182, getitem_308, sum_181, mul_994, sum_179, getitem_305, sum_178, mul_985, sum_176, getitem_302, sum_175, mul_976, sum_173, getitem_299, sum_172, mul_967, sum_170, getitem_296, sum_169, mul_958, sum_167, getitem_293, sum_166, mul_949, sum_164, getitem_290, sum_163, mul_940, sum_161, getitem_287, sum_160, mul_931, sum_158, getitem_284, sum_157, mul_922, sum_155, getitem_281, sum_154, mul_913, sum_152, getitem_278, sum_151, mul_904, sum_149, getitem_275, sum_148, mul_895, sum_146, getitem_272, sum_145, mul_886, sum_143, getitem_269, sum_142, mul_877, sum_140, getitem_266, sum_139, mul_868, sum_137, getitem_263, sum_136, mul_859, sum_134, getitem_260, sum_133, mul_850, sum_131, getitem_257, sum_130, mul_841, sum_128, getitem_254, sum_127, mul_832, sum_125, getitem_251, sum_124, mul_823, sum_122, getitem_248, sum_121, mul_814, sum_119, getitem_245, sum_118, mul_805, sum_116, getitem_242, sum_115, mul_796, sum_113, getitem_239, sum_112, mul_787, sum_110, getitem_236, sum_109, mul_778, sum_107, getitem_233, sum_106, mul_769, sum_104, getitem_230, sum_103, mul_760, sum_101, getitem_227, sum_100, mul_751, sum_98, getitem_224, sum_97, mul_742, sum_95, getitem_221, sum_94, mul_733, sum_92, getitem_218, sum_91, mul_724, sum_89, getitem_215, sum_88, mul_715, sum_86, getitem_212, sum_85, mul_706, sum_83, getitem_209, sum_82, mul_697, sum_80, getitem_206, sum_79, mul_688, sum_77, getitem_203, sum_76, mul_679, sum_74, getitem_200, sum_73, mul_670, sum_71, getitem_197, sum_70, mul_661, sum_68, getitem_194, sum_67, mul_652, sum_65, getitem_191, sum_64, mul_643, sum_62, getitem_188, sum_61, mul_634, sum_59, getitem_185, sum_58, mul_625, sum_56, getitem_182, sum_55, mul_616, sum_53, getitem_179, sum_52, mul_607, sum_50, getitem_176, sum_49, mul_598, sum_47, getitem_173, sum_46, mul_589, sum_44, getitem_170, sum_43, mul_580, sum_41, getitem_167, sum_40, mul_571, sum_38, getitem_164, sum_37, mul_562, sum_35, getitem_161, sum_34, mul_553, sum_32, getitem_158, sum_31, mul_544, sum_29, getitem_155, sum_28, mul_535, sum_26, getitem_152, sum_25, mul_526, sum_23, getitem_149, sum_22, mul_517, sum_20, getitem_146, sum_19, mul_508, sum_17, getitem_143, sum_16, mul_499, sum_14, getitem_140, sum_13, mul_490, sum_11, getitem_137, sum_10, mul_481, sum_8, getitem_134, sum_7, mul_472, sum_5, getitem_131, sum_4, mul_463, sum_2, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    