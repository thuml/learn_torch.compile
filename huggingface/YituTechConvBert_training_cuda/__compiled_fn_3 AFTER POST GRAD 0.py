from __future__ import annotations



def forward(self, primals_1: "f32[384, 1]", primals_2: "f32[384, 1]", primals_3: "f32[384, 1]", primals_4: "f32[384, 1]", primals_5: "f32[384, 1]", primals_6: "f32[384, 1]", primals_7: "f32[384, 1]", primals_8: "f32[384, 1]", primals_9: "f32[384, 1]", primals_10: "f32[384, 1]", primals_11: "f32[384, 1]", primals_12: "f32[384, 1]", primals_13: "f32[30522, 768]", primals_14: "f32[512, 768]", primals_15: "f32[2, 768]", primals_16: "f32[768]", primals_17: "f32[768]", primals_18: "f32[384, 768]", primals_19: "f32[384]", primals_20: "f32[384, 768]", primals_21: "f32[384]", primals_22: "f32[384, 768]", primals_23: "f32[384]", primals_24: "f32[768, 1, 9]", primals_25: "f32[384, 768, 1]", primals_26: "f32[54, 384]", primals_27: "f32[54]", primals_28: "f32[384, 768]", primals_29: "f32[384]", primals_30: "f32[768, 768]", primals_31: "f32[768]", primals_32: "f32[768]", primals_33: "f32[768]", primals_34: "f32[3072, 768]", primals_35: "f32[3072]", primals_36: "f32[768, 3072]", primals_37: "f32[768]", primals_38: "f32[768]", primals_39: "f32[768]", primals_40: "f32[384, 768]", primals_41: "f32[384]", primals_42: "f32[384, 768]", primals_43: "f32[384]", primals_44: "f32[384, 768]", primals_45: "f32[384]", primals_46: "f32[768, 1, 9]", primals_47: "f32[384, 768, 1]", primals_48: "f32[54, 384]", primals_49: "f32[54]", primals_50: "f32[384, 768]", primals_51: "f32[384]", primals_52: "f32[768, 768]", primals_53: "f32[768]", primals_54: "f32[768]", primals_55: "f32[768]", primals_56: "f32[3072, 768]", primals_57: "f32[3072]", primals_58: "f32[768, 3072]", primals_59: "f32[768]", primals_60: "f32[768]", primals_61: "f32[768]", primals_62: "f32[384, 768]", primals_63: "f32[384]", primals_64: "f32[384, 768]", primals_65: "f32[384]", primals_66: "f32[384, 768]", primals_67: "f32[384]", primals_68: "f32[768, 1, 9]", primals_69: "f32[384, 768, 1]", primals_70: "f32[54, 384]", primals_71: "f32[54]", primals_72: "f32[384, 768]", primals_73: "f32[384]", primals_74: "f32[768, 768]", primals_75: "f32[768]", primals_76: "f32[768]", primals_77: "f32[768]", primals_78: "f32[3072, 768]", primals_79: "f32[3072]", primals_80: "f32[768, 3072]", primals_81: "f32[768]", primals_82: "f32[768]", primals_83: "f32[768]", primals_84: "f32[384, 768]", primals_85: "f32[384]", primals_86: "f32[384, 768]", primals_87: "f32[384]", primals_88: "f32[384, 768]", primals_89: "f32[384]", primals_90: "f32[768, 1, 9]", primals_91: "f32[384, 768, 1]", primals_92: "f32[54, 384]", primals_93: "f32[54]", primals_94: "f32[384, 768]", primals_95: "f32[384]", primals_96: "f32[768, 768]", primals_97: "f32[768]", primals_98: "f32[768]", primals_99: "f32[768]", primals_100: "f32[3072, 768]", primals_101: "f32[3072]", primals_102: "f32[768, 3072]", primals_103: "f32[768]", primals_104: "f32[768]", primals_105: "f32[768]", primals_106: "f32[384, 768]", primals_107: "f32[384]", primals_108: "f32[384, 768]", primals_109: "f32[384]", primals_110: "f32[384, 768]", primals_111: "f32[384]", primals_112: "f32[768, 1, 9]", primals_113: "f32[384, 768, 1]", primals_114: "f32[54, 384]", primals_115: "f32[54]", primals_116: "f32[384, 768]", primals_117: "f32[384]", primals_118: "f32[768, 768]", primals_119: "f32[768]", primals_120: "f32[768]", primals_121: "f32[768]", primals_122: "f32[3072, 768]", primals_123: "f32[3072]", primals_124: "f32[768, 3072]", primals_125: "f32[768]", primals_126: "f32[768]", primals_127: "f32[768]", primals_128: "f32[384, 768]", primals_129: "f32[384]", primals_130: "f32[384, 768]", primals_131: "f32[384]", primals_132: "f32[384, 768]", primals_133: "f32[384]", primals_134: "f32[768, 1, 9]", primals_135: "f32[384, 768, 1]", primals_136: "f32[54, 384]", primals_137: "f32[54]", primals_138: "f32[384, 768]", primals_139: "f32[384]", primals_140: "f32[768, 768]", primals_141: "f32[768]", primals_142: "f32[768]", primals_143: "f32[768]", primals_144: "f32[3072, 768]", primals_145: "f32[3072]", primals_146: "f32[768, 3072]", primals_147: "f32[768]", primals_148: "f32[768]", primals_149: "f32[768]", primals_150: "f32[384, 768]", primals_151: "f32[384]", primals_152: "f32[384, 768]", primals_153: "f32[384]", primals_154: "f32[384, 768]", primals_155: "f32[384]", primals_156: "f32[768, 1, 9]", primals_157: "f32[384, 768, 1]", primals_158: "f32[54, 384]", primals_159: "f32[54]", primals_160: "f32[384, 768]", primals_161: "f32[384]", primals_162: "f32[768, 768]", primals_163: "f32[768]", primals_164: "f32[768]", primals_165: "f32[768]", primals_166: "f32[3072, 768]", primals_167: "f32[3072]", primals_168: "f32[768, 3072]", primals_169: "f32[768]", primals_170: "f32[768]", primals_171: "f32[768]", primals_172: "f32[384, 768]", primals_173: "f32[384]", primals_174: "f32[384, 768]", primals_175: "f32[384]", primals_176: "f32[384, 768]", primals_177: "f32[384]", primals_178: "f32[768, 1, 9]", primals_179: "f32[384, 768, 1]", primals_180: "f32[54, 384]", primals_181: "f32[54]", primals_182: "f32[384, 768]", primals_183: "f32[384]", primals_184: "f32[768, 768]", primals_185: "f32[768]", primals_186: "f32[768]", primals_187: "f32[768]", primals_188: "f32[3072, 768]", primals_189: "f32[3072]", primals_190: "f32[768, 3072]", primals_191: "f32[768]", primals_192: "f32[768]", primals_193: "f32[768]", primals_194: "f32[384, 768]", primals_195: "f32[384]", primals_196: "f32[384, 768]", primals_197: "f32[384]", primals_198: "f32[384, 768]", primals_199: "f32[384]", primals_200: "f32[768, 1, 9]", primals_201: "f32[384, 768, 1]", primals_202: "f32[54, 384]", primals_203: "f32[54]", primals_204: "f32[384, 768]", primals_205: "f32[384]", primals_206: "f32[768, 768]", primals_207: "f32[768]", primals_208: "f32[768]", primals_209: "f32[768]", primals_210: "f32[3072, 768]", primals_211: "f32[3072]", primals_212: "f32[768, 3072]", primals_213: "f32[768]", primals_214: "f32[768]", primals_215: "f32[768]", primals_216: "f32[384, 768]", primals_217: "f32[384]", primals_218: "f32[384, 768]", primals_219: "f32[384]", primals_220: "f32[384, 768]", primals_221: "f32[384]", primals_222: "f32[768, 1, 9]", primals_223: "f32[384, 768, 1]", primals_224: "f32[54, 384]", primals_225: "f32[54]", primals_226: "f32[384, 768]", primals_227: "f32[384]", primals_228: "f32[768, 768]", primals_229: "f32[768]", primals_230: "f32[768]", primals_231: "f32[768]", primals_232: "f32[3072, 768]", primals_233: "f32[3072]", primals_234: "f32[768, 3072]", primals_235: "f32[768]", primals_236: "f32[768]", primals_237: "f32[768]", primals_238: "f32[384, 768]", primals_239: "f32[384]", primals_240: "f32[384, 768]", primals_241: "f32[384]", primals_242: "f32[384, 768]", primals_243: "f32[384]", primals_244: "f32[768, 1, 9]", primals_245: "f32[384, 768, 1]", primals_246: "f32[54, 384]", primals_247: "f32[54]", primals_248: "f32[384, 768]", primals_249: "f32[384]", primals_250: "f32[768, 768]", primals_251: "f32[768]", primals_252: "f32[768]", primals_253: "f32[768]", primals_254: "f32[3072, 768]", primals_255: "f32[3072]", primals_256: "f32[768, 3072]", primals_257: "f32[768]", primals_258: "f32[768]", primals_259: "f32[768]", primals_260: "f32[384, 768]", primals_261: "f32[384]", primals_262: "f32[384, 768]", primals_263: "f32[384]", primals_264: "f32[384, 768]", primals_265: "f32[384]", primals_266: "f32[768, 1, 9]", primals_267: "f32[384, 768, 1]", primals_268: "f32[54, 384]", primals_269: "f32[54]", primals_270: "f32[384, 768]", primals_271: "f32[384]", primals_272: "f32[768, 768]", primals_273: "f32[768]", primals_274: "f32[768]", primals_275: "f32[768]", primals_276: "f32[3072, 768]", primals_277: "f32[3072]", primals_278: "f32[768, 3072]", primals_279: "f32[768]", primals_280: "f32[768]", primals_281: "f32[768]", primals_282: "f32[768, 768]", primals_283: "f32[768]", primals_284: "f32[768]", primals_285: "f32[768]", primals_286: "f32[30522, 768]", primals_287: "f32[30522]", primals_288: "i64[1, 512]", primals_289: "i64[1, 512]", primals_290: "i64[1, 512]", primals_291: "i64[1, 512]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:835, code: buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
    slice_1: "i64[1, 512]" = torch.ops.aten.slice.Tensor(primals_288, 0, 0, 9223372036854775807);  primals_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:836, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
    expand: "i64[1, 512]" = torch.ops.aten.expand.default(slice_1, [1, 512]);  slice_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:216, code: position_ids = self.position_ids[:, :seq_length]
    slice_4: "i64[1, 512]" = torch.ops.aten.slice.Tensor(primals_289, 0, 0, 9223372036854775807);  primals_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:230, code: inputs_embeds = self.word_embeddings(input_ids)
    embedding: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_13, primals_290, 0);  primals_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:231, code: position_embeddings = self.position_embeddings(position_ids)
    embedding_1: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_14, slice_4);  primals_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:232, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    embedding_2: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_15, expand);  primals_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:234, code: embeddings = inputs_embeds + position_embeddings + token_type_embeddings
    add: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
    add_1: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:235, code: embeddings = self.LayerNorm(embeddings)
    var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 512, 1]" = var_mean[0]
    getitem_1: "f32[1, 512, 1]" = var_mean[1];  var_mean = None
    add_2: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
    rsqrt: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
    sub_1: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
    mul_1: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = None
    mul_2: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_1, primals_16)
    add_3: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_2, primals_17);  mul_2 = primals_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:236, code: embeddings = self.dropout(embeddings)
    native_dropout = torch.ops.aten.native_dropout.default(add_3, 0.1, True);  add_3 = None
    getitem_2: "f32[1, 512, 768]" = native_dropout[0]
    getitem_3: "b8[1, 512, 768]" = native_dropout[1];  native_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view: "f32[512, 768]" = torch.ops.aten.reshape.default(getitem_2, [512, 768])
    permute: "f32[768, 384]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
    addmm: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_19, view, permute);  primals_19 = None
    view_1: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(addmm, [1, 512, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    permute_1: "f32[768, 384]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
    
    # No stacktrace found for following nodes
    mm_default_35: "f32[512, 384]" = torch.ops.aten.mm.default(view, permute_1)
    add_tensor_35: "f32[512, 384]" = torch.ops.aten.add.Tensor(mm_default_35, primals_21);  mm_default_35 = primals_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_3: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(add_tensor_35, [1, 512, 384]);  add_tensor_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    permute_2: "f32[768, 384]" = torch.ops.aten.permute.default(primals_22, [1, 0]);  primals_22 = None
    
    # No stacktrace found for following nodes
    mm_default_34: "f32[512, 384]" = torch.ops.aten.mm.default(view, permute_2)
    add_tensor_34: "f32[512, 384]" = torch.ops.aten.add.Tensor(mm_default_34, primals_23);  mm_default_34 = primals_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_5: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(add_tensor_34, [1, 512, 384]);  add_tensor_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_3: "f32[1, 768, 512]" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution: "f32[1, 768, 512]" = torch.ops.aten.convolution.default(permute_3, primals_24, None, [1], [4], [1], False, [0], 768)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_1: "f32[1, 384, 512]" = torch.ops.aten.convolution.default(convolution, primals_25, None, [1], [0], [1], False, [0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_4: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_1, primals_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_6: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_1, [1, 512, 6, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_5: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_7: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_3, [1, 512, 6, 64]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_6: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_8: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_5, [1, 512, 6, 64]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_7: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_8: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_4, [0, 2, 1]);  add_4 = None
    mul_3: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(permute_8, view_1);  permute_8 = view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    permute_9: "f32[384, 54]" = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
    view_9: "f32[512, 384]" = torch.ops.aten.reshape.default(mul_3, [512, 384]);  mul_3 = None
    mm: "f32[512, 54]" = torch.ops.aten.mm.default(view_9, permute_9)
    view_10: "f32[1, 512, 54]" = torch.ops.aten.reshape.default(mm, [1, 512, 54]);  mm = None
    add_5: "f32[1, 512, 54]" = torch.ops.aten.add.Tensor(view_10, primals_27);  view_10 = primals_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_11: "f32[3072, 9, 1]" = torch.ops.aten.reshape.default(add_5, [-1, 9, 1]);  add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    amax: "f32[3072, 1, 1]" = torch.ops.aten.amax.default(view_11, [1], True)
    sub_2: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(view_11, amax);  view_11 = amax = None
    exp: "f32[3072, 9, 1]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
    sum_1: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True)
    div: "f32[3072, 9, 1]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(div)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    permute_10: "f32[768, 384]" = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
    
    # No stacktrace found for following nodes
    mm_default_33: "f32[512, 384]" = torch.ops.aten.mm.default(view, permute_10)
    add_tensor_33: "f32[512, 384]" = torch.ops.aten.add.Tensor(mm_default_33, primals_29);  mm_default_33 = primals_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_13: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(add_tensor_33, [1, 512, 384]);  add_tensor_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_14: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(view_13, [1, -1, 384]);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    permute_11: "f32[1, 384, 512]" = torch.ops.aten.permute.default(view_14, [0, 2, 1]);  view_14 = None
    clone: "f32[1, 384, 512]" = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None
    unsqueeze_2: "f32[1, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone, -1);  clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    iota: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_3: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
    iota_1: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_4: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
    add_6: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_3, unsqueeze_4);  unsqueeze_3 = unsqueeze_4 = None
    full_default_1: "i64[1, 1]" = torch.ops.aten.full.default([1, 1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    constant_pad_nd: "f32[1, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_2, [0, 0, 4, 4], 0.0);  unsqueeze_2 = None
    unsqueeze_7: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_6, -1);  add_6 = None
    unsqueeze_8: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, -1);  unsqueeze_7 = None
    index: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(constant_pad_nd, [None, None, unsqueeze_8, full_default_1]);  constant_pad_nd = None
    permute_12: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index, [0, 1, 2, 4, 3, 5]);  index = None
    view_15: "f32[1, 3456, 512]" = torch.ops.aten.reshape.default(permute_12, [1, 3456, 512]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    permute_13: "f32[1, 512, 3456]" = torch.ops.aten.permute.default(view_15, [0, 2, 1]);  view_15 = None
    view_16: "f32[1, 512, 384, 9]" = torch.ops.aten.reshape.default(permute_13, [1, 512, 384, 9]);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    clone_1: "f32[1, 512, 384, 9]" = torch.ops.aten.clone.default(view_16, memory_format = torch.contiguous_format);  view_16 = None
    view_17: "f32[3072, 64, 9]" = torch.ops.aten.reshape.default(clone_1, [3072, 64, 9]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    expand_1: "f32[3072, 64, 9]" = torch.ops.aten.expand.default(view_17, [3072, 64, 9]);  view_17 = None
    expand_2: "f32[3072, 9, 1]" = torch.ops.aten.expand.default(div, [3072, 9, 1]);  div = None
    bmm: "f32[3072, 64, 1]" = torch.ops.aten.bmm.default(expand_1, expand_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    view_21: "f32[512, 384]" = torch.ops.aten.reshape.default(bmm, [-1, 384]);  bmm = None
    
    # No stacktrace found for following nodes
    clone_default_33: "f32[1, 6, 512, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    clone_default_34: "f32[1, 6, 512, 64]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
    clone_default_35: "f32[1, 6, 512, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    _scaled_dot_product_efficient_attention_default_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_33, clone_default_34, clone_default_35, None, True, 0.1, scale = 0.125)
    getitem_275: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_default_11[0]
    getitem_276: "f32[1, 6, 512]" = _scaled_dot_product_efficient_attention_default_11[1]
    getitem_277: "i64[]" = _scaled_dot_product_efficient_attention_default_11[2]
    getitem_278: "i64[]" = _scaled_dot_product_efficient_attention_default_11[3];  _scaled_dot_product_efficient_attention_default_11 = None
    alias_default_22: "f32[1, 6, 512, 64]" = torch.ops.aten.alias.default(getitem_275)
    alias_default_23: "f32[1, 6, 512, 64]" = torch.ops.aten.alias.default(alias_default_22);  alias_default_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_15: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_275, [0, 2, 1, 3]);  getitem_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_28: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_21, [1, -1, 6, 64]);  view_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    cat: "f32[1, 512, 12, 64]" = torch.ops.aten.cat.default([permute_15, view_28], 2);  permute_15 = view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_29: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(cat, [1, 512, 768]);  cat = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_30: "f32[512, 768]" = torch.ops.aten.reshape.default(view_29, [512, 768]);  view_29 = None
    permute_16: "f32[768, 768]" = torch.ops.aten.permute.default(primals_30, [1, 0]);  primals_30 = None
    addmm_4: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_31, view_30, permute_16);  primals_31 = None
    view_31: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_4, [1, 512, 768]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    native_dropout_2 = torch.ops.aten.native_dropout.default(view_31, 0.1, True);  view_31 = None
    getitem_6: "f32[1, 512, 768]" = native_dropout_2[0]
    getitem_7: "b8[1, 512, 768]" = native_dropout_2[1];  native_dropout_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_9: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_6, getitem_2);  getitem_6 = getitem_2 = None
    var_mean_1 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 512, 1]" = var_mean_1[0]
    getitem_9: "f32[1, 512, 1]" = var_mean_1[1];  var_mean_1 = None
    add_10: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
    rsqrt_1: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
    sub_4: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_9, getitem_9);  add_9 = getitem_9 = None
    mul_4: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_1);  sub_4 = None
    mul_5: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_4, primals_32)
    add_11: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_5, primals_33);  mul_5 = primals_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_32: "f32[512, 768]" = torch.ops.aten.reshape.default(add_11, [512, 768])
    permute_17: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
    addmm_5: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_35, view_32, permute_17);  primals_35 = None
    view_33: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_5, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_6: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_33, 0.5)
    mul_7: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_33, 0.7071067811865476);  view_33 = None
    erf: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
    add_12: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_8: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_6, add_12);  mul_6 = add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_34: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_8, [512, 3072]);  mul_8 = None
    permute_18: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_36, [1, 0]);  primals_36 = None
    addmm_6: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_37, view_34, permute_18);  primals_37 = None
    view_35: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_6, [1, 512, 768]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    native_dropout_3 = torch.ops.aten.native_dropout.default(view_35, 0.1, True);  view_35 = None
    getitem_10: "f32[1, 512, 768]" = native_dropout_3[0]
    getitem_11: "b8[1, 512, 768]" = native_dropout_3[1];  native_dropout_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_13: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_10, add_11);  getitem_10 = add_11 = None
    var_mean_2 = torch.ops.aten.var_mean.correction(add_13, [2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 512, 1]" = var_mean_2[0]
    getitem_13: "f32[1, 512, 1]" = var_mean_2[1];  var_mean_2 = None
    add_14: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
    rsqrt_2: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    sub_5: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_13, getitem_13);  add_13 = getitem_13 = None
    mul_9: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_2);  sub_5 = None
    mul_10: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_9, primals_38)
    add_15: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_10, primals_39);  mul_10 = primals_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_36: "f32[512, 768]" = torch.ops.aten.reshape.default(add_15, [512, 768])
    permute_19: "f32[768, 384]" = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
    addmm_7: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_41, view_36, permute_19);  primals_41 = None
    view_37: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(addmm_7, [1, 512, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    permute_20: "f32[768, 384]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
    
    # No stacktrace found for following nodes
    mm_default_32: "f32[512, 384]" = torch.ops.aten.mm.default(view_36, permute_20)
    add_tensor_32: "f32[512, 384]" = torch.ops.aten.add.Tensor(mm_default_32, primals_43);  mm_default_32 = primals_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_39: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(add_tensor_32, [1, 512, 384]);  add_tensor_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    permute_21: "f32[768, 384]" = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
    
    # No stacktrace found for following nodes
    mm_default_31: "f32[512, 384]" = torch.ops.aten.mm.default(view_36, permute_21)
    add_tensor_31: "f32[512, 384]" = torch.ops.aten.add.Tensor(mm_default_31, primals_45);  mm_default_31 = primals_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_41: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(add_tensor_31, [1, 512, 384]);  add_tensor_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_22: "f32[1, 768, 512]" = torch.ops.aten.permute.default(add_15, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_2: "f32[1, 768, 512]" = torch.ops.aten.convolution.default(permute_22, primals_46, None, [1], [4], [1], False, [0], 768)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_3: "f32[1, 384, 512]" = torch.ops.aten.convolution.default(convolution_2, primals_47, None, [1], [0], [1], False, [0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_16: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_3, primals_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_42: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_37, [1, 512, 6, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_24: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_43: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_39, [1, 512, 6, 64]);  view_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_25: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_43, [0, 2, 1, 3]);  view_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_44: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_41, [1, 512, 6, 64]);  view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_26: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_44, [0, 2, 1, 3]);  view_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_27: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_16, [0, 2, 1]);  add_16 = None
    mul_11: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(permute_27, view_37);  permute_27 = view_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    permute_28: "f32[384, 54]" = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
    view_45: "f32[512, 384]" = torch.ops.aten.reshape.default(mul_11, [512, 384]);  mul_11 = None
    mm_1: "f32[512, 54]" = torch.ops.aten.mm.default(view_45, permute_28)
    view_46: "f32[1, 512, 54]" = torch.ops.aten.reshape.default(mm_1, [1, 512, 54]);  mm_1 = None
    add_17: "f32[1, 512, 54]" = torch.ops.aten.add.Tensor(view_46, primals_49);  view_46 = primals_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_47: "f32[3072, 9, 1]" = torch.ops.aten.reshape.default(add_17, [-1, 9, 1]);  add_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    amax_2: "f32[3072, 1, 1]" = torch.ops.aten.amax.default(view_47, [1], True)
    sub_6: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(view_47, amax_2);  view_47 = amax_2 = None
    exp_2: "f32[3072, 9, 1]" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
    sum_3: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [1], True)
    div_3: "f32[3072, 9, 1]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_2: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(div_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    permute_29: "f32[768, 384]" = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
    
    # No stacktrace found for following nodes
    mm_default_30: "f32[512, 384]" = torch.ops.aten.mm.default(view_36, permute_29)
    add_tensor_30: "f32[512, 384]" = torch.ops.aten.add.Tensor(mm_default_30, primals_51);  mm_default_30 = primals_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_49: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(add_tensor_30, [1, 512, 384]);  add_tensor_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_50: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(view_49, [1, -1, 384]);  view_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    permute_30: "f32[1, 384, 512]" = torch.ops.aten.permute.default(view_50, [0, 2, 1]);  view_50 = None
    clone_3: "f32[1, 384, 512]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
    unsqueeze_9: "f32[1, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_3, -1);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    constant_pad_nd_1: "f32[1, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_9, [0, 0, 4, 4], 0.0);  unsqueeze_9 = None
    index_1: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(constant_pad_nd_1, [None, None, unsqueeze_8, full_default_1]);  constant_pad_nd_1 = None
    permute_31: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index_1, [0, 1, 2, 4, 3, 5]);  index_1 = None
    view_51: "f32[1, 3456, 512]" = torch.ops.aten.reshape.default(permute_31, [1, 3456, 512]);  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    permute_32: "f32[1, 512, 3456]" = torch.ops.aten.permute.default(view_51, [0, 2, 1]);  view_51 = None
    view_52: "f32[1, 512, 384, 9]" = torch.ops.aten.reshape.default(permute_32, [1, 512, 384, 9]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    clone_4: "f32[1, 512, 384, 9]" = torch.ops.aten.clone.default(view_52, memory_format = torch.contiguous_format);  view_52 = None
    view_53: "f32[3072, 64, 9]" = torch.ops.aten.reshape.default(clone_4, [3072, 64, 9]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    expand_7: "f32[3072, 64, 9]" = torch.ops.aten.expand.default(view_53, [3072, 64, 9]);  view_53 = None
    expand_8: "f32[3072, 9, 1]" = torch.ops.aten.expand.default(div_3, [3072, 9, 1]);  div_3 = None
    bmm_3: "f32[3072, 64, 1]" = torch.ops.aten.bmm.default(expand_7, expand_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    view_57: "f32[512, 384]" = torch.ops.aten.reshape.default(bmm_3, [-1, 384]);  bmm_3 = None
    
    # No stacktrace found for following nodes
    clone_default_30: "f32[1, 6, 512, 64]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
    clone_default_31: "f32[1, 6, 512, 64]" = torch.ops.aten.clone.default(permute_25, memory_format = torch.contiguous_format);  permute_25 = None
    clone_default_32: "f32[1, 6, 512, 64]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
    _scaled_dot_product_efficient_attention_default_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_30, clone_default_31, clone_default_32, None, True, 0.1, scale = 0.125)
    getitem_268: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_default_10[0]
    getitem_269: "f32[1, 6, 512]" = _scaled_dot_product_efficient_attention_default_10[1]
    getitem_270: "i64[]" = _scaled_dot_product_efficient_attention_default_10[2]
    getitem_271: "i64[]" = _scaled_dot_product_efficient_attention_default_10[3];  _scaled_dot_product_efficient_attention_default_10 = None
    alias_default_20: "f32[1, 6, 512, 64]" = torch.ops.aten.alias.default(getitem_268)
    alias_default_21: "f32[1, 6, 512, 64]" = torch.ops.aten.alias.default(alias_default_20);  alias_default_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_34: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_268, [0, 2, 1, 3]);  getitem_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_64: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_57, [1, -1, 6, 64]);  view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    cat_1: "f32[1, 512, 12, 64]" = torch.ops.aten.cat.default([permute_34, view_64], 2);  permute_34 = view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_65: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(cat_1, [1, 512, 768]);  cat_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_66: "f32[512, 768]" = torch.ops.aten.reshape.default(view_65, [512, 768]);  view_65 = None
    permute_35: "f32[768, 768]" = torch.ops.aten.permute.default(primals_52, [1, 0]);  primals_52 = None
    addmm_11: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_53, view_66, permute_35);  primals_53 = None
    view_67: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_11, [1, 512, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    native_dropout_5 = torch.ops.aten.native_dropout.default(view_67, 0.1, True);  view_67 = None
    getitem_16: "f32[1, 512, 768]" = native_dropout_5[0]
    getitem_17: "b8[1, 512, 768]" = native_dropout_5[1];  native_dropout_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_21: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_16, add_15);  getitem_16 = add_15 = None
    var_mean_3 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 512, 1]" = var_mean_3[0]
    getitem_19: "f32[1, 512, 1]" = var_mean_3[1];  var_mean_3 = None
    add_22: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
    rsqrt_3: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
    sub_8: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_21, getitem_19);  add_21 = getitem_19 = None
    mul_12: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_3);  sub_8 = None
    mul_13: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_12, primals_54)
    add_23: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_13, primals_55);  mul_13 = primals_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_68: "f32[512, 768]" = torch.ops.aten.reshape.default(add_23, [512, 768])
    permute_36: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
    addmm_12: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_57, view_68, permute_36);  primals_57 = None
    view_69: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_12, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_14: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_69, 0.5)
    mul_15: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_69, 0.7071067811865476);  view_69 = None
    erf_1: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_15);  mul_15 = None
    add_24: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_16: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_14, add_24);  mul_14 = add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_70: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_16, [512, 3072]);  mul_16 = None
    permute_37: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
    addmm_13: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_59, view_70, permute_37);  primals_59 = None
    view_71: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_13, [1, 512, 768]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    native_dropout_6 = torch.ops.aten.native_dropout.default(view_71, 0.1, True);  view_71 = None
    getitem_20: "f32[1, 512, 768]" = native_dropout_6[0]
    getitem_21: "b8[1, 512, 768]" = native_dropout_6[1];  native_dropout_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_25: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_20, add_23);  getitem_20 = add_23 = None
    var_mean_4 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 512, 1]" = var_mean_4[0]
    getitem_23: "f32[1, 512, 1]" = var_mean_4[1];  var_mean_4 = None
    add_26: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
    rsqrt_4: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    sub_9: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_25, getitem_23);  add_25 = getitem_23 = None
    mul_17: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_4);  sub_9 = None
    mul_18: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_17, primals_60)
    add_27: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_18, primals_61);  mul_18 = primals_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_72: "f32[512, 768]" = torch.ops.aten.reshape.default(add_27, [512, 768])
    permute_38: "f32[768, 384]" = torch.ops.aten.permute.default(primals_62, [1, 0]);  primals_62 = None
    addmm_14: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_63, view_72, permute_38);  primals_63 = None
    view_73: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(addmm_14, [1, 512, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    permute_39: "f32[768, 384]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
    
    # No stacktrace found for following nodes
    mm_default_29: "f32[512, 384]" = torch.ops.aten.mm.default(view_72, permute_39)
    add_tensor_29: "f32[512, 384]" = torch.ops.aten.add.Tensor(mm_default_29, primals_65);  mm_default_29 = primals_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_75: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(add_tensor_29, [1, 512, 384]);  add_tensor_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    permute_40: "f32[768, 384]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
    
    # No stacktrace found for following nodes
    mm_default_28: "f32[512, 384]" = torch.ops.aten.mm.default(view_72, permute_40)
    add_tensor_28: "f32[512, 384]" = torch.ops.aten.add.Tensor(mm_default_28, primals_67);  mm_default_28 = primals_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_77: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(add_tensor_28, [1, 512, 384]);  add_tensor_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_41: "f32[1, 768, 512]" = torch.ops.aten.permute.default(add_27, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_4: "f32[1, 768, 512]" = torch.ops.aten.convolution.default(permute_41, primals_68, None, [1], [4], [1], False, [0], 768)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_5: "f32[1, 384, 512]" = torch.ops.aten.convolution.default(convolution_4, primals_69, None, [1], [0], [1], False, [0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_28: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_5, primals_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_78: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_73, [1, 512, 6, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_43: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_79: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_75, [1, 512, 6, 64]);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_44: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_79, [0, 2, 1, 3]);  view_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_80: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_77, [1, 512, 6, 64]);  view_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_45: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_46: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_28, [0, 2, 1]);  add_28 = None
    mul_19: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(permute_46, view_73);  permute_46 = view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    permute_47: "f32[384, 54]" = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
    view_81: "f32[512, 384]" = torch.ops.aten.reshape.default(mul_19, [512, 384]);  mul_19 = None
    mm_2: "f32[512, 54]" = torch.ops.aten.mm.default(view_81, permute_47)
    view_82: "f32[1, 512, 54]" = torch.ops.aten.reshape.default(mm_2, [1, 512, 54]);  mm_2 = None
    add_29: "f32[1, 512, 54]" = torch.ops.aten.add.Tensor(view_82, primals_71);  view_82 = primals_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_83: "f32[3072, 9, 1]" = torch.ops.aten.reshape.default(add_29, [-1, 9, 1]);  add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    amax_4: "f32[3072, 1, 1]" = torch.ops.aten.amax.default(view_83, [1], True)
    sub_10: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(view_83, amax_4);  view_83 = amax_4 = None
    exp_4: "f32[3072, 9, 1]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_5: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [1], True)
    div_6: "f32[3072, 9, 1]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    alias_4: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(div_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    permute_48: "f32[768, 384]" = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
    
    # No stacktrace found for following nodes
    mm_default_27: "f32[512, 384]" = torch.ops.aten.mm.default(view_72, permute_48)
    add_tensor_27: "f32[512, 384]" = torch.ops.aten.add.Tensor(mm_default_27, primals_73);  mm_default_27 = primals_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_85: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(add_tensor_27, [1, 512, 384]);  add_tensor_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_86: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(view_85, [1, -1, 384]);  view_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    permute_49: "f32[1, 384, 512]" = torch.ops.aten.permute.default(view_86, [0, 2, 1]);  view_86 = None
    clone_6: "f32[1, 384, 512]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    unsqueeze_16: "f32[1, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_6, -1);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    constant_pad_nd_2: "f32[1, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_16, [0, 0, 4, 4], 0.0);  unsqueeze_16 = None
    index_2: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(constant_pad_nd_2, [None, None, unsqueeze_8, full_default_1]);  constant_pad_nd_2 = None
    permute_50: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index_2, [0, 1, 2, 4, 3, 5]);  index_2 = None
    view_87: "f32[1, 3456, 512]" = torch.ops.aten.reshape.default(permute_50, [1, 3456, 512]);  permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    permute_51: "f32[1, 512, 3456]" = torch.ops.aten.permute.default(view_87, [0, 2, 1]);  view_87 = None
    view_88: "f32[1, 512, 384, 9]" = torch.ops.aten.reshape.default(permute_51, [1, 512, 384, 9]);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    clone_7: "f32[1, 512, 384, 9]" = torch.ops.aten.clone.default(view_88, memory_format = torch.contiguous_format);  view_88 = None
    view_89: "f32[3072, 64, 9]" = torch.ops.aten.reshape.default(clone_7, [3072, 64, 9]);  clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    expand_13: "f32[3072, 64, 9]" = torch.ops.aten.expand.default(view_89, [3072, 64, 9]);  view_89 = None
    expand_14: "f32[3072, 9, 1]" = torch.ops.aten.expand.default(div_6, [3072, 9, 1]);  div_6 = None
    bmm_6: "f32[3072, 64, 1]" = torch.ops.aten.bmm.default(expand_13, expand_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    view_93: "f32[512, 384]" = torch.ops.aten.reshape.default(bmm_6, [-1, 384]);  bmm_6 = None
    
    # No stacktrace found for following nodes
    clone_default_27: "f32[1, 6, 512, 64]" = torch.ops.aten.clone.default(permute_43, memory_format = torch.contiguous_format);  permute_43 = None
    clone_default_28: "f32[1, 6, 512, 64]" = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None
    clone_default_29: "f32[1, 6, 512, 64]" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
    _scaled_dot_product_efficient_attention_default_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_27, clone_default_28, clone_default_29, None, True, 0.1, scale = 0.125)
    getitem_261: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_default_9[0]
    getitem_262: "f32[1, 6, 512]" = _scaled_dot_product_efficient_attention_default_9[1]
    getitem_263: "i64[]" = _scaled_dot_product_efficient_attention_default_9[2]
    getitem_264: "i64[]" = _scaled_dot_product_efficient_attention_default_9[3];  _scaled_dot_product_efficient_attention_default_9 = None
    alias_default_18: "f32[1, 6, 512, 64]" = torch.ops.aten.alias.default(getitem_261)
    alias_default_19: "f32[1, 6, 512, 64]" = torch.ops.aten.alias.default(alias_default_18);  alias_default_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_53: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_261, [0, 2, 1, 3]);  getitem_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_100: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_93, [1, -1, 6, 64]);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    cat_2: "f32[1, 512, 12, 64]" = torch.ops.aten.cat.default([permute_53, view_100], 2);  permute_53 = view_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_101: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(cat_2, [1, 512, 768]);  cat_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_102: "f32[512, 768]" = torch.ops.aten.reshape.default(view_101, [512, 768]);  view_101 = None
    permute_54: "f32[768, 768]" = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
    addmm_18: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_75, view_102, permute_54);  primals_75 = None
    view_103: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_18, [1, 512, 768]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    native_dropout_8 = torch.ops.aten.native_dropout.default(view_103, 0.1, True);  view_103 = None
    getitem_26: "f32[1, 512, 768]" = native_dropout_8[0]
    getitem_27: "b8[1, 512, 768]" = native_dropout_8[1];  native_dropout_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_33: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_26, add_27);  getitem_26 = add_27 = None
    var_mean_5 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 512, 1]" = var_mean_5[0]
    getitem_29: "f32[1, 512, 1]" = var_mean_5[1];  var_mean_5 = None
    add_34: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
    rsqrt_5: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
    sub_12: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_33, getitem_29);  add_33 = getitem_29 = None
    mul_20: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_5);  sub_12 = None
    mul_21: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_20, primals_76)
    add_35: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_21, primals_77);  mul_21 = primals_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_104: "f32[512, 768]" = torch.ops.aten.reshape.default(add_35, [512, 768])
    permute_55: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_78, [1, 0]);  primals_78 = None
    addmm_19: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_79, view_104, permute_55);  primals_79 = None
    view_105: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_19, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_22: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_105, 0.5)
    mul_23: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_105, 0.7071067811865476);  view_105 = None
    erf_2: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_23);  mul_23 = None
    add_36: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_24: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_22, add_36);  mul_22 = add_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_106: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_24, [512, 3072]);  mul_24 = None
    permute_56: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
    addmm_20: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_81, view_106, permute_56);  primals_81 = None
    view_107: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_20, [1, 512, 768]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    native_dropout_9 = torch.ops.aten.native_dropout.default(view_107, 0.1, True);  view_107 = None
    getitem_30: "f32[1, 512, 768]" = native_dropout_9[0]
    getitem_31: "b8[1, 512, 768]" = native_dropout_9[1];  native_dropout_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_37: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_30, add_35);  getitem_30 = add_35 = None
    var_mean_6 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 512, 1]" = var_mean_6[0]
    getitem_33: "f32[1, 512, 1]" = var_mean_6[1];  var_mean_6 = None
    add_38: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
    rsqrt_6: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    sub_13: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_37, getitem_33);  add_37 = getitem_33 = None
    mul_25: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_6);  sub_13 = None
    mul_26: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_25, primals_82)
    add_39: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_26, primals_83);  mul_26 = primals_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_108: "f32[512, 768]" = torch.ops.aten.reshape.default(add_39, [512, 768])
    permute_57: "f32[768, 384]" = torch.ops.aten.permute.default(primals_84, [1, 0]);  primals_84 = None
    addmm_21: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_85, view_108, permute_57);  primals_85 = None
    view_109: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(addmm_21, [1, 512, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    permute_58: "f32[768, 384]" = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
    
    # No stacktrace found for following nodes
    mm_default_26: "f32[512, 384]" = torch.ops.aten.mm.default(view_108, permute_58)
    add_tensor_26: "f32[512, 384]" = torch.ops.aten.add.Tensor(mm_default_26, primals_87);  mm_default_26 = primals_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_111: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(add_tensor_26, [1, 512, 384]);  add_tensor_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    permute_59: "f32[768, 384]" = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
    
    # No stacktrace found for following nodes
    mm_default_25: "f32[512, 384]" = torch.ops.aten.mm.default(view_108, permute_59)
    add_tensor_25: "f32[512, 384]" = torch.ops.aten.add.Tensor(mm_default_25, primals_89);  mm_default_25 = primals_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_113: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(add_tensor_25, [1, 512, 384]);  add_tensor_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_60: "f32[1, 768, 512]" = torch.ops.aten.permute.default(add_39, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_6: "f32[1, 768, 512]" = torch.ops.aten.convolution.default(permute_60, primals_90, None, [1], [4], [1], False, [0], 768)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_7: "f32[1, 384, 512]" = torch.ops.aten.convolution.default(convolution_6, primals_91, None, [1], [0], [1], False, [0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_40: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_7, primals_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_114: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_109, [1, 512, 6, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_62: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_115: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_111, [1, 512, 6, 64]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_63: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_115, [0, 2, 1, 3]);  view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_116: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_113, [1, 512, 6, 64]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_64: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_65: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_40, [0, 2, 1]);  add_40 = None
    mul_27: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(permute_65, view_109);  permute_65 = view_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    permute_66: "f32[384, 54]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
    view_117: "f32[512, 384]" = torch.ops.aten.reshape.default(mul_27, [512, 384]);  mul_27 = None
    mm_3: "f32[512, 54]" = torch.ops.aten.mm.default(view_117, permute_66)
    view_118: "f32[1, 512, 54]" = torch.ops.aten.reshape.default(mm_3, [1, 512, 54]);  mm_3 = None
    add_41: "f32[1, 512, 54]" = torch.ops.aten.add.Tensor(view_118, primals_93);  view_118 = primals_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_119: "f32[3072, 9, 1]" = torch.ops.aten.reshape.default(add_41, [-1, 9, 1]);  add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    amax_6: "f32[3072, 1, 1]" = torch.ops.aten.amax.default(view_119, [1], True)
    sub_14: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(view_119, amax_6);  view_119 = amax_6 = None
    exp_6: "f32[3072, 9, 1]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
    sum_7: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [1], True)
    div_9: "f32[3072, 9, 1]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    alias_6: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(div_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    permute_67: "f32[768, 384]" = torch.ops.aten.permute.default(primals_94, [1, 0]);  primals_94 = None
    
    # No stacktrace found for following nodes
    mm_default_24: "f32[512, 384]" = torch.ops.aten.mm.default(view_108, permute_67)
    add_tensor_24: "f32[512, 384]" = torch.ops.aten.add.Tensor(mm_default_24, primals_95);  mm_default_24 = primals_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_121: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(add_tensor_24, [1, 512, 384]);  add_tensor_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_122: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(view_121, [1, -1, 384]);  view_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    permute_68: "f32[1, 384, 512]" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
    clone_9: "f32[1, 384, 512]" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
    unsqueeze_23: "f32[1, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_9, -1);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    constant_pad_nd_3: "f32[1, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_23, [0, 0, 4, 4], 0.0);  unsqueeze_23 = None
    index_3: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(constant_pad_nd_3, [None, None, unsqueeze_8, full_default_1]);  constant_pad_nd_3 = None
    permute_69: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index_3, [0, 1, 2, 4, 3, 5]);  index_3 = None
    view_123: "f32[1, 3456, 512]" = torch.ops.aten.reshape.default(permute_69, [1, 3456, 512]);  permute_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    permute_70: "f32[1, 512, 3456]" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None
    view_124: "f32[1, 512, 384, 9]" = torch.ops.aten.reshape.default(permute_70, [1, 512, 384, 9]);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    clone_10: "f32[1, 512, 384, 9]" = torch.ops.aten.clone.default(view_124, memory_format = torch.contiguous_format);  view_124 = None
    view_125: "f32[3072, 64, 9]" = torch.ops.aten.reshape.default(clone_10, [3072, 64, 9]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    expand_19: "f32[3072, 64, 9]" = torch.ops.aten.expand.default(view_125, [3072, 64, 9]);  view_125 = None
    expand_20: "f32[3072, 9, 1]" = torch.ops.aten.expand.default(div_9, [3072, 9, 1]);  div_9 = None
    bmm_9: "f32[3072, 64, 1]" = torch.ops.aten.bmm.default(expand_19, expand_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    view_129: "f32[512, 384]" = torch.ops.aten.reshape.default(bmm_9, [-1, 384]);  bmm_9 = None
    
    # No stacktrace found for following nodes
    clone_default_24: "f32[1, 6, 512, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    clone_default_25: "f32[1, 6, 512, 64]" = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None
    clone_default_26: "f32[1, 6, 512, 64]" = torch.ops.aten.clone.default(permute_64, memory_format = torch.contiguous_format);  permute_64 = None
    _scaled_dot_product_efficient_attention_default_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_24, clone_default_25, clone_default_26, None, True, 0.1, scale = 0.125)
    getitem_254: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_default_8[0]
    getitem_255: "f32[1, 6, 512]" = _scaled_dot_product_efficient_attention_default_8[1]
    getitem_256: "i64[]" = _scaled_dot_product_efficient_attention_default_8[2]
    getitem_257: "i64[]" = _scaled_dot_product_efficient_attention_default_8[3];  _scaled_dot_product_efficient_attention_default_8 = None
    alias_default_16: "f32[1, 6, 512, 64]" = torch.ops.aten.alias.default(getitem_254)
    alias_default_17: "f32[1, 6, 512, 64]" = torch.ops.aten.alias.default(alias_default_16);  alias_default_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_72: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_254, [0, 2, 1, 3]);  getitem_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_136: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_129, [1, -1, 6, 64]);  view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    cat_3: "f32[1, 512, 12, 64]" = torch.ops.aten.cat.default([permute_72, view_136], 2);  permute_72 = view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_137: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(cat_3, [1, 512, 768]);  cat_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_138: "f32[512, 768]" = torch.ops.aten.reshape.default(view_137, [512, 768]);  view_137 = None
    permute_73: "f32[768, 768]" = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
    addmm_25: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_97, view_138, permute_73);  primals_97 = None
    view_139: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_25, [1, 512, 768]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    native_dropout_11 = torch.ops.aten.native_dropout.default(view_139, 0.1, True);  view_139 = None
    getitem_36: "f32[1, 512, 768]" = native_dropout_11[0]
    getitem_37: "b8[1, 512, 768]" = native_dropout_11[1];  native_dropout_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_45: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_36, add_39);  getitem_36 = add_39 = None
    var_mean_7 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
    getitem_38: "f32[1, 512, 1]" = var_mean_7[0]
    getitem_39: "f32[1, 512, 1]" = var_mean_7[1];  var_mean_7 = None
    add_46: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
    rsqrt_7: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    sub_16: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_45, getitem_39);  add_45 = getitem_39 = None
    mul_28: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_7);  sub_16 = None
    mul_29: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_28, primals_98)
    add_47: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_29, primals_99);  mul_29 = primals_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_140: "f32[512, 768]" = torch.ops.aten.reshape.default(add_47, [512, 768])
    permute_74: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_100, [1, 0]);  primals_100 = None
    addmm_26: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_101, view_140, permute_74);  primals_101 = None
    view_141: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_26, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_30: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_141, 0.5)
    mul_31: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_141, 0.7071067811865476);  view_141 = None
    erf_3: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_31);  mul_31 = None
    add_48: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_32: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_30, add_48);  mul_30 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_142: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_32, [512, 3072]);  mul_32 = None
    permute_75: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_102, [1, 0]);  primals_102 = None
    addmm_27: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_103, view_142, permute_75);  primals_103 = None
    view_143: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_27, [1, 512, 768]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    native_dropout_12 = torch.ops.aten.native_dropout.default(view_143, 0.1, True);  view_143 = None
    getitem_40: "f32[1, 512, 768]" = native_dropout_12[0]
    getitem_41: "b8[1, 512, 768]" = native_dropout_12[1];  native_dropout_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_49: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_40, add_47);  getitem_40 = add_47 = None
    var_mean_8 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
    getitem_42: "f32[1, 512, 1]" = var_mean_8[0]
    getitem_43: "f32[1, 512, 1]" = var_mean_8[1];  var_mean_8 = None
    add_50: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
    rsqrt_8: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
    sub_17: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_49, getitem_43);  add_49 = getitem_43 = None
    mul_33: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_8);  sub_17 = None
    mul_34: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_33, primals_104)
    add_51: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_34, primals_105);  mul_34 = primals_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_144: "f32[512, 768]" = torch.ops.aten.reshape.default(add_51, [512, 768])
    permute_76: "f32[768, 384]" = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
    addmm_28: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_107, view_144, permute_76);  primals_107 = None
    view_145: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(addmm_28, [1, 512, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    permute_77: "f32[768, 384]" = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
    
    # No stacktrace found for following nodes
    mm_default_23: "f32[512, 384]" = torch.ops.aten.mm.default(view_144, permute_77)
    add_tensor_23: "f32[512, 384]" = torch.ops.aten.add.Tensor(mm_default_23, primals_109);  mm_default_23 = primals_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_147: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(add_tensor_23, [1, 512, 384]);  add_tensor_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    permute_78: "f32[768, 384]" = torch.ops.aten.permute.default(primals_110, [1, 0]);  primals_110 = None
    
    # No stacktrace found for following nodes
    mm_default_22: "f32[512, 384]" = torch.ops.aten.mm.default(view_144, permute_78)
    add_tensor_22: "f32[512, 384]" = torch.ops.aten.add.Tensor(mm_default_22, primals_111);  mm_default_22 = primals_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_149: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(add_tensor_22, [1, 512, 384]);  add_tensor_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_79: "f32[1, 768, 512]" = torch.ops.aten.permute.default(add_51, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_8: "f32[1, 768, 512]" = torch.ops.aten.convolution.default(permute_79, primals_112, None, [1], [4], [1], False, [0], 768)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_9: "f32[1, 384, 512]" = torch.ops.aten.convolution.default(convolution_8, primals_113, None, [1], [0], [1], False, [0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_52: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_9, primals_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_150: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_145, [1, 512, 6, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_81: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_150, [0, 2, 1, 3]);  view_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_151: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_147, [1, 512, 6, 64]);  view_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_82: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_151, [0, 2, 1, 3]);  view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_152: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_149, [1, 512, 6, 64]);  view_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_83: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_152, [0, 2, 1, 3]);  view_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_84: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_52, [0, 2, 1]);  add_52 = None
    mul_35: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(permute_84, view_145);  permute_84 = view_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    permute_85: "f32[384, 54]" = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
    view_153: "f32[512, 384]" = torch.ops.aten.reshape.default(mul_35, [512, 384]);  mul_35 = None
    mm_4: "f32[512, 54]" = torch.ops.aten.mm.default(view_153, permute_85)
    view_154: "f32[1, 512, 54]" = torch.ops.aten.reshape.default(mm_4, [1, 512, 54]);  mm_4 = None
    add_53: "f32[1, 512, 54]" = torch.ops.aten.add.Tensor(view_154, primals_115);  view_154 = primals_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_155: "f32[3072, 9, 1]" = torch.ops.aten.reshape.default(add_53, [-1, 9, 1]);  add_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    amax_8: "f32[3072, 1, 1]" = torch.ops.aten.amax.default(view_155, [1], True)
    sub_18: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(view_155, amax_8);  view_155 = amax_8 = None
    exp_8: "f32[3072, 9, 1]" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
    sum_9: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [1], True)
    div_12: "f32[3072, 9, 1]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    alias_8: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(div_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    permute_86: "f32[768, 384]" = torch.ops.aten.permute.default(primals_116, [1, 0]);  primals_116 = None
    
    # No stacktrace found for following nodes
    mm_default_21: "f32[512, 384]" = torch.ops.aten.mm.default(view_144, permute_86)
    add_tensor_21: "f32[512, 384]" = torch.ops.aten.add.Tensor(mm_default_21, primals_117);  mm_default_21 = primals_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_157: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(add_tensor_21, [1, 512, 384]);  add_tensor_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_158: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(view_157, [1, -1, 384]);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    permute_87: "f32[1, 384, 512]" = torch.ops.aten.permute.default(view_158, [0, 2, 1]);  view_158 = None
    clone_12: "f32[1, 384, 512]" = torch.ops.aten.clone.default(permute_87, memory_format = torch.contiguous_format);  permute_87 = None
    unsqueeze_30: "f32[1, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_12, -1);  clone_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    constant_pad_nd_4: "f32[1, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_30, [0, 0, 4, 4], 0.0);  unsqueeze_30 = None
    index_4: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(constant_pad_nd_4, [None, None, unsqueeze_8, full_default_1]);  constant_pad_nd_4 = None
    permute_88: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index_4, [0, 1, 2, 4, 3, 5]);  index_4 = None
    view_159: "f32[1, 3456, 512]" = torch.ops.aten.reshape.default(permute_88, [1, 3456, 512]);  permute_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    permute_89: "f32[1, 512, 3456]" = torch.ops.aten.permute.default(view_159, [0, 2, 1]);  view_159 = None
    view_160: "f32[1, 512, 384, 9]" = torch.ops.aten.reshape.default(permute_89, [1, 512, 384, 9]);  permute_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    clone_13: "f32[1, 512, 384, 9]" = torch.ops.aten.clone.default(view_160, memory_format = torch.contiguous_format);  view_160 = None
    view_161: "f32[3072, 64, 9]" = torch.ops.aten.reshape.default(clone_13, [3072, 64, 9]);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    expand_25: "f32[3072, 64, 9]" = torch.ops.aten.expand.default(view_161, [3072, 64, 9]);  view_161 = None
    expand_26: "f32[3072, 9, 1]" = torch.ops.aten.expand.default(div_12, [3072, 9, 1]);  div_12 = None
    bmm_12: "f32[3072, 64, 1]" = torch.ops.aten.bmm.default(expand_25, expand_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    view_165: "f32[512, 384]" = torch.ops.aten.reshape.default(bmm_12, [-1, 384]);  bmm_12 = None
    
    # No stacktrace found for following nodes
    clone_default_21: "f32[1, 6, 512, 64]" = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
    clone_default_22: "f32[1, 6, 512, 64]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
    clone_default_23: "f32[1, 6, 512, 64]" = torch.ops.aten.clone.default(permute_83, memory_format = torch.contiguous_format);  permute_83 = None
    _scaled_dot_product_efficient_attention_default_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_21, clone_default_22, clone_default_23, None, True, 0.1, scale = 0.125)
    getitem_247: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_default_7[0]
    getitem_248: "f32[1, 6, 512]" = _scaled_dot_product_efficient_attention_default_7[1]
    getitem_249: "i64[]" = _scaled_dot_product_efficient_attention_default_7[2]
    getitem_250: "i64[]" = _scaled_dot_product_efficient_attention_default_7[3];  _scaled_dot_product_efficient_attention_default_7 = None
    alias_default_14: "f32[1, 6, 512, 64]" = torch.ops.aten.alias.default(getitem_247)
    alias_default_15: "f32[1, 6, 512, 64]" = torch.ops.aten.alias.default(alias_default_14);  alias_default_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_91: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_247, [0, 2, 1, 3]);  getitem_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_172: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_165, [1, -1, 6, 64]);  view_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    cat_4: "f32[1, 512, 12, 64]" = torch.ops.aten.cat.default([permute_91, view_172], 2);  permute_91 = view_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_173: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(cat_4, [1, 512, 768]);  cat_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_174: "f32[512, 768]" = torch.ops.aten.reshape.default(view_173, [512, 768]);  view_173 = None
    permute_92: "f32[768, 768]" = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
    addmm_32: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_119, view_174, permute_92);  primals_119 = None
    view_175: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_32, [1, 512, 768]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    native_dropout_14 = torch.ops.aten.native_dropout.default(view_175, 0.1, True);  view_175 = None
    getitem_46: "f32[1, 512, 768]" = native_dropout_14[0]
    getitem_47: "b8[1, 512, 768]" = native_dropout_14[1];  native_dropout_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_57: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_46, add_51);  getitem_46 = add_51 = None
    var_mean_9 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
    getitem_48: "f32[1, 512, 1]" = var_mean_9[0]
    getitem_49: "f32[1, 512, 1]" = var_mean_9[1];  var_mean_9 = None
    add_58: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
    rsqrt_9: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    sub_20: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_57, getitem_49);  add_57 = getitem_49 = None
    mul_36: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_9);  sub_20 = None
    mul_37: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_36, primals_120)
    add_59: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_37, primals_121);  mul_37 = primals_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_176: "f32[512, 768]" = torch.ops.aten.reshape.default(add_59, [512, 768])
    permute_93: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
    addmm_33: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_123, view_176, permute_93);  primals_123 = None
    view_177: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_33, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_38: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_177, 0.5)
    mul_39: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_177, 0.7071067811865476);  view_177 = None
    erf_4: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_39);  mul_39 = None
    add_60: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_40: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_38, add_60);  mul_38 = add_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_178: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_40, [512, 3072]);  mul_40 = None
    permute_94: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_124, [1, 0]);  primals_124 = None
    addmm_34: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_125, view_178, permute_94);  primals_125 = None
    view_179: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_34, [1, 512, 768]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    native_dropout_15 = torch.ops.aten.native_dropout.default(view_179, 0.1, True);  view_179 = None
    getitem_50: "f32[1, 512, 768]" = native_dropout_15[0]
    getitem_51: "b8[1, 512, 768]" = native_dropout_15[1];  native_dropout_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_61: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_50, add_59);  getitem_50 = add_59 = None
    var_mean_10 = torch.ops.aten.var_mean.correction(add_61, [2], correction = 0, keepdim = True)
    getitem_52: "f32[1, 512, 1]" = var_mean_10[0]
    getitem_53: "f32[1, 512, 1]" = var_mean_10[1];  var_mean_10 = None
    add_62: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-12);  getitem_52 = None
    rsqrt_10: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    sub_21: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_61, getitem_53);  add_61 = getitem_53 = None
    mul_41: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_10);  sub_21 = None
    mul_42: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_41, primals_126)
    add_63: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_42, primals_127);  mul_42 = primals_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_180: "f32[512, 768]" = torch.ops.aten.reshape.default(add_63, [512, 768])
    permute_95: "f32[768, 384]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
    addmm_35: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_129, view_180, permute_95);  primals_129 = None
    view_181: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(addmm_35, [1, 512, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    permute_96: "f32[768, 384]" = torch.ops.aten.permute.default(primals_130, [1, 0]);  primals_130 = None
    
    # No stacktrace found for following nodes
    mm_default_20: "f32[512, 384]" = torch.ops.aten.mm.default(view_180, permute_96)
    add_tensor_20: "f32[512, 384]" = torch.ops.aten.add.Tensor(mm_default_20, primals_131);  mm_default_20 = primals_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_183: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(add_tensor_20, [1, 512, 384]);  add_tensor_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    permute_97: "f32[768, 384]" = torch.ops.aten.permute.default(primals_132, [1, 0]);  primals_132 = None
    
    # No stacktrace found for following nodes
    mm_default_19: "f32[512, 384]" = torch.ops.aten.mm.default(view_180, permute_97)
    add_tensor_19: "f32[512, 384]" = torch.ops.aten.add.Tensor(mm_default_19, primals_133);  mm_default_19 = primals_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_185: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(add_tensor_19, [1, 512, 384]);  add_tensor_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_98: "f32[1, 768, 512]" = torch.ops.aten.permute.default(add_63, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_10: "f32[1, 768, 512]" = torch.ops.aten.convolution.default(permute_98, primals_134, None, [1], [4], [1], False, [0], 768)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_11: "f32[1, 384, 512]" = torch.ops.aten.convolution.default(convolution_10, primals_135, None, [1], [0], [1], False, [0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_64: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_11, primals_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_186: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_181, [1, 512, 6, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_100: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_187: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_183, [1, 512, 6, 64]);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_101: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_187, [0, 2, 1, 3]);  view_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_188: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_185, [1, 512, 6, 64]);  view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_102: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_188, [0, 2, 1, 3]);  view_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_103: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_64, [0, 2, 1]);  add_64 = None
    mul_43: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(permute_103, view_181);  permute_103 = view_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    permute_104: "f32[384, 54]" = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
    view_189: "f32[512, 384]" = torch.ops.aten.reshape.default(mul_43, [512, 384]);  mul_43 = None
    mm_5: "f32[512, 54]" = torch.ops.aten.mm.default(view_189, permute_104)
    view_190: "f32[1, 512, 54]" = torch.ops.aten.reshape.default(mm_5, [1, 512, 54]);  mm_5 = None
    add_65: "f32[1, 512, 54]" = torch.ops.aten.add.Tensor(view_190, primals_137);  view_190 = primals_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_191: "f32[3072, 9, 1]" = torch.ops.aten.reshape.default(add_65, [-1, 9, 1]);  add_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    amax_10: "f32[3072, 1, 1]" = torch.ops.aten.amax.default(view_191, [1], True)
    sub_22: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(view_191, amax_10);  view_191 = amax_10 = None
    exp_10: "f32[3072, 9, 1]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_11: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [1], True)
    div_15: "f32[3072, 9, 1]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    alias_10: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(div_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    permute_105: "f32[768, 384]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
    
    # No stacktrace found for following nodes
    mm_default_18: "f32[512, 384]" = torch.ops.aten.mm.default(view_180, permute_105)
    add_tensor_18: "f32[512, 384]" = torch.ops.aten.add.Tensor(mm_default_18, primals_139);  mm_default_18 = primals_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_193: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(add_tensor_18, [1, 512, 384]);  add_tensor_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_194: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(view_193, [1, -1, 384]);  view_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    permute_106: "f32[1, 384, 512]" = torch.ops.aten.permute.default(view_194, [0, 2, 1]);  view_194 = None
    clone_15: "f32[1, 384, 512]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
    unsqueeze_37: "f32[1, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_15, -1);  clone_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    constant_pad_nd_5: "f32[1, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_37, [0, 0, 4, 4], 0.0);  unsqueeze_37 = None
    index_5: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(constant_pad_nd_5, [None, None, unsqueeze_8, full_default_1]);  constant_pad_nd_5 = None
    permute_107: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index_5, [0, 1, 2, 4, 3, 5]);  index_5 = None
    view_195: "f32[1, 3456, 512]" = torch.ops.aten.reshape.default(permute_107, [1, 3456, 512]);  permute_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    permute_108: "f32[1, 512, 3456]" = torch.ops.aten.permute.default(view_195, [0, 2, 1]);  view_195 = None
    view_196: "f32[1, 512, 384, 9]" = torch.ops.aten.reshape.default(permute_108, [1, 512, 384, 9]);  permute_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    clone_16: "f32[1, 512, 384, 9]" = torch.ops.aten.clone.default(view_196, memory_format = torch.contiguous_format);  view_196 = None
    view_197: "f32[3072, 64, 9]" = torch.ops.aten.reshape.default(clone_16, [3072, 64, 9]);  clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    expand_31: "f32[3072, 64, 9]" = torch.ops.aten.expand.default(view_197, [3072, 64, 9]);  view_197 = None
    expand_32: "f32[3072, 9, 1]" = torch.ops.aten.expand.default(div_15, [3072, 9, 1]);  div_15 = None
    bmm_15: "f32[3072, 64, 1]" = torch.ops.aten.bmm.default(expand_31, expand_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    view_201: "f32[512, 384]" = torch.ops.aten.reshape.default(bmm_15, [-1, 384]);  bmm_15 = None
    
    # No stacktrace found for following nodes
    clone_default_18: "f32[1, 6, 512, 64]" = torch.ops.aten.clone.default(permute_100, memory_format = torch.contiguous_format);  permute_100 = None
    clone_default_19: "f32[1, 6, 512, 64]" = torch.ops.aten.clone.default(permute_101, memory_format = torch.contiguous_format);  permute_101 = None
    clone_default_20: "f32[1, 6, 512, 64]" = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format);  permute_102 = None
    _scaled_dot_product_efficient_attention_default_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_18, clone_default_19, clone_default_20, None, True, 0.1, scale = 0.125)
    getitem_240: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_default_6[0]
    getitem_241: "f32[1, 6, 512]" = _scaled_dot_product_efficient_attention_default_6[1]
    getitem_242: "i64[]" = _scaled_dot_product_efficient_attention_default_6[2]
    getitem_243: "i64[]" = _scaled_dot_product_efficient_attention_default_6[3];  _scaled_dot_product_efficient_attention_default_6 = None
    alias_default_12: "f32[1, 6, 512, 64]" = torch.ops.aten.alias.default(getitem_240)
    alias_default_13: "f32[1, 6, 512, 64]" = torch.ops.aten.alias.default(alias_default_12);  alias_default_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_110: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_240, [0, 2, 1, 3]);  getitem_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_208: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_201, [1, -1, 6, 64]);  view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    cat_5: "f32[1, 512, 12, 64]" = torch.ops.aten.cat.default([permute_110, view_208], 2);  permute_110 = view_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_209: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(cat_5, [1, 512, 768]);  cat_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_210: "f32[512, 768]" = torch.ops.aten.reshape.default(view_209, [512, 768]);  view_209 = None
    permute_111: "f32[768, 768]" = torch.ops.aten.permute.default(primals_140, [1, 0]);  primals_140 = None
    addmm_39: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_141, view_210, permute_111);  primals_141 = None
    view_211: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_39, [1, 512, 768]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    native_dropout_17 = torch.ops.aten.native_dropout.default(view_211, 0.1, True);  view_211 = None
    getitem_56: "f32[1, 512, 768]" = native_dropout_17[0]
    getitem_57: "b8[1, 512, 768]" = native_dropout_17[1];  native_dropout_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_69: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_56, add_63);  getitem_56 = add_63 = None
    var_mean_11 = torch.ops.aten.var_mean.correction(add_69, [2], correction = 0, keepdim = True)
    getitem_58: "f32[1, 512, 1]" = var_mean_11[0]
    getitem_59: "f32[1, 512, 1]" = var_mean_11[1];  var_mean_11 = None
    add_70: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-12);  getitem_58 = None
    rsqrt_11: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
    sub_24: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_69, getitem_59);  add_69 = getitem_59 = None
    mul_44: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_11);  sub_24 = None
    mul_45: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_44, primals_142)
    add_71: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_45, primals_143);  mul_45 = primals_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_212: "f32[512, 768]" = torch.ops.aten.reshape.default(add_71, [512, 768])
    permute_112: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_144, [1, 0]);  primals_144 = None
    addmm_40: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_145, view_212, permute_112);  primals_145 = None
    view_213: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_40, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_46: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_213, 0.5)
    mul_47: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_213, 0.7071067811865476);  view_213 = None
    erf_5: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
    add_72: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_48: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_46, add_72);  mul_46 = add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_214: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_48, [512, 3072]);  mul_48 = None
    permute_113: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_146, [1, 0]);  primals_146 = None
    addmm_41: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_147, view_214, permute_113);  primals_147 = None
    view_215: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_41, [1, 512, 768]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    native_dropout_18 = torch.ops.aten.native_dropout.default(view_215, 0.1, True);  view_215 = None
    getitem_60: "f32[1, 512, 768]" = native_dropout_18[0]
    getitem_61: "b8[1, 512, 768]" = native_dropout_18[1];  native_dropout_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_73: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_60, add_71);  getitem_60 = add_71 = None
    var_mean_12 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
    getitem_62: "f32[1, 512, 1]" = var_mean_12[0]
    getitem_63: "f32[1, 512, 1]" = var_mean_12[1];  var_mean_12 = None
    add_74: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-12);  getitem_62 = None
    rsqrt_12: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_25: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_73, getitem_63);  add_73 = getitem_63 = None
    mul_49: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_12);  sub_25 = None
    mul_50: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_49, primals_148)
    add_75: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_50, primals_149);  mul_50 = primals_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_216: "f32[512, 768]" = torch.ops.aten.reshape.default(add_75, [512, 768])
    permute_114: "f32[768, 384]" = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
    addmm_42: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_151, view_216, permute_114);  primals_151 = None
    view_217: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(addmm_42, [1, 512, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    permute_115: "f32[768, 384]" = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None
    
    # No stacktrace found for following nodes
    mm_default_17: "f32[512, 384]" = torch.ops.aten.mm.default(view_216, permute_115)
    add_tensor_17: "f32[512, 384]" = torch.ops.aten.add.Tensor(mm_default_17, primals_153);  mm_default_17 = primals_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_219: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(add_tensor_17, [1, 512, 384]);  add_tensor_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    permute_116: "f32[768, 384]" = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
    
    # No stacktrace found for following nodes
    mm_default_16: "f32[512, 384]" = torch.ops.aten.mm.default(view_216, permute_116)
    add_tensor_16: "f32[512, 384]" = torch.ops.aten.add.Tensor(mm_default_16, primals_155);  mm_default_16 = primals_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_221: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(add_tensor_16, [1, 512, 384]);  add_tensor_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_117: "f32[1, 768, 512]" = torch.ops.aten.permute.default(add_75, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_12: "f32[1, 768, 512]" = torch.ops.aten.convolution.default(permute_117, primals_156, None, [1], [4], [1], False, [0], 768)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_13: "f32[1, 384, 512]" = torch.ops.aten.convolution.default(convolution_12, primals_157, None, [1], [0], [1], False, [0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_76: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_13, primals_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_222: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_217, [1, 512, 6, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_119: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_223: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_219, [1, 512, 6, 64]);  view_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_120: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_223, [0, 2, 1, 3]);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_224: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_221, [1, 512, 6, 64]);  view_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_121: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_122: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_76, [0, 2, 1]);  add_76 = None
    mul_51: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(permute_122, view_217);  permute_122 = view_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    permute_123: "f32[384, 54]" = torch.ops.aten.permute.default(primals_158, [1, 0]);  primals_158 = None
    view_225: "f32[512, 384]" = torch.ops.aten.reshape.default(mul_51, [512, 384]);  mul_51 = None
    mm_6: "f32[512, 54]" = torch.ops.aten.mm.default(view_225, permute_123)
    view_226: "f32[1, 512, 54]" = torch.ops.aten.reshape.default(mm_6, [1, 512, 54]);  mm_6 = None
    add_77: "f32[1, 512, 54]" = torch.ops.aten.add.Tensor(view_226, primals_159);  view_226 = primals_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_227: "f32[3072, 9, 1]" = torch.ops.aten.reshape.default(add_77, [-1, 9, 1]);  add_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    amax_12: "f32[3072, 1, 1]" = torch.ops.aten.amax.default(view_227, [1], True)
    sub_26: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(view_227, amax_12);  view_227 = amax_12 = None
    exp_12: "f32[3072, 9, 1]" = torch.ops.aten.exp.default(sub_26);  sub_26 = None
    sum_13: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [1], True)
    div_18: "f32[3072, 9, 1]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    alias_12: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(div_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    permute_124: "f32[768, 384]" = torch.ops.aten.permute.default(primals_160, [1, 0]);  primals_160 = None
    
    # No stacktrace found for following nodes
    mm_default_15: "f32[512, 384]" = torch.ops.aten.mm.default(view_216, permute_124)
    add_tensor_15: "f32[512, 384]" = torch.ops.aten.add.Tensor(mm_default_15, primals_161);  mm_default_15 = primals_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_229: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(add_tensor_15, [1, 512, 384]);  add_tensor_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_230: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(view_229, [1, -1, 384]);  view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    permute_125: "f32[1, 384, 512]" = torch.ops.aten.permute.default(view_230, [0, 2, 1]);  view_230 = None
    clone_18: "f32[1, 384, 512]" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
    unsqueeze_44: "f32[1, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_18, -1);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    constant_pad_nd_6: "f32[1, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_44, [0, 0, 4, 4], 0.0);  unsqueeze_44 = None
    index_6: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(constant_pad_nd_6, [None, None, unsqueeze_8, full_default_1]);  constant_pad_nd_6 = None
    permute_126: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index_6, [0, 1, 2, 4, 3, 5]);  index_6 = None
    view_231: "f32[1, 3456, 512]" = torch.ops.aten.reshape.default(permute_126, [1, 3456, 512]);  permute_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    permute_127: "f32[1, 512, 3456]" = torch.ops.aten.permute.default(view_231, [0, 2, 1]);  view_231 = None
    view_232: "f32[1, 512, 384, 9]" = torch.ops.aten.reshape.default(permute_127, [1, 512, 384, 9]);  permute_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    clone_19: "f32[1, 512, 384, 9]" = torch.ops.aten.clone.default(view_232, memory_format = torch.contiguous_format);  view_232 = None
    view_233: "f32[3072, 64, 9]" = torch.ops.aten.reshape.default(clone_19, [3072, 64, 9]);  clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    expand_37: "f32[3072, 64, 9]" = torch.ops.aten.expand.default(view_233, [3072, 64, 9]);  view_233 = None
    expand_38: "f32[3072, 9, 1]" = torch.ops.aten.expand.default(div_18, [3072, 9, 1]);  div_18 = None
    bmm_18: "f32[3072, 64, 1]" = torch.ops.aten.bmm.default(expand_37, expand_38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    view_237: "f32[512, 384]" = torch.ops.aten.reshape.default(bmm_18, [-1, 384]);  bmm_18 = None
    
    # No stacktrace found for following nodes
    clone_default_15: "f32[1, 6, 512, 64]" = torch.ops.aten.clone.default(permute_119, memory_format = torch.contiguous_format);  permute_119 = None
    clone_default_16: "f32[1, 6, 512, 64]" = torch.ops.aten.clone.default(permute_120, memory_format = torch.contiguous_format);  permute_120 = None
    clone_default_17: "f32[1, 6, 512, 64]" = torch.ops.aten.clone.default(permute_121, memory_format = torch.contiguous_format);  permute_121 = None
    _scaled_dot_product_efficient_attention_default_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_15, clone_default_16, clone_default_17, None, True, 0.1, scale = 0.125)
    getitem_233: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_default_5[0]
    getitem_234: "f32[1, 6, 512]" = _scaled_dot_product_efficient_attention_default_5[1]
    getitem_235: "i64[]" = _scaled_dot_product_efficient_attention_default_5[2]
    getitem_236: "i64[]" = _scaled_dot_product_efficient_attention_default_5[3];  _scaled_dot_product_efficient_attention_default_5 = None
    alias_default_10: "f32[1, 6, 512, 64]" = torch.ops.aten.alias.default(getitem_233)
    alias_default_11: "f32[1, 6, 512, 64]" = torch.ops.aten.alias.default(alias_default_10);  alias_default_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_129: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_233, [0, 2, 1, 3]);  getitem_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_244: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_237, [1, -1, 6, 64]);  view_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    cat_6: "f32[1, 512, 12, 64]" = torch.ops.aten.cat.default([permute_129, view_244], 2);  permute_129 = view_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_245: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(cat_6, [1, 512, 768]);  cat_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_246: "f32[512, 768]" = torch.ops.aten.reshape.default(view_245, [512, 768]);  view_245 = None
    permute_130: "f32[768, 768]" = torch.ops.aten.permute.default(primals_162, [1, 0]);  primals_162 = None
    addmm_46: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_163, view_246, permute_130);  primals_163 = None
    view_247: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_46, [1, 512, 768]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    native_dropout_20 = torch.ops.aten.native_dropout.default(view_247, 0.1, True);  view_247 = None
    getitem_66: "f32[1, 512, 768]" = native_dropout_20[0]
    getitem_67: "b8[1, 512, 768]" = native_dropout_20[1];  native_dropout_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_81: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_66, add_75);  getitem_66 = add_75 = None
    var_mean_13 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
    getitem_68: "f32[1, 512, 1]" = var_mean_13[0]
    getitem_69: "f32[1, 512, 1]" = var_mean_13[1];  var_mean_13 = None
    add_82: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-12);  getitem_68 = None
    rsqrt_13: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
    sub_28: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_81, getitem_69);  add_81 = getitem_69 = None
    mul_52: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_13);  sub_28 = None
    mul_53: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_52, primals_164)
    add_83: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_53, primals_165);  mul_53 = primals_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_248: "f32[512, 768]" = torch.ops.aten.reshape.default(add_83, [512, 768])
    permute_131: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_166, [1, 0]);  primals_166 = None
    addmm_47: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_167, view_248, permute_131);  primals_167 = None
    view_249: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_47, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_54: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_249, 0.5)
    mul_55: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_249, 0.7071067811865476);  view_249 = None
    erf_6: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
    add_84: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_56: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_54, add_84);  mul_54 = add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_250: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_56, [512, 3072]);  mul_56 = None
    permute_132: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_168, [1, 0]);  primals_168 = None
    addmm_48: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_169, view_250, permute_132);  primals_169 = None
    view_251: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_48, [1, 512, 768]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    native_dropout_21 = torch.ops.aten.native_dropout.default(view_251, 0.1, True);  view_251 = None
    getitem_70: "f32[1, 512, 768]" = native_dropout_21[0]
    getitem_71: "b8[1, 512, 768]" = native_dropout_21[1];  native_dropout_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_85: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_70, add_83);  getitem_70 = add_83 = None
    var_mean_14 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
    getitem_72: "f32[1, 512, 1]" = var_mean_14[0]
    getitem_73: "f32[1, 512, 1]" = var_mean_14[1];  var_mean_14 = None
    add_86: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-12);  getitem_72 = None
    rsqrt_14: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
    sub_29: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_85, getitem_73);  add_85 = getitem_73 = None
    mul_57: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_14);  sub_29 = None
    mul_58: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_57, primals_170)
    add_87: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_58, primals_171);  mul_58 = primals_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_252: "f32[512, 768]" = torch.ops.aten.reshape.default(add_87, [512, 768])
    permute_133: "f32[768, 384]" = torch.ops.aten.permute.default(primals_172, [1, 0]);  primals_172 = None
    addmm_49: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_173, view_252, permute_133);  primals_173 = None
    view_253: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(addmm_49, [1, 512, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    permute_134: "f32[768, 384]" = torch.ops.aten.permute.default(primals_174, [1, 0]);  primals_174 = None
    
    # No stacktrace found for following nodes
    mm_default_14: "f32[512, 384]" = torch.ops.aten.mm.default(view_252, permute_134)
    add_tensor_14: "f32[512, 384]" = torch.ops.aten.add.Tensor(mm_default_14, primals_175);  mm_default_14 = primals_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_255: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(add_tensor_14, [1, 512, 384]);  add_tensor_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    permute_135: "f32[768, 384]" = torch.ops.aten.permute.default(primals_176, [1, 0]);  primals_176 = None
    
    # No stacktrace found for following nodes
    mm_default_13: "f32[512, 384]" = torch.ops.aten.mm.default(view_252, permute_135)
    add_tensor_13: "f32[512, 384]" = torch.ops.aten.add.Tensor(mm_default_13, primals_177);  mm_default_13 = primals_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_257: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(add_tensor_13, [1, 512, 384]);  add_tensor_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_136: "f32[1, 768, 512]" = torch.ops.aten.permute.default(add_87, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_14: "f32[1, 768, 512]" = torch.ops.aten.convolution.default(permute_136, primals_178, None, [1], [4], [1], False, [0], 768)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_15: "f32[1, 384, 512]" = torch.ops.aten.convolution.default(convolution_14, primals_179, None, [1], [0], [1], False, [0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_88: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_15, primals_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_258: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_253, [1, 512, 6, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_138: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_259: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_255, [1, 512, 6, 64]);  view_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_139: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_259, [0, 2, 1, 3]);  view_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_260: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_257, [1, 512, 6, 64]);  view_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_140: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_260, [0, 2, 1, 3]);  view_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_141: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_88, [0, 2, 1]);  add_88 = None
    mul_59: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(permute_141, view_253);  permute_141 = view_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    permute_142: "f32[384, 54]" = torch.ops.aten.permute.default(primals_180, [1, 0]);  primals_180 = None
    view_261: "f32[512, 384]" = torch.ops.aten.reshape.default(mul_59, [512, 384]);  mul_59 = None
    mm_7: "f32[512, 54]" = torch.ops.aten.mm.default(view_261, permute_142)
    view_262: "f32[1, 512, 54]" = torch.ops.aten.reshape.default(mm_7, [1, 512, 54]);  mm_7 = None
    add_89: "f32[1, 512, 54]" = torch.ops.aten.add.Tensor(view_262, primals_181);  view_262 = primals_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_263: "f32[3072, 9, 1]" = torch.ops.aten.reshape.default(add_89, [-1, 9, 1]);  add_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    amax_14: "f32[3072, 1, 1]" = torch.ops.aten.amax.default(view_263, [1], True)
    sub_30: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(view_263, amax_14);  view_263 = amax_14 = None
    exp_14: "f32[3072, 9, 1]" = torch.ops.aten.exp.default(sub_30);  sub_30 = None
    sum_15: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [1], True)
    div_21: "f32[3072, 9, 1]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
    alias_14: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(div_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    permute_143: "f32[768, 384]" = torch.ops.aten.permute.default(primals_182, [1, 0]);  primals_182 = None
    
    # No stacktrace found for following nodes
    mm_default_12: "f32[512, 384]" = torch.ops.aten.mm.default(view_252, permute_143)
    add_tensor_12: "f32[512, 384]" = torch.ops.aten.add.Tensor(mm_default_12, primals_183);  mm_default_12 = primals_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_265: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(add_tensor_12, [1, 512, 384]);  add_tensor_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_266: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(view_265, [1, -1, 384]);  view_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    permute_144: "f32[1, 384, 512]" = torch.ops.aten.permute.default(view_266, [0, 2, 1]);  view_266 = None
    clone_21: "f32[1, 384, 512]" = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None
    unsqueeze_51: "f32[1, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_21, -1);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    constant_pad_nd_7: "f32[1, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_51, [0, 0, 4, 4], 0.0);  unsqueeze_51 = None
    index_7: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(constant_pad_nd_7, [None, None, unsqueeze_8, full_default_1]);  constant_pad_nd_7 = None
    permute_145: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index_7, [0, 1, 2, 4, 3, 5]);  index_7 = None
    view_267: "f32[1, 3456, 512]" = torch.ops.aten.reshape.default(permute_145, [1, 3456, 512]);  permute_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    permute_146: "f32[1, 512, 3456]" = torch.ops.aten.permute.default(view_267, [0, 2, 1]);  view_267 = None
    view_268: "f32[1, 512, 384, 9]" = torch.ops.aten.reshape.default(permute_146, [1, 512, 384, 9]);  permute_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    clone_22: "f32[1, 512, 384, 9]" = torch.ops.aten.clone.default(view_268, memory_format = torch.contiguous_format);  view_268 = None
    view_269: "f32[3072, 64, 9]" = torch.ops.aten.reshape.default(clone_22, [3072, 64, 9]);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    expand_43: "f32[3072, 64, 9]" = torch.ops.aten.expand.default(view_269, [3072, 64, 9]);  view_269 = None
    expand_44: "f32[3072, 9, 1]" = torch.ops.aten.expand.default(div_21, [3072, 9, 1]);  div_21 = None
    bmm_21: "f32[3072, 64, 1]" = torch.ops.aten.bmm.default(expand_43, expand_44)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    view_273: "f32[512, 384]" = torch.ops.aten.reshape.default(bmm_21, [-1, 384]);  bmm_21 = None
    
    # No stacktrace found for following nodes
    clone_default_12: "f32[1, 6, 512, 64]" = torch.ops.aten.clone.default(permute_138, memory_format = torch.contiguous_format);  permute_138 = None
    clone_default_13: "f32[1, 6, 512, 64]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
    clone_default_14: "f32[1, 6, 512, 64]" = torch.ops.aten.clone.default(permute_140, memory_format = torch.contiguous_format);  permute_140 = None
    _scaled_dot_product_efficient_attention_default_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_12, clone_default_13, clone_default_14, None, True, 0.1, scale = 0.125)
    getitem_226: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_default_4[0]
    getitem_227: "f32[1, 6, 512]" = _scaled_dot_product_efficient_attention_default_4[1]
    getitem_228: "i64[]" = _scaled_dot_product_efficient_attention_default_4[2]
    getitem_229: "i64[]" = _scaled_dot_product_efficient_attention_default_4[3];  _scaled_dot_product_efficient_attention_default_4 = None
    alias_default_8: "f32[1, 6, 512, 64]" = torch.ops.aten.alias.default(getitem_226)
    alias_default_9: "f32[1, 6, 512, 64]" = torch.ops.aten.alias.default(alias_default_8);  alias_default_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_148: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_226, [0, 2, 1, 3]);  getitem_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_280: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_273, [1, -1, 6, 64]);  view_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    cat_7: "f32[1, 512, 12, 64]" = torch.ops.aten.cat.default([permute_148, view_280], 2);  permute_148 = view_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_281: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(cat_7, [1, 512, 768]);  cat_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_282: "f32[512, 768]" = torch.ops.aten.reshape.default(view_281, [512, 768]);  view_281 = None
    permute_149: "f32[768, 768]" = torch.ops.aten.permute.default(primals_184, [1, 0]);  primals_184 = None
    addmm_53: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_185, view_282, permute_149);  primals_185 = None
    view_283: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_53, [1, 512, 768]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    native_dropout_23 = torch.ops.aten.native_dropout.default(view_283, 0.1, True);  view_283 = None
    getitem_76: "f32[1, 512, 768]" = native_dropout_23[0]
    getitem_77: "b8[1, 512, 768]" = native_dropout_23[1];  native_dropout_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_93: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_76, add_87);  getitem_76 = add_87 = None
    var_mean_15 = torch.ops.aten.var_mean.correction(add_93, [2], correction = 0, keepdim = True)
    getitem_78: "f32[1, 512, 1]" = var_mean_15[0]
    getitem_79: "f32[1, 512, 1]" = var_mean_15[1];  var_mean_15 = None
    add_94: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-12);  getitem_78 = None
    rsqrt_15: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
    sub_32: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_93, getitem_79);  add_93 = getitem_79 = None
    mul_60: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_15);  sub_32 = None
    mul_61: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_60, primals_186)
    add_95: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_61, primals_187);  mul_61 = primals_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_284: "f32[512, 768]" = torch.ops.aten.reshape.default(add_95, [512, 768])
    permute_150: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_188, [1, 0]);  primals_188 = None
    addmm_54: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_189, view_284, permute_150);  primals_189 = None
    view_285: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_54, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_62: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_285, 0.5)
    mul_63: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_285, 0.7071067811865476);  view_285 = None
    erf_7: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
    add_96: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_64: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_62, add_96);  mul_62 = add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_286: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_64, [512, 3072]);  mul_64 = None
    permute_151: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_190, [1, 0]);  primals_190 = None
    addmm_55: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_191, view_286, permute_151);  primals_191 = None
    view_287: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_55, [1, 512, 768]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    native_dropout_24 = torch.ops.aten.native_dropout.default(view_287, 0.1, True);  view_287 = None
    getitem_80: "f32[1, 512, 768]" = native_dropout_24[0]
    getitem_81: "b8[1, 512, 768]" = native_dropout_24[1];  native_dropout_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_97: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_80, add_95);  getitem_80 = add_95 = None
    var_mean_16 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
    getitem_82: "f32[1, 512, 1]" = var_mean_16[0]
    getitem_83: "f32[1, 512, 1]" = var_mean_16[1];  var_mean_16 = None
    add_98: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-12);  getitem_82 = None
    rsqrt_16: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    sub_33: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_97, getitem_83);  add_97 = getitem_83 = None
    mul_65: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_16);  sub_33 = None
    mul_66: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_65, primals_192)
    add_99: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_66, primals_193);  mul_66 = primals_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_288: "f32[512, 768]" = torch.ops.aten.reshape.default(add_99, [512, 768])
    permute_152: "f32[768, 384]" = torch.ops.aten.permute.default(primals_194, [1, 0]);  primals_194 = None
    addmm_56: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_195, view_288, permute_152);  primals_195 = None
    view_289: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(addmm_56, [1, 512, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    permute_153: "f32[768, 384]" = torch.ops.aten.permute.default(primals_196, [1, 0]);  primals_196 = None
    
    # No stacktrace found for following nodes
    mm_default_11: "f32[512, 384]" = torch.ops.aten.mm.default(view_288, permute_153)
    add_tensor_11: "f32[512, 384]" = torch.ops.aten.add.Tensor(mm_default_11, primals_197);  mm_default_11 = primals_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_291: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(add_tensor_11, [1, 512, 384]);  add_tensor_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    permute_154: "f32[768, 384]" = torch.ops.aten.permute.default(primals_198, [1, 0]);  primals_198 = None
    
    # No stacktrace found for following nodes
    mm_default_10: "f32[512, 384]" = torch.ops.aten.mm.default(view_288, permute_154)
    add_tensor_10: "f32[512, 384]" = torch.ops.aten.add.Tensor(mm_default_10, primals_199);  mm_default_10 = primals_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_293: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(add_tensor_10, [1, 512, 384]);  add_tensor_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_155: "f32[1, 768, 512]" = torch.ops.aten.permute.default(add_99, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_16: "f32[1, 768, 512]" = torch.ops.aten.convolution.default(permute_155, primals_200, None, [1], [4], [1], False, [0], 768)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_17: "f32[1, 384, 512]" = torch.ops.aten.convolution.default(convolution_16, primals_201, None, [1], [0], [1], False, [0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_100: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_17, primals_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_294: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_289, [1, 512, 6, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_157: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_295: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_291, [1, 512, 6, 64]);  view_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_158: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_295, [0, 2, 1, 3]);  view_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_296: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_293, [1, 512, 6, 64]);  view_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_159: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_296, [0, 2, 1, 3]);  view_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_160: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_100, [0, 2, 1]);  add_100 = None
    mul_67: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(permute_160, view_289);  permute_160 = view_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    permute_161: "f32[384, 54]" = torch.ops.aten.permute.default(primals_202, [1, 0]);  primals_202 = None
    view_297: "f32[512, 384]" = torch.ops.aten.reshape.default(mul_67, [512, 384]);  mul_67 = None
    mm_8: "f32[512, 54]" = torch.ops.aten.mm.default(view_297, permute_161)
    view_298: "f32[1, 512, 54]" = torch.ops.aten.reshape.default(mm_8, [1, 512, 54]);  mm_8 = None
    add_101: "f32[1, 512, 54]" = torch.ops.aten.add.Tensor(view_298, primals_203);  view_298 = primals_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_299: "f32[3072, 9, 1]" = torch.ops.aten.reshape.default(add_101, [-1, 9, 1]);  add_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    amax_16: "f32[3072, 1, 1]" = torch.ops.aten.amax.default(view_299, [1], True)
    sub_34: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(view_299, amax_16);  view_299 = amax_16 = None
    exp_16: "f32[3072, 9, 1]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
    sum_17: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [1], True)
    div_24: "f32[3072, 9, 1]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
    alias_16: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(div_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    permute_162: "f32[768, 384]" = torch.ops.aten.permute.default(primals_204, [1, 0]);  primals_204 = None
    
    # No stacktrace found for following nodes
    mm_default_9: "f32[512, 384]" = torch.ops.aten.mm.default(view_288, permute_162)
    add_tensor_9: "f32[512, 384]" = torch.ops.aten.add.Tensor(mm_default_9, primals_205);  mm_default_9 = primals_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_301: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(add_tensor_9, [1, 512, 384]);  add_tensor_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_302: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(view_301, [1, -1, 384]);  view_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    permute_163: "f32[1, 384, 512]" = torch.ops.aten.permute.default(view_302, [0, 2, 1]);  view_302 = None
    clone_24: "f32[1, 384, 512]" = torch.ops.aten.clone.default(permute_163, memory_format = torch.contiguous_format);  permute_163 = None
    unsqueeze_58: "f32[1, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_24, -1);  clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    constant_pad_nd_8: "f32[1, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_58, [0, 0, 4, 4], 0.0);  unsqueeze_58 = None
    index_8: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(constant_pad_nd_8, [None, None, unsqueeze_8, full_default_1]);  constant_pad_nd_8 = None
    permute_164: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index_8, [0, 1, 2, 4, 3, 5]);  index_8 = None
    view_303: "f32[1, 3456, 512]" = torch.ops.aten.reshape.default(permute_164, [1, 3456, 512]);  permute_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    permute_165: "f32[1, 512, 3456]" = torch.ops.aten.permute.default(view_303, [0, 2, 1]);  view_303 = None
    view_304: "f32[1, 512, 384, 9]" = torch.ops.aten.reshape.default(permute_165, [1, 512, 384, 9]);  permute_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    clone_25: "f32[1, 512, 384, 9]" = torch.ops.aten.clone.default(view_304, memory_format = torch.contiguous_format);  view_304 = None
    view_305: "f32[3072, 64, 9]" = torch.ops.aten.reshape.default(clone_25, [3072, 64, 9]);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    expand_49: "f32[3072, 64, 9]" = torch.ops.aten.expand.default(view_305, [3072, 64, 9]);  view_305 = None
    expand_50: "f32[3072, 9, 1]" = torch.ops.aten.expand.default(div_24, [3072, 9, 1]);  div_24 = None
    bmm_24: "f32[3072, 64, 1]" = torch.ops.aten.bmm.default(expand_49, expand_50)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    view_309: "f32[512, 384]" = torch.ops.aten.reshape.default(bmm_24, [-1, 384]);  bmm_24 = None
    
    # No stacktrace found for following nodes
    clone_default_9: "f32[1, 6, 512, 64]" = torch.ops.aten.clone.default(permute_157, memory_format = torch.contiguous_format);  permute_157 = None
    clone_default_10: "f32[1, 6, 512, 64]" = torch.ops.aten.clone.default(permute_158, memory_format = torch.contiguous_format);  permute_158 = None
    clone_default_11: "f32[1, 6, 512, 64]" = torch.ops.aten.clone.default(permute_159, memory_format = torch.contiguous_format);  permute_159 = None
    _scaled_dot_product_efficient_attention_default_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_9, clone_default_10, clone_default_11, None, True, 0.1, scale = 0.125)
    getitem_219: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_default_3[0]
    getitem_220: "f32[1, 6, 512]" = _scaled_dot_product_efficient_attention_default_3[1]
    getitem_221: "i64[]" = _scaled_dot_product_efficient_attention_default_3[2]
    getitem_222: "i64[]" = _scaled_dot_product_efficient_attention_default_3[3];  _scaled_dot_product_efficient_attention_default_3 = None
    alias_default_6: "f32[1, 6, 512, 64]" = torch.ops.aten.alias.default(getitem_219)
    alias_default_7: "f32[1, 6, 512, 64]" = torch.ops.aten.alias.default(alias_default_6);  alias_default_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_167: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_219, [0, 2, 1, 3]);  getitem_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_316: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_309, [1, -1, 6, 64]);  view_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    cat_8: "f32[1, 512, 12, 64]" = torch.ops.aten.cat.default([permute_167, view_316], 2);  permute_167 = view_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_317: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(cat_8, [1, 512, 768]);  cat_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_318: "f32[512, 768]" = torch.ops.aten.reshape.default(view_317, [512, 768]);  view_317 = None
    permute_168: "f32[768, 768]" = torch.ops.aten.permute.default(primals_206, [1, 0]);  primals_206 = None
    addmm_60: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_207, view_318, permute_168);  primals_207 = None
    view_319: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_60, [1, 512, 768]);  addmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    native_dropout_26 = torch.ops.aten.native_dropout.default(view_319, 0.1, True);  view_319 = None
    getitem_86: "f32[1, 512, 768]" = native_dropout_26[0]
    getitem_87: "b8[1, 512, 768]" = native_dropout_26[1];  native_dropout_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_105: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_86, add_99);  getitem_86 = add_99 = None
    var_mean_17 = torch.ops.aten.var_mean.correction(add_105, [2], correction = 0, keepdim = True)
    getitem_88: "f32[1, 512, 1]" = var_mean_17[0]
    getitem_89: "f32[1, 512, 1]" = var_mean_17[1];  var_mean_17 = None
    add_106: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-12);  getitem_88 = None
    rsqrt_17: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
    sub_36: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_105, getitem_89);  add_105 = getitem_89 = None
    mul_68: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_17);  sub_36 = None
    mul_69: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_68, primals_208)
    add_107: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_69, primals_209);  mul_69 = primals_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_320: "f32[512, 768]" = torch.ops.aten.reshape.default(add_107, [512, 768])
    permute_169: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_210, [1, 0]);  primals_210 = None
    addmm_61: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_211, view_320, permute_169);  primals_211 = None
    view_321: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_61, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_70: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_321, 0.5)
    mul_71: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_321, 0.7071067811865476);  view_321 = None
    erf_8: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_71);  mul_71 = None
    add_108: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_72: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_70, add_108);  mul_70 = add_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_322: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_72, [512, 3072]);  mul_72 = None
    permute_170: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_212, [1, 0]);  primals_212 = None
    addmm_62: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_213, view_322, permute_170);  primals_213 = None
    view_323: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_62, [1, 512, 768]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    native_dropout_27 = torch.ops.aten.native_dropout.default(view_323, 0.1, True);  view_323 = None
    getitem_90: "f32[1, 512, 768]" = native_dropout_27[0]
    getitem_91: "b8[1, 512, 768]" = native_dropout_27[1];  native_dropout_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_109: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_90, add_107);  getitem_90 = add_107 = None
    var_mean_18 = torch.ops.aten.var_mean.correction(add_109, [2], correction = 0, keepdim = True)
    getitem_92: "f32[1, 512, 1]" = var_mean_18[0]
    getitem_93: "f32[1, 512, 1]" = var_mean_18[1];  var_mean_18 = None
    add_110: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-12);  getitem_92 = None
    rsqrt_18: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
    sub_37: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_109, getitem_93);  add_109 = getitem_93 = None
    mul_73: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_18);  sub_37 = None
    mul_74: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_73, primals_214)
    add_111: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_74, primals_215);  mul_74 = primals_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_324: "f32[512, 768]" = torch.ops.aten.reshape.default(add_111, [512, 768])
    permute_171: "f32[768, 384]" = torch.ops.aten.permute.default(primals_216, [1, 0]);  primals_216 = None
    addmm_63: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_217, view_324, permute_171);  primals_217 = None
    view_325: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(addmm_63, [1, 512, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    permute_172: "f32[768, 384]" = torch.ops.aten.permute.default(primals_218, [1, 0]);  primals_218 = None
    
    # No stacktrace found for following nodes
    mm_default_8: "f32[512, 384]" = torch.ops.aten.mm.default(view_324, permute_172)
    add_tensor_8: "f32[512, 384]" = torch.ops.aten.add.Tensor(mm_default_8, primals_219);  mm_default_8 = primals_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_327: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(add_tensor_8, [1, 512, 384]);  add_tensor_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    permute_173: "f32[768, 384]" = torch.ops.aten.permute.default(primals_220, [1, 0]);  primals_220 = None
    
    # No stacktrace found for following nodes
    mm_default_7: "f32[512, 384]" = torch.ops.aten.mm.default(view_324, permute_173)
    add_tensor_7: "f32[512, 384]" = torch.ops.aten.add.Tensor(mm_default_7, primals_221);  mm_default_7 = primals_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_329: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(add_tensor_7, [1, 512, 384]);  add_tensor_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_174: "f32[1, 768, 512]" = torch.ops.aten.permute.default(add_111, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_18: "f32[1, 768, 512]" = torch.ops.aten.convolution.default(permute_174, primals_222, None, [1], [4], [1], False, [0], 768)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_19: "f32[1, 384, 512]" = torch.ops.aten.convolution.default(convolution_18, primals_223, None, [1], [0], [1], False, [0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_112: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_19, primals_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_330: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_325, [1, 512, 6, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_176: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_331: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_327, [1, 512, 6, 64]);  view_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_177: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_331, [0, 2, 1, 3]);  view_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_332: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_329, [1, 512, 6, 64]);  view_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_178: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_332, [0, 2, 1, 3]);  view_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_179: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_112, [0, 2, 1]);  add_112 = None
    mul_75: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(permute_179, view_325);  permute_179 = view_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    permute_180: "f32[384, 54]" = torch.ops.aten.permute.default(primals_224, [1, 0]);  primals_224 = None
    view_333: "f32[512, 384]" = torch.ops.aten.reshape.default(mul_75, [512, 384]);  mul_75 = None
    mm_9: "f32[512, 54]" = torch.ops.aten.mm.default(view_333, permute_180)
    view_334: "f32[1, 512, 54]" = torch.ops.aten.reshape.default(mm_9, [1, 512, 54]);  mm_9 = None
    add_113: "f32[1, 512, 54]" = torch.ops.aten.add.Tensor(view_334, primals_225);  view_334 = primals_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_335: "f32[3072, 9, 1]" = torch.ops.aten.reshape.default(add_113, [-1, 9, 1]);  add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    amax_18: "f32[3072, 1, 1]" = torch.ops.aten.amax.default(view_335, [1], True)
    sub_38: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(view_335, amax_18);  view_335 = amax_18 = None
    exp_18: "f32[3072, 9, 1]" = torch.ops.aten.exp.default(sub_38);  sub_38 = None
    sum_19: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [1], True)
    div_27: "f32[3072, 9, 1]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
    alias_18: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(div_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    permute_181: "f32[768, 384]" = torch.ops.aten.permute.default(primals_226, [1, 0]);  primals_226 = None
    
    # No stacktrace found for following nodes
    mm_default_6: "f32[512, 384]" = torch.ops.aten.mm.default(view_324, permute_181)
    add_tensor_6: "f32[512, 384]" = torch.ops.aten.add.Tensor(mm_default_6, primals_227);  mm_default_6 = primals_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_337: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(add_tensor_6, [1, 512, 384]);  add_tensor_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_338: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(view_337, [1, -1, 384]);  view_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    permute_182: "f32[1, 384, 512]" = torch.ops.aten.permute.default(view_338, [0, 2, 1]);  view_338 = None
    clone_27: "f32[1, 384, 512]" = torch.ops.aten.clone.default(permute_182, memory_format = torch.contiguous_format);  permute_182 = None
    unsqueeze_65: "f32[1, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_27, -1);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    constant_pad_nd_9: "f32[1, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_65, [0, 0, 4, 4], 0.0);  unsqueeze_65 = None
    index_9: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(constant_pad_nd_9, [None, None, unsqueeze_8, full_default_1]);  constant_pad_nd_9 = None
    permute_183: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index_9, [0, 1, 2, 4, 3, 5]);  index_9 = None
    view_339: "f32[1, 3456, 512]" = torch.ops.aten.reshape.default(permute_183, [1, 3456, 512]);  permute_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    permute_184: "f32[1, 512, 3456]" = torch.ops.aten.permute.default(view_339, [0, 2, 1]);  view_339 = None
    view_340: "f32[1, 512, 384, 9]" = torch.ops.aten.reshape.default(permute_184, [1, 512, 384, 9]);  permute_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    clone_28: "f32[1, 512, 384, 9]" = torch.ops.aten.clone.default(view_340, memory_format = torch.contiguous_format);  view_340 = None
    view_341: "f32[3072, 64, 9]" = torch.ops.aten.reshape.default(clone_28, [3072, 64, 9]);  clone_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    expand_55: "f32[3072, 64, 9]" = torch.ops.aten.expand.default(view_341, [3072, 64, 9]);  view_341 = None
    expand_56: "f32[3072, 9, 1]" = torch.ops.aten.expand.default(div_27, [3072, 9, 1]);  div_27 = None
    bmm_27: "f32[3072, 64, 1]" = torch.ops.aten.bmm.default(expand_55, expand_56)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    view_345: "f32[512, 384]" = torch.ops.aten.reshape.default(bmm_27, [-1, 384]);  bmm_27 = None
    
    # No stacktrace found for following nodes
    clone_default_6: "f32[1, 6, 512, 64]" = torch.ops.aten.clone.default(permute_176, memory_format = torch.contiguous_format);  permute_176 = None
    clone_default_7: "f32[1, 6, 512, 64]" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
    clone_default_8: "f32[1, 6, 512, 64]" = torch.ops.aten.clone.default(permute_178, memory_format = torch.contiguous_format);  permute_178 = None
    _scaled_dot_product_efficient_attention_default_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_6, clone_default_7, clone_default_8, None, True, 0.1, scale = 0.125)
    getitem_212: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_default_2[0]
    getitem_213: "f32[1, 6, 512]" = _scaled_dot_product_efficient_attention_default_2[1]
    getitem_214: "i64[]" = _scaled_dot_product_efficient_attention_default_2[2]
    getitem_215: "i64[]" = _scaled_dot_product_efficient_attention_default_2[3];  _scaled_dot_product_efficient_attention_default_2 = None
    alias_default_4: "f32[1, 6, 512, 64]" = torch.ops.aten.alias.default(getitem_212)
    alias_default_5: "f32[1, 6, 512, 64]" = torch.ops.aten.alias.default(alias_default_4);  alias_default_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_186: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_212, [0, 2, 1, 3]);  getitem_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_352: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_345, [1, -1, 6, 64]);  view_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    cat_9: "f32[1, 512, 12, 64]" = torch.ops.aten.cat.default([permute_186, view_352], 2);  permute_186 = view_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_353: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(cat_9, [1, 512, 768]);  cat_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_354: "f32[512, 768]" = torch.ops.aten.reshape.default(view_353, [512, 768]);  view_353 = None
    permute_187: "f32[768, 768]" = torch.ops.aten.permute.default(primals_228, [1, 0]);  primals_228 = None
    addmm_67: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_229, view_354, permute_187);  primals_229 = None
    view_355: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_67, [1, 512, 768]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    native_dropout_29 = torch.ops.aten.native_dropout.default(view_355, 0.1, True);  view_355 = None
    getitem_96: "f32[1, 512, 768]" = native_dropout_29[0]
    getitem_97: "b8[1, 512, 768]" = native_dropout_29[1];  native_dropout_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_117: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_96, add_111);  getitem_96 = add_111 = None
    var_mean_19 = torch.ops.aten.var_mean.correction(add_117, [2], correction = 0, keepdim = True)
    getitem_98: "f32[1, 512, 1]" = var_mean_19[0]
    getitem_99: "f32[1, 512, 1]" = var_mean_19[1];  var_mean_19 = None
    add_118: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-12);  getitem_98 = None
    rsqrt_19: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
    sub_40: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_117, getitem_99);  add_117 = getitem_99 = None
    mul_76: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_19);  sub_40 = None
    mul_77: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_76, primals_230)
    add_119: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_77, primals_231);  mul_77 = primals_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_356: "f32[512, 768]" = torch.ops.aten.reshape.default(add_119, [512, 768])
    permute_188: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_232, [1, 0]);  primals_232 = None
    addmm_68: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_233, view_356, permute_188);  primals_233 = None
    view_357: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_68, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_78: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_357, 0.5)
    mul_79: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_357, 0.7071067811865476);  view_357 = None
    erf_9: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_79);  mul_79 = None
    add_120: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_80: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_78, add_120);  mul_78 = add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_358: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_80, [512, 3072]);  mul_80 = None
    permute_189: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_234, [1, 0]);  primals_234 = None
    addmm_69: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_235, view_358, permute_189);  primals_235 = None
    view_359: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_69, [1, 512, 768]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    native_dropout_30 = torch.ops.aten.native_dropout.default(view_359, 0.1, True);  view_359 = None
    getitem_100: "f32[1, 512, 768]" = native_dropout_30[0]
    getitem_101: "b8[1, 512, 768]" = native_dropout_30[1];  native_dropout_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_121: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_100, add_119);  getitem_100 = add_119 = None
    var_mean_20 = torch.ops.aten.var_mean.correction(add_121, [2], correction = 0, keepdim = True)
    getitem_102: "f32[1, 512, 1]" = var_mean_20[0]
    getitem_103: "f32[1, 512, 1]" = var_mean_20[1];  var_mean_20 = None
    add_122: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-12);  getitem_102 = None
    rsqrt_20: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
    sub_41: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_121, getitem_103);  add_121 = getitem_103 = None
    mul_81: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_20);  sub_41 = None
    mul_82: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_81, primals_236)
    add_123: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_82, primals_237);  mul_82 = primals_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_360: "f32[512, 768]" = torch.ops.aten.reshape.default(add_123, [512, 768])
    permute_190: "f32[768, 384]" = torch.ops.aten.permute.default(primals_238, [1, 0]);  primals_238 = None
    addmm_70: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_239, view_360, permute_190);  primals_239 = None
    view_361: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(addmm_70, [1, 512, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    permute_191: "f32[768, 384]" = torch.ops.aten.permute.default(primals_240, [1, 0]);  primals_240 = None
    
    # No stacktrace found for following nodes
    mm_default_5: "f32[512, 384]" = torch.ops.aten.mm.default(view_360, permute_191)
    add_tensor_5: "f32[512, 384]" = torch.ops.aten.add.Tensor(mm_default_5, primals_241);  mm_default_5 = primals_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_363: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(add_tensor_5, [1, 512, 384]);  add_tensor_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    permute_192: "f32[768, 384]" = torch.ops.aten.permute.default(primals_242, [1, 0]);  primals_242 = None
    
    # No stacktrace found for following nodes
    mm_default_4: "f32[512, 384]" = torch.ops.aten.mm.default(view_360, permute_192)
    add_tensor_4: "f32[512, 384]" = torch.ops.aten.add.Tensor(mm_default_4, primals_243);  mm_default_4 = primals_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_365: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(add_tensor_4, [1, 512, 384]);  add_tensor_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_193: "f32[1, 768, 512]" = torch.ops.aten.permute.default(add_123, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_20: "f32[1, 768, 512]" = torch.ops.aten.convolution.default(permute_193, primals_244, None, [1], [4], [1], False, [0], 768)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_21: "f32[1, 384, 512]" = torch.ops.aten.convolution.default(convolution_20, primals_245, None, [1], [0], [1], False, [0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_124: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_21, primals_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_366: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_361, [1, 512, 6, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_195: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_366, [0, 2, 1, 3]);  view_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_367: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_363, [1, 512, 6, 64]);  view_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_196: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_367, [0, 2, 1, 3]);  view_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_368: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_365, [1, 512, 6, 64]);  view_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_197: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_368, [0, 2, 1, 3]);  view_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_198: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_124, [0, 2, 1]);  add_124 = None
    mul_83: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(permute_198, view_361);  permute_198 = view_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    permute_199: "f32[384, 54]" = torch.ops.aten.permute.default(primals_246, [1, 0]);  primals_246 = None
    view_369: "f32[512, 384]" = torch.ops.aten.reshape.default(mul_83, [512, 384]);  mul_83 = None
    mm_10: "f32[512, 54]" = torch.ops.aten.mm.default(view_369, permute_199)
    view_370: "f32[1, 512, 54]" = torch.ops.aten.reshape.default(mm_10, [1, 512, 54]);  mm_10 = None
    add_125: "f32[1, 512, 54]" = torch.ops.aten.add.Tensor(view_370, primals_247);  view_370 = primals_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_371: "f32[3072, 9, 1]" = torch.ops.aten.reshape.default(add_125, [-1, 9, 1]);  add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    amax_20: "f32[3072, 1, 1]" = torch.ops.aten.amax.default(view_371, [1], True)
    sub_42: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(view_371, amax_20);  view_371 = amax_20 = None
    exp_20: "f32[3072, 9, 1]" = torch.ops.aten.exp.default(sub_42);  sub_42 = None
    sum_21: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [1], True)
    div_30: "f32[3072, 9, 1]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
    alias_20: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(div_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    permute_200: "f32[768, 384]" = torch.ops.aten.permute.default(primals_248, [1, 0]);  primals_248 = None
    
    # No stacktrace found for following nodes
    mm_default_3: "f32[512, 384]" = torch.ops.aten.mm.default(view_360, permute_200)
    add_tensor_3: "f32[512, 384]" = torch.ops.aten.add.Tensor(mm_default_3, primals_249);  mm_default_3 = primals_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_373: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(add_tensor_3, [1, 512, 384]);  add_tensor_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_374: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(view_373, [1, -1, 384]);  view_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    permute_201: "f32[1, 384, 512]" = torch.ops.aten.permute.default(view_374, [0, 2, 1]);  view_374 = None
    clone_30: "f32[1, 384, 512]" = torch.ops.aten.clone.default(permute_201, memory_format = torch.contiguous_format);  permute_201 = None
    unsqueeze_72: "f32[1, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_30, -1);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    constant_pad_nd_10: "f32[1, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_72, [0, 0, 4, 4], 0.0);  unsqueeze_72 = None
    index_10: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(constant_pad_nd_10, [None, None, unsqueeze_8, full_default_1]);  constant_pad_nd_10 = None
    permute_202: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index_10, [0, 1, 2, 4, 3, 5]);  index_10 = None
    view_375: "f32[1, 3456, 512]" = torch.ops.aten.reshape.default(permute_202, [1, 3456, 512]);  permute_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    permute_203: "f32[1, 512, 3456]" = torch.ops.aten.permute.default(view_375, [0, 2, 1]);  view_375 = None
    view_376: "f32[1, 512, 384, 9]" = torch.ops.aten.reshape.default(permute_203, [1, 512, 384, 9]);  permute_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    clone_31: "f32[1, 512, 384, 9]" = torch.ops.aten.clone.default(view_376, memory_format = torch.contiguous_format);  view_376 = None
    view_377: "f32[3072, 64, 9]" = torch.ops.aten.reshape.default(clone_31, [3072, 64, 9]);  clone_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    expand_61: "f32[3072, 64, 9]" = torch.ops.aten.expand.default(view_377, [3072, 64, 9]);  view_377 = None
    expand_62: "f32[3072, 9, 1]" = torch.ops.aten.expand.default(div_30, [3072, 9, 1]);  div_30 = None
    bmm_30: "f32[3072, 64, 1]" = torch.ops.aten.bmm.default(expand_61, expand_62)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    view_381: "f32[512, 384]" = torch.ops.aten.reshape.default(bmm_30, [-1, 384]);  bmm_30 = None
    
    # No stacktrace found for following nodes
    clone_default_3: "f32[1, 6, 512, 64]" = torch.ops.aten.clone.default(permute_195, memory_format = torch.contiguous_format);  permute_195 = None
    clone_default_4: "f32[1, 6, 512, 64]" = torch.ops.aten.clone.default(permute_196, memory_format = torch.contiguous_format);  permute_196 = None
    clone_default_5: "f32[1, 6, 512, 64]" = torch.ops.aten.clone.default(permute_197, memory_format = torch.contiguous_format);  permute_197 = None
    _scaled_dot_product_efficient_attention_default_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_3, clone_default_4, clone_default_5, None, True, 0.1, scale = 0.125)
    getitem_205: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_default_1[0]
    getitem_206: "f32[1, 6, 512]" = _scaled_dot_product_efficient_attention_default_1[1]
    getitem_207: "i64[]" = _scaled_dot_product_efficient_attention_default_1[2]
    getitem_208: "i64[]" = _scaled_dot_product_efficient_attention_default_1[3];  _scaled_dot_product_efficient_attention_default_1 = None
    alias_default_2: "f32[1, 6, 512, 64]" = torch.ops.aten.alias.default(getitem_205)
    alias_default_3: "f32[1, 6, 512, 64]" = torch.ops.aten.alias.default(alias_default_2);  alias_default_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_205: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_205, [0, 2, 1, 3]);  getitem_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_388: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_381, [1, -1, 6, 64]);  view_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    cat_10: "f32[1, 512, 12, 64]" = torch.ops.aten.cat.default([permute_205, view_388], 2);  permute_205 = view_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_389: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(cat_10, [1, 512, 768]);  cat_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_390: "f32[512, 768]" = torch.ops.aten.reshape.default(view_389, [512, 768]);  view_389 = None
    permute_206: "f32[768, 768]" = torch.ops.aten.permute.default(primals_250, [1, 0]);  primals_250 = None
    addmm_74: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_251, view_390, permute_206);  primals_251 = None
    view_391: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_74, [1, 512, 768]);  addmm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    native_dropout_32 = torch.ops.aten.native_dropout.default(view_391, 0.1, True);  view_391 = None
    getitem_106: "f32[1, 512, 768]" = native_dropout_32[0]
    getitem_107: "b8[1, 512, 768]" = native_dropout_32[1];  native_dropout_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_129: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_106, add_123);  getitem_106 = add_123 = None
    var_mean_21 = torch.ops.aten.var_mean.correction(add_129, [2], correction = 0, keepdim = True)
    getitem_108: "f32[1, 512, 1]" = var_mean_21[0]
    getitem_109: "f32[1, 512, 1]" = var_mean_21[1];  var_mean_21 = None
    add_130: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-12);  getitem_108 = None
    rsqrt_21: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
    sub_44: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_129, getitem_109);  add_129 = getitem_109 = None
    mul_84: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_21);  sub_44 = None
    mul_85: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_84, primals_252)
    add_131: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_85, primals_253);  mul_85 = primals_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_392: "f32[512, 768]" = torch.ops.aten.reshape.default(add_131, [512, 768])
    permute_207: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_254, [1, 0]);  primals_254 = None
    addmm_75: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_255, view_392, permute_207);  primals_255 = None
    view_393: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_75, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_86: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_393, 0.5)
    mul_87: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_393, 0.7071067811865476);  view_393 = None
    erf_10: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
    add_132: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_88: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_86, add_132);  mul_86 = add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_394: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_88, [512, 3072]);  mul_88 = None
    permute_208: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_256, [1, 0]);  primals_256 = None
    addmm_76: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_257, view_394, permute_208);  primals_257 = None
    view_395: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_76, [1, 512, 768]);  addmm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    native_dropout_33 = torch.ops.aten.native_dropout.default(view_395, 0.1, True);  view_395 = None
    getitem_110: "f32[1, 512, 768]" = native_dropout_33[0]
    getitem_111: "b8[1, 512, 768]" = native_dropout_33[1];  native_dropout_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_133: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_110, add_131);  getitem_110 = add_131 = None
    var_mean_22 = torch.ops.aten.var_mean.correction(add_133, [2], correction = 0, keepdim = True)
    getitem_112: "f32[1, 512, 1]" = var_mean_22[0]
    getitem_113: "f32[1, 512, 1]" = var_mean_22[1];  var_mean_22 = None
    add_134: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-12);  getitem_112 = None
    rsqrt_22: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
    sub_45: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_133, getitem_113);  add_133 = getitem_113 = None
    mul_89: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_22);  sub_45 = None
    mul_90: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_89, primals_258)
    add_135: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_90, primals_259);  mul_90 = primals_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_396: "f32[512, 768]" = torch.ops.aten.reshape.default(add_135, [512, 768])
    permute_209: "f32[768, 384]" = torch.ops.aten.permute.default(primals_260, [1, 0]);  primals_260 = None
    addmm_77: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_261, view_396, permute_209);  primals_261 = None
    view_397: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(addmm_77, [1, 512, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    permute_210: "f32[768, 384]" = torch.ops.aten.permute.default(primals_262, [1, 0]);  primals_262 = None
    
    # No stacktrace found for following nodes
    mm_default_2: "f32[512, 384]" = torch.ops.aten.mm.default(view_396, permute_210)
    add_tensor_2: "f32[512, 384]" = torch.ops.aten.add.Tensor(mm_default_2, primals_263);  mm_default_2 = primals_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_399: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(add_tensor_2, [1, 512, 384]);  add_tensor_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    permute_211: "f32[768, 384]" = torch.ops.aten.permute.default(primals_264, [1, 0]);  primals_264 = None
    
    # No stacktrace found for following nodes
    mm_default_1: "f32[512, 384]" = torch.ops.aten.mm.default(view_396, permute_211)
    add_tensor_1: "f32[512, 384]" = torch.ops.aten.add.Tensor(mm_default_1, primals_265);  mm_default_1 = primals_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_401: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(add_tensor_1, [1, 512, 384]);  add_tensor_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_212: "f32[1, 768, 512]" = torch.ops.aten.permute.default(add_135, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_22: "f32[1, 768, 512]" = torch.ops.aten.convolution.default(permute_212, primals_266, None, [1], [4], [1], False, [0], 768)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_23: "f32[1, 384, 512]" = torch.ops.aten.convolution.default(convolution_22, primals_267, None, [1], [0], [1], False, [0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_136: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_23, primals_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_402: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_397, [1, 512, 6, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_214: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_403: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_399, [1, 512, 6, 64]);  view_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_215: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_403, [0, 2, 1, 3]);  view_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_404: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_401, [1, 512, 6, 64]);  view_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_216: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_404, [0, 2, 1, 3]);  view_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_217: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_136, [0, 2, 1]);  add_136 = None
    mul_91: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(permute_217, view_397);  permute_217 = view_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    permute_218: "f32[384, 54]" = torch.ops.aten.permute.default(primals_268, [1, 0]);  primals_268 = None
    view_405: "f32[512, 384]" = torch.ops.aten.reshape.default(mul_91, [512, 384]);  mul_91 = None
    mm_11: "f32[512, 54]" = torch.ops.aten.mm.default(view_405, permute_218)
    view_406: "f32[1, 512, 54]" = torch.ops.aten.reshape.default(mm_11, [1, 512, 54]);  mm_11 = None
    add_137: "f32[1, 512, 54]" = torch.ops.aten.add.Tensor(view_406, primals_269);  view_406 = primals_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_407: "f32[3072, 9, 1]" = torch.ops.aten.reshape.default(add_137, [-1, 9, 1]);  add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    amax_22: "f32[3072, 1, 1]" = torch.ops.aten.amax.default(view_407, [1], True)
    sub_46: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(view_407, amax_22);  view_407 = amax_22 = None
    exp_22: "f32[3072, 9, 1]" = torch.ops.aten.exp.default(sub_46);  sub_46 = None
    sum_23: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [1], True)
    div_33: "f32[3072, 9, 1]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
    alias_22: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(div_33)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    permute_219: "f32[768, 384]" = torch.ops.aten.permute.default(primals_270, [1, 0]);  primals_270 = None
    
    # No stacktrace found for following nodes
    mm_default: "f32[512, 384]" = torch.ops.aten.mm.default(view_396, permute_219)
    add_tensor: "f32[512, 384]" = torch.ops.aten.add.Tensor(mm_default, primals_271);  mm_default = primals_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_409: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(add_tensor, [1, 512, 384]);  add_tensor = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_410: "f32[1, 512, 384]" = torch.ops.aten.reshape.default(view_409, [1, -1, 384]);  view_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    permute_220: "f32[1, 384, 512]" = torch.ops.aten.permute.default(view_410, [0, 2, 1]);  view_410 = None
    clone_33: "f32[1, 384, 512]" = torch.ops.aten.clone.default(permute_220, memory_format = torch.contiguous_format);  permute_220 = None
    unsqueeze_79: "f32[1, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_33, -1);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    constant_pad_nd_11: "f32[1, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_79, [0, 0, 4, 4], 0.0);  unsqueeze_79 = None
    index_11: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(constant_pad_nd_11, [None, None, unsqueeze_8, full_default_1]);  constant_pad_nd_11 = None
    permute_221: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index_11, [0, 1, 2, 4, 3, 5]);  index_11 = None
    view_411: "f32[1, 3456, 512]" = torch.ops.aten.reshape.default(permute_221, [1, 3456, 512]);  permute_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    permute_222: "f32[1, 512, 3456]" = torch.ops.aten.permute.default(view_411, [0, 2, 1]);  view_411 = None
    view_412: "f32[1, 512, 384, 9]" = torch.ops.aten.reshape.default(permute_222, [1, 512, 384, 9]);  permute_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    clone_34: "f32[1, 512, 384, 9]" = torch.ops.aten.clone.default(view_412, memory_format = torch.contiguous_format);  view_412 = None
    view_413: "f32[3072, 64, 9]" = torch.ops.aten.reshape.default(clone_34, [3072, 64, 9]);  clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    expand_67: "f32[3072, 64, 9]" = torch.ops.aten.expand.default(view_413, [3072, 64, 9]);  view_413 = None
    expand_68: "f32[3072, 9, 1]" = torch.ops.aten.expand.default(div_33, [3072, 9, 1]);  div_33 = None
    bmm_33: "f32[3072, 64, 1]" = torch.ops.aten.bmm.default(expand_67, expand_68)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    view_417: "f32[512, 384]" = torch.ops.aten.reshape.default(bmm_33, [-1, 384]);  bmm_33 = None
    
    # No stacktrace found for following nodes
    clone_default: "f32[1, 6, 512, 64]" = torch.ops.aten.clone.default(permute_214, memory_format = torch.contiguous_format);  permute_214 = None
    clone_default_1: "f32[1, 6, 512, 64]" = torch.ops.aten.clone.default(permute_215, memory_format = torch.contiguous_format);  permute_215 = None
    clone_default_2: "f32[1, 6, 512, 64]" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None
    _scaled_dot_product_efficient_attention_default = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default, clone_default_1, clone_default_2, None, True, 0.1, scale = 0.125)
    getitem_198: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_default[0]
    getitem_199: "f32[1, 6, 512]" = _scaled_dot_product_efficient_attention_default[1]
    getitem_200: "i64[]" = _scaled_dot_product_efficient_attention_default[2]
    getitem_201: "i64[]" = _scaled_dot_product_efficient_attention_default[3];  _scaled_dot_product_efficient_attention_default = None
    alias_default: "f32[1, 6, 512, 64]" = torch.ops.aten.alias.default(getitem_198)
    alias_default_1: "f32[1, 6, 512, 64]" = torch.ops.aten.alias.default(alias_default);  alias_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_224: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_198, [0, 2, 1, 3]);  getitem_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_424: "f32[1, 512, 6, 64]" = torch.ops.aten.reshape.default(view_417, [1, -1, 6, 64]);  view_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    cat_11: "f32[1, 512, 12, 64]" = torch.ops.aten.cat.default([permute_224, view_424], 2);  permute_224 = view_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_425: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(cat_11, [1, 512, 768]);  cat_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_426: "f32[512, 768]" = torch.ops.aten.reshape.default(view_425, [512, 768]);  view_425 = None
    permute_225: "f32[768, 768]" = torch.ops.aten.permute.default(primals_272, [1, 0]);  primals_272 = None
    addmm_81: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_273, view_426, permute_225);  primals_273 = None
    view_427: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_81, [1, 512, 768]);  addmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    native_dropout_35 = torch.ops.aten.native_dropout.default(view_427, 0.1, True);  view_427 = None
    getitem_116: "f32[1, 512, 768]" = native_dropout_35[0]
    getitem_117: "b8[1, 512, 768]" = native_dropout_35[1];  native_dropout_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_141: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_116, add_135);  getitem_116 = add_135 = None
    var_mean_23 = torch.ops.aten.var_mean.correction(add_141, [2], correction = 0, keepdim = True)
    getitem_118: "f32[1, 512, 1]" = var_mean_23[0]
    getitem_119: "f32[1, 512, 1]" = var_mean_23[1];  var_mean_23 = None
    add_142: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-12);  getitem_118 = None
    rsqrt_23: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
    sub_48: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_141, getitem_119);  add_141 = getitem_119 = None
    mul_92: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_23);  sub_48 = None
    mul_93: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_92, primals_274)
    add_143: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_93, primals_275);  mul_93 = primals_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_428: "f32[512, 768]" = torch.ops.aten.reshape.default(add_143, [512, 768])
    permute_226: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_276, [1, 0]);  primals_276 = None
    addmm_82: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_277, view_428, permute_226);  primals_277 = None
    view_429: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_82, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_94: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_429, 0.5)
    mul_95: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_429, 0.7071067811865476);  view_429 = None
    erf_11: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_95);  mul_95 = None
    add_144: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_96: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_94, add_144);  mul_94 = add_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_430: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_96, [512, 3072]);  mul_96 = None
    permute_227: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_278, [1, 0]);  primals_278 = None
    addmm_83: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_279, view_430, permute_227);  primals_279 = None
    view_431: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_83, [1, 512, 768]);  addmm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    native_dropout_36 = torch.ops.aten.native_dropout.default(view_431, 0.1, True);  view_431 = None
    getitem_120: "f32[1, 512, 768]" = native_dropout_36[0]
    getitem_121: "b8[1, 512, 768]" = native_dropout_36[1];  native_dropout_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_145: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_120, add_143);  getitem_120 = add_143 = None
    var_mean_24 = torch.ops.aten.var_mean.correction(add_145, [2], correction = 0, keepdim = True)
    getitem_122: "f32[1, 512, 1]" = var_mean_24[0]
    getitem_123: "f32[1, 512, 1]" = var_mean_24[1];  var_mean_24 = None
    add_146: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-12);  getitem_122 = None
    rsqrt_24: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
    sub_49: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_145, getitem_123);  add_145 = getitem_123 = None
    mul_97: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_24);  sub_49 = None
    mul_98: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_97, primals_280)
    add_147: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_98, primals_281);  mul_98 = primals_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:873, code: hidden_states = self.dense(generator_hidden_states)
    view_432: "f32[512, 768]" = torch.ops.aten.reshape.default(add_147, [512, 768]);  add_147 = None
    permute_228: "f32[768, 768]" = torch.ops.aten.permute.default(primals_282, [1, 0]);  primals_282 = None
    addmm_84: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_283, view_432, permute_228);  primals_283 = None
    view_433: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_84, [1, 512, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_99: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_433, 0.5)
    mul_100: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_433, 0.7071067811865476);  view_433 = None
    erf_12: "f32[1, 512, 768]" = torch.ops.aten.erf.default(mul_100);  mul_100 = None
    add_148: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_101: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_99, add_148);  mul_99 = add_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:875, code: hidden_states = self.LayerNorm(hidden_states)
    var_mean_25 = torch.ops.aten.var_mean.correction(mul_101, [2], correction = 0, keepdim = True)
    getitem_124: "f32[1, 512, 1]" = var_mean_25[0]
    getitem_125: "f32[1, 512, 1]" = var_mean_25[1];  var_mean_25 = None
    add_149: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-12);  getitem_124 = None
    rsqrt_25: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
    sub_50: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_101, getitem_125);  mul_101 = getitem_125 = None
    mul_102: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_25);  sub_50 = None
    mul_103: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_102, primals_284)
    add_150: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_103, primals_285);  mul_103 = primals_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:941, code: prediction_scores = self.generator_lm_head(prediction_scores)
    view_434: "f32[512, 768]" = torch.ops.aten.reshape.default(add_150, [512, 768]);  add_150 = None
    permute_229: "f32[768, 30522]" = torch.ops.aten.permute.default(primals_286, [1, 0]);  primals_286 = None
    addmm_85: "f32[512, 30522]" = torch.ops.aten.addmm.default(primals_287, view_434, permute_229);  primals_287 = None
    view_435: "f32[1, 512, 30522]" = torch.ops.aten.reshape.default(addmm_85, [1, 512, 30522]);  addmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:947, code: loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    view_436: "f32[512, 30522]" = torch.ops.aten.reshape.default(view_435, [-1, 30522])
    view_437: "i64[512]" = torch.ops.aten.reshape.default(primals_291, [-1])
    amax_24: "f32[512, 1]" = torch.ops.aten.amax.default(view_436, [1], True)
    sub_51: "f32[512, 30522]" = torch.ops.aten.sub.Tensor(view_436, amax_24);  view_436 = amax_24 = None
    exp_24: "f32[512, 30522]" = torch.ops.aten.exp.default(sub_51)
    sum_25: "f32[512, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
    log: "f32[512, 1]" = torch.ops.aten.log.default(sum_25);  sum_25 = None
    sub_52: "f32[512, 30522]" = torch.ops.aten.sub.Tensor(sub_51, log);  sub_51 = log = None
    ne: "b8[512]" = torch.ops.aten.ne.Scalar(view_437, -100)
    full_default_13: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where: "i64[512]" = torch.ops.aten.where.self(ne, view_437, full_default_13);  view_437 = full_default_13 = None
    unsqueeze_86: "i64[512, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
    gather: "f32[512, 1]" = torch.ops.aten.gather.default(sub_52, 1, unsqueeze_86);  unsqueeze_86 = None
    squeeze: "f32[512]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[512]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    full_default_14: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where_1: "f32[512]" = torch.ops.aten.where.self(ne, neg, full_default_14);  neg = full_default_14 = None
    sum_26: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
    sum_27: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
    div_36: "f32[]" = torch.ops.aten.div.Tensor(sum_27, convert_element_type);  sum_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:941, code: prediction_scores = self.generator_lm_head(prediction_scores)
    permute_230: "f32[30522, 768]" = torch.ops.aten.permute.default(permute_229, [1, 0]);  permute_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:875, code: hidden_states = self.LayerNorm(hidden_states)
    div_38: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 768);  rsqrt_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:873, code: hidden_states = self.dense(generator_hidden_states)
    permute_234: "f32[768, 768]" = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_39: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    permute_238: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_227, [1, 0]);  permute_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    permute_242: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_226, [1, 0]);  permute_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_40: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    permute_246: "f32[768, 768]" = torch.ops.aten.permute.default(permute_225, [1, 0]);  permute_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    permute_256: "f32[3072, 9, 64]" = torch.ops.aten.permute.default(expand_67, [0, 2, 1]);  expand_67 = None
    permute_257: "f32[3072, 1, 9]" = torch.ops.aten.permute.default(expand_68, [0, 2, 1]);  expand_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    permute_261: "f32[384, 768]" = torch.ops.aten.permute.default(permute_219, [1, 0]);  permute_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    alias_27: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(alias_22);  alias_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    permute_275: "f32[384, 768]" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    permute_279: "f32[384, 768]" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    permute_283: "f32[384, 768]" = torch.ops.aten.permute.default(permute_209, [1, 0]);  permute_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_42: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    permute_287: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_208, [1, 0]);  permute_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    permute_291: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_43: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    permute_295: "f32[768, 768]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    permute_305: "f32[3072, 9, 64]" = torch.ops.aten.permute.default(expand_61, [0, 2, 1]);  expand_61 = None
    permute_306: "f32[3072, 1, 9]" = torch.ops.aten.permute.default(expand_62, [0, 2, 1]);  expand_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    permute_310: "f32[384, 768]" = torch.ops.aten.permute.default(permute_200, [1, 0]);  permute_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    alias_29: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(alias_20);  alias_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    permute_324: "f32[384, 768]" = torch.ops.aten.permute.default(permute_192, [1, 0]);  permute_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    permute_328: "f32[384, 768]" = torch.ops.aten.permute.default(permute_191, [1, 0]);  permute_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    permute_332: "f32[384, 768]" = torch.ops.aten.permute.default(permute_190, [1, 0]);  permute_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_45: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    permute_336: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_189, [1, 0]);  permute_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    permute_340: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_188, [1, 0]);  permute_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_46: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    permute_344: "f32[768, 768]" = torch.ops.aten.permute.default(permute_187, [1, 0]);  permute_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    permute_354: "f32[3072, 9, 64]" = torch.ops.aten.permute.default(expand_55, [0, 2, 1]);  expand_55 = None
    permute_355: "f32[3072, 1, 9]" = torch.ops.aten.permute.default(expand_56, [0, 2, 1]);  expand_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    permute_359: "f32[384, 768]" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    alias_31: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(alias_18);  alias_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    permute_373: "f32[384, 768]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    permute_377: "f32[384, 768]" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    permute_381: "f32[384, 768]" = torch.ops.aten.permute.default(permute_171, [1, 0]);  permute_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_48: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    permute_385: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_170, [1, 0]);  permute_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    permute_389: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_169, [1, 0]);  permute_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_49: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    permute_393: "f32[768, 768]" = torch.ops.aten.permute.default(permute_168, [1, 0]);  permute_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    permute_403: "f32[3072, 9, 64]" = torch.ops.aten.permute.default(expand_49, [0, 2, 1]);  expand_49 = None
    permute_404: "f32[3072, 1, 9]" = torch.ops.aten.permute.default(expand_50, [0, 2, 1]);  expand_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    permute_408: "f32[384, 768]" = torch.ops.aten.permute.default(permute_162, [1, 0]);  permute_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    alias_33: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(alias_16);  alias_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    permute_422: "f32[384, 768]" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    permute_426: "f32[384, 768]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    permute_430: "f32[384, 768]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_51: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    permute_434: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    permute_438: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_150, [1, 0]);  permute_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_52: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    permute_442: "f32[768, 768]" = torch.ops.aten.permute.default(permute_149, [1, 0]);  permute_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    permute_452: "f32[3072, 9, 64]" = torch.ops.aten.permute.default(expand_43, [0, 2, 1]);  expand_43 = None
    permute_453: "f32[3072, 1, 9]" = torch.ops.aten.permute.default(expand_44, [0, 2, 1]);  expand_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    permute_457: "f32[384, 768]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    alias_35: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    permute_471: "f32[384, 768]" = torch.ops.aten.permute.default(permute_135, [1, 0]);  permute_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    permute_475: "f32[384, 768]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    permute_479: "f32[384, 768]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_54: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    permute_483: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    permute_487: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_55: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    permute_491: "f32[768, 768]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    permute_501: "f32[3072, 9, 64]" = torch.ops.aten.permute.default(expand_37, [0, 2, 1]);  expand_37 = None
    permute_502: "f32[3072, 1, 9]" = torch.ops.aten.permute.default(expand_38, [0, 2, 1]);  expand_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    permute_506: "f32[384, 768]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    alias_37: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    permute_520: "f32[384, 768]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    permute_524: "f32[384, 768]" = torch.ops.aten.permute.default(permute_115, [1, 0]);  permute_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    permute_528: "f32[384, 768]" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_57: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    permute_532: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_113, [1, 0]);  permute_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    permute_536: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_58: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    permute_540: "f32[768, 768]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    permute_550: "f32[3072, 9, 64]" = torch.ops.aten.permute.default(expand_31, [0, 2, 1]);  expand_31 = None
    permute_551: "f32[3072, 1, 9]" = torch.ops.aten.permute.default(expand_32, [0, 2, 1]);  expand_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    permute_555: "f32[384, 768]" = torch.ops.aten.permute.default(permute_105, [1, 0]);  permute_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    alias_39: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    permute_569: "f32[384, 768]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    permute_573: "f32[384, 768]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    permute_577: "f32[384, 768]" = torch.ops.aten.permute.default(permute_95, [1, 0]);  permute_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_60: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    permute_581: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_94, [1, 0]);  permute_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    permute_585: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_61: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    permute_589: "f32[768, 768]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    permute_599: "f32[3072, 9, 64]" = torch.ops.aten.permute.default(expand_25, [0, 2, 1]);  expand_25 = None
    permute_600: "f32[3072, 1, 9]" = torch.ops.aten.permute.default(expand_26, [0, 2, 1]);  expand_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    permute_604: "f32[384, 768]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    alias_41: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    permute_618: "f32[384, 768]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    permute_622: "f32[384, 768]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    permute_626: "f32[384, 768]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_63: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    permute_630: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    permute_634: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_64: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    permute_638: "f32[768, 768]" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    permute_648: "f32[3072, 9, 64]" = torch.ops.aten.permute.default(expand_19, [0, 2, 1]);  expand_19 = None
    permute_649: "f32[3072, 1, 9]" = torch.ops.aten.permute.default(expand_20, [0, 2, 1]);  expand_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    permute_653: "f32[384, 768]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    alias_43: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    permute_667: "f32[384, 768]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    permute_671: "f32[384, 768]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    permute_675: "f32[384, 768]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_66: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    permute_679: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    permute_683: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_67: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    permute_687: "f32[768, 768]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    permute_697: "f32[3072, 9, 64]" = torch.ops.aten.permute.default(expand_13, [0, 2, 1]);  expand_13 = None
    permute_698: "f32[3072, 1, 9]" = torch.ops.aten.permute.default(expand_14, [0, 2, 1]);  expand_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    permute_702: "f32[384, 768]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    alias_45: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    permute_716: "f32[384, 768]" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    permute_720: "f32[384, 768]" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    permute_724: "f32[384, 768]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_69: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    permute_728: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    permute_732: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_70: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    permute_736: "f32[768, 768]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    permute_746: "f32[3072, 9, 64]" = torch.ops.aten.permute.default(expand_7, [0, 2, 1]);  expand_7 = None
    permute_747: "f32[3072, 1, 9]" = torch.ops.aten.permute.default(expand_8, [0, 2, 1]);  expand_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    permute_751: "f32[384, 768]" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    alias_47: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    permute_765: "f32[384, 768]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    permute_769: "f32[384, 768]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    permute_773: "f32[384, 768]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_72: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    permute_777: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    permute_781: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_73: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    permute_785: "f32[768, 768]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    permute_795: "f32[3072, 9, 64]" = torch.ops.aten.permute.default(expand_1, [0, 2, 1]);  expand_1 = None
    permute_796: "f32[3072, 1, 9]" = torch.ops.aten.permute.default(expand_2, [0, 2, 1]);  expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    permute_800: "f32[384, 768]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    alias_49: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(alias);  alias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    permute_814: "f32[384, 768]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    permute_818: "f32[384, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    permute_822: "f32[384, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:235, code: embeddings = self.LayerNorm(embeddings)
    div_75: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
    return [div_36, view_435, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_16, primals_24, primals_25, primals_32, primals_38, primals_46, primals_47, primals_54, primals_60, primals_68, primals_69, primals_76, primals_82, primals_90, primals_91, primals_98, primals_104, primals_112, primals_113, primals_120, primals_126, primals_134, primals_135, primals_142, primals_148, primals_156, primals_157, primals_164, primals_170, primals_178, primals_179, primals_186, primals_192, primals_200, primals_201, primals_208, primals_214, primals_222, primals_223, primals_230, primals_236, primals_244, primals_245, primals_252, primals_258, primals_266, primals_267, primals_274, primals_280, primals_284, primals_290, primals_291, expand, slice_4, mul_1, getitem_3, view, addmm, permute_3, convolution, convolution_1, permute_9, view_9, full_default_1, unsqueeze_8, clone_default_33, clone_default_34, clone_default_35, getitem_276, getitem_277, getitem_278, alias_default_23, view_30, getitem_7, mul_4, view_32, addmm_5, view_34, getitem_11, mul_9, view_36, addmm_7, permute_22, convolution_2, convolution_3, permute_28, view_45, clone_default_30, clone_default_31, clone_default_32, getitem_269, getitem_270, getitem_271, alias_default_21, view_66, getitem_17, mul_12, view_68, addmm_12, view_70, getitem_21, mul_17, view_72, addmm_14, permute_41, convolution_4, convolution_5, permute_47, view_81, clone_default_27, clone_default_28, clone_default_29, getitem_262, getitem_263, getitem_264, alias_default_19, view_102, getitem_27, mul_20, view_104, addmm_19, view_106, getitem_31, mul_25, view_108, addmm_21, permute_60, convolution_6, convolution_7, permute_66, view_117, clone_default_24, clone_default_25, clone_default_26, getitem_255, getitem_256, getitem_257, alias_default_17, view_138, getitem_37, mul_28, view_140, addmm_26, view_142, getitem_41, mul_33, view_144, addmm_28, permute_79, convolution_8, convolution_9, permute_85, view_153, clone_default_21, clone_default_22, clone_default_23, getitem_248, getitem_249, getitem_250, alias_default_15, view_174, getitem_47, mul_36, view_176, addmm_33, view_178, getitem_51, mul_41, view_180, addmm_35, permute_98, convolution_10, convolution_11, permute_104, view_189, clone_default_18, clone_default_19, clone_default_20, getitem_241, getitem_242, getitem_243, alias_default_13, view_210, getitem_57, mul_44, view_212, addmm_40, view_214, getitem_61, mul_49, view_216, addmm_42, permute_117, convolution_12, convolution_13, permute_123, view_225, clone_default_15, clone_default_16, clone_default_17, getitem_234, getitem_235, getitem_236, alias_default_11, view_246, getitem_67, mul_52, view_248, addmm_47, view_250, getitem_71, mul_57, view_252, addmm_49, permute_136, convolution_14, convolution_15, permute_142, view_261, clone_default_12, clone_default_13, clone_default_14, getitem_227, getitem_228, getitem_229, alias_default_9, view_282, getitem_77, mul_60, view_284, addmm_54, view_286, getitem_81, mul_65, view_288, addmm_56, permute_155, convolution_16, convolution_17, permute_161, view_297, clone_default_9, clone_default_10, clone_default_11, getitem_220, getitem_221, getitem_222, alias_default_7, view_318, getitem_87, mul_68, view_320, addmm_61, view_322, getitem_91, mul_73, view_324, addmm_63, permute_174, convolution_18, convolution_19, permute_180, view_333, clone_default_6, clone_default_7, clone_default_8, getitem_213, getitem_214, getitem_215, alias_default_5, view_354, getitem_97, mul_76, view_356, addmm_68, view_358, getitem_101, mul_81, view_360, addmm_70, permute_193, convolution_20, convolution_21, permute_199, view_369, clone_default_3, clone_default_4, clone_default_5, getitem_206, getitem_207, getitem_208, alias_default_3, view_390, getitem_107, mul_84, view_392, addmm_75, view_394, getitem_111, mul_89, view_396, addmm_77, permute_212, convolution_22, convolution_23, permute_218, view_405, clone_default, clone_default_1, clone_default_2, getitem_199, getitem_200, getitem_201, alias_default_1, view_426, getitem_117, mul_92, view_428, addmm_82, view_430, getitem_121, mul_97, view_432, addmm_84, mul_102, view_434, sub_52, convert_element_type, permute_230, div_38, permute_234, div_39, permute_238, permute_242, div_40, permute_246, permute_256, permute_257, permute_261, alias_27, permute_275, permute_279, permute_283, div_42, permute_287, permute_291, div_43, permute_295, permute_305, permute_306, permute_310, alias_29, permute_324, permute_328, permute_332, div_45, permute_336, permute_340, div_46, permute_344, permute_354, permute_355, permute_359, alias_31, permute_373, permute_377, permute_381, div_48, permute_385, permute_389, div_49, permute_393, permute_403, permute_404, permute_408, alias_33, permute_422, permute_426, permute_430, div_51, permute_434, permute_438, div_52, permute_442, permute_452, permute_453, permute_457, alias_35, permute_471, permute_475, permute_479, div_54, permute_483, permute_487, div_55, permute_491, permute_501, permute_502, permute_506, alias_37, permute_520, permute_524, permute_528, div_57, permute_532, permute_536, div_58, permute_540, permute_550, permute_551, permute_555, alias_39, permute_569, permute_573, permute_577, div_60, permute_581, permute_585, div_61, permute_589, permute_599, permute_600, permute_604, alias_41, permute_618, permute_622, permute_626, div_63, permute_630, permute_634, div_64, permute_638, permute_648, permute_649, permute_653, alias_43, permute_667, permute_671, permute_675, div_66, permute_679, permute_683, div_67, permute_687, permute_697, permute_698, permute_702, alias_45, permute_716, permute_720, permute_724, div_69, permute_728, permute_732, div_70, permute_736, permute_746, permute_747, permute_751, alias_47, permute_765, permute_769, permute_773, div_72, permute_777, permute_781, div_73, permute_785, permute_795, permute_796, permute_800, alias_49, permute_814, permute_818, permute_822, div_75]
    