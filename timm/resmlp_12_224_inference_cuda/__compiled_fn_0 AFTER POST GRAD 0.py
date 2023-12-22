from __future__ import annotations



def forward(self, arg0_1: "f32[384]", arg1_1: "f32[1, 1, 384]", arg2_1: "f32[1, 1, 384]", arg3_1: "f32[384]", arg4_1: "f32[1, 1, 384]", arg5_1: "f32[1, 1, 384]", arg6_1: "f32[384]", arg7_1: "f32[1, 1, 384]", arg8_1: "f32[1, 1, 384]", arg9_1: "f32[384]", arg10_1: "f32[1, 1, 384]", arg11_1: "f32[1, 1, 384]", arg12_1: "f32[384]", arg13_1: "f32[1, 1, 384]", arg14_1: "f32[1, 1, 384]", arg15_1: "f32[384]", arg16_1: "f32[1, 1, 384]", arg17_1: "f32[1, 1, 384]", arg18_1: "f32[384]", arg19_1: "f32[1, 1, 384]", arg20_1: "f32[1, 1, 384]", arg21_1: "f32[384]", arg22_1: "f32[1, 1, 384]", arg23_1: "f32[1, 1, 384]", arg24_1: "f32[384]", arg25_1: "f32[1, 1, 384]", arg26_1: "f32[1, 1, 384]", arg27_1: "f32[384]", arg28_1: "f32[1, 1, 384]", arg29_1: "f32[1, 1, 384]", arg30_1: "f32[384]", arg31_1: "f32[1, 1, 384]", arg32_1: "f32[1, 1, 384]", arg33_1: "f32[384]", arg34_1: "f32[1, 1, 384]", arg35_1: "f32[1, 1, 384]", arg36_1: "f32[384]", arg37_1: "f32[1, 1, 384]", arg38_1: "f32[1, 1, 384]", arg39_1: "f32[384]", arg40_1: "f32[1, 1, 384]", arg41_1: "f32[1, 1, 384]", arg42_1: "f32[384]", arg43_1: "f32[1, 1, 384]", arg44_1: "f32[1, 1, 384]", arg45_1: "f32[384]", arg46_1: "f32[1, 1, 384]", arg47_1: "f32[1, 1, 384]", arg48_1: "f32[384]", arg49_1: "f32[1, 1, 384]", arg50_1: "f32[1, 1, 384]", arg51_1: "f32[384]", arg52_1: "f32[1, 1, 384]", arg53_1: "f32[1, 1, 384]", arg54_1: "f32[384]", arg55_1: "f32[1, 1, 384]", arg56_1: "f32[1, 1, 384]", arg57_1: "f32[384]", arg58_1: "f32[1, 1, 384]", arg59_1: "f32[1, 1, 384]", arg60_1: "f32[384]", arg61_1: "f32[1, 1, 384]", arg62_1: "f32[1, 1, 384]", arg63_1: "f32[384]", arg64_1: "f32[1, 1, 384]", arg65_1: "f32[1, 1, 384]", arg66_1: "f32[384]", arg67_1: "f32[1, 1, 384]", arg68_1: "f32[1, 1, 384]", arg69_1: "f32[384]", arg70_1: "f32[1, 1, 384]", arg71_1: "f32[1, 1, 384]", arg72_1: "f32[1, 1, 384]", arg73_1: "f32[1, 1, 384]", arg74_1: "f32[384, 3, 16, 16]", arg75_1: "f32[384]", arg76_1: "f32[196, 196]", arg77_1: "f32[196]", arg78_1: "f32[1536, 384]", arg79_1: "f32[1536]", arg80_1: "f32[384, 1536]", arg81_1: "f32[384]", arg82_1: "f32[196, 196]", arg83_1: "f32[196]", arg84_1: "f32[1536, 384]", arg85_1: "f32[1536]", arg86_1: "f32[384, 1536]", arg87_1: "f32[384]", arg88_1: "f32[196, 196]", arg89_1: "f32[196]", arg90_1: "f32[1536, 384]", arg91_1: "f32[1536]", arg92_1: "f32[384, 1536]", arg93_1: "f32[384]", arg94_1: "f32[196, 196]", arg95_1: "f32[196]", arg96_1: "f32[1536, 384]", arg97_1: "f32[1536]", arg98_1: "f32[384, 1536]", arg99_1: "f32[384]", arg100_1: "f32[196, 196]", arg101_1: "f32[196]", arg102_1: "f32[1536, 384]", arg103_1: "f32[1536]", arg104_1: "f32[384, 1536]", arg105_1: "f32[384]", arg106_1: "f32[196, 196]", arg107_1: "f32[196]", arg108_1: "f32[1536, 384]", arg109_1: "f32[1536]", arg110_1: "f32[384, 1536]", arg111_1: "f32[384]", arg112_1: "f32[196, 196]", arg113_1: "f32[196]", arg114_1: "f32[1536, 384]", arg115_1: "f32[1536]", arg116_1: "f32[384, 1536]", arg117_1: "f32[384]", arg118_1: "f32[196, 196]", arg119_1: "f32[196]", arg120_1: "f32[1536, 384]", arg121_1: "f32[1536]", arg122_1: "f32[384, 1536]", arg123_1: "f32[384]", arg124_1: "f32[196, 196]", arg125_1: "f32[196]", arg126_1: "f32[1536, 384]", arg127_1: "f32[1536]", arg128_1: "f32[384, 1536]", arg129_1: "f32[384]", arg130_1: "f32[196, 196]", arg131_1: "f32[196]", arg132_1: "f32[1536, 384]", arg133_1: "f32[1536]", arg134_1: "f32[384, 1536]", arg135_1: "f32[384]", arg136_1: "f32[196, 196]", arg137_1: "f32[196]", arg138_1: "f32[1536, 384]", arg139_1: "f32[1536]", arg140_1: "f32[384, 1536]", arg141_1: "f32[384]", arg142_1: "f32[196, 196]", arg143_1: "f32[196]", arg144_1: "f32[1536, 384]", arg145_1: "f32[1536]", arg146_1: "f32[384, 1536]", arg147_1: "f32[384]", arg148_1: "f32[1000, 384]", arg149_1: "f32[1000]", arg150_1: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_108: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg73_1, 1);  arg73_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_102: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg71_1, 1);  arg71_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_99: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg68_1, 1);  arg68_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_93: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg65_1, 1);  arg65_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_90: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg62_1, 1);  arg62_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_84: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg59_1, 1);  arg59_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_81: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg56_1, 1);  arg56_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_75: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg53_1, 1);  arg53_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_72: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg50_1, 1);  arg50_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_66: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg47_1, 1);  arg47_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_63: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg44_1, 1);  arg44_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_57: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg41_1, 1);  arg41_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_54: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg38_1, 1);  arg38_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_48: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg35_1, 1);  arg35_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_45: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg32_1, 1);  arg32_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_39: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg29_1, 1);  arg29_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_36: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg26_1, 1);  arg26_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_30: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg23_1, 1);  arg23_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_27: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg20_1, 1);  arg20_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_21: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg17_1, 1);  arg17_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_18: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg14_1, 1);  arg14_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_12: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg11_1, 1);  arg11_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_9: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg8_1, 1);  arg8_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_3: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg5_1, 1);  arg5_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg2_1, 1);  arg2_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(arg150_1, arg74_1, arg75_1, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  arg150_1 = arg74_1 = arg75_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    view: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(convolution, [8, 384, 196]);  convolution = None
    permute: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_1: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul, permute);  mul = None
    add: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg1_1, mul_1);  arg1_1 = mul_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_1: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add, [0, 2, 1]);  add = None
    view_1: "f32[3072, 196]" = torch.ops.aten.reshape.default(permute_1, [3072, 196]);  permute_1 = None
    permute_2: "f32[196, 196]" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
    
    # No stacktrace found for following nodes
    mm_default_23: "f32[3072, 196]" = torch.ops.aten.mm.default(view_1, permute_2);  view_1 = permute_2 = None
    add_tensor_23: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_23, arg77_1);  mm_default_23 = arg77_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    view_2: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_23, [8, 384, 196]);  add_tensor_23 = None
    permute_3: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_2, [0, 2, 1]);  view_2 = None
    mul_2: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg0_1, permute_3);  arg0_1 = permute_3 = None
    add_1: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(permute, mul_2);  permute = mul_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_4: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_3, add_1);  mul_3 = None
    add_2: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg4_1, mul_4);  arg4_1 = mul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    clone: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_2, memory_format = torch.contiguous_format);  add_2 = None
    view_3: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone, [1568, 384]);  clone = None
    permute_4: "f32[384, 1536]" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
    mm: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_3, permute_4);  view_3 = permute_4 = None
    view_4: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm, [8, 196, 1536]);  mm = None
    add_3: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_4, arg79_1);  view_4 = arg79_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_5: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_3, 0.5)
    mul_6: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_3, 0.7071067811865476);  add_3 = None
    erf: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
    add_4: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_7: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_5, add_4);  mul_5 = add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_5: "f32[1568, 1536]" = torch.ops.aten.reshape.default(mul_7, [1568, 1536]);  mul_7 = None
    permute_5: "f32[1536, 384]" = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
    
    # No stacktrace found for following nodes
    mm_default_22: "f32[1568, 384]" = torch.ops.aten.mm.default(view_5, permute_5);  view_5 = permute_5 = None
    add_tensor_22: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_22, arg81_1);  mm_default_22 = arg81_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_6: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_22, [8, 196, 384]);  add_tensor_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_8: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg3_1, view_6);  arg3_1 = view_6 = None
    add_5: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_1, mul_8);  add_1 = mul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_10: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_9, add_5);  mul_9 = None
    add_6: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg7_1, mul_10);  arg7_1 = mul_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_6: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_6, [0, 2, 1]);  add_6 = None
    view_7: "f32[3072, 196]" = torch.ops.aten.reshape.default(permute_6, [3072, 196]);  permute_6 = None
    permute_7: "f32[196, 196]" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
    
    # No stacktrace found for following nodes
    mm_default_21: "f32[3072, 196]" = torch.ops.aten.mm.default(view_7, permute_7);  view_7 = permute_7 = None
    add_tensor_21: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_21, arg83_1);  mm_default_21 = arg83_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    view_8: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_21, [8, 384, 196]);  add_tensor_21 = None
    permute_8: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_8, [0, 2, 1]);  view_8 = None
    mul_11: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg6_1, permute_8);  arg6_1 = permute_8 = None
    add_7: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_5, mul_11);  add_5 = mul_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_13: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_12, add_7);  mul_12 = None
    add_8: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg10_1, mul_13);  arg10_1 = mul_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    clone_3: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_8, memory_format = torch.contiguous_format);  add_8 = None
    view_9: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_3, [1568, 384]);  clone_3 = None
    permute_9: "f32[384, 1536]" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
    mm_1: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_9, permute_9);  view_9 = permute_9 = None
    view_10: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_1, [8, 196, 1536]);  mm_1 = None
    add_9: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_10, arg85_1);  view_10 = arg85_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_14: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_9, 0.5)
    mul_15: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_9, 0.7071067811865476);  add_9 = None
    erf_1: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_15);  mul_15 = None
    add_10: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_16: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_14, add_10);  mul_14 = add_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_11: "f32[1568, 1536]" = torch.ops.aten.reshape.default(mul_16, [1568, 1536]);  mul_16 = None
    permute_10: "f32[1536, 384]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
    
    # No stacktrace found for following nodes
    mm_default_20: "f32[1568, 384]" = torch.ops.aten.mm.default(view_11, permute_10);  view_11 = permute_10 = None
    add_tensor_20: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_20, arg87_1);  mm_default_20 = arg87_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_12: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_20, [8, 196, 384]);  add_tensor_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_17: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg9_1, view_12);  arg9_1 = view_12 = None
    add_11: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_7, mul_17);  add_7 = mul_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_19: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_18, add_11);  mul_18 = None
    add_12: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg13_1, mul_19);  arg13_1 = mul_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_11: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_12, [0, 2, 1]);  add_12 = None
    view_13: "f32[3072, 196]" = torch.ops.aten.reshape.default(permute_11, [3072, 196]);  permute_11 = None
    permute_12: "f32[196, 196]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
    
    # No stacktrace found for following nodes
    mm_default_19: "f32[3072, 196]" = torch.ops.aten.mm.default(view_13, permute_12);  view_13 = permute_12 = None
    add_tensor_19: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_19, arg89_1);  mm_default_19 = arg89_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    view_14: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_19, [8, 384, 196]);  add_tensor_19 = None
    permute_13: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_14, [0, 2, 1]);  view_14 = None
    mul_20: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg12_1, permute_13);  arg12_1 = permute_13 = None
    add_13: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_11, mul_20);  add_11 = mul_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_22: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_21, add_13);  mul_21 = None
    add_14: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg16_1, mul_22);  arg16_1 = mul_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    clone_6: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_14, memory_format = torch.contiguous_format);  add_14 = None
    view_15: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_6, [1568, 384]);  clone_6 = None
    permute_14: "f32[384, 1536]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
    mm_2: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_15, permute_14);  view_15 = permute_14 = None
    view_16: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_2, [8, 196, 1536]);  mm_2 = None
    add_15: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_16, arg91_1);  view_16 = arg91_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_23: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_15, 0.5)
    mul_24: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_15, 0.7071067811865476);  add_15 = None
    erf_2: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_24);  mul_24 = None
    add_16: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_25: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_23, add_16);  mul_23 = add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_17: "f32[1568, 1536]" = torch.ops.aten.reshape.default(mul_25, [1568, 1536]);  mul_25 = None
    permute_15: "f32[1536, 384]" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
    
    # No stacktrace found for following nodes
    mm_default_18: "f32[1568, 384]" = torch.ops.aten.mm.default(view_17, permute_15);  view_17 = permute_15 = None
    add_tensor_18: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_18, arg93_1);  mm_default_18 = arg93_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_18: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_18, [8, 196, 384]);  add_tensor_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_26: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg15_1, view_18);  arg15_1 = view_18 = None
    add_17: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_13, mul_26);  add_13 = mul_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_28: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_27, add_17);  mul_27 = None
    add_18: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg19_1, mul_28);  arg19_1 = mul_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_16: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_18, [0, 2, 1]);  add_18 = None
    view_19: "f32[3072, 196]" = torch.ops.aten.reshape.default(permute_16, [3072, 196]);  permute_16 = None
    permute_17: "f32[196, 196]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
    
    # No stacktrace found for following nodes
    mm_default_17: "f32[3072, 196]" = torch.ops.aten.mm.default(view_19, permute_17);  view_19 = permute_17 = None
    add_tensor_17: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_17, arg95_1);  mm_default_17 = arg95_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    view_20: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_17, [8, 384, 196]);  add_tensor_17 = None
    permute_18: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_20, [0, 2, 1]);  view_20 = None
    mul_29: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg18_1, permute_18);  arg18_1 = permute_18 = None
    add_19: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_17, mul_29);  add_17 = mul_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_31: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_30, add_19);  mul_30 = None
    add_20: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg22_1, mul_31);  arg22_1 = mul_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    clone_9: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_20, memory_format = torch.contiguous_format);  add_20 = None
    view_21: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_9, [1568, 384]);  clone_9 = None
    permute_19: "f32[384, 1536]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
    mm_3: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_21, permute_19);  view_21 = permute_19 = None
    view_22: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_3, [8, 196, 1536]);  mm_3 = None
    add_21: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_22, arg97_1);  view_22 = arg97_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_32: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_21, 0.5)
    mul_33: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_21, 0.7071067811865476);  add_21 = None
    erf_3: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
    add_22: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_34: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_32, add_22);  mul_32 = add_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_23: "f32[1568, 1536]" = torch.ops.aten.reshape.default(mul_34, [1568, 1536]);  mul_34 = None
    permute_20: "f32[1536, 384]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
    
    # No stacktrace found for following nodes
    mm_default_16: "f32[1568, 384]" = torch.ops.aten.mm.default(view_23, permute_20);  view_23 = permute_20 = None
    add_tensor_16: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_16, arg99_1);  mm_default_16 = arg99_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_24: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_16, [8, 196, 384]);  add_tensor_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_35: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg21_1, view_24);  arg21_1 = view_24 = None
    add_23: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_19, mul_35);  add_19 = mul_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_37: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_36, add_23);  mul_36 = None
    add_24: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg25_1, mul_37);  arg25_1 = mul_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_21: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_24, [0, 2, 1]);  add_24 = None
    view_25: "f32[3072, 196]" = torch.ops.aten.reshape.default(permute_21, [3072, 196]);  permute_21 = None
    permute_22: "f32[196, 196]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
    
    # No stacktrace found for following nodes
    mm_default_15: "f32[3072, 196]" = torch.ops.aten.mm.default(view_25, permute_22);  view_25 = permute_22 = None
    add_tensor_15: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_15, arg101_1);  mm_default_15 = arg101_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    view_26: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_15, [8, 384, 196]);  add_tensor_15 = None
    permute_23: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_26, [0, 2, 1]);  view_26 = None
    mul_38: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg24_1, permute_23);  arg24_1 = permute_23 = None
    add_25: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_23, mul_38);  add_23 = mul_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_40: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_39, add_25);  mul_39 = None
    add_26: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg28_1, mul_40);  arg28_1 = mul_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    clone_12: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_26, memory_format = torch.contiguous_format);  add_26 = None
    view_27: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_12, [1568, 384]);  clone_12 = None
    permute_24: "f32[384, 1536]" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
    mm_4: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_27, permute_24);  view_27 = permute_24 = None
    view_28: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_4, [8, 196, 1536]);  mm_4 = None
    add_27: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_28, arg103_1);  view_28 = arg103_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_41: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_27, 0.5)
    mul_42: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_27, 0.7071067811865476);  add_27 = None
    erf_4: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_42);  mul_42 = None
    add_28: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_43: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_41, add_28);  mul_41 = add_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_29: "f32[1568, 1536]" = torch.ops.aten.reshape.default(mul_43, [1568, 1536]);  mul_43 = None
    permute_25: "f32[1536, 384]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
    
    # No stacktrace found for following nodes
    mm_default_14: "f32[1568, 384]" = torch.ops.aten.mm.default(view_29, permute_25);  view_29 = permute_25 = None
    add_tensor_14: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_14, arg105_1);  mm_default_14 = arg105_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_30: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_14, [8, 196, 384]);  add_tensor_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_44: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg27_1, view_30);  arg27_1 = view_30 = None
    add_29: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_25, mul_44);  add_25 = mul_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_46: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_45, add_29);  mul_45 = None
    add_30: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg31_1, mul_46);  arg31_1 = mul_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_26: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_30, [0, 2, 1]);  add_30 = None
    view_31: "f32[3072, 196]" = torch.ops.aten.reshape.default(permute_26, [3072, 196]);  permute_26 = None
    permute_27: "f32[196, 196]" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
    
    # No stacktrace found for following nodes
    mm_default_13: "f32[3072, 196]" = torch.ops.aten.mm.default(view_31, permute_27);  view_31 = permute_27 = None
    add_tensor_13: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_13, arg107_1);  mm_default_13 = arg107_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    view_32: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_13, [8, 384, 196]);  add_tensor_13 = None
    permute_28: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None
    mul_47: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg30_1, permute_28);  arg30_1 = permute_28 = None
    add_31: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_29, mul_47);  add_29 = mul_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_49: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_48, add_31);  mul_48 = None
    add_32: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg34_1, mul_49);  arg34_1 = mul_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    clone_15: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_32, memory_format = torch.contiguous_format);  add_32 = None
    view_33: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_15, [1568, 384]);  clone_15 = None
    permute_29: "f32[384, 1536]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
    mm_5: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_33, permute_29);  view_33 = permute_29 = None
    view_34: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_5, [8, 196, 1536]);  mm_5 = None
    add_33: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_34, arg109_1);  view_34 = arg109_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_50: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_33, 0.5)
    mul_51: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_33, 0.7071067811865476);  add_33 = None
    erf_5: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
    add_34: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_52: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_50, add_34);  mul_50 = add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_35: "f32[1568, 1536]" = torch.ops.aten.reshape.default(mul_52, [1568, 1536]);  mul_52 = None
    permute_30: "f32[1536, 384]" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
    
    # No stacktrace found for following nodes
    mm_default_12: "f32[1568, 384]" = torch.ops.aten.mm.default(view_35, permute_30);  view_35 = permute_30 = None
    add_tensor_12: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_12, arg111_1);  mm_default_12 = arg111_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_36: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_12, [8, 196, 384]);  add_tensor_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_53: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg33_1, view_36);  arg33_1 = view_36 = None
    add_35: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_31, mul_53);  add_31 = mul_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_55: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_54, add_35);  mul_54 = None
    add_36: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg37_1, mul_55);  arg37_1 = mul_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_31: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_36, [0, 2, 1]);  add_36 = None
    view_37: "f32[3072, 196]" = torch.ops.aten.reshape.default(permute_31, [3072, 196]);  permute_31 = None
    permute_32: "f32[196, 196]" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
    
    # No stacktrace found for following nodes
    mm_default_11: "f32[3072, 196]" = torch.ops.aten.mm.default(view_37, permute_32);  view_37 = permute_32 = None
    add_tensor_11: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_11, arg113_1);  mm_default_11 = arg113_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    view_38: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_11, [8, 384, 196]);  add_tensor_11 = None
    permute_33: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_38, [0, 2, 1]);  view_38 = None
    mul_56: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg36_1, permute_33);  arg36_1 = permute_33 = None
    add_37: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_35, mul_56);  add_35 = mul_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_58: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_57, add_37);  mul_57 = None
    add_38: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg40_1, mul_58);  arg40_1 = mul_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    clone_18: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_38, memory_format = torch.contiguous_format);  add_38 = None
    view_39: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_18, [1568, 384]);  clone_18 = None
    permute_34: "f32[384, 1536]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
    mm_6: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_39, permute_34);  view_39 = permute_34 = None
    view_40: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_6, [8, 196, 1536]);  mm_6 = None
    add_39: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_40, arg115_1);  view_40 = arg115_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_59: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_39, 0.5)
    mul_60: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_39, 0.7071067811865476);  add_39 = None
    erf_6: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_60);  mul_60 = None
    add_40: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_61: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_59, add_40);  mul_59 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_41: "f32[1568, 1536]" = torch.ops.aten.reshape.default(mul_61, [1568, 1536]);  mul_61 = None
    permute_35: "f32[1536, 384]" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
    
    # No stacktrace found for following nodes
    mm_default_10: "f32[1568, 384]" = torch.ops.aten.mm.default(view_41, permute_35);  view_41 = permute_35 = None
    add_tensor_10: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_10, arg117_1);  mm_default_10 = arg117_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_42: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_10, [8, 196, 384]);  add_tensor_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_62: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg39_1, view_42);  arg39_1 = view_42 = None
    add_41: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_37, mul_62);  add_37 = mul_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_64: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_63, add_41);  mul_63 = None
    add_42: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg43_1, mul_64);  arg43_1 = mul_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_36: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_42, [0, 2, 1]);  add_42 = None
    view_43: "f32[3072, 196]" = torch.ops.aten.reshape.default(permute_36, [3072, 196]);  permute_36 = None
    permute_37: "f32[196, 196]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
    
    # No stacktrace found for following nodes
    mm_default_9: "f32[3072, 196]" = torch.ops.aten.mm.default(view_43, permute_37);  view_43 = permute_37 = None
    add_tensor_9: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_9, arg119_1);  mm_default_9 = arg119_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    view_44: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_9, [8, 384, 196]);  add_tensor_9 = None
    permute_38: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_44, [0, 2, 1]);  view_44 = None
    mul_65: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg42_1, permute_38);  arg42_1 = permute_38 = None
    add_43: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_41, mul_65);  add_41 = mul_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_67: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_66, add_43);  mul_66 = None
    add_44: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg46_1, mul_67);  arg46_1 = mul_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    clone_21: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_44, memory_format = torch.contiguous_format);  add_44 = None
    view_45: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_21, [1568, 384]);  clone_21 = None
    permute_39: "f32[384, 1536]" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
    mm_7: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_45, permute_39);  view_45 = permute_39 = None
    view_46: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_7, [8, 196, 1536]);  mm_7 = None
    add_45: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_46, arg121_1);  view_46 = arg121_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_68: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_45, 0.5)
    mul_69: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_45, 0.7071067811865476);  add_45 = None
    erf_7: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
    add_46: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_70: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_68, add_46);  mul_68 = add_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_47: "f32[1568, 1536]" = torch.ops.aten.reshape.default(mul_70, [1568, 1536]);  mul_70 = None
    permute_40: "f32[1536, 384]" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
    
    # No stacktrace found for following nodes
    mm_default_8: "f32[1568, 384]" = torch.ops.aten.mm.default(view_47, permute_40);  view_47 = permute_40 = None
    add_tensor_8: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_8, arg123_1);  mm_default_8 = arg123_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_48: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_8, [8, 196, 384]);  add_tensor_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_71: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg45_1, view_48);  arg45_1 = view_48 = None
    add_47: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_43, mul_71);  add_43 = mul_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_73: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_72, add_47);  mul_72 = None
    add_48: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg49_1, mul_73);  arg49_1 = mul_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_41: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_48, [0, 2, 1]);  add_48 = None
    view_49: "f32[3072, 196]" = torch.ops.aten.reshape.default(permute_41, [3072, 196]);  permute_41 = None
    permute_42: "f32[196, 196]" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
    
    # No stacktrace found for following nodes
    mm_default_7: "f32[3072, 196]" = torch.ops.aten.mm.default(view_49, permute_42);  view_49 = permute_42 = None
    add_tensor_7: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_7, arg125_1);  mm_default_7 = arg125_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    view_50: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_7, [8, 384, 196]);  add_tensor_7 = None
    permute_43: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_50, [0, 2, 1]);  view_50 = None
    mul_74: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg48_1, permute_43);  arg48_1 = permute_43 = None
    add_49: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_47, mul_74);  add_47 = mul_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_76: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_75, add_49);  mul_75 = None
    add_50: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg52_1, mul_76);  arg52_1 = mul_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    clone_24: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_50, memory_format = torch.contiguous_format);  add_50 = None
    view_51: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_24, [1568, 384]);  clone_24 = None
    permute_44: "f32[384, 1536]" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
    mm_8: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_51, permute_44);  view_51 = permute_44 = None
    view_52: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_8, [8, 196, 1536]);  mm_8 = None
    add_51: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_52, arg127_1);  view_52 = arg127_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_77: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_51, 0.5)
    mul_78: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_51, 0.7071067811865476);  add_51 = None
    erf_8: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_78);  mul_78 = None
    add_52: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_79: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_77, add_52);  mul_77 = add_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_53: "f32[1568, 1536]" = torch.ops.aten.reshape.default(mul_79, [1568, 1536]);  mul_79 = None
    permute_45: "f32[1536, 384]" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
    
    # No stacktrace found for following nodes
    mm_default_6: "f32[1568, 384]" = torch.ops.aten.mm.default(view_53, permute_45);  view_53 = permute_45 = None
    add_tensor_6: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_6, arg129_1);  mm_default_6 = arg129_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_54: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_6, [8, 196, 384]);  add_tensor_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_80: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg51_1, view_54);  arg51_1 = view_54 = None
    add_53: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_49, mul_80);  add_49 = mul_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_82: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_81, add_53);  mul_81 = None
    add_54: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg55_1, mul_82);  arg55_1 = mul_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_46: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_54, [0, 2, 1]);  add_54 = None
    view_55: "f32[3072, 196]" = torch.ops.aten.reshape.default(permute_46, [3072, 196]);  permute_46 = None
    permute_47: "f32[196, 196]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
    
    # No stacktrace found for following nodes
    mm_default_5: "f32[3072, 196]" = torch.ops.aten.mm.default(view_55, permute_47);  view_55 = permute_47 = None
    add_tensor_5: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_5, arg131_1);  mm_default_5 = arg131_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    view_56: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_5, [8, 384, 196]);  add_tensor_5 = None
    permute_48: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
    mul_83: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg54_1, permute_48);  arg54_1 = permute_48 = None
    add_55: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_53, mul_83);  add_53 = mul_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_85: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_84, add_55);  mul_84 = None
    add_56: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg58_1, mul_85);  arg58_1 = mul_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    clone_27: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_56, memory_format = torch.contiguous_format);  add_56 = None
    view_57: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_27, [1568, 384]);  clone_27 = None
    permute_49: "f32[384, 1536]" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
    mm_9: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_57, permute_49);  view_57 = permute_49 = None
    view_58: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_9, [8, 196, 1536]);  mm_9 = None
    add_57: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_58, arg133_1);  view_58 = arg133_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_86: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_57, 0.5)
    mul_87: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_57, 0.7071067811865476);  add_57 = None
    erf_9: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
    add_58: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_88: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_86, add_58);  mul_86 = add_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_59: "f32[1568, 1536]" = torch.ops.aten.reshape.default(mul_88, [1568, 1536]);  mul_88 = None
    permute_50: "f32[1536, 384]" = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
    
    # No stacktrace found for following nodes
    mm_default_4: "f32[1568, 384]" = torch.ops.aten.mm.default(view_59, permute_50);  view_59 = permute_50 = None
    add_tensor_4: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_4, arg135_1);  mm_default_4 = arg135_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_60: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_4, [8, 196, 384]);  add_tensor_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_89: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg57_1, view_60);  arg57_1 = view_60 = None
    add_59: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_55, mul_89);  add_55 = mul_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_91: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_90, add_59);  mul_90 = None
    add_60: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg61_1, mul_91);  arg61_1 = mul_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_51: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_60, [0, 2, 1]);  add_60 = None
    view_61: "f32[3072, 196]" = torch.ops.aten.reshape.default(permute_51, [3072, 196]);  permute_51 = None
    permute_52: "f32[196, 196]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
    
    # No stacktrace found for following nodes
    mm_default_3: "f32[3072, 196]" = torch.ops.aten.mm.default(view_61, permute_52);  view_61 = permute_52 = None
    add_tensor_3: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_3, arg137_1);  mm_default_3 = arg137_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    view_62: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_3, [8, 384, 196]);  add_tensor_3 = None
    permute_53: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_62, [0, 2, 1]);  view_62 = None
    mul_92: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg60_1, permute_53);  arg60_1 = permute_53 = None
    add_61: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_59, mul_92);  add_59 = mul_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_94: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_93, add_61);  mul_93 = None
    add_62: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg64_1, mul_94);  arg64_1 = mul_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    clone_30: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_62, memory_format = torch.contiguous_format);  add_62 = None
    view_63: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_30, [1568, 384]);  clone_30 = None
    permute_54: "f32[384, 1536]" = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
    mm_10: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_63, permute_54);  view_63 = permute_54 = None
    view_64: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_10, [8, 196, 1536]);  mm_10 = None
    add_63: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_64, arg139_1);  view_64 = arg139_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_95: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_63, 0.5)
    mul_96: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_63, 0.7071067811865476);  add_63 = None
    erf_10: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_96);  mul_96 = None
    add_64: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_97: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_95, add_64);  mul_95 = add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_65: "f32[1568, 1536]" = torch.ops.aten.reshape.default(mul_97, [1568, 1536]);  mul_97 = None
    permute_55: "f32[1536, 384]" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
    
    # No stacktrace found for following nodes
    mm_default_2: "f32[1568, 384]" = torch.ops.aten.mm.default(view_65, permute_55);  view_65 = permute_55 = None
    add_tensor_2: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_2, arg141_1);  mm_default_2 = arg141_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_66: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_2, [8, 196, 384]);  add_tensor_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_98: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg63_1, view_66);  arg63_1 = view_66 = None
    add_65: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_61, mul_98);  add_61 = mul_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_100: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_99, add_65);  mul_99 = None
    add_66: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg67_1, mul_100);  arg67_1 = mul_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_56: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_66, [0, 2, 1]);  add_66 = None
    view_67: "f32[3072, 196]" = torch.ops.aten.reshape.default(permute_56, [3072, 196]);  permute_56 = None
    permute_57: "f32[196, 196]" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
    
    # No stacktrace found for following nodes
    mm_default_1: "f32[3072, 196]" = torch.ops.aten.mm.default(view_67, permute_57);  view_67 = permute_57 = None
    add_tensor_1: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_1, arg143_1);  mm_default_1 = arg143_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    view_68: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_1, [8, 384, 196]);  add_tensor_1 = None
    permute_58: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_68, [0, 2, 1]);  view_68 = None
    mul_101: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg66_1, permute_58);  arg66_1 = permute_58 = None
    add_67: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_65, mul_101);  add_65 = mul_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_103: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_102, add_67);  mul_102 = None
    add_68: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg70_1, mul_103);  arg70_1 = mul_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    clone_33: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_68, memory_format = torch.contiguous_format);  add_68 = None
    view_69: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_33, [1568, 384]);  clone_33 = None
    permute_59: "f32[384, 1536]" = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
    mm_11: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_69, permute_59);  view_69 = permute_59 = None
    view_70: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_11, [8, 196, 1536]);  mm_11 = None
    add_69: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_70, arg145_1);  view_70 = arg145_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_104: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_69, 0.5)
    mul_105: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_69, 0.7071067811865476);  add_69 = None
    erf_11: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_105);  mul_105 = None
    add_70: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_106: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_104, add_70);  mul_104 = add_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_71: "f32[1568, 1536]" = torch.ops.aten.reshape.default(mul_106, [1568, 1536]);  mul_106 = None
    permute_60: "f32[1536, 384]" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
    
    # No stacktrace found for following nodes
    mm_default: "f32[1568, 384]" = torch.ops.aten.mm.default(view_71, permute_60);  view_71 = permute_60 = None
    add_tensor: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default, arg147_1);  mm_default = arg147_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_72: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor, [8, 196, 384]);  add_tensor = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_107: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg69_1, view_72);  arg69_1 = view_72 = None
    add_71: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_67, mul_107);  add_67 = mul_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_109: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_108, add_71);  mul_108 = add_71 = None
    add_72: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg72_1, mul_109);  arg72_1 = mul_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:271, code: x = x.mean(dim=1)
    mean: "f32[8, 384]" = torch.ops.aten.mean.dim(add_72, [1]);  add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:273, code: return x if pre_logits else self.head(x)
    permute_61: "f32[384, 1000]" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
    addmm_24: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg149_1, mean, permute_61);  arg149_1 = mean = permute_61 = None
    return (addmm_24,)
    