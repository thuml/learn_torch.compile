from __future__ import annotations



def forward(self, L_cloned_inputs_0_ : torch.Tensor):
    l_cloned_inputs_0_ = L_cloned_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    x = self.L__mod___stem_proj(l_cloned_inputs_0_);  l_cloned_inputs_0_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    flatten = x.flatten(2);  x = None
    x_1 = flatten.transpose(1, 2);  flatten = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    x_3 = self.L__mod___stem_norm(x_1);  x_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___0___ls1 = self.getattr_L__mod___blocks___0___ls1
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    getattr_l__mod___blocks___0___norm1_beta = self.getattr_L__mod___blocks___0___norm1_beta
    getattr_l__mod___blocks___0___norm1_alpha = self.getattr_L__mod___blocks___0___norm1_alpha
    addcmul = torch.addcmul(getattr_l__mod___blocks___0___norm1_beta, getattr_l__mod___blocks___0___norm1_alpha, x_3);  getattr_l__mod___blocks___0___norm1_beta = getattr_l__mod___blocks___0___norm1_alpha = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_1 = addcmul.transpose(1, 2);  addcmul = None
    getattr_l__mod___blocks___0___linear_tokens = self.getattr_L__mod___blocks___0___linear_tokens(transpose_1);  transpose_1 = None
    transpose_2 = getattr_l__mod___blocks___0___linear_tokens.transpose(1, 2);  getattr_l__mod___blocks___0___linear_tokens = None
    mul = getattr_l__mod___blocks___0___ls1 * transpose_2;  getattr_l__mod___blocks___0___ls1 = transpose_2 = None
    getattr_l__mod___blocks___0___drop_path = self.getattr_L__mod___blocks___0___drop_path(mul);  mul = None
    x_4 = x_3 + getattr_l__mod___blocks___0___drop_path;  x_3 = getattr_l__mod___blocks___0___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___0___ls2 = self.getattr_L__mod___blocks___0___ls2
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    getattr_l__mod___blocks___0___norm2_beta = self.getattr_L__mod___blocks___0___norm2_beta
    getattr_l__mod___blocks___0___norm2_alpha = self.getattr_L__mod___blocks___0___norm2_alpha
    addcmul_1 = torch.addcmul(getattr_l__mod___blocks___0___norm2_beta, getattr_l__mod___blocks___0___norm2_alpha, x_4);  getattr_l__mod___blocks___0___norm2_beta = getattr_l__mod___blocks___0___norm2_alpha = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_5 = self.getattr_L__mod___blocks___0___mlp_channels_fc1(addcmul_1);  addcmul_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_6 = self.getattr_L__mod___blocks___0___mlp_channels_act(x_5);  x_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_7 = self.getattr_L__mod___blocks___0___mlp_channels_drop1(x_6);  x_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_8 = self.getattr_L__mod___blocks___0___mlp_channels_norm(x_7);  x_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_9 = self.getattr_L__mod___blocks___0___mlp_channels_fc2(x_8);  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_10 = self.getattr_L__mod___blocks___0___mlp_channels_drop2(x_9);  x_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_1 = getattr_l__mod___blocks___0___ls2 * x_10;  getattr_l__mod___blocks___0___ls2 = x_10 = None
    getattr_l__mod___blocks___0___drop_path_1 = self.getattr_L__mod___blocks___0___drop_path(mul_1);  mul_1 = None
    x_11 = x_4 + getattr_l__mod___blocks___0___drop_path_1;  x_4 = getattr_l__mod___blocks___0___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___1___ls1 = self.getattr_L__mod___blocks___1___ls1
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    getattr_l__mod___blocks___1___norm1_beta = self.getattr_L__mod___blocks___1___norm1_beta
    getattr_l__mod___blocks___1___norm1_alpha = self.getattr_L__mod___blocks___1___norm1_alpha
    addcmul_2 = torch.addcmul(getattr_l__mod___blocks___1___norm1_beta, getattr_l__mod___blocks___1___norm1_alpha, x_11);  getattr_l__mod___blocks___1___norm1_beta = getattr_l__mod___blocks___1___norm1_alpha = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_3 = addcmul_2.transpose(1, 2);  addcmul_2 = None
    getattr_l__mod___blocks___1___linear_tokens = self.getattr_L__mod___blocks___1___linear_tokens(transpose_3);  transpose_3 = None
    transpose_4 = getattr_l__mod___blocks___1___linear_tokens.transpose(1, 2);  getattr_l__mod___blocks___1___linear_tokens = None
    mul_2 = getattr_l__mod___blocks___1___ls1 * transpose_4;  getattr_l__mod___blocks___1___ls1 = transpose_4 = None
    getattr_l__mod___blocks___1___drop_path = self.getattr_L__mod___blocks___1___drop_path(mul_2);  mul_2 = None
    x_12 = x_11 + getattr_l__mod___blocks___1___drop_path;  x_11 = getattr_l__mod___blocks___1___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___1___ls2 = self.getattr_L__mod___blocks___1___ls2
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    getattr_l__mod___blocks___1___norm2_beta = self.getattr_L__mod___blocks___1___norm2_beta
    getattr_l__mod___blocks___1___norm2_alpha = self.getattr_L__mod___blocks___1___norm2_alpha
    addcmul_3 = torch.addcmul(getattr_l__mod___blocks___1___norm2_beta, getattr_l__mod___blocks___1___norm2_alpha, x_12);  getattr_l__mod___blocks___1___norm2_beta = getattr_l__mod___blocks___1___norm2_alpha = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_13 = self.getattr_L__mod___blocks___1___mlp_channels_fc1(addcmul_3);  addcmul_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_14 = self.getattr_L__mod___blocks___1___mlp_channels_act(x_13);  x_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_15 = self.getattr_L__mod___blocks___1___mlp_channels_drop1(x_14);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_16 = self.getattr_L__mod___blocks___1___mlp_channels_norm(x_15);  x_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_17 = self.getattr_L__mod___blocks___1___mlp_channels_fc2(x_16);  x_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_18 = self.getattr_L__mod___blocks___1___mlp_channels_drop2(x_17);  x_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_3 = getattr_l__mod___blocks___1___ls2 * x_18;  getattr_l__mod___blocks___1___ls2 = x_18 = None
    getattr_l__mod___blocks___1___drop_path_1 = self.getattr_L__mod___blocks___1___drop_path(mul_3);  mul_3 = None
    x_19 = x_12 + getattr_l__mod___blocks___1___drop_path_1;  x_12 = getattr_l__mod___blocks___1___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___2___ls1 = self.getattr_L__mod___blocks___2___ls1
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    getattr_l__mod___blocks___2___norm1_beta = self.getattr_L__mod___blocks___2___norm1_beta
    getattr_l__mod___blocks___2___norm1_alpha = self.getattr_L__mod___blocks___2___norm1_alpha
    addcmul_4 = torch.addcmul(getattr_l__mod___blocks___2___norm1_beta, getattr_l__mod___blocks___2___norm1_alpha, x_19);  getattr_l__mod___blocks___2___norm1_beta = getattr_l__mod___blocks___2___norm1_alpha = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_5 = addcmul_4.transpose(1, 2);  addcmul_4 = None
    getattr_l__mod___blocks___2___linear_tokens = self.getattr_L__mod___blocks___2___linear_tokens(transpose_5);  transpose_5 = None
    transpose_6 = getattr_l__mod___blocks___2___linear_tokens.transpose(1, 2);  getattr_l__mod___blocks___2___linear_tokens = None
    mul_4 = getattr_l__mod___blocks___2___ls1 * transpose_6;  getattr_l__mod___blocks___2___ls1 = transpose_6 = None
    getattr_l__mod___blocks___2___drop_path = self.getattr_L__mod___blocks___2___drop_path(mul_4);  mul_4 = None
    x_20 = x_19 + getattr_l__mod___blocks___2___drop_path;  x_19 = getattr_l__mod___blocks___2___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___2___ls2 = self.getattr_L__mod___blocks___2___ls2
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    getattr_l__mod___blocks___2___norm2_beta = self.getattr_L__mod___blocks___2___norm2_beta
    getattr_l__mod___blocks___2___norm2_alpha = self.getattr_L__mod___blocks___2___norm2_alpha
    addcmul_5 = torch.addcmul(getattr_l__mod___blocks___2___norm2_beta, getattr_l__mod___blocks___2___norm2_alpha, x_20);  getattr_l__mod___blocks___2___norm2_beta = getattr_l__mod___blocks___2___norm2_alpha = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_21 = self.getattr_L__mod___blocks___2___mlp_channels_fc1(addcmul_5);  addcmul_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_22 = self.getattr_L__mod___blocks___2___mlp_channels_act(x_21);  x_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_23 = self.getattr_L__mod___blocks___2___mlp_channels_drop1(x_22);  x_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_24 = self.getattr_L__mod___blocks___2___mlp_channels_norm(x_23);  x_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_25 = self.getattr_L__mod___blocks___2___mlp_channels_fc2(x_24);  x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_26 = self.getattr_L__mod___blocks___2___mlp_channels_drop2(x_25);  x_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_5 = getattr_l__mod___blocks___2___ls2 * x_26;  getattr_l__mod___blocks___2___ls2 = x_26 = None
    getattr_l__mod___blocks___2___drop_path_1 = self.getattr_L__mod___blocks___2___drop_path(mul_5);  mul_5 = None
    x_27 = x_20 + getattr_l__mod___blocks___2___drop_path_1;  x_20 = getattr_l__mod___blocks___2___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___3___ls1 = self.getattr_L__mod___blocks___3___ls1
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    getattr_l__mod___blocks___3___norm1_beta = self.getattr_L__mod___blocks___3___norm1_beta
    getattr_l__mod___blocks___3___norm1_alpha = self.getattr_L__mod___blocks___3___norm1_alpha
    addcmul_6 = torch.addcmul(getattr_l__mod___blocks___3___norm1_beta, getattr_l__mod___blocks___3___norm1_alpha, x_27);  getattr_l__mod___blocks___3___norm1_beta = getattr_l__mod___blocks___3___norm1_alpha = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_7 = addcmul_6.transpose(1, 2);  addcmul_6 = None
    getattr_l__mod___blocks___3___linear_tokens = self.getattr_L__mod___blocks___3___linear_tokens(transpose_7);  transpose_7 = None
    transpose_8 = getattr_l__mod___blocks___3___linear_tokens.transpose(1, 2);  getattr_l__mod___blocks___3___linear_tokens = None
    mul_6 = getattr_l__mod___blocks___3___ls1 * transpose_8;  getattr_l__mod___blocks___3___ls1 = transpose_8 = None
    getattr_l__mod___blocks___3___drop_path = self.getattr_L__mod___blocks___3___drop_path(mul_6);  mul_6 = None
    x_28 = x_27 + getattr_l__mod___blocks___3___drop_path;  x_27 = getattr_l__mod___blocks___3___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___3___ls2 = self.getattr_L__mod___blocks___3___ls2
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    getattr_l__mod___blocks___3___norm2_beta = self.getattr_L__mod___blocks___3___norm2_beta
    getattr_l__mod___blocks___3___norm2_alpha = self.getattr_L__mod___blocks___3___norm2_alpha
    addcmul_7 = torch.addcmul(getattr_l__mod___blocks___3___norm2_beta, getattr_l__mod___blocks___3___norm2_alpha, x_28);  getattr_l__mod___blocks___3___norm2_beta = getattr_l__mod___blocks___3___norm2_alpha = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_29 = self.getattr_L__mod___blocks___3___mlp_channels_fc1(addcmul_7);  addcmul_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_30 = self.getattr_L__mod___blocks___3___mlp_channels_act(x_29);  x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_31 = self.getattr_L__mod___blocks___3___mlp_channels_drop1(x_30);  x_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_32 = self.getattr_L__mod___blocks___3___mlp_channels_norm(x_31);  x_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_33 = self.getattr_L__mod___blocks___3___mlp_channels_fc2(x_32);  x_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_34 = self.getattr_L__mod___blocks___3___mlp_channels_drop2(x_33);  x_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_7 = getattr_l__mod___blocks___3___ls2 * x_34;  getattr_l__mod___blocks___3___ls2 = x_34 = None
    getattr_l__mod___blocks___3___drop_path_1 = self.getattr_L__mod___blocks___3___drop_path(mul_7);  mul_7 = None
    x_35 = x_28 + getattr_l__mod___blocks___3___drop_path_1;  x_28 = getattr_l__mod___blocks___3___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___4___ls1 = self.getattr_L__mod___blocks___4___ls1
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    getattr_l__mod___blocks___4___norm1_beta = self.getattr_L__mod___blocks___4___norm1_beta
    getattr_l__mod___blocks___4___norm1_alpha = self.getattr_L__mod___blocks___4___norm1_alpha
    addcmul_8 = torch.addcmul(getattr_l__mod___blocks___4___norm1_beta, getattr_l__mod___blocks___4___norm1_alpha, x_35);  getattr_l__mod___blocks___4___norm1_beta = getattr_l__mod___blocks___4___norm1_alpha = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_9 = addcmul_8.transpose(1, 2);  addcmul_8 = None
    getattr_l__mod___blocks___4___linear_tokens = self.getattr_L__mod___blocks___4___linear_tokens(transpose_9);  transpose_9 = None
    transpose_10 = getattr_l__mod___blocks___4___linear_tokens.transpose(1, 2);  getattr_l__mod___blocks___4___linear_tokens = None
    mul_8 = getattr_l__mod___blocks___4___ls1 * transpose_10;  getattr_l__mod___blocks___4___ls1 = transpose_10 = None
    getattr_l__mod___blocks___4___drop_path = self.getattr_L__mod___blocks___4___drop_path(mul_8);  mul_8 = None
    x_36 = x_35 + getattr_l__mod___blocks___4___drop_path;  x_35 = getattr_l__mod___blocks___4___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___4___ls2 = self.getattr_L__mod___blocks___4___ls2
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    getattr_l__mod___blocks___4___norm2_beta = self.getattr_L__mod___blocks___4___norm2_beta
    getattr_l__mod___blocks___4___norm2_alpha = self.getattr_L__mod___blocks___4___norm2_alpha
    addcmul_9 = torch.addcmul(getattr_l__mod___blocks___4___norm2_beta, getattr_l__mod___blocks___4___norm2_alpha, x_36);  getattr_l__mod___blocks___4___norm2_beta = getattr_l__mod___blocks___4___norm2_alpha = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_37 = self.getattr_L__mod___blocks___4___mlp_channels_fc1(addcmul_9);  addcmul_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_38 = self.getattr_L__mod___blocks___4___mlp_channels_act(x_37);  x_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_39 = self.getattr_L__mod___blocks___4___mlp_channels_drop1(x_38);  x_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_40 = self.getattr_L__mod___blocks___4___mlp_channels_norm(x_39);  x_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_41 = self.getattr_L__mod___blocks___4___mlp_channels_fc2(x_40);  x_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_42 = self.getattr_L__mod___blocks___4___mlp_channels_drop2(x_41);  x_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_9 = getattr_l__mod___blocks___4___ls2 * x_42;  getattr_l__mod___blocks___4___ls2 = x_42 = None
    getattr_l__mod___blocks___4___drop_path_1 = self.getattr_L__mod___blocks___4___drop_path(mul_9);  mul_9 = None
    x_43 = x_36 + getattr_l__mod___blocks___4___drop_path_1;  x_36 = getattr_l__mod___blocks___4___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___5___ls1 = self.getattr_L__mod___blocks___5___ls1
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    getattr_l__mod___blocks___5___norm1_beta = self.getattr_L__mod___blocks___5___norm1_beta
    getattr_l__mod___blocks___5___norm1_alpha = self.getattr_L__mod___blocks___5___norm1_alpha
    addcmul_10 = torch.addcmul(getattr_l__mod___blocks___5___norm1_beta, getattr_l__mod___blocks___5___norm1_alpha, x_43);  getattr_l__mod___blocks___5___norm1_beta = getattr_l__mod___blocks___5___norm1_alpha = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_11 = addcmul_10.transpose(1, 2);  addcmul_10 = None
    getattr_l__mod___blocks___5___linear_tokens = self.getattr_L__mod___blocks___5___linear_tokens(transpose_11);  transpose_11 = None
    transpose_12 = getattr_l__mod___blocks___5___linear_tokens.transpose(1, 2);  getattr_l__mod___blocks___5___linear_tokens = None
    mul_10 = getattr_l__mod___blocks___5___ls1 * transpose_12;  getattr_l__mod___blocks___5___ls1 = transpose_12 = None
    getattr_l__mod___blocks___5___drop_path = self.getattr_L__mod___blocks___5___drop_path(mul_10);  mul_10 = None
    x_44 = x_43 + getattr_l__mod___blocks___5___drop_path;  x_43 = getattr_l__mod___blocks___5___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___5___ls2 = self.getattr_L__mod___blocks___5___ls2
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    getattr_l__mod___blocks___5___norm2_beta = self.getattr_L__mod___blocks___5___norm2_beta
    getattr_l__mod___blocks___5___norm2_alpha = self.getattr_L__mod___blocks___5___norm2_alpha
    addcmul_11 = torch.addcmul(getattr_l__mod___blocks___5___norm2_beta, getattr_l__mod___blocks___5___norm2_alpha, x_44);  getattr_l__mod___blocks___5___norm2_beta = getattr_l__mod___blocks___5___norm2_alpha = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_45 = self.getattr_L__mod___blocks___5___mlp_channels_fc1(addcmul_11);  addcmul_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_46 = self.getattr_L__mod___blocks___5___mlp_channels_act(x_45);  x_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_47 = self.getattr_L__mod___blocks___5___mlp_channels_drop1(x_46);  x_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_48 = self.getattr_L__mod___blocks___5___mlp_channels_norm(x_47);  x_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_49 = self.getattr_L__mod___blocks___5___mlp_channels_fc2(x_48);  x_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_50 = self.getattr_L__mod___blocks___5___mlp_channels_drop2(x_49);  x_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_11 = getattr_l__mod___blocks___5___ls2 * x_50;  getattr_l__mod___blocks___5___ls2 = x_50 = None
    getattr_l__mod___blocks___5___drop_path_1 = self.getattr_L__mod___blocks___5___drop_path(mul_11);  mul_11 = None
    x_51 = x_44 + getattr_l__mod___blocks___5___drop_path_1;  x_44 = getattr_l__mod___blocks___5___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___6___ls1 = self.getattr_L__mod___blocks___6___ls1
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    getattr_l__mod___blocks___6___norm1_beta = self.getattr_L__mod___blocks___6___norm1_beta
    getattr_l__mod___blocks___6___norm1_alpha = self.getattr_L__mod___blocks___6___norm1_alpha
    addcmul_12 = torch.addcmul(getattr_l__mod___blocks___6___norm1_beta, getattr_l__mod___blocks___6___norm1_alpha, x_51);  getattr_l__mod___blocks___6___norm1_beta = getattr_l__mod___blocks___6___norm1_alpha = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_13 = addcmul_12.transpose(1, 2);  addcmul_12 = None
    getattr_l__mod___blocks___6___linear_tokens = self.getattr_L__mod___blocks___6___linear_tokens(transpose_13);  transpose_13 = None
    transpose_14 = getattr_l__mod___blocks___6___linear_tokens.transpose(1, 2);  getattr_l__mod___blocks___6___linear_tokens = None
    mul_12 = getattr_l__mod___blocks___6___ls1 * transpose_14;  getattr_l__mod___blocks___6___ls1 = transpose_14 = None
    getattr_l__mod___blocks___6___drop_path = self.getattr_L__mod___blocks___6___drop_path(mul_12);  mul_12 = None
    x_52 = x_51 + getattr_l__mod___blocks___6___drop_path;  x_51 = getattr_l__mod___blocks___6___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___6___ls2 = self.getattr_L__mod___blocks___6___ls2
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    getattr_l__mod___blocks___6___norm2_beta = self.getattr_L__mod___blocks___6___norm2_beta
    getattr_l__mod___blocks___6___norm2_alpha = self.getattr_L__mod___blocks___6___norm2_alpha
    addcmul_13 = torch.addcmul(getattr_l__mod___blocks___6___norm2_beta, getattr_l__mod___blocks___6___norm2_alpha, x_52);  getattr_l__mod___blocks___6___norm2_beta = getattr_l__mod___blocks___6___norm2_alpha = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_53 = self.getattr_L__mod___blocks___6___mlp_channels_fc1(addcmul_13);  addcmul_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_54 = self.getattr_L__mod___blocks___6___mlp_channels_act(x_53);  x_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_55 = self.getattr_L__mod___blocks___6___mlp_channels_drop1(x_54);  x_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_56 = self.getattr_L__mod___blocks___6___mlp_channels_norm(x_55);  x_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_57 = self.getattr_L__mod___blocks___6___mlp_channels_fc2(x_56);  x_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_58 = self.getattr_L__mod___blocks___6___mlp_channels_drop2(x_57);  x_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_13 = getattr_l__mod___blocks___6___ls2 * x_58;  getattr_l__mod___blocks___6___ls2 = x_58 = None
    getattr_l__mod___blocks___6___drop_path_1 = self.getattr_L__mod___blocks___6___drop_path(mul_13);  mul_13 = None
    x_59 = x_52 + getattr_l__mod___blocks___6___drop_path_1;  x_52 = getattr_l__mod___blocks___6___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___7___ls1 = self.getattr_L__mod___blocks___7___ls1
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    getattr_l__mod___blocks___7___norm1_beta = self.getattr_L__mod___blocks___7___norm1_beta
    getattr_l__mod___blocks___7___norm1_alpha = self.getattr_L__mod___blocks___7___norm1_alpha
    addcmul_14 = torch.addcmul(getattr_l__mod___blocks___7___norm1_beta, getattr_l__mod___blocks___7___norm1_alpha, x_59);  getattr_l__mod___blocks___7___norm1_beta = getattr_l__mod___blocks___7___norm1_alpha = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_15 = addcmul_14.transpose(1, 2);  addcmul_14 = None
    getattr_l__mod___blocks___7___linear_tokens = self.getattr_L__mod___blocks___7___linear_tokens(transpose_15);  transpose_15 = None
    transpose_16 = getattr_l__mod___blocks___7___linear_tokens.transpose(1, 2);  getattr_l__mod___blocks___7___linear_tokens = None
    mul_14 = getattr_l__mod___blocks___7___ls1 * transpose_16;  getattr_l__mod___blocks___7___ls1 = transpose_16 = None
    getattr_l__mod___blocks___7___drop_path = self.getattr_L__mod___blocks___7___drop_path(mul_14);  mul_14 = None
    x_60 = x_59 + getattr_l__mod___blocks___7___drop_path;  x_59 = getattr_l__mod___blocks___7___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___7___ls2 = self.getattr_L__mod___blocks___7___ls2
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    getattr_l__mod___blocks___7___norm2_beta = self.getattr_L__mod___blocks___7___norm2_beta
    getattr_l__mod___blocks___7___norm2_alpha = self.getattr_L__mod___blocks___7___norm2_alpha
    addcmul_15 = torch.addcmul(getattr_l__mod___blocks___7___norm2_beta, getattr_l__mod___blocks___7___norm2_alpha, x_60);  getattr_l__mod___blocks___7___norm2_beta = getattr_l__mod___blocks___7___norm2_alpha = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_61 = self.getattr_L__mod___blocks___7___mlp_channels_fc1(addcmul_15);  addcmul_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_62 = self.getattr_L__mod___blocks___7___mlp_channels_act(x_61);  x_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_63 = self.getattr_L__mod___blocks___7___mlp_channels_drop1(x_62);  x_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_64 = self.getattr_L__mod___blocks___7___mlp_channels_norm(x_63);  x_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_65 = self.getattr_L__mod___blocks___7___mlp_channels_fc2(x_64);  x_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_66 = self.getattr_L__mod___blocks___7___mlp_channels_drop2(x_65);  x_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_15 = getattr_l__mod___blocks___7___ls2 * x_66;  getattr_l__mod___blocks___7___ls2 = x_66 = None
    getattr_l__mod___blocks___7___drop_path_1 = self.getattr_L__mod___blocks___7___drop_path(mul_15);  mul_15 = None
    x_67 = x_60 + getattr_l__mod___blocks___7___drop_path_1;  x_60 = getattr_l__mod___blocks___7___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___8___ls1 = self.getattr_L__mod___blocks___8___ls1
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    getattr_l__mod___blocks___8___norm1_beta = self.getattr_L__mod___blocks___8___norm1_beta
    getattr_l__mod___blocks___8___norm1_alpha = self.getattr_L__mod___blocks___8___norm1_alpha
    addcmul_16 = torch.addcmul(getattr_l__mod___blocks___8___norm1_beta, getattr_l__mod___blocks___8___norm1_alpha, x_67);  getattr_l__mod___blocks___8___norm1_beta = getattr_l__mod___blocks___8___norm1_alpha = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_17 = addcmul_16.transpose(1, 2);  addcmul_16 = None
    getattr_l__mod___blocks___8___linear_tokens = self.getattr_L__mod___blocks___8___linear_tokens(transpose_17);  transpose_17 = None
    transpose_18 = getattr_l__mod___blocks___8___linear_tokens.transpose(1, 2);  getattr_l__mod___blocks___8___linear_tokens = None
    mul_16 = getattr_l__mod___blocks___8___ls1 * transpose_18;  getattr_l__mod___blocks___8___ls1 = transpose_18 = None
    getattr_l__mod___blocks___8___drop_path = self.getattr_L__mod___blocks___8___drop_path(mul_16);  mul_16 = None
    x_68 = x_67 + getattr_l__mod___blocks___8___drop_path;  x_67 = getattr_l__mod___blocks___8___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___8___ls2 = self.getattr_L__mod___blocks___8___ls2
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    getattr_l__mod___blocks___8___norm2_beta = self.getattr_L__mod___blocks___8___norm2_beta
    getattr_l__mod___blocks___8___norm2_alpha = self.getattr_L__mod___blocks___8___norm2_alpha
    addcmul_17 = torch.addcmul(getattr_l__mod___blocks___8___norm2_beta, getattr_l__mod___blocks___8___norm2_alpha, x_68);  getattr_l__mod___blocks___8___norm2_beta = getattr_l__mod___blocks___8___norm2_alpha = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_69 = self.getattr_L__mod___blocks___8___mlp_channels_fc1(addcmul_17);  addcmul_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_70 = self.getattr_L__mod___blocks___8___mlp_channels_act(x_69);  x_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_71 = self.getattr_L__mod___blocks___8___mlp_channels_drop1(x_70);  x_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_72 = self.getattr_L__mod___blocks___8___mlp_channels_norm(x_71);  x_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_73 = self.getattr_L__mod___blocks___8___mlp_channels_fc2(x_72);  x_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_74 = self.getattr_L__mod___blocks___8___mlp_channels_drop2(x_73);  x_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_17 = getattr_l__mod___blocks___8___ls2 * x_74;  getattr_l__mod___blocks___8___ls2 = x_74 = None
    getattr_l__mod___blocks___8___drop_path_1 = self.getattr_L__mod___blocks___8___drop_path(mul_17);  mul_17 = None
    x_75 = x_68 + getattr_l__mod___blocks___8___drop_path_1;  x_68 = getattr_l__mod___blocks___8___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___9___ls1 = self.getattr_L__mod___blocks___9___ls1
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    getattr_l__mod___blocks___9___norm1_beta = self.getattr_L__mod___blocks___9___norm1_beta
    getattr_l__mod___blocks___9___norm1_alpha = self.getattr_L__mod___blocks___9___norm1_alpha
    addcmul_18 = torch.addcmul(getattr_l__mod___blocks___9___norm1_beta, getattr_l__mod___blocks___9___norm1_alpha, x_75);  getattr_l__mod___blocks___9___norm1_beta = getattr_l__mod___blocks___9___norm1_alpha = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_19 = addcmul_18.transpose(1, 2);  addcmul_18 = None
    getattr_l__mod___blocks___9___linear_tokens = self.getattr_L__mod___blocks___9___linear_tokens(transpose_19);  transpose_19 = None
    transpose_20 = getattr_l__mod___blocks___9___linear_tokens.transpose(1, 2);  getattr_l__mod___blocks___9___linear_tokens = None
    mul_18 = getattr_l__mod___blocks___9___ls1 * transpose_20;  getattr_l__mod___blocks___9___ls1 = transpose_20 = None
    getattr_l__mod___blocks___9___drop_path = self.getattr_L__mod___blocks___9___drop_path(mul_18);  mul_18 = None
    x_76 = x_75 + getattr_l__mod___blocks___9___drop_path;  x_75 = getattr_l__mod___blocks___9___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___9___ls2 = self.getattr_L__mod___blocks___9___ls2
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    getattr_l__mod___blocks___9___norm2_beta = self.getattr_L__mod___blocks___9___norm2_beta
    getattr_l__mod___blocks___9___norm2_alpha = self.getattr_L__mod___blocks___9___norm2_alpha
    addcmul_19 = torch.addcmul(getattr_l__mod___blocks___9___norm2_beta, getattr_l__mod___blocks___9___norm2_alpha, x_76);  getattr_l__mod___blocks___9___norm2_beta = getattr_l__mod___blocks___9___norm2_alpha = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_77 = self.getattr_L__mod___blocks___9___mlp_channels_fc1(addcmul_19);  addcmul_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_78 = self.getattr_L__mod___blocks___9___mlp_channels_act(x_77);  x_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_79 = self.getattr_L__mod___blocks___9___mlp_channels_drop1(x_78);  x_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_80 = self.getattr_L__mod___blocks___9___mlp_channels_norm(x_79);  x_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_81 = self.getattr_L__mod___blocks___9___mlp_channels_fc2(x_80);  x_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_82 = self.getattr_L__mod___blocks___9___mlp_channels_drop2(x_81);  x_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_19 = getattr_l__mod___blocks___9___ls2 * x_82;  getattr_l__mod___blocks___9___ls2 = x_82 = None
    getattr_l__mod___blocks___9___drop_path_1 = self.getattr_L__mod___blocks___9___drop_path(mul_19);  mul_19 = None
    x_83 = x_76 + getattr_l__mod___blocks___9___drop_path_1;  x_76 = getattr_l__mod___blocks___9___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___10___ls1 = self.getattr_L__mod___blocks___10___ls1
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    getattr_l__mod___blocks___10___norm1_beta = self.getattr_L__mod___blocks___10___norm1_beta
    getattr_l__mod___blocks___10___norm1_alpha = self.getattr_L__mod___blocks___10___norm1_alpha
    addcmul_20 = torch.addcmul(getattr_l__mod___blocks___10___norm1_beta, getattr_l__mod___blocks___10___norm1_alpha, x_83);  getattr_l__mod___blocks___10___norm1_beta = getattr_l__mod___blocks___10___norm1_alpha = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_21 = addcmul_20.transpose(1, 2);  addcmul_20 = None
    getattr_l__mod___blocks___10___linear_tokens = self.getattr_L__mod___blocks___10___linear_tokens(transpose_21);  transpose_21 = None
    transpose_22 = getattr_l__mod___blocks___10___linear_tokens.transpose(1, 2);  getattr_l__mod___blocks___10___linear_tokens = None
    mul_20 = getattr_l__mod___blocks___10___ls1 * transpose_22;  getattr_l__mod___blocks___10___ls1 = transpose_22 = None
    getattr_l__mod___blocks___10___drop_path = self.getattr_L__mod___blocks___10___drop_path(mul_20);  mul_20 = None
    x_84 = x_83 + getattr_l__mod___blocks___10___drop_path;  x_83 = getattr_l__mod___blocks___10___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___10___ls2 = self.getattr_L__mod___blocks___10___ls2
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    getattr_l__mod___blocks___10___norm2_beta = self.getattr_L__mod___blocks___10___norm2_beta
    getattr_l__mod___blocks___10___norm2_alpha = self.getattr_L__mod___blocks___10___norm2_alpha
    addcmul_21 = torch.addcmul(getattr_l__mod___blocks___10___norm2_beta, getattr_l__mod___blocks___10___norm2_alpha, x_84);  getattr_l__mod___blocks___10___norm2_beta = getattr_l__mod___blocks___10___norm2_alpha = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_85 = self.getattr_L__mod___blocks___10___mlp_channels_fc1(addcmul_21);  addcmul_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_86 = self.getattr_L__mod___blocks___10___mlp_channels_act(x_85);  x_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_87 = self.getattr_L__mod___blocks___10___mlp_channels_drop1(x_86);  x_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_88 = self.getattr_L__mod___blocks___10___mlp_channels_norm(x_87);  x_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_89 = self.getattr_L__mod___blocks___10___mlp_channels_fc2(x_88);  x_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_90 = self.getattr_L__mod___blocks___10___mlp_channels_drop2(x_89);  x_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_21 = getattr_l__mod___blocks___10___ls2 * x_90;  getattr_l__mod___blocks___10___ls2 = x_90 = None
    getattr_l__mod___blocks___10___drop_path_1 = self.getattr_L__mod___blocks___10___drop_path(mul_21);  mul_21 = None
    x_91 = x_84 + getattr_l__mod___blocks___10___drop_path_1;  x_84 = getattr_l__mod___blocks___10___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___11___ls1 = self.getattr_L__mod___blocks___11___ls1
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    getattr_l__mod___blocks___11___norm1_beta = self.getattr_L__mod___blocks___11___norm1_beta
    getattr_l__mod___blocks___11___norm1_alpha = self.getattr_L__mod___blocks___11___norm1_alpha
    addcmul_22 = torch.addcmul(getattr_l__mod___blocks___11___norm1_beta, getattr_l__mod___blocks___11___norm1_alpha, x_91);  getattr_l__mod___blocks___11___norm1_beta = getattr_l__mod___blocks___11___norm1_alpha = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_23 = addcmul_22.transpose(1, 2);  addcmul_22 = None
    getattr_l__mod___blocks___11___linear_tokens = self.getattr_L__mod___blocks___11___linear_tokens(transpose_23);  transpose_23 = None
    transpose_24 = getattr_l__mod___blocks___11___linear_tokens.transpose(1, 2);  getattr_l__mod___blocks___11___linear_tokens = None
    mul_22 = getattr_l__mod___blocks___11___ls1 * transpose_24;  getattr_l__mod___blocks___11___ls1 = transpose_24 = None
    getattr_l__mod___blocks___11___drop_path = self.getattr_L__mod___blocks___11___drop_path(mul_22);  mul_22 = None
    x_92 = x_91 + getattr_l__mod___blocks___11___drop_path;  x_91 = getattr_l__mod___blocks___11___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___11___ls2 = self.getattr_L__mod___blocks___11___ls2
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    getattr_l__mod___blocks___11___norm2_beta = self.getattr_L__mod___blocks___11___norm2_beta
    getattr_l__mod___blocks___11___norm2_alpha = self.getattr_L__mod___blocks___11___norm2_alpha
    addcmul_23 = torch.addcmul(getattr_l__mod___blocks___11___norm2_beta, getattr_l__mod___blocks___11___norm2_alpha, x_92);  getattr_l__mod___blocks___11___norm2_beta = getattr_l__mod___blocks___11___norm2_alpha = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_93 = self.getattr_L__mod___blocks___11___mlp_channels_fc1(addcmul_23);  addcmul_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_94 = self.getattr_L__mod___blocks___11___mlp_channels_act(x_93);  x_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_95 = self.getattr_L__mod___blocks___11___mlp_channels_drop1(x_94);  x_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_96 = self.getattr_L__mod___blocks___11___mlp_channels_norm(x_95);  x_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_97 = self.getattr_L__mod___blocks___11___mlp_channels_fc2(x_96);  x_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_98 = self.getattr_L__mod___blocks___11___mlp_channels_drop2(x_97);  x_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_23 = getattr_l__mod___blocks___11___ls2 * x_98;  getattr_l__mod___blocks___11___ls2 = x_98 = None
    getattr_l__mod___blocks___11___drop_path_1 = self.getattr_L__mod___blocks___11___drop_path(mul_23);  mul_23 = None
    x_100 = x_92 + getattr_l__mod___blocks___11___drop_path_1;  x_92 = getattr_l__mod___blocks___11___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    l__mod___norm_beta = self.L__mod___norm_beta
    l__mod___norm_alpha = self.L__mod___norm_alpha
    x_102 = torch.addcmul(l__mod___norm_beta, l__mod___norm_alpha, x_100);  l__mod___norm_beta = l__mod___norm_alpha = x_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:271, code: x = x.mean(dim=1)
    x_103 = x_102.mean(dim = 1);  x_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:272, code: x = self.head_drop(x)
    x_104 = self.L__mod___head_drop(x_103);  x_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:273, code: return x if pre_logits else self.head(x)
    pred = self.L__mod___head(x_104);  x_104 = None
    return (pred,)
    