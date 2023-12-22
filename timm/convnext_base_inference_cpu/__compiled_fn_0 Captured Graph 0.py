from __future__ import annotations



def forward(self, L_inputs_0_ : torch.Tensor):
    l_inputs_0_ = L_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:411, code: x = self.stem(x)
    l__mod___stem_0 = self.L__mod___stem_0(l_inputs_0_);  l_inputs_0_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:68, code: x = x.permute(0, 2, 3, 1)
    x = l__mod___stem_0.permute(0, 2, 3, 1);  l__mod___stem_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_l__mod___stem___1___weight = self.getattr_L__mod___stem___1___weight
    getattr_l__mod___stem___1___bias = self.getattr_L__mod___stem___1___bias
    x_1 = torch.nn.functional.layer_norm(x, (128,), getattr_l__mod___stem___1___weight, getattr_l__mod___stem___1___bias, 1e-06);  x = getattr_l__mod___stem___1___weight = getattr_l__mod___stem___1___bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:73, code: x = x.permute(0, 3, 1, 2)
    x_3 = x_1.permute(0, 3, 1, 2);  x_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:229, code: x = self.downsample(x)
    shortcut = self.getattr_L__mod___stages___0___downsample(x_3);  x_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    x_5 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_dw(shortcut)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    x_6 = x_5.permute(0, 2, 3, 1);  x_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___0___blocks___0___norm_weight = self.getattr_getattr_L__mod___stages___0___blocks___0___norm_weight
    getattr_getattr_l__mod___stages___0___blocks___0___norm_bias = self.getattr_getattr_L__mod___stages___0___blocks___0___norm_bias
    x_8 = torch.nn.functional.layer_norm(x_6, (128,), getattr_getattr_l__mod___stages___0___blocks___0___norm_weight, getattr_getattr_l__mod___stages___0___blocks___0___norm_bias, 1e-06);  x_6 = getattr_getattr_l__mod___stages___0___blocks___0___norm_weight = getattr_getattr_l__mod___stages___0___blocks___0___norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_9 = self.getattr_getattr_L__mod___stages___0___blocks___0___mlp_fc1(x_8);  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    x_10 = torch._C._nn.gelu(x_9);  x_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_11 = self.getattr_getattr_L__mod___stages___0___blocks___0___mlp_drop1(x_10);  x_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_12 = self.getattr_getattr_L__mod___stages___0___blocks___0___mlp_norm(x_11);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_13 = self.getattr_getattr_L__mod___stages___0___blocks___0___mlp_fc2(x_12);  x_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_15 = self.getattr_getattr_L__mod___stages___0___blocks___0___mlp_drop2(x_13);  x_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    x_16 = x_15.permute(0, 3, 1, 2);  x_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:161, code: if self.gamma is not None:
    getattr_getattr_l__mod___stages___0___blocks___0___gamma = self.getattr_getattr_L__mod___stages___0___blocks___0___gamma
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    reshape = getattr_getattr_l__mod___stages___0___blocks___0___gamma.reshape(1, -1, 1, 1);  getattr_getattr_l__mod___stages___0___blocks___0___gamma = None
    x_17 = x_16.mul(reshape);  x_16 = reshape = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___0___blocks___0___drop_path = self.getattr_getattr_L__mod___stages___0___blocks___0___drop_path(x_17);  x_17 = None
    getattr_getattr_l__mod___stages___0___blocks___0___shortcut = self.getattr_getattr_L__mod___stages___0___blocks___0___shortcut(shortcut);  shortcut = None
    shortcut_1 = getattr_getattr_l__mod___stages___0___blocks___0___drop_path + getattr_getattr_l__mod___stages___0___blocks___0___shortcut;  getattr_getattr_l__mod___stages___0___blocks___0___drop_path = getattr_getattr_l__mod___stages___0___blocks___0___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    x_19 = self.getattr_getattr_L__mod___stages___0___blocks___1___conv_dw(shortcut_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    x_20 = x_19.permute(0, 2, 3, 1);  x_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___0___blocks___1___norm_weight = self.getattr_getattr_L__mod___stages___0___blocks___1___norm_weight
    getattr_getattr_l__mod___stages___0___blocks___1___norm_bias = self.getattr_getattr_L__mod___stages___0___blocks___1___norm_bias
    x_22 = torch.nn.functional.layer_norm(x_20, (128,), getattr_getattr_l__mod___stages___0___blocks___1___norm_weight, getattr_getattr_l__mod___stages___0___blocks___1___norm_bias, 1e-06);  x_20 = getattr_getattr_l__mod___stages___0___blocks___1___norm_weight = getattr_getattr_l__mod___stages___0___blocks___1___norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_23 = self.getattr_getattr_L__mod___stages___0___blocks___1___mlp_fc1(x_22);  x_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    x_24 = torch._C._nn.gelu(x_23);  x_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_25 = self.getattr_getattr_L__mod___stages___0___blocks___1___mlp_drop1(x_24);  x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_26 = self.getattr_getattr_L__mod___stages___0___blocks___1___mlp_norm(x_25);  x_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_27 = self.getattr_getattr_L__mod___stages___0___blocks___1___mlp_fc2(x_26);  x_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_29 = self.getattr_getattr_L__mod___stages___0___blocks___1___mlp_drop2(x_27);  x_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    x_30 = x_29.permute(0, 3, 1, 2);  x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:161, code: if self.gamma is not None:
    getattr_getattr_l__mod___stages___0___blocks___1___gamma = self.getattr_getattr_L__mod___stages___0___blocks___1___gamma
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    reshape_1 = getattr_getattr_l__mod___stages___0___blocks___1___gamma.reshape(1, -1, 1, 1);  getattr_getattr_l__mod___stages___0___blocks___1___gamma = None
    x_31 = x_30.mul(reshape_1);  x_30 = reshape_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___0___blocks___1___drop_path = self.getattr_getattr_L__mod___stages___0___blocks___1___drop_path(x_31);  x_31 = None
    getattr_getattr_l__mod___stages___0___blocks___1___shortcut = self.getattr_getattr_L__mod___stages___0___blocks___1___shortcut(shortcut_1);  shortcut_1 = None
    shortcut_2 = getattr_getattr_l__mod___stages___0___blocks___1___drop_path + getattr_getattr_l__mod___stages___0___blocks___1___shortcut;  getattr_getattr_l__mod___stages___0___blocks___1___drop_path = getattr_getattr_l__mod___stages___0___blocks___1___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    x_33 = self.getattr_getattr_L__mod___stages___0___blocks___2___conv_dw(shortcut_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    x_34 = x_33.permute(0, 2, 3, 1);  x_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___0___blocks___2___norm_weight = self.getattr_getattr_L__mod___stages___0___blocks___2___norm_weight
    getattr_getattr_l__mod___stages___0___blocks___2___norm_bias = self.getattr_getattr_L__mod___stages___0___blocks___2___norm_bias
    x_36 = torch.nn.functional.layer_norm(x_34, (128,), getattr_getattr_l__mod___stages___0___blocks___2___norm_weight, getattr_getattr_l__mod___stages___0___blocks___2___norm_bias, 1e-06);  x_34 = getattr_getattr_l__mod___stages___0___blocks___2___norm_weight = getattr_getattr_l__mod___stages___0___blocks___2___norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_37 = self.getattr_getattr_L__mod___stages___0___blocks___2___mlp_fc1(x_36);  x_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    x_38 = torch._C._nn.gelu(x_37);  x_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_39 = self.getattr_getattr_L__mod___stages___0___blocks___2___mlp_drop1(x_38);  x_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_40 = self.getattr_getattr_L__mod___stages___0___blocks___2___mlp_norm(x_39);  x_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_41 = self.getattr_getattr_L__mod___stages___0___blocks___2___mlp_fc2(x_40);  x_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_43 = self.getattr_getattr_L__mod___stages___0___blocks___2___mlp_drop2(x_41);  x_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    x_44 = x_43.permute(0, 3, 1, 2);  x_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:161, code: if self.gamma is not None:
    getattr_getattr_l__mod___stages___0___blocks___2___gamma = self.getattr_getattr_L__mod___stages___0___blocks___2___gamma
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    reshape_2 = getattr_getattr_l__mod___stages___0___blocks___2___gamma.reshape(1, -1, 1, 1);  getattr_getattr_l__mod___stages___0___blocks___2___gamma = None
    x_45 = x_44.mul(reshape_2);  x_44 = reshape_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___0___blocks___2___drop_path = self.getattr_getattr_L__mod___stages___0___blocks___2___drop_path(x_45);  x_45 = None
    getattr_getattr_l__mod___stages___0___blocks___2___shortcut = self.getattr_getattr_L__mod___stages___0___blocks___2___shortcut(shortcut_2);  shortcut_2 = None
    x_47 = getattr_getattr_l__mod___stages___0___blocks___2___drop_path + getattr_getattr_l__mod___stages___0___blocks___2___shortcut;  getattr_getattr_l__mod___stages___0___blocks___2___drop_path = getattr_getattr_l__mod___stages___0___blocks___2___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:68, code: x = x.permute(0, 2, 3, 1)
    x_48 = x_47.permute(0, 2, 3, 1);  x_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___1___downsample___0___weight = self.getattr_getattr_L__mod___stages___1___downsample___0___weight
    getattr_getattr_l__mod___stages___1___downsample___0___bias = self.getattr_getattr_L__mod___stages___1___downsample___0___bias
    x_49 = torch.nn.functional.layer_norm(x_48, (128,), getattr_getattr_l__mod___stages___1___downsample___0___weight, getattr_getattr_l__mod___stages___1___downsample___0___bias, 1e-06);  x_48 = getattr_getattr_l__mod___stages___1___downsample___0___weight = getattr_getattr_l__mod___stages___1___downsample___0___bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:73, code: x = x.permute(0, 3, 1, 2)
    x_50 = x_49.permute(0, 3, 1, 2);  x_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:229, code: x = self.downsample(x)
    shortcut_3 = self.getattr_L__mod___stages___1___downsample_1(x_50);  x_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    x_52 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_dw(shortcut_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    x_53 = x_52.permute(0, 2, 3, 1);  x_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___1___blocks___0___norm_weight = self.getattr_getattr_L__mod___stages___1___blocks___0___norm_weight
    getattr_getattr_l__mod___stages___1___blocks___0___norm_bias = self.getattr_getattr_L__mod___stages___1___blocks___0___norm_bias
    x_55 = torch.nn.functional.layer_norm(x_53, (256,), getattr_getattr_l__mod___stages___1___blocks___0___norm_weight, getattr_getattr_l__mod___stages___1___blocks___0___norm_bias, 1e-06);  x_53 = getattr_getattr_l__mod___stages___1___blocks___0___norm_weight = getattr_getattr_l__mod___stages___1___blocks___0___norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_56 = self.getattr_getattr_L__mod___stages___1___blocks___0___mlp_fc1(x_55);  x_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    x_57 = torch._C._nn.gelu(x_56);  x_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_58 = self.getattr_getattr_L__mod___stages___1___blocks___0___mlp_drop1(x_57);  x_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_59 = self.getattr_getattr_L__mod___stages___1___blocks___0___mlp_norm(x_58);  x_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_60 = self.getattr_getattr_L__mod___stages___1___blocks___0___mlp_fc2(x_59);  x_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_62 = self.getattr_getattr_L__mod___stages___1___blocks___0___mlp_drop2(x_60);  x_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    x_63 = x_62.permute(0, 3, 1, 2);  x_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:161, code: if self.gamma is not None:
    getattr_getattr_l__mod___stages___1___blocks___0___gamma = self.getattr_getattr_L__mod___stages___1___blocks___0___gamma
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    reshape_3 = getattr_getattr_l__mod___stages___1___blocks___0___gamma.reshape(1, -1, 1, 1);  getattr_getattr_l__mod___stages___1___blocks___0___gamma = None
    x_64 = x_63.mul(reshape_3);  x_63 = reshape_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___1___blocks___0___drop_path = self.getattr_getattr_L__mod___stages___1___blocks___0___drop_path(x_64);  x_64 = None
    getattr_getattr_l__mod___stages___1___blocks___0___shortcut = self.getattr_getattr_L__mod___stages___1___blocks___0___shortcut(shortcut_3);  shortcut_3 = None
    shortcut_4 = getattr_getattr_l__mod___stages___1___blocks___0___drop_path + getattr_getattr_l__mod___stages___1___blocks___0___shortcut;  getattr_getattr_l__mod___stages___1___blocks___0___drop_path = getattr_getattr_l__mod___stages___1___blocks___0___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    x_66 = self.getattr_getattr_L__mod___stages___1___blocks___1___conv_dw(shortcut_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    x_67 = x_66.permute(0, 2, 3, 1);  x_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___1___blocks___1___norm_weight = self.getattr_getattr_L__mod___stages___1___blocks___1___norm_weight
    getattr_getattr_l__mod___stages___1___blocks___1___norm_bias = self.getattr_getattr_L__mod___stages___1___blocks___1___norm_bias
    x_69 = torch.nn.functional.layer_norm(x_67, (256,), getattr_getattr_l__mod___stages___1___blocks___1___norm_weight, getattr_getattr_l__mod___stages___1___blocks___1___norm_bias, 1e-06);  x_67 = getattr_getattr_l__mod___stages___1___blocks___1___norm_weight = getattr_getattr_l__mod___stages___1___blocks___1___norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_70 = self.getattr_getattr_L__mod___stages___1___blocks___1___mlp_fc1(x_69);  x_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    x_71 = torch._C._nn.gelu(x_70);  x_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_72 = self.getattr_getattr_L__mod___stages___1___blocks___1___mlp_drop1(x_71);  x_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_73 = self.getattr_getattr_L__mod___stages___1___blocks___1___mlp_norm(x_72);  x_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_74 = self.getattr_getattr_L__mod___stages___1___blocks___1___mlp_fc2(x_73);  x_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_76 = self.getattr_getattr_L__mod___stages___1___blocks___1___mlp_drop2(x_74);  x_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    x_77 = x_76.permute(0, 3, 1, 2);  x_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:161, code: if self.gamma is not None:
    getattr_getattr_l__mod___stages___1___blocks___1___gamma = self.getattr_getattr_L__mod___stages___1___blocks___1___gamma
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    reshape_4 = getattr_getattr_l__mod___stages___1___blocks___1___gamma.reshape(1, -1, 1, 1);  getattr_getattr_l__mod___stages___1___blocks___1___gamma = None
    x_78 = x_77.mul(reshape_4);  x_77 = reshape_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___1___blocks___1___drop_path = self.getattr_getattr_L__mod___stages___1___blocks___1___drop_path(x_78);  x_78 = None
    getattr_getattr_l__mod___stages___1___blocks___1___shortcut = self.getattr_getattr_L__mod___stages___1___blocks___1___shortcut(shortcut_4);  shortcut_4 = None
    shortcut_5 = getattr_getattr_l__mod___stages___1___blocks___1___drop_path + getattr_getattr_l__mod___stages___1___blocks___1___shortcut;  getattr_getattr_l__mod___stages___1___blocks___1___drop_path = getattr_getattr_l__mod___stages___1___blocks___1___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    x_80 = self.getattr_getattr_L__mod___stages___1___blocks___2___conv_dw(shortcut_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    x_81 = x_80.permute(0, 2, 3, 1);  x_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___1___blocks___2___norm_weight = self.getattr_getattr_L__mod___stages___1___blocks___2___norm_weight
    getattr_getattr_l__mod___stages___1___blocks___2___norm_bias = self.getattr_getattr_L__mod___stages___1___blocks___2___norm_bias
    x_83 = torch.nn.functional.layer_norm(x_81, (256,), getattr_getattr_l__mod___stages___1___blocks___2___norm_weight, getattr_getattr_l__mod___stages___1___blocks___2___norm_bias, 1e-06);  x_81 = getattr_getattr_l__mod___stages___1___blocks___2___norm_weight = getattr_getattr_l__mod___stages___1___blocks___2___norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_84 = self.getattr_getattr_L__mod___stages___1___blocks___2___mlp_fc1(x_83);  x_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    x_85 = torch._C._nn.gelu(x_84);  x_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_86 = self.getattr_getattr_L__mod___stages___1___blocks___2___mlp_drop1(x_85);  x_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_87 = self.getattr_getattr_L__mod___stages___1___blocks___2___mlp_norm(x_86);  x_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_88 = self.getattr_getattr_L__mod___stages___1___blocks___2___mlp_fc2(x_87);  x_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_90 = self.getattr_getattr_L__mod___stages___1___blocks___2___mlp_drop2(x_88);  x_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    x_91 = x_90.permute(0, 3, 1, 2);  x_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:161, code: if self.gamma is not None:
    getattr_getattr_l__mod___stages___1___blocks___2___gamma = self.getattr_getattr_L__mod___stages___1___blocks___2___gamma
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    reshape_5 = getattr_getattr_l__mod___stages___1___blocks___2___gamma.reshape(1, -1, 1, 1);  getattr_getattr_l__mod___stages___1___blocks___2___gamma = None
    x_92 = x_91.mul(reshape_5);  x_91 = reshape_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___1___blocks___2___drop_path = self.getattr_getattr_L__mod___stages___1___blocks___2___drop_path(x_92);  x_92 = None
    getattr_getattr_l__mod___stages___1___blocks___2___shortcut = self.getattr_getattr_L__mod___stages___1___blocks___2___shortcut(shortcut_5);  shortcut_5 = None
    x_94 = getattr_getattr_l__mod___stages___1___blocks___2___drop_path + getattr_getattr_l__mod___stages___1___blocks___2___shortcut;  getattr_getattr_l__mod___stages___1___blocks___2___drop_path = getattr_getattr_l__mod___stages___1___blocks___2___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:68, code: x = x.permute(0, 2, 3, 1)
    x_95 = x_94.permute(0, 2, 3, 1);  x_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___downsample___0___weight = self.getattr_getattr_L__mod___stages___2___downsample___0___weight
    getattr_getattr_l__mod___stages___2___downsample___0___bias = self.getattr_getattr_L__mod___stages___2___downsample___0___bias
    x_96 = torch.nn.functional.layer_norm(x_95, (256,), getattr_getattr_l__mod___stages___2___downsample___0___weight, getattr_getattr_l__mod___stages___2___downsample___0___bias, 1e-06);  x_95 = getattr_getattr_l__mod___stages___2___downsample___0___weight = getattr_getattr_l__mod___stages___2___downsample___0___bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:73, code: x = x.permute(0, 3, 1, 2)
    x_97 = x_96.permute(0, 3, 1, 2);  x_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:229, code: x = self.downsample(x)
    shortcut_6 = self.getattr_L__mod___stages___2___downsample_1(x_97);  x_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    x_99 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_dw(shortcut_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    x_100 = x_99.permute(0, 2, 3, 1);  x_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___0___norm_weight = self.getattr_getattr_L__mod___stages___2___blocks___0___norm_weight
    getattr_getattr_l__mod___stages___2___blocks___0___norm_bias = self.getattr_getattr_L__mod___stages___2___blocks___0___norm_bias
    x_102 = torch.nn.functional.layer_norm(x_100, (512,), getattr_getattr_l__mod___stages___2___blocks___0___norm_weight, getattr_getattr_l__mod___stages___2___blocks___0___norm_bias, 1e-06);  x_100 = getattr_getattr_l__mod___stages___2___blocks___0___norm_weight = getattr_getattr_l__mod___stages___2___blocks___0___norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_103 = self.getattr_getattr_L__mod___stages___2___blocks___0___mlp_fc1(x_102);  x_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    x_104 = torch._C._nn.gelu(x_103);  x_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_105 = self.getattr_getattr_L__mod___stages___2___blocks___0___mlp_drop1(x_104);  x_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_106 = self.getattr_getattr_L__mod___stages___2___blocks___0___mlp_norm(x_105);  x_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_107 = self.getattr_getattr_L__mod___stages___2___blocks___0___mlp_fc2(x_106);  x_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_109 = self.getattr_getattr_L__mod___stages___2___blocks___0___mlp_drop2(x_107);  x_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    x_110 = x_109.permute(0, 3, 1, 2);  x_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:161, code: if self.gamma is not None:
    getattr_getattr_l__mod___stages___2___blocks___0___gamma = self.getattr_getattr_L__mod___stages___2___blocks___0___gamma
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    reshape_6 = getattr_getattr_l__mod___stages___2___blocks___0___gamma.reshape(1, -1, 1, 1);  getattr_getattr_l__mod___stages___2___blocks___0___gamma = None
    x_111 = x_110.mul(reshape_6);  x_110 = reshape_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___2___blocks___0___drop_path = self.getattr_getattr_L__mod___stages___2___blocks___0___drop_path(x_111);  x_111 = None
    getattr_getattr_l__mod___stages___2___blocks___0___shortcut = self.getattr_getattr_L__mod___stages___2___blocks___0___shortcut(shortcut_6);  shortcut_6 = None
    shortcut_7 = getattr_getattr_l__mod___stages___2___blocks___0___drop_path + getattr_getattr_l__mod___stages___2___blocks___0___shortcut;  getattr_getattr_l__mod___stages___2___blocks___0___drop_path = getattr_getattr_l__mod___stages___2___blocks___0___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    x_113 = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_dw(shortcut_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    x_114 = x_113.permute(0, 2, 3, 1);  x_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___1___norm_weight = self.getattr_getattr_L__mod___stages___2___blocks___1___norm_weight
    getattr_getattr_l__mod___stages___2___blocks___1___norm_bias = self.getattr_getattr_L__mod___stages___2___blocks___1___norm_bias
    x_116 = torch.nn.functional.layer_norm(x_114, (512,), getattr_getattr_l__mod___stages___2___blocks___1___norm_weight, getattr_getattr_l__mod___stages___2___blocks___1___norm_bias, 1e-06);  x_114 = getattr_getattr_l__mod___stages___2___blocks___1___norm_weight = getattr_getattr_l__mod___stages___2___blocks___1___norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_117 = self.getattr_getattr_L__mod___stages___2___blocks___1___mlp_fc1(x_116);  x_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    x_118 = torch._C._nn.gelu(x_117);  x_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_119 = self.getattr_getattr_L__mod___stages___2___blocks___1___mlp_drop1(x_118);  x_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_120 = self.getattr_getattr_L__mod___stages___2___blocks___1___mlp_norm(x_119);  x_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_121 = self.getattr_getattr_L__mod___stages___2___blocks___1___mlp_fc2(x_120);  x_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_123 = self.getattr_getattr_L__mod___stages___2___blocks___1___mlp_drop2(x_121);  x_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    x_124 = x_123.permute(0, 3, 1, 2);  x_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:161, code: if self.gamma is not None:
    getattr_getattr_l__mod___stages___2___blocks___1___gamma = self.getattr_getattr_L__mod___stages___2___blocks___1___gamma
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    reshape_7 = getattr_getattr_l__mod___stages___2___blocks___1___gamma.reshape(1, -1, 1, 1);  getattr_getattr_l__mod___stages___2___blocks___1___gamma = None
    x_125 = x_124.mul(reshape_7);  x_124 = reshape_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___2___blocks___1___drop_path = self.getattr_getattr_L__mod___stages___2___blocks___1___drop_path(x_125);  x_125 = None
    getattr_getattr_l__mod___stages___2___blocks___1___shortcut = self.getattr_getattr_L__mod___stages___2___blocks___1___shortcut(shortcut_7);  shortcut_7 = None
    shortcut_8 = getattr_getattr_l__mod___stages___2___blocks___1___drop_path + getattr_getattr_l__mod___stages___2___blocks___1___shortcut;  getattr_getattr_l__mod___stages___2___blocks___1___drop_path = getattr_getattr_l__mod___stages___2___blocks___1___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    x_127 = self.getattr_getattr_L__mod___stages___2___blocks___2___conv_dw(shortcut_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    x_128 = x_127.permute(0, 2, 3, 1);  x_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___2___norm_weight = self.getattr_getattr_L__mod___stages___2___blocks___2___norm_weight
    getattr_getattr_l__mod___stages___2___blocks___2___norm_bias = self.getattr_getattr_L__mod___stages___2___blocks___2___norm_bias
    x_130 = torch.nn.functional.layer_norm(x_128, (512,), getattr_getattr_l__mod___stages___2___blocks___2___norm_weight, getattr_getattr_l__mod___stages___2___blocks___2___norm_bias, 1e-06);  x_128 = getattr_getattr_l__mod___stages___2___blocks___2___norm_weight = getattr_getattr_l__mod___stages___2___blocks___2___norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_131 = self.getattr_getattr_L__mod___stages___2___blocks___2___mlp_fc1(x_130);  x_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    x_132 = torch._C._nn.gelu(x_131);  x_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_133 = self.getattr_getattr_L__mod___stages___2___blocks___2___mlp_drop1(x_132);  x_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_134 = self.getattr_getattr_L__mod___stages___2___blocks___2___mlp_norm(x_133);  x_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_135 = self.getattr_getattr_L__mod___stages___2___blocks___2___mlp_fc2(x_134);  x_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_137 = self.getattr_getattr_L__mod___stages___2___blocks___2___mlp_drop2(x_135);  x_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    x_138 = x_137.permute(0, 3, 1, 2);  x_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:161, code: if self.gamma is not None:
    getattr_getattr_l__mod___stages___2___blocks___2___gamma = self.getattr_getattr_L__mod___stages___2___blocks___2___gamma
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    reshape_8 = getattr_getattr_l__mod___stages___2___blocks___2___gamma.reshape(1, -1, 1, 1);  getattr_getattr_l__mod___stages___2___blocks___2___gamma = None
    x_139 = x_138.mul(reshape_8);  x_138 = reshape_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___2___blocks___2___drop_path = self.getattr_getattr_L__mod___stages___2___blocks___2___drop_path(x_139);  x_139 = None
    getattr_getattr_l__mod___stages___2___blocks___2___shortcut = self.getattr_getattr_L__mod___stages___2___blocks___2___shortcut(shortcut_8);  shortcut_8 = None
    shortcut_9 = getattr_getattr_l__mod___stages___2___blocks___2___drop_path + getattr_getattr_l__mod___stages___2___blocks___2___shortcut;  getattr_getattr_l__mod___stages___2___blocks___2___drop_path = getattr_getattr_l__mod___stages___2___blocks___2___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    x_141 = self.getattr_getattr_L__mod___stages___2___blocks___3___conv_dw(shortcut_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    x_142 = x_141.permute(0, 2, 3, 1);  x_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___3___norm_weight = self.getattr_getattr_L__mod___stages___2___blocks___3___norm_weight
    getattr_getattr_l__mod___stages___2___blocks___3___norm_bias = self.getattr_getattr_L__mod___stages___2___blocks___3___norm_bias
    x_144 = torch.nn.functional.layer_norm(x_142, (512,), getattr_getattr_l__mod___stages___2___blocks___3___norm_weight, getattr_getattr_l__mod___stages___2___blocks___3___norm_bias, 1e-06);  x_142 = getattr_getattr_l__mod___stages___2___blocks___3___norm_weight = getattr_getattr_l__mod___stages___2___blocks___3___norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_145 = self.getattr_getattr_L__mod___stages___2___blocks___3___mlp_fc1(x_144);  x_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    x_146 = torch._C._nn.gelu(x_145);  x_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_147 = self.getattr_getattr_L__mod___stages___2___blocks___3___mlp_drop1(x_146);  x_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_148 = self.getattr_getattr_L__mod___stages___2___blocks___3___mlp_norm(x_147);  x_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_149 = self.getattr_getattr_L__mod___stages___2___blocks___3___mlp_fc2(x_148);  x_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_151 = self.getattr_getattr_L__mod___stages___2___blocks___3___mlp_drop2(x_149);  x_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    x_152 = x_151.permute(0, 3, 1, 2);  x_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:161, code: if self.gamma is not None:
    getattr_getattr_l__mod___stages___2___blocks___3___gamma = self.getattr_getattr_L__mod___stages___2___blocks___3___gamma
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    reshape_9 = getattr_getattr_l__mod___stages___2___blocks___3___gamma.reshape(1, -1, 1, 1);  getattr_getattr_l__mod___stages___2___blocks___3___gamma = None
    x_153 = x_152.mul(reshape_9);  x_152 = reshape_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___2___blocks___3___drop_path = self.getattr_getattr_L__mod___stages___2___blocks___3___drop_path(x_153);  x_153 = None
    getattr_getattr_l__mod___stages___2___blocks___3___shortcut = self.getattr_getattr_L__mod___stages___2___blocks___3___shortcut(shortcut_9);  shortcut_9 = None
    shortcut_10 = getattr_getattr_l__mod___stages___2___blocks___3___drop_path + getattr_getattr_l__mod___stages___2___blocks___3___shortcut;  getattr_getattr_l__mod___stages___2___blocks___3___drop_path = getattr_getattr_l__mod___stages___2___blocks___3___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    x_155 = self.getattr_getattr_L__mod___stages___2___blocks___4___conv_dw(shortcut_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    x_156 = x_155.permute(0, 2, 3, 1);  x_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___4___norm_weight = self.getattr_getattr_L__mod___stages___2___blocks___4___norm_weight
    getattr_getattr_l__mod___stages___2___blocks___4___norm_bias = self.getattr_getattr_L__mod___stages___2___blocks___4___norm_bias
    x_158 = torch.nn.functional.layer_norm(x_156, (512,), getattr_getattr_l__mod___stages___2___blocks___4___norm_weight, getattr_getattr_l__mod___stages___2___blocks___4___norm_bias, 1e-06);  x_156 = getattr_getattr_l__mod___stages___2___blocks___4___norm_weight = getattr_getattr_l__mod___stages___2___blocks___4___norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_159 = self.getattr_getattr_L__mod___stages___2___blocks___4___mlp_fc1(x_158);  x_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    x_160 = torch._C._nn.gelu(x_159);  x_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_161 = self.getattr_getattr_L__mod___stages___2___blocks___4___mlp_drop1(x_160);  x_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_162 = self.getattr_getattr_L__mod___stages___2___blocks___4___mlp_norm(x_161);  x_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_163 = self.getattr_getattr_L__mod___stages___2___blocks___4___mlp_fc2(x_162);  x_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_165 = self.getattr_getattr_L__mod___stages___2___blocks___4___mlp_drop2(x_163);  x_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    x_166 = x_165.permute(0, 3, 1, 2);  x_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:161, code: if self.gamma is not None:
    getattr_getattr_l__mod___stages___2___blocks___4___gamma = self.getattr_getattr_L__mod___stages___2___blocks___4___gamma
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    reshape_10 = getattr_getattr_l__mod___stages___2___blocks___4___gamma.reshape(1, -1, 1, 1);  getattr_getattr_l__mod___stages___2___blocks___4___gamma = None
    x_167 = x_166.mul(reshape_10);  x_166 = reshape_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___2___blocks___4___drop_path = self.getattr_getattr_L__mod___stages___2___blocks___4___drop_path(x_167);  x_167 = None
    getattr_getattr_l__mod___stages___2___blocks___4___shortcut = self.getattr_getattr_L__mod___stages___2___blocks___4___shortcut(shortcut_10);  shortcut_10 = None
    shortcut_11 = getattr_getattr_l__mod___stages___2___blocks___4___drop_path + getattr_getattr_l__mod___stages___2___blocks___4___shortcut;  getattr_getattr_l__mod___stages___2___blocks___4___drop_path = getattr_getattr_l__mod___stages___2___blocks___4___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    x_169 = self.getattr_getattr_L__mod___stages___2___blocks___5___conv_dw(shortcut_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    x_170 = x_169.permute(0, 2, 3, 1);  x_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___5___norm_weight = self.getattr_getattr_L__mod___stages___2___blocks___5___norm_weight
    getattr_getattr_l__mod___stages___2___blocks___5___norm_bias = self.getattr_getattr_L__mod___stages___2___blocks___5___norm_bias
    x_172 = torch.nn.functional.layer_norm(x_170, (512,), getattr_getattr_l__mod___stages___2___blocks___5___norm_weight, getattr_getattr_l__mod___stages___2___blocks___5___norm_bias, 1e-06);  x_170 = getattr_getattr_l__mod___stages___2___blocks___5___norm_weight = getattr_getattr_l__mod___stages___2___blocks___5___norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_173 = self.getattr_getattr_L__mod___stages___2___blocks___5___mlp_fc1(x_172);  x_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    x_174 = torch._C._nn.gelu(x_173);  x_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_175 = self.getattr_getattr_L__mod___stages___2___blocks___5___mlp_drop1(x_174);  x_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_176 = self.getattr_getattr_L__mod___stages___2___blocks___5___mlp_norm(x_175);  x_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_177 = self.getattr_getattr_L__mod___stages___2___blocks___5___mlp_fc2(x_176);  x_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_179 = self.getattr_getattr_L__mod___stages___2___blocks___5___mlp_drop2(x_177);  x_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    x_180 = x_179.permute(0, 3, 1, 2);  x_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:161, code: if self.gamma is not None:
    getattr_getattr_l__mod___stages___2___blocks___5___gamma = self.getattr_getattr_L__mod___stages___2___blocks___5___gamma
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    reshape_11 = getattr_getattr_l__mod___stages___2___blocks___5___gamma.reshape(1, -1, 1, 1);  getattr_getattr_l__mod___stages___2___blocks___5___gamma = None
    x_181 = x_180.mul(reshape_11);  x_180 = reshape_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___2___blocks___5___drop_path = self.getattr_getattr_L__mod___stages___2___blocks___5___drop_path(x_181);  x_181 = None
    getattr_getattr_l__mod___stages___2___blocks___5___shortcut = self.getattr_getattr_L__mod___stages___2___blocks___5___shortcut(shortcut_11);  shortcut_11 = None
    shortcut_12 = getattr_getattr_l__mod___stages___2___blocks___5___drop_path + getattr_getattr_l__mod___stages___2___blocks___5___shortcut;  getattr_getattr_l__mod___stages___2___blocks___5___drop_path = getattr_getattr_l__mod___stages___2___blocks___5___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    x_183 = self.getattr_getattr_L__mod___stages___2___blocks___6___conv_dw(shortcut_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    x_184 = x_183.permute(0, 2, 3, 1);  x_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___6___norm_weight = self.getattr_getattr_L__mod___stages___2___blocks___6___norm_weight
    getattr_getattr_l__mod___stages___2___blocks___6___norm_bias = self.getattr_getattr_L__mod___stages___2___blocks___6___norm_bias
    x_186 = torch.nn.functional.layer_norm(x_184, (512,), getattr_getattr_l__mod___stages___2___blocks___6___norm_weight, getattr_getattr_l__mod___stages___2___blocks___6___norm_bias, 1e-06);  x_184 = getattr_getattr_l__mod___stages___2___blocks___6___norm_weight = getattr_getattr_l__mod___stages___2___blocks___6___norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_187 = self.getattr_getattr_L__mod___stages___2___blocks___6___mlp_fc1(x_186);  x_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    x_188 = torch._C._nn.gelu(x_187);  x_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_189 = self.getattr_getattr_L__mod___stages___2___blocks___6___mlp_drop1(x_188);  x_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_190 = self.getattr_getattr_L__mod___stages___2___blocks___6___mlp_norm(x_189);  x_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_191 = self.getattr_getattr_L__mod___stages___2___blocks___6___mlp_fc2(x_190);  x_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_193 = self.getattr_getattr_L__mod___stages___2___blocks___6___mlp_drop2(x_191);  x_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    x_194 = x_193.permute(0, 3, 1, 2);  x_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:161, code: if self.gamma is not None:
    getattr_getattr_l__mod___stages___2___blocks___6___gamma = self.getattr_getattr_L__mod___stages___2___blocks___6___gamma
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    reshape_12 = getattr_getattr_l__mod___stages___2___blocks___6___gamma.reshape(1, -1, 1, 1);  getattr_getattr_l__mod___stages___2___blocks___6___gamma = None
    x_195 = x_194.mul(reshape_12);  x_194 = reshape_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___2___blocks___6___drop_path = self.getattr_getattr_L__mod___stages___2___blocks___6___drop_path(x_195);  x_195 = None
    getattr_getattr_l__mod___stages___2___blocks___6___shortcut = self.getattr_getattr_L__mod___stages___2___blocks___6___shortcut(shortcut_12);  shortcut_12 = None
    shortcut_13 = getattr_getattr_l__mod___stages___2___blocks___6___drop_path + getattr_getattr_l__mod___stages___2___blocks___6___shortcut;  getattr_getattr_l__mod___stages___2___blocks___6___drop_path = getattr_getattr_l__mod___stages___2___blocks___6___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    x_197 = self.getattr_getattr_L__mod___stages___2___blocks___7___conv_dw(shortcut_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    x_198 = x_197.permute(0, 2, 3, 1);  x_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___7___norm_weight = self.getattr_getattr_L__mod___stages___2___blocks___7___norm_weight
    getattr_getattr_l__mod___stages___2___blocks___7___norm_bias = self.getattr_getattr_L__mod___stages___2___blocks___7___norm_bias
    x_200 = torch.nn.functional.layer_norm(x_198, (512,), getattr_getattr_l__mod___stages___2___blocks___7___norm_weight, getattr_getattr_l__mod___stages___2___blocks___7___norm_bias, 1e-06);  x_198 = getattr_getattr_l__mod___stages___2___blocks___7___norm_weight = getattr_getattr_l__mod___stages___2___blocks___7___norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_201 = self.getattr_getattr_L__mod___stages___2___blocks___7___mlp_fc1(x_200);  x_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    x_202 = torch._C._nn.gelu(x_201);  x_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_203 = self.getattr_getattr_L__mod___stages___2___blocks___7___mlp_drop1(x_202);  x_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_204 = self.getattr_getattr_L__mod___stages___2___blocks___7___mlp_norm(x_203);  x_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_205 = self.getattr_getattr_L__mod___stages___2___blocks___7___mlp_fc2(x_204);  x_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_207 = self.getattr_getattr_L__mod___stages___2___blocks___7___mlp_drop2(x_205);  x_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    x_208 = x_207.permute(0, 3, 1, 2);  x_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:161, code: if self.gamma is not None:
    getattr_getattr_l__mod___stages___2___blocks___7___gamma = self.getattr_getattr_L__mod___stages___2___blocks___7___gamma
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    reshape_13 = getattr_getattr_l__mod___stages___2___blocks___7___gamma.reshape(1, -1, 1, 1);  getattr_getattr_l__mod___stages___2___blocks___7___gamma = None
    x_209 = x_208.mul(reshape_13);  x_208 = reshape_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___2___blocks___7___drop_path = self.getattr_getattr_L__mod___stages___2___blocks___7___drop_path(x_209);  x_209 = None
    getattr_getattr_l__mod___stages___2___blocks___7___shortcut = self.getattr_getattr_L__mod___stages___2___blocks___7___shortcut(shortcut_13);  shortcut_13 = None
    shortcut_14 = getattr_getattr_l__mod___stages___2___blocks___7___drop_path + getattr_getattr_l__mod___stages___2___blocks___7___shortcut;  getattr_getattr_l__mod___stages___2___blocks___7___drop_path = getattr_getattr_l__mod___stages___2___blocks___7___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    x_211 = self.getattr_getattr_L__mod___stages___2___blocks___8___conv_dw(shortcut_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    x_212 = x_211.permute(0, 2, 3, 1);  x_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___8___norm_weight = self.getattr_getattr_L__mod___stages___2___blocks___8___norm_weight
    getattr_getattr_l__mod___stages___2___blocks___8___norm_bias = self.getattr_getattr_L__mod___stages___2___blocks___8___norm_bias
    x_214 = torch.nn.functional.layer_norm(x_212, (512,), getattr_getattr_l__mod___stages___2___blocks___8___norm_weight, getattr_getattr_l__mod___stages___2___blocks___8___norm_bias, 1e-06);  x_212 = getattr_getattr_l__mod___stages___2___blocks___8___norm_weight = getattr_getattr_l__mod___stages___2___blocks___8___norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_215 = self.getattr_getattr_L__mod___stages___2___blocks___8___mlp_fc1(x_214);  x_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    x_216 = torch._C._nn.gelu(x_215);  x_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_217 = self.getattr_getattr_L__mod___stages___2___blocks___8___mlp_drop1(x_216);  x_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_218 = self.getattr_getattr_L__mod___stages___2___blocks___8___mlp_norm(x_217);  x_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_219 = self.getattr_getattr_L__mod___stages___2___blocks___8___mlp_fc2(x_218);  x_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_221 = self.getattr_getattr_L__mod___stages___2___blocks___8___mlp_drop2(x_219);  x_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    x_222 = x_221.permute(0, 3, 1, 2);  x_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:161, code: if self.gamma is not None:
    getattr_getattr_l__mod___stages___2___blocks___8___gamma = self.getattr_getattr_L__mod___stages___2___blocks___8___gamma
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    reshape_14 = getattr_getattr_l__mod___stages___2___blocks___8___gamma.reshape(1, -1, 1, 1);  getattr_getattr_l__mod___stages___2___blocks___8___gamma = None
    x_223 = x_222.mul(reshape_14);  x_222 = reshape_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___2___blocks___8___drop_path = self.getattr_getattr_L__mod___stages___2___blocks___8___drop_path(x_223);  x_223 = None
    getattr_getattr_l__mod___stages___2___blocks___8___shortcut = self.getattr_getattr_L__mod___stages___2___blocks___8___shortcut(shortcut_14);  shortcut_14 = None
    shortcut_15 = getattr_getattr_l__mod___stages___2___blocks___8___drop_path + getattr_getattr_l__mod___stages___2___blocks___8___shortcut;  getattr_getattr_l__mod___stages___2___blocks___8___drop_path = getattr_getattr_l__mod___stages___2___blocks___8___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    x_225 = self.getattr_getattr_L__mod___stages___2___blocks___9___conv_dw(shortcut_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    x_226 = x_225.permute(0, 2, 3, 1);  x_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___9___norm_weight = self.getattr_getattr_L__mod___stages___2___blocks___9___norm_weight
    getattr_getattr_l__mod___stages___2___blocks___9___norm_bias = self.getattr_getattr_L__mod___stages___2___blocks___9___norm_bias
    x_228 = torch.nn.functional.layer_norm(x_226, (512,), getattr_getattr_l__mod___stages___2___blocks___9___norm_weight, getattr_getattr_l__mod___stages___2___blocks___9___norm_bias, 1e-06);  x_226 = getattr_getattr_l__mod___stages___2___blocks___9___norm_weight = getattr_getattr_l__mod___stages___2___blocks___9___norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_229 = self.getattr_getattr_L__mod___stages___2___blocks___9___mlp_fc1(x_228);  x_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    x_230 = torch._C._nn.gelu(x_229);  x_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_231 = self.getattr_getattr_L__mod___stages___2___blocks___9___mlp_drop1(x_230);  x_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_232 = self.getattr_getattr_L__mod___stages___2___blocks___9___mlp_norm(x_231);  x_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_233 = self.getattr_getattr_L__mod___stages___2___blocks___9___mlp_fc2(x_232);  x_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_235 = self.getattr_getattr_L__mod___stages___2___blocks___9___mlp_drop2(x_233);  x_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    x_236 = x_235.permute(0, 3, 1, 2);  x_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:161, code: if self.gamma is not None:
    getattr_getattr_l__mod___stages___2___blocks___9___gamma = self.getattr_getattr_L__mod___stages___2___blocks___9___gamma
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    reshape_15 = getattr_getattr_l__mod___stages___2___blocks___9___gamma.reshape(1, -1, 1, 1);  getattr_getattr_l__mod___stages___2___blocks___9___gamma = None
    x_237 = x_236.mul(reshape_15);  x_236 = reshape_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___2___blocks___9___drop_path = self.getattr_getattr_L__mod___stages___2___blocks___9___drop_path(x_237);  x_237 = None
    getattr_getattr_l__mod___stages___2___blocks___9___shortcut = self.getattr_getattr_L__mod___stages___2___blocks___9___shortcut(shortcut_15);  shortcut_15 = None
    shortcut_16 = getattr_getattr_l__mod___stages___2___blocks___9___drop_path + getattr_getattr_l__mod___stages___2___blocks___9___shortcut;  getattr_getattr_l__mod___stages___2___blocks___9___drop_path = getattr_getattr_l__mod___stages___2___blocks___9___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    x_239 = self.getattr_getattr_L__mod___stages___2___blocks___10___conv_dw(shortcut_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    x_240 = x_239.permute(0, 2, 3, 1);  x_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___10___norm_weight = self.getattr_getattr_L__mod___stages___2___blocks___10___norm_weight
    getattr_getattr_l__mod___stages___2___blocks___10___norm_bias = self.getattr_getattr_L__mod___stages___2___blocks___10___norm_bias
    x_242 = torch.nn.functional.layer_norm(x_240, (512,), getattr_getattr_l__mod___stages___2___blocks___10___norm_weight, getattr_getattr_l__mod___stages___2___blocks___10___norm_bias, 1e-06);  x_240 = getattr_getattr_l__mod___stages___2___blocks___10___norm_weight = getattr_getattr_l__mod___stages___2___blocks___10___norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_243 = self.getattr_getattr_L__mod___stages___2___blocks___10___mlp_fc1(x_242);  x_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    x_244 = torch._C._nn.gelu(x_243);  x_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_245 = self.getattr_getattr_L__mod___stages___2___blocks___10___mlp_drop1(x_244);  x_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_246 = self.getattr_getattr_L__mod___stages___2___blocks___10___mlp_norm(x_245);  x_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_247 = self.getattr_getattr_L__mod___stages___2___blocks___10___mlp_fc2(x_246);  x_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_249 = self.getattr_getattr_L__mod___stages___2___blocks___10___mlp_drop2(x_247);  x_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    x_250 = x_249.permute(0, 3, 1, 2);  x_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:161, code: if self.gamma is not None:
    getattr_getattr_l__mod___stages___2___blocks___10___gamma = self.getattr_getattr_L__mod___stages___2___blocks___10___gamma
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    reshape_16 = getattr_getattr_l__mod___stages___2___blocks___10___gamma.reshape(1, -1, 1, 1);  getattr_getattr_l__mod___stages___2___blocks___10___gamma = None
    x_251 = x_250.mul(reshape_16);  x_250 = reshape_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___2___blocks___10___drop_path = self.getattr_getattr_L__mod___stages___2___blocks___10___drop_path(x_251);  x_251 = None
    getattr_getattr_l__mod___stages___2___blocks___10___shortcut = self.getattr_getattr_L__mod___stages___2___blocks___10___shortcut(shortcut_16);  shortcut_16 = None
    shortcut_17 = getattr_getattr_l__mod___stages___2___blocks___10___drop_path + getattr_getattr_l__mod___stages___2___blocks___10___shortcut;  getattr_getattr_l__mod___stages___2___blocks___10___drop_path = getattr_getattr_l__mod___stages___2___blocks___10___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    x_253 = self.getattr_getattr_L__mod___stages___2___blocks___11___conv_dw(shortcut_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    x_254 = x_253.permute(0, 2, 3, 1);  x_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___11___norm_weight = self.getattr_getattr_L__mod___stages___2___blocks___11___norm_weight
    getattr_getattr_l__mod___stages___2___blocks___11___norm_bias = self.getattr_getattr_L__mod___stages___2___blocks___11___norm_bias
    x_256 = torch.nn.functional.layer_norm(x_254, (512,), getattr_getattr_l__mod___stages___2___blocks___11___norm_weight, getattr_getattr_l__mod___stages___2___blocks___11___norm_bias, 1e-06);  x_254 = getattr_getattr_l__mod___stages___2___blocks___11___norm_weight = getattr_getattr_l__mod___stages___2___blocks___11___norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_257 = self.getattr_getattr_L__mod___stages___2___blocks___11___mlp_fc1(x_256);  x_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    x_258 = torch._C._nn.gelu(x_257);  x_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_259 = self.getattr_getattr_L__mod___stages___2___blocks___11___mlp_drop1(x_258);  x_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_260 = self.getattr_getattr_L__mod___stages___2___blocks___11___mlp_norm(x_259);  x_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_261 = self.getattr_getattr_L__mod___stages___2___blocks___11___mlp_fc2(x_260);  x_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_263 = self.getattr_getattr_L__mod___stages___2___blocks___11___mlp_drop2(x_261);  x_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    x_264 = x_263.permute(0, 3, 1, 2);  x_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:161, code: if self.gamma is not None:
    getattr_getattr_l__mod___stages___2___blocks___11___gamma = self.getattr_getattr_L__mod___stages___2___blocks___11___gamma
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    reshape_17 = getattr_getattr_l__mod___stages___2___blocks___11___gamma.reshape(1, -1, 1, 1);  getattr_getattr_l__mod___stages___2___blocks___11___gamma = None
    x_265 = x_264.mul(reshape_17);  x_264 = reshape_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___2___blocks___11___drop_path = self.getattr_getattr_L__mod___stages___2___blocks___11___drop_path(x_265);  x_265 = None
    getattr_getattr_l__mod___stages___2___blocks___11___shortcut = self.getattr_getattr_L__mod___stages___2___blocks___11___shortcut(shortcut_17);  shortcut_17 = None
    shortcut_18 = getattr_getattr_l__mod___stages___2___blocks___11___drop_path + getattr_getattr_l__mod___stages___2___blocks___11___shortcut;  getattr_getattr_l__mod___stages___2___blocks___11___drop_path = getattr_getattr_l__mod___stages___2___blocks___11___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    x_267 = self.getattr_getattr_L__mod___stages___2___blocks___12___conv_dw(shortcut_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    x_268 = x_267.permute(0, 2, 3, 1);  x_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___12___norm_weight = self.getattr_getattr_L__mod___stages___2___blocks___12___norm_weight
    getattr_getattr_l__mod___stages___2___blocks___12___norm_bias = self.getattr_getattr_L__mod___stages___2___blocks___12___norm_bias
    x_270 = torch.nn.functional.layer_norm(x_268, (512,), getattr_getattr_l__mod___stages___2___blocks___12___norm_weight, getattr_getattr_l__mod___stages___2___blocks___12___norm_bias, 1e-06);  x_268 = getattr_getattr_l__mod___stages___2___blocks___12___norm_weight = getattr_getattr_l__mod___stages___2___blocks___12___norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_271 = self.getattr_getattr_L__mod___stages___2___blocks___12___mlp_fc1(x_270);  x_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    x_272 = torch._C._nn.gelu(x_271);  x_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_273 = self.getattr_getattr_L__mod___stages___2___blocks___12___mlp_drop1(x_272);  x_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_274 = self.getattr_getattr_L__mod___stages___2___blocks___12___mlp_norm(x_273);  x_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_275 = self.getattr_getattr_L__mod___stages___2___blocks___12___mlp_fc2(x_274);  x_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_277 = self.getattr_getattr_L__mod___stages___2___blocks___12___mlp_drop2(x_275);  x_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    x_278 = x_277.permute(0, 3, 1, 2);  x_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:161, code: if self.gamma is not None:
    getattr_getattr_l__mod___stages___2___blocks___12___gamma = self.getattr_getattr_L__mod___stages___2___blocks___12___gamma
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    reshape_18 = getattr_getattr_l__mod___stages___2___blocks___12___gamma.reshape(1, -1, 1, 1);  getattr_getattr_l__mod___stages___2___blocks___12___gamma = None
    x_279 = x_278.mul(reshape_18);  x_278 = reshape_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___2___blocks___12___drop_path = self.getattr_getattr_L__mod___stages___2___blocks___12___drop_path(x_279);  x_279 = None
    getattr_getattr_l__mod___stages___2___blocks___12___shortcut = self.getattr_getattr_L__mod___stages___2___blocks___12___shortcut(shortcut_18);  shortcut_18 = None
    shortcut_19 = getattr_getattr_l__mod___stages___2___blocks___12___drop_path + getattr_getattr_l__mod___stages___2___blocks___12___shortcut;  getattr_getattr_l__mod___stages___2___blocks___12___drop_path = getattr_getattr_l__mod___stages___2___blocks___12___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    x_281 = self.getattr_getattr_L__mod___stages___2___blocks___13___conv_dw(shortcut_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    x_282 = x_281.permute(0, 2, 3, 1);  x_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___13___norm_weight = self.getattr_getattr_L__mod___stages___2___blocks___13___norm_weight
    getattr_getattr_l__mod___stages___2___blocks___13___norm_bias = self.getattr_getattr_L__mod___stages___2___blocks___13___norm_bias
    x_284 = torch.nn.functional.layer_norm(x_282, (512,), getattr_getattr_l__mod___stages___2___blocks___13___norm_weight, getattr_getattr_l__mod___stages___2___blocks___13___norm_bias, 1e-06);  x_282 = getattr_getattr_l__mod___stages___2___blocks___13___norm_weight = getattr_getattr_l__mod___stages___2___blocks___13___norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_285 = self.getattr_getattr_L__mod___stages___2___blocks___13___mlp_fc1(x_284);  x_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    x_286 = torch._C._nn.gelu(x_285);  x_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_287 = self.getattr_getattr_L__mod___stages___2___blocks___13___mlp_drop1(x_286);  x_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_288 = self.getattr_getattr_L__mod___stages___2___blocks___13___mlp_norm(x_287);  x_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_289 = self.getattr_getattr_L__mod___stages___2___blocks___13___mlp_fc2(x_288);  x_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_291 = self.getattr_getattr_L__mod___stages___2___blocks___13___mlp_drop2(x_289);  x_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    x_292 = x_291.permute(0, 3, 1, 2);  x_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:161, code: if self.gamma is not None:
    getattr_getattr_l__mod___stages___2___blocks___13___gamma = self.getattr_getattr_L__mod___stages___2___blocks___13___gamma
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    reshape_19 = getattr_getattr_l__mod___stages___2___blocks___13___gamma.reshape(1, -1, 1, 1);  getattr_getattr_l__mod___stages___2___blocks___13___gamma = None
    x_293 = x_292.mul(reshape_19);  x_292 = reshape_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___2___blocks___13___drop_path = self.getattr_getattr_L__mod___stages___2___blocks___13___drop_path(x_293);  x_293 = None
    getattr_getattr_l__mod___stages___2___blocks___13___shortcut = self.getattr_getattr_L__mod___stages___2___blocks___13___shortcut(shortcut_19);  shortcut_19 = None
    shortcut_20 = getattr_getattr_l__mod___stages___2___blocks___13___drop_path + getattr_getattr_l__mod___stages___2___blocks___13___shortcut;  getattr_getattr_l__mod___stages___2___blocks___13___drop_path = getattr_getattr_l__mod___stages___2___blocks___13___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    x_295 = self.getattr_getattr_L__mod___stages___2___blocks___14___conv_dw(shortcut_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    x_296 = x_295.permute(0, 2, 3, 1);  x_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___14___norm_weight = self.getattr_getattr_L__mod___stages___2___blocks___14___norm_weight
    getattr_getattr_l__mod___stages___2___blocks___14___norm_bias = self.getattr_getattr_L__mod___stages___2___blocks___14___norm_bias
    x_298 = torch.nn.functional.layer_norm(x_296, (512,), getattr_getattr_l__mod___stages___2___blocks___14___norm_weight, getattr_getattr_l__mod___stages___2___blocks___14___norm_bias, 1e-06);  x_296 = getattr_getattr_l__mod___stages___2___blocks___14___norm_weight = getattr_getattr_l__mod___stages___2___blocks___14___norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_299 = self.getattr_getattr_L__mod___stages___2___blocks___14___mlp_fc1(x_298);  x_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    x_300 = torch._C._nn.gelu(x_299);  x_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_301 = self.getattr_getattr_L__mod___stages___2___blocks___14___mlp_drop1(x_300);  x_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_302 = self.getattr_getattr_L__mod___stages___2___blocks___14___mlp_norm(x_301);  x_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_303 = self.getattr_getattr_L__mod___stages___2___blocks___14___mlp_fc2(x_302);  x_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_305 = self.getattr_getattr_L__mod___stages___2___blocks___14___mlp_drop2(x_303);  x_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    x_306 = x_305.permute(0, 3, 1, 2);  x_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:161, code: if self.gamma is not None:
    getattr_getattr_l__mod___stages___2___blocks___14___gamma = self.getattr_getattr_L__mod___stages___2___blocks___14___gamma
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    reshape_20 = getattr_getattr_l__mod___stages___2___blocks___14___gamma.reshape(1, -1, 1, 1);  getattr_getattr_l__mod___stages___2___blocks___14___gamma = None
    x_307 = x_306.mul(reshape_20);  x_306 = reshape_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___2___blocks___14___drop_path = self.getattr_getattr_L__mod___stages___2___blocks___14___drop_path(x_307);  x_307 = None
    getattr_getattr_l__mod___stages___2___blocks___14___shortcut = self.getattr_getattr_L__mod___stages___2___blocks___14___shortcut(shortcut_20);  shortcut_20 = None
    shortcut_21 = getattr_getattr_l__mod___stages___2___blocks___14___drop_path + getattr_getattr_l__mod___stages___2___blocks___14___shortcut;  getattr_getattr_l__mod___stages___2___blocks___14___drop_path = getattr_getattr_l__mod___stages___2___blocks___14___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    x_309 = self.getattr_getattr_L__mod___stages___2___blocks___15___conv_dw(shortcut_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    x_310 = x_309.permute(0, 2, 3, 1);  x_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___15___norm_weight = self.getattr_getattr_L__mod___stages___2___blocks___15___norm_weight
    getattr_getattr_l__mod___stages___2___blocks___15___norm_bias = self.getattr_getattr_L__mod___stages___2___blocks___15___norm_bias
    x_312 = torch.nn.functional.layer_norm(x_310, (512,), getattr_getattr_l__mod___stages___2___blocks___15___norm_weight, getattr_getattr_l__mod___stages___2___blocks___15___norm_bias, 1e-06);  x_310 = getattr_getattr_l__mod___stages___2___blocks___15___norm_weight = getattr_getattr_l__mod___stages___2___blocks___15___norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_313 = self.getattr_getattr_L__mod___stages___2___blocks___15___mlp_fc1(x_312);  x_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    x_314 = torch._C._nn.gelu(x_313);  x_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_315 = self.getattr_getattr_L__mod___stages___2___blocks___15___mlp_drop1(x_314);  x_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_316 = self.getattr_getattr_L__mod___stages___2___blocks___15___mlp_norm(x_315);  x_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_317 = self.getattr_getattr_L__mod___stages___2___blocks___15___mlp_fc2(x_316);  x_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_319 = self.getattr_getattr_L__mod___stages___2___blocks___15___mlp_drop2(x_317);  x_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    x_320 = x_319.permute(0, 3, 1, 2);  x_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:161, code: if self.gamma is not None:
    getattr_getattr_l__mod___stages___2___blocks___15___gamma = self.getattr_getattr_L__mod___stages___2___blocks___15___gamma
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    reshape_21 = getattr_getattr_l__mod___stages___2___blocks___15___gamma.reshape(1, -1, 1, 1);  getattr_getattr_l__mod___stages___2___blocks___15___gamma = None
    x_321 = x_320.mul(reshape_21);  x_320 = reshape_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___2___blocks___15___drop_path = self.getattr_getattr_L__mod___stages___2___blocks___15___drop_path(x_321);  x_321 = None
    getattr_getattr_l__mod___stages___2___blocks___15___shortcut = self.getattr_getattr_L__mod___stages___2___blocks___15___shortcut(shortcut_21);  shortcut_21 = None
    shortcut_22 = getattr_getattr_l__mod___stages___2___blocks___15___drop_path + getattr_getattr_l__mod___stages___2___blocks___15___shortcut;  getattr_getattr_l__mod___stages___2___blocks___15___drop_path = getattr_getattr_l__mod___stages___2___blocks___15___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    x_323 = self.getattr_getattr_L__mod___stages___2___blocks___16___conv_dw(shortcut_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    x_324 = x_323.permute(0, 2, 3, 1);  x_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___16___norm_weight = self.getattr_getattr_L__mod___stages___2___blocks___16___norm_weight
    getattr_getattr_l__mod___stages___2___blocks___16___norm_bias = self.getattr_getattr_L__mod___stages___2___blocks___16___norm_bias
    x_326 = torch.nn.functional.layer_norm(x_324, (512,), getattr_getattr_l__mod___stages___2___blocks___16___norm_weight, getattr_getattr_l__mod___stages___2___blocks___16___norm_bias, 1e-06);  x_324 = getattr_getattr_l__mod___stages___2___blocks___16___norm_weight = getattr_getattr_l__mod___stages___2___blocks___16___norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_327 = self.getattr_getattr_L__mod___stages___2___blocks___16___mlp_fc1(x_326);  x_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    x_328 = torch._C._nn.gelu(x_327);  x_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_329 = self.getattr_getattr_L__mod___stages___2___blocks___16___mlp_drop1(x_328);  x_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_330 = self.getattr_getattr_L__mod___stages___2___blocks___16___mlp_norm(x_329);  x_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_331 = self.getattr_getattr_L__mod___stages___2___blocks___16___mlp_fc2(x_330);  x_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_333 = self.getattr_getattr_L__mod___stages___2___blocks___16___mlp_drop2(x_331);  x_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    x_334 = x_333.permute(0, 3, 1, 2);  x_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:161, code: if self.gamma is not None:
    getattr_getattr_l__mod___stages___2___blocks___16___gamma = self.getattr_getattr_L__mod___stages___2___blocks___16___gamma
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    reshape_22 = getattr_getattr_l__mod___stages___2___blocks___16___gamma.reshape(1, -1, 1, 1);  getattr_getattr_l__mod___stages___2___blocks___16___gamma = None
    x_335 = x_334.mul(reshape_22);  x_334 = reshape_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___2___blocks___16___drop_path = self.getattr_getattr_L__mod___stages___2___blocks___16___drop_path(x_335);  x_335 = None
    getattr_getattr_l__mod___stages___2___blocks___16___shortcut = self.getattr_getattr_L__mod___stages___2___blocks___16___shortcut(shortcut_22);  shortcut_22 = None
    shortcut_23 = getattr_getattr_l__mod___stages___2___blocks___16___drop_path + getattr_getattr_l__mod___stages___2___blocks___16___shortcut;  getattr_getattr_l__mod___stages___2___blocks___16___drop_path = getattr_getattr_l__mod___stages___2___blocks___16___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    x_337 = self.getattr_getattr_L__mod___stages___2___blocks___17___conv_dw(shortcut_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    x_338 = x_337.permute(0, 2, 3, 1);  x_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___17___norm_weight = self.getattr_getattr_L__mod___stages___2___blocks___17___norm_weight
    getattr_getattr_l__mod___stages___2___blocks___17___norm_bias = self.getattr_getattr_L__mod___stages___2___blocks___17___norm_bias
    x_340 = torch.nn.functional.layer_norm(x_338, (512,), getattr_getattr_l__mod___stages___2___blocks___17___norm_weight, getattr_getattr_l__mod___stages___2___blocks___17___norm_bias, 1e-06);  x_338 = getattr_getattr_l__mod___stages___2___blocks___17___norm_weight = getattr_getattr_l__mod___stages___2___blocks___17___norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_341 = self.getattr_getattr_L__mod___stages___2___blocks___17___mlp_fc1(x_340);  x_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    x_342 = torch._C._nn.gelu(x_341);  x_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_343 = self.getattr_getattr_L__mod___stages___2___blocks___17___mlp_drop1(x_342);  x_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_344 = self.getattr_getattr_L__mod___stages___2___blocks___17___mlp_norm(x_343);  x_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_345 = self.getattr_getattr_L__mod___stages___2___blocks___17___mlp_fc2(x_344);  x_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_347 = self.getattr_getattr_L__mod___stages___2___blocks___17___mlp_drop2(x_345);  x_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    x_348 = x_347.permute(0, 3, 1, 2);  x_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:161, code: if self.gamma is not None:
    getattr_getattr_l__mod___stages___2___blocks___17___gamma = self.getattr_getattr_L__mod___stages___2___blocks___17___gamma
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    reshape_23 = getattr_getattr_l__mod___stages___2___blocks___17___gamma.reshape(1, -1, 1, 1);  getattr_getattr_l__mod___stages___2___blocks___17___gamma = None
    x_349 = x_348.mul(reshape_23);  x_348 = reshape_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___2___blocks___17___drop_path = self.getattr_getattr_L__mod___stages___2___blocks___17___drop_path(x_349);  x_349 = None
    getattr_getattr_l__mod___stages___2___blocks___17___shortcut = self.getattr_getattr_L__mod___stages___2___blocks___17___shortcut(shortcut_23);  shortcut_23 = None
    shortcut_24 = getattr_getattr_l__mod___stages___2___blocks___17___drop_path + getattr_getattr_l__mod___stages___2___blocks___17___shortcut;  getattr_getattr_l__mod___stages___2___blocks___17___drop_path = getattr_getattr_l__mod___stages___2___blocks___17___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    x_351 = self.getattr_getattr_L__mod___stages___2___blocks___18___conv_dw(shortcut_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    x_352 = x_351.permute(0, 2, 3, 1);  x_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___18___norm_weight = self.getattr_getattr_L__mod___stages___2___blocks___18___norm_weight
    getattr_getattr_l__mod___stages___2___blocks___18___norm_bias = self.getattr_getattr_L__mod___stages___2___blocks___18___norm_bias
    x_354 = torch.nn.functional.layer_norm(x_352, (512,), getattr_getattr_l__mod___stages___2___blocks___18___norm_weight, getattr_getattr_l__mod___stages___2___blocks___18___norm_bias, 1e-06);  x_352 = getattr_getattr_l__mod___stages___2___blocks___18___norm_weight = getattr_getattr_l__mod___stages___2___blocks___18___norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_355 = self.getattr_getattr_L__mod___stages___2___blocks___18___mlp_fc1(x_354);  x_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    x_356 = torch._C._nn.gelu(x_355);  x_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_357 = self.getattr_getattr_L__mod___stages___2___blocks___18___mlp_drop1(x_356);  x_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_358 = self.getattr_getattr_L__mod___stages___2___blocks___18___mlp_norm(x_357);  x_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_359 = self.getattr_getattr_L__mod___stages___2___blocks___18___mlp_fc2(x_358);  x_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_361 = self.getattr_getattr_L__mod___stages___2___blocks___18___mlp_drop2(x_359);  x_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    x_362 = x_361.permute(0, 3, 1, 2);  x_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:161, code: if self.gamma is not None:
    getattr_getattr_l__mod___stages___2___blocks___18___gamma = self.getattr_getattr_L__mod___stages___2___blocks___18___gamma
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    reshape_24 = getattr_getattr_l__mod___stages___2___blocks___18___gamma.reshape(1, -1, 1, 1);  getattr_getattr_l__mod___stages___2___blocks___18___gamma = None
    x_363 = x_362.mul(reshape_24);  x_362 = reshape_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___2___blocks___18___drop_path = self.getattr_getattr_L__mod___stages___2___blocks___18___drop_path(x_363);  x_363 = None
    getattr_getattr_l__mod___stages___2___blocks___18___shortcut = self.getattr_getattr_L__mod___stages___2___blocks___18___shortcut(shortcut_24);  shortcut_24 = None
    shortcut_25 = getattr_getattr_l__mod___stages___2___blocks___18___drop_path + getattr_getattr_l__mod___stages___2___blocks___18___shortcut;  getattr_getattr_l__mod___stages___2___blocks___18___drop_path = getattr_getattr_l__mod___stages___2___blocks___18___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    x_365 = self.getattr_getattr_L__mod___stages___2___blocks___19___conv_dw(shortcut_25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    x_366 = x_365.permute(0, 2, 3, 1);  x_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___19___norm_weight = self.getattr_getattr_L__mod___stages___2___blocks___19___norm_weight
    getattr_getattr_l__mod___stages___2___blocks___19___norm_bias = self.getattr_getattr_L__mod___stages___2___blocks___19___norm_bias
    x_368 = torch.nn.functional.layer_norm(x_366, (512,), getattr_getattr_l__mod___stages___2___blocks___19___norm_weight, getattr_getattr_l__mod___stages___2___blocks___19___norm_bias, 1e-06);  x_366 = getattr_getattr_l__mod___stages___2___blocks___19___norm_weight = getattr_getattr_l__mod___stages___2___blocks___19___norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_369 = self.getattr_getattr_L__mod___stages___2___blocks___19___mlp_fc1(x_368);  x_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    x_370 = torch._C._nn.gelu(x_369);  x_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_371 = self.getattr_getattr_L__mod___stages___2___blocks___19___mlp_drop1(x_370);  x_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_372 = self.getattr_getattr_L__mod___stages___2___blocks___19___mlp_norm(x_371);  x_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_373 = self.getattr_getattr_L__mod___stages___2___blocks___19___mlp_fc2(x_372);  x_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_375 = self.getattr_getattr_L__mod___stages___2___blocks___19___mlp_drop2(x_373);  x_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    x_376 = x_375.permute(0, 3, 1, 2);  x_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:161, code: if self.gamma is not None:
    getattr_getattr_l__mod___stages___2___blocks___19___gamma = self.getattr_getattr_L__mod___stages___2___blocks___19___gamma
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    reshape_25 = getattr_getattr_l__mod___stages___2___blocks___19___gamma.reshape(1, -1, 1, 1);  getattr_getattr_l__mod___stages___2___blocks___19___gamma = None
    x_377 = x_376.mul(reshape_25);  x_376 = reshape_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___2___blocks___19___drop_path = self.getattr_getattr_L__mod___stages___2___blocks___19___drop_path(x_377);  x_377 = None
    getattr_getattr_l__mod___stages___2___blocks___19___shortcut = self.getattr_getattr_L__mod___stages___2___blocks___19___shortcut(shortcut_25);  shortcut_25 = None
    shortcut_26 = getattr_getattr_l__mod___stages___2___blocks___19___drop_path + getattr_getattr_l__mod___stages___2___blocks___19___shortcut;  getattr_getattr_l__mod___stages___2___blocks___19___drop_path = getattr_getattr_l__mod___stages___2___blocks___19___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    x_379 = self.getattr_getattr_L__mod___stages___2___blocks___20___conv_dw(shortcut_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    x_380 = x_379.permute(0, 2, 3, 1);  x_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___20___norm_weight = self.getattr_getattr_L__mod___stages___2___blocks___20___norm_weight
    getattr_getattr_l__mod___stages___2___blocks___20___norm_bias = self.getattr_getattr_L__mod___stages___2___blocks___20___norm_bias
    x_382 = torch.nn.functional.layer_norm(x_380, (512,), getattr_getattr_l__mod___stages___2___blocks___20___norm_weight, getattr_getattr_l__mod___stages___2___blocks___20___norm_bias, 1e-06);  x_380 = getattr_getattr_l__mod___stages___2___blocks___20___norm_weight = getattr_getattr_l__mod___stages___2___blocks___20___norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_383 = self.getattr_getattr_L__mod___stages___2___blocks___20___mlp_fc1(x_382);  x_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    x_384 = torch._C._nn.gelu(x_383);  x_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_385 = self.getattr_getattr_L__mod___stages___2___blocks___20___mlp_drop1(x_384);  x_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_386 = self.getattr_getattr_L__mod___stages___2___blocks___20___mlp_norm(x_385);  x_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_387 = self.getattr_getattr_L__mod___stages___2___blocks___20___mlp_fc2(x_386);  x_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_389 = self.getattr_getattr_L__mod___stages___2___blocks___20___mlp_drop2(x_387);  x_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    x_390 = x_389.permute(0, 3, 1, 2);  x_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:161, code: if self.gamma is not None:
    getattr_getattr_l__mod___stages___2___blocks___20___gamma = self.getattr_getattr_L__mod___stages___2___blocks___20___gamma
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    reshape_26 = getattr_getattr_l__mod___stages___2___blocks___20___gamma.reshape(1, -1, 1, 1);  getattr_getattr_l__mod___stages___2___blocks___20___gamma = None
    x_391 = x_390.mul(reshape_26);  x_390 = reshape_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___2___blocks___20___drop_path = self.getattr_getattr_L__mod___stages___2___blocks___20___drop_path(x_391);  x_391 = None
    getattr_getattr_l__mod___stages___2___blocks___20___shortcut = self.getattr_getattr_L__mod___stages___2___blocks___20___shortcut(shortcut_26);  shortcut_26 = None
    shortcut_27 = getattr_getattr_l__mod___stages___2___blocks___20___drop_path + getattr_getattr_l__mod___stages___2___blocks___20___shortcut;  getattr_getattr_l__mod___stages___2___blocks___20___drop_path = getattr_getattr_l__mod___stages___2___blocks___20___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    x_393 = self.getattr_getattr_L__mod___stages___2___blocks___21___conv_dw(shortcut_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    x_394 = x_393.permute(0, 2, 3, 1);  x_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___21___norm_weight = self.getattr_getattr_L__mod___stages___2___blocks___21___norm_weight
    getattr_getattr_l__mod___stages___2___blocks___21___norm_bias = self.getattr_getattr_L__mod___stages___2___blocks___21___norm_bias
    x_396 = torch.nn.functional.layer_norm(x_394, (512,), getattr_getattr_l__mod___stages___2___blocks___21___norm_weight, getattr_getattr_l__mod___stages___2___blocks___21___norm_bias, 1e-06);  x_394 = getattr_getattr_l__mod___stages___2___blocks___21___norm_weight = getattr_getattr_l__mod___stages___2___blocks___21___norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_397 = self.getattr_getattr_L__mod___stages___2___blocks___21___mlp_fc1(x_396);  x_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    x_398 = torch._C._nn.gelu(x_397);  x_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_399 = self.getattr_getattr_L__mod___stages___2___blocks___21___mlp_drop1(x_398);  x_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_400 = self.getattr_getattr_L__mod___stages___2___blocks___21___mlp_norm(x_399);  x_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_401 = self.getattr_getattr_L__mod___stages___2___blocks___21___mlp_fc2(x_400);  x_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_403 = self.getattr_getattr_L__mod___stages___2___blocks___21___mlp_drop2(x_401);  x_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    x_404 = x_403.permute(0, 3, 1, 2);  x_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:161, code: if self.gamma is not None:
    getattr_getattr_l__mod___stages___2___blocks___21___gamma = self.getattr_getattr_L__mod___stages___2___blocks___21___gamma
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    reshape_27 = getattr_getattr_l__mod___stages___2___blocks___21___gamma.reshape(1, -1, 1, 1);  getattr_getattr_l__mod___stages___2___blocks___21___gamma = None
    x_405 = x_404.mul(reshape_27);  x_404 = reshape_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___2___blocks___21___drop_path = self.getattr_getattr_L__mod___stages___2___blocks___21___drop_path(x_405);  x_405 = None
    getattr_getattr_l__mod___stages___2___blocks___21___shortcut = self.getattr_getattr_L__mod___stages___2___blocks___21___shortcut(shortcut_27);  shortcut_27 = None
    shortcut_28 = getattr_getattr_l__mod___stages___2___blocks___21___drop_path + getattr_getattr_l__mod___stages___2___blocks___21___shortcut;  getattr_getattr_l__mod___stages___2___blocks___21___drop_path = getattr_getattr_l__mod___stages___2___blocks___21___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    x_407 = self.getattr_getattr_L__mod___stages___2___blocks___22___conv_dw(shortcut_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    x_408 = x_407.permute(0, 2, 3, 1);  x_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___22___norm_weight = self.getattr_getattr_L__mod___stages___2___blocks___22___norm_weight
    getattr_getattr_l__mod___stages___2___blocks___22___norm_bias = self.getattr_getattr_L__mod___stages___2___blocks___22___norm_bias
    x_410 = torch.nn.functional.layer_norm(x_408, (512,), getattr_getattr_l__mod___stages___2___blocks___22___norm_weight, getattr_getattr_l__mod___stages___2___blocks___22___norm_bias, 1e-06);  x_408 = getattr_getattr_l__mod___stages___2___blocks___22___norm_weight = getattr_getattr_l__mod___stages___2___blocks___22___norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_411 = self.getattr_getattr_L__mod___stages___2___blocks___22___mlp_fc1(x_410);  x_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    x_412 = torch._C._nn.gelu(x_411);  x_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_413 = self.getattr_getattr_L__mod___stages___2___blocks___22___mlp_drop1(x_412);  x_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_414 = self.getattr_getattr_L__mod___stages___2___blocks___22___mlp_norm(x_413);  x_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_415 = self.getattr_getattr_L__mod___stages___2___blocks___22___mlp_fc2(x_414);  x_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_417 = self.getattr_getattr_L__mod___stages___2___blocks___22___mlp_drop2(x_415);  x_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    x_418 = x_417.permute(0, 3, 1, 2);  x_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:161, code: if self.gamma is not None:
    getattr_getattr_l__mod___stages___2___blocks___22___gamma = self.getattr_getattr_L__mod___stages___2___blocks___22___gamma
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    reshape_28 = getattr_getattr_l__mod___stages___2___blocks___22___gamma.reshape(1, -1, 1, 1);  getattr_getattr_l__mod___stages___2___blocks___22___gamma = None
    x_419 = x_418.mul(reshape_28);  x_418 = reshape_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___2___blocks___22___drop_path = self.getattr_getattr_L__mod___stages___2___blocks___22___drop_path(x_419);  x_419 = None
    getattr_getattr_l__mod___stages___2___blocks___22___shortcut = self.getattr_getattr_L__mod___stages___2___blocks___22___shortcut(shortcut_28);  shortcut_28 = None
    shortcut_29 = getattr_getattr_l__mod___stages___2___blocks___22___drop_path + getattr_getattr_l__mod___stages___2___blocks___22___shortcut;  getattr_getattr_l__mod___stages___2___blocks___22___drop_path = getattr_getattr_l__mod___stages___2___blocks___22___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    x_421 = self.getattr_getattr_L__mod___stages___2___blocks___23___conv_dw(shortcut_29)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    x_422 = x_421.permute(0, 2, 3, 1);  x_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___23___norm_weight = self.getattr_getattr_L__mod___stages___2___blocks___23___norm_weight
    getattr_getattr_l__mod___stages___2___blocks___23___norm_bias = self.getattr_getattr_L__mod___stages___2___blocks___23___norm_bias
    x_424 = torch.nn.functional.layer_norm(x_422, (512,), getattr_getattr_l__mod___stages___2___blocks___23___norm_weight, getattr_getattr_l__mod___stages___2___blocks___23___norm_bias, 1e-06);  x_422 = getattr_getattr_l__mod___stages___2___blocks___23___norm_weight = getattr_getattr_l__mod___stages___2___blocks___23___norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_425 = self.getattr_getattr_L__mod___stages___2___blocks___23___mlp_fc1(x_424);  x_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    x_426 = torch._C._nn.gelu(x_425);  x_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_427 = self.getattr_getattr_L__mod___stages___2___blocks___23___mlp_drop1(x_426);  x_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_428 = self.getattr_getattr_L__mod___stages___2___blocks___23___mlp_norm(x_427);  x_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_429 = self.getattr_getattr_L__mod___stages___2___blocks___23___mlp_fc2(x_428);  x_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_431 = self.getattr_getattr_L__mod___stages___2___blocks___23___mlp_drop2(x_429);  x_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    x_432 = x_431.permute(0, 3, 1, 2);  x_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:161, code: if self.gamma is not None:
    getattr_getattr_l__mod___stages___2___blocks___23___gamma = self.getattr_getattr_L__mod___stages___2___blocks___23___gamma
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    reshape_29 = getattr_getattr_l__mod___stages___2___blocks___23___gamma.reshape(1, -1, 1, 1);  getattr_getattr_l__mod___stages___2___blocks___23___gamma = None
    x_433 = x_432.mul(reshape_29);  x_432 = reshape_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___2___blocks___23___drop_path = self.getattr_getattr_L__mod___stages___2___blocks___23___drop_path(x_433);  x_433 = None
    getattr_getattr_l__mod___stages___2___blocks___23___shortcut = self.getattr_getattr_L__mod___stages___2___blocks___23___shortcut(shortcut_29);  shortcut_29 = None
    shortcut_30 = getattr_getattr_l__mod___stages___2___blocks___23___drop_path + getattr_getattr_l__mod___stages___2___blocks___23___shortcut;  getattr_getattr_l__mod___stages___2___blocks___23___drop_path = getattr_getattr_l__mod___stages___2___blocks___23___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    x_435 = self.getattr_getattr_L__mod___stages___2___blocks___24___conv_dw(shortcut_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    x_436 = x_435.permute(0, 2, 3, 1);  x_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___24___norm_weight = self.getattr_getattr_L__mod___stages___2___blocks___24___norm_weight
    getattr_getattr_l__mod___stages___2___blocks___24___norm_bias = self.getattr_getattr_L__mod___stages___2___blocks___24___norm_bias
    x_438 = torch.nn.functional.layer_norm(x_436, (512,), getattr_getattr_l__mod___stages___2___blocks___24___norm_weight, getattr_getattr_l__mod___stages___2___blocks___24___norm_bias, 1e-06);  x_436 = getattr_getattr_l__mod___stages___2___blocks___24___norm_weight = getattr_getattr_l__mod___stages___2___blocks___24___norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_439 = self.getattr_getattr_L__mod___stages___2___blocks___24___mlp_fc1(x_438);  x_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    x_440 = torch._C._nn.gelu(x_439);  x_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_441 = self.getattr_getattr_L__mod___stages___2___blocks___24___mlp_drop1(x_440);  x_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_442 = self.getattr_getattr_L__mod___stages___2___blocks___24___mlp_norm(x_441);  x_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_443 = self.getattr_getattr_L__mod___stages___2___blocks___24___mlp_fc2(x_442);  x_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_445 = self.getattr_getattr_L__mod___stages___2___blocks___24___mlp_drop2(x_443);  x_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    x_446 = x_445.permute(0, 3, 1, 2);  x_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:161, code: if self.gamma is not None:
    getattr_getattr_l__mod___stages___2___blocks___24___gamma = self.getattr_getattr_L__mod___stages___2___blocks___24___gamma
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    reshape_30 = getattr_getattr_l__mod___stages___2___blocks___24___gamma.reshape(1, -1, 1, 1);  getattr_getattr_l__mod___stages___2___blocks___24___gamma = None
    x_447 = x_446.mul(reshape_30);  x_446 = reshape_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___2___blocks___24___drop_path = self.getattr_getattr_L__mod___stages___2___blocks___24___drop_path(x_447);  x_447 = None
    getattr_getattr_l__mod___stages___2___blocks___24___shortcut = self.getattr_getattr_L__mod___stages___2___blocks___24___shortcut(shortcut_30);  shortcut_30 = None
    shortcut_31 = getattr_getattr_l__mod___stages___2___blocks___24___drop_path + getattr_getattr_l__mod___stages___2___blocks___24___shortcut;  getattr_getattr_l__mod___stages___2___blocks___24___drop_path = getattr_getattr_l__mod___stages___2___blocks___24___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    x_449 = self.getattr_getattr_L__mod___stages___2___blocks___25___conv_dw(shortcut_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    x_450 = x_449.permute(0, 2, 3, 1);  x_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___25___norm_weight = self.getattr_getattr_L__mod___stages___2___blocks___25___norm_weight
    getattr_getattr_l__mod___stages___2___blocks___25___norm_bias = self.getattr_getattr_L__mod___stages___2___blocks___25___norm_bias
    x_452 = torch.nn.functional.layer_norm(x_450, (512,), getattr_getattr_l__mod___stages___2___blocks___25___norm_weight, getattr_getattr_l__mod___stages___2___blocks___25___norm_bias, 1e-06);  x_450 = getattr_getattr_l__mod___stages___2___blocks___25___norm_weight = getattr_getattr_l__mod___stages___2___blocks___25___norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_453 = self.getattr_getattr_L__mod___stages___2___blocks___25___mlp_fc1(x_452);  x_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    x_454 = torch._C._nn.gelu(x_453);  x_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_455 = self.getattr_getattr_L__mod___stages___2___blocks___25___mlp_drop1(x_454);  x_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_456 = self.getattr_getattr_L__mod___stages___2___blocks___25___mlp_norm(x_455);  x_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_457 = self.getattr_getattr_L__mod___stages___2___blocks___25___mlp_fc2(x_456);  x_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_459 = self.getattr_getattr_L__mod___stages___2___blocks___25___mlp_drop2(x_457);  x_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    x_460 = x_459.permute(0, 3, 1, 2);  x_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:161, code: if self.gamma is not None:
    getattr_getattr_l__mod___stages___2___blocks___25___gamma = self.getattr_getattr_L__mod___stages___2___blocks___25___gamma
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    reshape_31 = getattr_getattr_l__mod___stages___2___blocks___25___gamma.reshape(1, -1, 1, 1);  getattr_getattr_l__mod___stages___2___blocks___25___gamma = None
    x_461 = x_460.mul(reshape_31);  x_460 = reshape_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___2___blocks___25___drop_path = self.getattr_getattr_L__mod___stages___2___blocks___25___drop_path(x_461);  x_461 = None
    getattr_getattr_l__mod___stages___2___blocks___25___shortcut = self.getattr_getattr_L__mod___stages___2___blocks___25___shortcut(shortcut_31);  shortcut_31 = None
    shortcut_32 = getattr_getattr_l__mod___stages___2___blocks___25___drop_path + getattr_getattr_l__mod___stages___2___blocks___25___shortcut;  getattr_getattr_l__mod___stages___2___blocks___25___drop_path = getattr_getattr_l__mod___stages___2___blocks___25___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    x_463 = self.getattr_getattr_L__mod___stages___2___blocks___26___conv_dw(shortcut_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    x_464 = x_463.permute(0, 2, 3, 1);  x_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___26___norm_weight = self.getattr_getattr_L__mod___stages___2___blocks___26___norm_weight
    getattr_getattr_l__mod___stages___2___blocks___26___norm_bias = self.getattr_getattr_L__mod___stages___2___blocks___26___norm_bias
    x_466 = torch.nn.functional.layer_norm(x_464, (512,), getattr_getattr_l__mod___stages___2___blocks___26___norm_weight, getattr_getattr_l__mod___stages___2___blocks___26___norm_bias, 1e-06);  x_464 = getattr_getattr_l__mod___stages___2___blocks___26___norm_weight = getattr_getattr_l__mod___stages___2___blocks___26___norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_467 = self.getattr_getattr_L__mod___stages___2___blocks___26___mlp_fc1(x_466);  x_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    x_468 = torch._C._nn.gelu(x_467);  x_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_469 = self.getattr_getattr_L__mod___stages___2___blocks___26___mlp_drop1(x_468);  x_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_470 = self.getattr_getattr_L__mod___stages___2___blocks___26___mlp_norm(x_469);  x_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_471 = self.getattr_getattr_L__mod___stages___2___blocks___26___mlp_fc2(x_470);  x_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_473 = self.getattr_getattr_L__mod___stages___2___blocks___26___mlp_drop2(x_471);  x_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    x_474 = x_473.permute(0, 3, 1, 2);  x_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:161, code: if self.gamma is not None:
    getattr_getattr_l__mod___stages___2___blocks___26___gamma = self.getattr_getattr_L__mod___stages___2___blocks___26___gamma
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    reshape_32 = getattr_getattr_l__mod___stages___2___blocks___26___gamma.reshape(1, -1, 1, 1);  getattr_getattr_l__mod___stages___2___blocks___26___gamma = None
    x_475 = x_474.mul(reshape_32);  x_474 = reshape_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___2___blocks___26___drop_path = self.getattr_getattr_L__mod___stages___2___blocks___26___drop_path(x_475);  x_475 = None
    getattr_getattr_l__mod___stages___2___blocks___26___shortcut = self.getattr_getattr_L__mod___stages___2___blocks___26___shortcut(shortcut_32);  shortcut_32 = None
    x_477 = getattr_getattr_l__mod___stages___2___blocks___26___drop_path + getattr_getattr_l__mod___stages___2___blocks___26___shortcut;  getattr_getattr_l__mod___stages___2___blocks___26___drop_path = getattr_getattr_l__mod___stages___2___blocks___26___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:68, code: x = x.permute(0, 2, 3, 1)
    x_478 = x_477.permute(0, 2, 3, 1);  x_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___3___downsample___0___weight = self.getattr_getattr_L__mod___stages___3___downsample___0___weight
    getattr_getattr_l__mod___stages___3___downsample___0___bias = self.getattr_getattr_L__mod___stages___3___downsample___0___bias
    x_479 = torch.nn.functional.layer_norm(x_478, (512,), getattr_getattr_l__mod___stages___3___downsample___0___weight, getattr_getattr_l__mod___stages___3___downsample___0___bias, 1e-06);  x_478 = getattr_getattr_l__mod___stages___3___downsample___0___weight = getattr_getattr_l__mod___stages___3___downsample___0___bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:73, code: x = x.permute(0, 3, 1, 2)
    x_480 = x_479.permute(0, 3, 1, 2);  x_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:229, code: x = self.downsample(x)
    shortcut_33 = self.getattr_L__mod___stages___3___downsample_1(x_480);  x_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    x_482 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_dw(shortcut_33)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    x_483 = x_482.permute(0, 2, 3, 1);  x_482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___3___blocks___0___norm_weight = self.getattr_getattr_L__mod___stages___3___blocks___0___norm_weight
    getattr_getattr_l__mod___stages___3___blocks___0___norm_bias = self.getattr_getattr_L__mod___stages___3___blocks___0___norm_bias
    x_485 = torch.nn.functional.layer_norm(x_483, (1024,), getattr_getattr_l__mod___stages___3___blocks___0___norm_weight, getattr_getattr_l__mod___stages___3___blocks___0___norm_bias, 1e-06);  x_483 = getattr_getattr_l__mod___stages___3___blocks___0___norm_weight = getattr_getattr_l__mod___stages___3___blocks___0___norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_486 = self.getattr_getattr_L__mod___stages___3___blocks___0___mlp_fc1(x_485);  x_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    x_487 = torch._C._nn.gelu(x_486);  x_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_488 = self.getattr_getattr_L__mod___stages___3___blocks___0___mlp_drop1(x_487);  x_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_489 = self.getattr_getattr_L__mod___stages___3___blocks___0___mlp_norm(x_488);  x_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_490 = self.getattr_getattr_L__mod___stages___3___blocks___0___mlp_fc2(x_489);  x_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_492 = self.getattr_getattr_L__mod___stages___3___blocks___0___mlp_drop2(x_490);  x_490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    x_493 = x_492.permute(0, 3, 1, 2);  x_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:161, code: if self.gamma is not None:
    getattr_getattr_l__mod___stages___3___blocks___0___gamma = self.getattr_getattr_L__mod___stages___3___blocks___0___gamma
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    reshape_33 = getattr_getattr_l__mod___stages___3___blocks___0___gamma.reshape(1, -1, 1, 1);  getattr_getattr_l__mod___stages___3___blocks___0___gamma = None
    x_494 = x_493.mul(reshape_33);  x_493 = reshape_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___3___blocks___0___drop_path = self.getattr_getattr_L__mod___stages___3___blocks___0___drop_path(x_494);  x_494 = None
    getattr_getattr_l__mod___stages___3___blocks___0___shortcut = self.getattr_getattr_L__mod___stages___3___blocks___0___shortcut(shortcut_33);  shortcut_33 = None
    shortcut_34 = getattr_getattr_l__mod___stages___3___blocks___0___drop_path + getattr_getattr_l__mod___stages___3___blocks___0___shortcut;  getattr_getattr_l__mod___stages___3___blocks___0___drop_path = getattr_getattr_l__mod___stages___3___blocks___0___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    x_496 = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_dw(shortcut_34)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    x_497 = x_496.permute(0, 2, 3, 1);  x_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___3___blocks___1___norm_weight = self.getattr_getattr_L__mod___stages___3___blocks___1___norm_weight
    getattr_getattr_l__mod___stages___3___blocks___1___norm_bias = self.getattr_getattr_L__mod___stages___3___blocks___1___norm_bias
    x_499 = torch.nn.functional.layer_norm(x_497, (1024,), getattr_getattr_l__mod___stages___3___blocks___1___norm_weight, getattr_getattr_l__mod___stages___3___blocks___1___norm_bias, 1e-06);  x_497 = getattr_getattr_l__mod___stages___3___blocks___1___norm_weight = getattr_getattr_l__mod___stages___3___blocks___1___norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_500 = self.getattr_getattr_L__mod___stages___3___blocks___1___mlp_fc1(x_499);  x_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    x_501 = torch._C._nn.gelu(x_500);  x_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_502 = self.getattr_getattr_L__mod___stages___3___blocks___1___mlp_drop1(x_501);  x_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_503 = self.getattr_getattr_L__mod___stages___3___blocks___1___mlp_norm(x_502);  x_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_504 = self.getattr_getattr_L__mod___stages___3___blocks___1___mlp_fc2(x_503);  x_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_506 = self.getattr_getattr_L__mod___stages___3___blocks___1___mlp_drop2(x_504);  x_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    x_507 = x_506.permute(0, 3, 1, 2);  x_506 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:161, code: if self.gamma is not None:
    getattr_getattr_l__mod___stages___3___blocks___1___gamma = self.getattr_getattr_L__mod___stages___3___blocks___1___gamma
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    reshape_34 = getattr_getattr_l__mod___stages___3___blocks___1___gamma.reshape(1, -1, 1, 1);  getattr_getattr_l__mod___stages___3___blocks___1___gamma = None
    x_508 = x_507.mul(reshape_34);  x_507 = reshape_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___3___blocks___1___drop_path = self.getattr_getattr_L__mod___stages___3___blocks___1___drop_path(x_508);  x_508 = None
    getattr_getattr_l__mod___stages___3___blocks___1___shortcut = self.getattr_getattr_L__mod___stages___3___blocks___1___shortcut(shortcut_34);  shortcut_34 = None
    shortcut_35 = getattr_getattr_l__mod___stages___3___blocks___1___drop_path + getattr_getattr_l__mod___stages___3___blocks___1___shortcut;  getattr_getattr_l__mod___stages___3___blocks___1___drop_path = getattr_getattr_l__mod___stages___3___blocks___1___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    x_510 = self.getattr_getattr_L__mod___stages___3___blocks___2___conv_dw(shortcut_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    x_511 = x_510.permute(0, 2, 3, 1);  x_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___3___blocks___2___norm_weight = self.getattr_getattr_L__mod___stages___3___blocks___2___norm_weight
    getattr_getattr_l__mod___stages___3___blocks___2___norm_bias = self.getattr_getattr_L__mod___stages___3___blocks___2___norm_bias
    x_513 = torch.nn.functional.layer_norm(x_511, (1024,), getattr_getattr_l__mod___stages___3___blocks___2___norm_weight, getattr_getattr_l__mod___stages___3___blocks___2___norm_bias, 1e-06);  x_511 = getattr_getattr_l__mod___stages___3___blocks___2___norm_weight = getattr_getattr_l__mod___stages___3___blocks___2___norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_514 = self.getattr_getattr_L__mod___stages___3___blocks___2___mlp_fc1(x_513);  x_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    x_515 = torch._C._nn.gelu(x_514);  x_514 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_516 = self.getattr_getattr_L__mod___stages___3___blocks___2___mlp_drop1(x_515);  x_515 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_517 = self.getattr_getattr_L__mod___stages___3___blocks___2___mlp_norm(x_516);  x_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_518 = self.getattr_getattr_L__mod___stages___3___blocks___2___mlp_fc2(x_517);  x_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_520 = self.getattr_getattr_L__mod___stages___3___blocks___2___mlp_drop2(x_518);  x_518 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    x_521 = x_520.permute(0, 3, 1, 2);  x_520 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:161, code: if self.gamma is not None:
    getattr_getattr_l__mod___stages___3___blocks___2___gamma = self.getattr_getattr_L__mod___stages___3___blocks___2___gamma
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    reshape_35 = getattr_getattr_l__mod___stages___3___blocks___2___gamma.reshape(1, -1, 1, 1);  getattr_getattr_l__mod___stages___3___blocks___2___gamma = None
    x_522 = x_521.mul(reshape_35);  x_521 = reshape_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___3___blocks___2___drop_path = self.getattr_getattr_L__mod___stages___3___blocks___2___drop_path(x_522);  x_522 = None
    getattr_getattr_l__mod___stages___3___blocks___2___shortcut = self.getattr_getattr_L__mod___stages___3___blocks___2___shortcut(shortcut_35);  shortcut_35 = None
    x_525 = getattr_getattr_l__mod___stages___3___blocks___2___drop_path + getattr_getattr_l__mod___stages___3___blocks___2___shortcut;  getattr_getattr_l__mod___stages___3___blocks___2___drop_path = getattr_getattr_l__mod___stages___3___blocks___2___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:413, code: x = self.norm_pre(x)
    x_527 = self.L__mod___norm_pre(x_525);  x_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    x_528 = self.L__mod___head_global_pool_pool(x_527);  x_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    x_530 = self.L__mod___head_global_pool_flatten(x_528);  x_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:68, code: x = x.permute(0, 2, 3, 1)
    x_531 = x_530.permute(0, 2, 3, 1);  x_530 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___head_norm_weight = self.L__mod___head_norm_weight
    l__mod___head_norm_bias = self.L__mod___head_norm_bias
    x_532 = torch.nn.functional.layer_norm(x_531, (1024,), l__mod___head_norm_weight, l__mod___head_norm_bias, 1e-06);  x_531 = l__mod___head_norm_weight = l__mod___head_norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:73, code: x = x.permute(0, 3, 1, 2)
    x_534 = x_532.permute(0, 3, 1, 2);  x_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:202, code: x = self.flatten(x)
    x_535 = self.L__mod___head_flatten(x_534);  x_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:203, code: x = self.pre_logits(x)
    x_536 = self.L__mod___head_pre_logits(x_535);  x_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:204, code: x = self.drop(x)
    x_537 = self.L__mod___head_drop(x_536);  x_536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:207, code: x = self.fc(x)
    x_539 = self.L__mod___head_fc(x_537);  x_537 = None
    return (x_539,)
    